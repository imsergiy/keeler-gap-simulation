"""Fast finite-source, conservative transport and failure-contract controls."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad
from collision.native.backend import NativeGaussianEnskog
from continuum import pilot as p
from continuum.contact_operator import ContactOperator, cayley, weighted_gradient, stress_force


class ContinuumTests(unittest.TestCase):
    def test_fractional_tag_and_constant_fraction_flux(self):
        c = p.Config(nx=8, ny=16, yhalf=200., y_center_km=-100.)
        x, y, dx, dy = p.geometry(c)
        q = p.initial(x, y, c)
        expected = 30*quad(lambda xx: 1/(1+np.exp(-2*(xx-20)/c.width)), 20, 21)[0]
        self.assertAlmostEqual(q[9].sum()*dx*dy, expected, places=11)
        self.assertTrue(np.all((q[9] >= 0) & (q[9] <= q[0])))
        q[9] = .37*q[0]
        faces = p.limited_faces(q, q[:, :, :1], q[:, :, -1:], q[:, :1, :], q[:, -1:, :])[:4]
        fluid_faces = p.limited_faces_fluid(q[:9], q[:9, :, :1], q[:9, :, -1:], q[:9, :1, :], q[:9, -1:, :])[:4]
        for field, fluid in zip(faces, fluid_faces):
            np.testing.assert_array_equal(field[:9], fluid)
            np.testing.assert_allclose(field[9], .37*field[0], atol=1e-16)
            self.assertTrue(p.valid(field, x)[0])
        for axis in (0, 1):
            flux = p.face(faces[0], faces[1], x, axis)
            np.testing.assert_allclose(flux[9], .37*flux[0], atol=1e-13)

    def test_face_contraction_preserves_cell_averages(self):
        c = p.Config(nx=5, ny=5, moon=False)
        x, y, _, _ = p.geometry(c); q = p.initial(x, y, c); q /= q[0]
        w = np.broadcast_to(np.linspace(-.2, .2, 5)[None, :], q[0].shape)
        q[2] = w; q[4] = q[0]*.005**2/.75/(4*np.sqrt(2))
        q[5] = w*w+.5*.005**2/.75
        xm, xp, ym, yp, stats = p.limited_faces(q, q[:, :, :1], q[:, :, -1:], q[:, :1, :], q[:, -1:, :])
        self.assertGreater(stats['limited_cells'], 0)
        for face in (xm, xp, ym, yp):
            self.assertTrue(np.all(p.valid_cells(face)))
        np.testing.assert_allclose((xm+xp)/2, q, atol=1e-17, rtol=0)
        np.testing.assert_allclose((ym+yp)/2, q, atol=1e-17, rtol=0)

    def test_transport_budgets_vertical_rotation_and_vacuum(self):
        c = p.Config(nx=8, ny=8, moon=False, collisions=False)
        x, y, dx, dy = p.geometry(c); q = p.initial(x, y, c)
        tr, boundary, _, jr, bj, _ = p.transport(q, c, x, y, dx, dy)
        self.assertLess(abs(tr[0].sum()*dx*dy+boundary[0]), 1e-10)
        self.assertLess(abs(jr.sum()*dx*dy+bj), 1e-9)
        q[6] = q[0]*.00004; q[7] = q[0]*.000008; q[8] = q[0]*.00003
        t = .37
        z, _ = p.source_step(q, c, x, y, t)
        M = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        expected = M@np.array([[.00004, .000008], [.000008, .00003]])@M.T
        actual = np.array([[z[6, 1, 1], z[7, 1, 1]], [z[7, 1, 1], z[8, 1, 1]]])/q[0, 1, 1]
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-18)
        self.assertTrue(p.valid(q*0, x)[0])
        bad = q*0; bad[1] = 1
        self.assertFalse(p.valid(bad, x)[0])

    def test_contact_boundary_work_and_positive_map(self):
        rng = np.random.default_rng(57)
        N = rng.uniform(.1, 2, (10, 12)); N[4, 5] = 0
        U = rng.normal(size=(10, 12, 2)); r = rng.normal(size=(10, 12, 3, 3))
        T = N[..., None, None]*(r@r.swapaxes(-1, -2))
        A = weighted_gradient(N, U, .3, .7)
        F, _, be = stress_force(N[1:-1, 1:-1], T[1:-1, 1:-1], U[1:-1, 1:-1], .3, .7)
        residual = (np.sum(U[2:-2, 2:-2]*F[..., :2])-np.sum(T[2:-2, 2:-2]*A[1:-1, 1:-1]))*.3*.7+be
        self.assertLess(abs(residual), 1e-12)
        np.testing.assert_array_equal(F[2, 3], 0)
        matrices = []
        for _ in range(3):
            r = rng.normal(size=(8, 3, 3)); matrices.append(r@r.swapaxes(-1, -2))
        for dt in (.01, .3, 3.):
            self.assertGreaterEqual(np.linalg.eigvalsh(cayley(*matrices, dt)).min(), 0)
        c = p.Config(nx=8, ny=8, xlo=25, xhi=45)
        x, y, dx, dy = p.geometry(c); q = p.initial(x, y, c); q[:, 3, 3] = 0
        native = NativeGaussianEnskog()
        errors = []
        for dt in (.02, .01, .005):
            op = ContactOperator(native.loss_gain, physics=p); z = q.copy(); residual = 0.; boundary = 0.
            for _ in range(round(.04/dt)):
                z, _ = op(z, c, x, dt)
                residual += op.last['energy_residual']; boundary += op.last['canonical_boundary']
                np.testing.assert_array_equal(z[:, 3, 3], 0.)
                self.assertTrue(p.valid(z, x)[0])
            self.assertLess(abs(np.sum(z[2]-q[2])*dx*dy+boundary), 1e-10)
            errors.append(abs(residual))
        self.assertLess(errors[-1], errors[0])

    def test_wall_cap_checkpoint_and_unique_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'bounded'
            c = p.Config(nx=8, ny=8, end=1., moon=False, collisions=False,
                         output=str(out), max_wall_seconds=1e-9)
            operator = ContactOperator(NativeGaussianEnskog().loss_gain, physics=p)
            summary = p.run(c, operator)
            self.assertEqual(summary['completion'], 'wall_cap')
            self.assertLess(summary['time_omega'], c.end)
            self.assertTrue((out/'checkpoint_latest.npz').is_file())
            self.assertTrue((out/'executed_source.py').is_file())
            self.assertLess(abs(summary['mass_residual']), 1e-9)
            self.assertLess(abs(summary['tag_mass_residual']), 1e-10)
            with self.assertRaises(FileExistsError):
                p.run(c, operator)


class BoundaryReconstructionTests(unittest.TestCase):
    def test_face_data_do_not_create_hot_near_vacuum_boundary_state(self):
        c = p.Config(nx=5, ny=5, moon=False, collisions=False)
        x, y, dx, dy = p.geometry(c)
        q = np.zeros((10, 5, 5))
        q[0] = np.array([1., 5., 9., 13., 17.])*1e-6
        q[3] = np.array([1., 1.5, 2., 2.5, 3.])*1e-6
        q[5] = .5*q[0]; q[6] = q[0]; q[8] = .7*q[0]; q[9] = .2*q[0]
        self.assertTrue(p.valid(q, x)[0])
        xm, xp, ym, yp, _ = p.limited_faces(q, q[:, :, :1]*0, q[:, :, -1:], q[:, :1, :], q[:, -1:, :])
        np.testing.assert_array_equal(xm[:9, :, 0], q[:9, :, 0])
        np.testing.assert_array_equal(xp[:9, :, -1], q[:9, :, -1])
        np.testing.assert_array_equal(ym[:9, 0, :], q[:9, 0, :])
        np.testing.assert_array_equal(yp[:9, -1, :], q[:9, -1, :])
        _, boundary_speed = p.flux(xm[:, :, :1], x[:, :1], 0)
        self.assertLess(boundary_speed.max(), 2.)
        np.testing.assert_allclose((xm+xp)/2, q, atol=1e-20, rtol=0)
        np.testing.assert_allclose((ym+yp)/2, q, atol=1e-20, rtol=0)
        tr, boundary, _, jr, bj, _ = p.transport(q, c, x, y, dx, dy)
        self.assertLess(abs(tr[0].sum()*dx*dy+boundary[0]), 1e-10)
        self.assertLess(abs(tr[9].sum()*dx*dy+boundary[9]), 1e-10)
        self.assertLess(abs(jr.sum()*dx*dy+bj), 1e-9)
