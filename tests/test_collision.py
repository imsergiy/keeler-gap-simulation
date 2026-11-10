"""Independent angular/moment identities and selected native parity controls."""
import unittest
import numpy as np
from scipy.integrate import quad
from collision.source import GaussianEnskog, truncated_moments
from collision.native.backend import NativeGaussianEnskog


class CollisionTests(unittest.TestCase):
    def test_angular_integral_and_truncated_gaussian(self):
        for order in (7, 11, 17):
            rule = GaussianEnskog(order)
            n, w = rule.normals, rule.weights
            self.assertAlmostEqual(w.sum(), 4*np.pi, places=12)
            np.testing.assert_allclose(np.einsum('n,ni,nj->ij', w, n, n),
                                       4*np.pi/3*np.eye(3), atol=2e-14)
            self.assertAlmostEqual(np.sum(w*n[:, 0]**4), 4*np.pi/5, places=12)
        native = NativeGaussianEnskog()
        for mean in (-8., -4.01, -4., 0., 3.):
            expected = [quad(lambda x: x**j*np.exp(-.5*(x-mean)**2)/np.sqrt(2*np.pi),
                             0, np.inf, epsabs=1e-28, epsrel=1e-10)[0] for j in (1, 2, 3)]
            np.testing.assert_allclose(native.normal_moments(mean, 1.), expected, rtol=2e-10)
            np.testing.assert_allclose(np.array(truncated_moments(mean, 1.)), expected, rtol=2e-10)
        np.testing.assert_array_equal(native.normal_moments(2., 0.), [2., 4., 8.])
        np.testing.assert_array_equal(native.normal_moments(-2., 0.), [0., 0., 0.])

    def test_native_reference_energy_loss_gain_and_psd(self):
        shear = np.zeros((3, 3)); shear[1, 0] = -1.5
        states = [
            (np.diag([2., .5, 1.]), shear, 1., .1, 2., .5, 1.),
            (np.array([[2., .2, 0], [.2, 1., .1], [0, .1, .7]]), shear, .5, .3, .2, 1., 1.7),
            (np.zeros((3, 3)), shear, 1., .1, 3., .5, 1.),
            (1e-4*np.eye(3), 100*np.eye(3), 1., .01, 1., .5, 1.),
            (np.eye(3), shear, 1., 0., 1., .5, 1.),
        ]
        for order, scheme in ((7, 'lebedev'), (17, 'lebedev'), (16, 'legendre')):
            reference = GaussianEnskog(order, scheme)
            native = NativeGaussianEnskog(order, scheme)
            for args in states:
                with self.subTest(order=order, scheme=scheme, height=args[4]):
                    C, A = args[:2]
                    expected, actual = reference.loss_gain(*args), native.loss_gain(*args)
                    for key in expected:
                        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-10, atol=1e-14)
                    np.testing.assert_allclose(-actual['L']@C-C@actual['L']+actual['B'],
                                               actual['Q'], rtol=2e-12, atol=1e-13)
                    for key in ('L', 'B'):
                        self.assertGreaterEqual(np.linalg.eigvalsh(actual[key]).min(),
                                                -1e-12*max(1., np.linalg.norm(actual[key])))
                    residual = .5*np.trace(actual['Q'])+np.sum(actual['P']*A)+actual['heat']
                    self.assertLess(abs(residual), 1e-11*max(1., abs(actual['heat'])))
        for evaluator in (GaussianEnskog(13), NativeGaussianEnskog(13)):
            with self.assertRaisesRegex(ValueError, 'nonnegative angular weights'):
                evaluator.loss_gain(*states[0])
        with self.assertRaises(ValueError):
            NativeGaussianEnskog()(np.diag([-1., 1., 1.]), shear, 1., .1, 1., .5)

    def test_isotropic_elastic_limit_and_explicit_threads(self):
        C, A = 2*np.eye(3), np.zeros((3, 3))
        args = (np.stack([C, C]), A, 1., .1, 1e9, 1.)
        result = NativeGaussianEnskog()(*args)
        np.testing.assert_allclose(result['Q'], 0., atol=1e-23)
        np.testing.assert_array_equal(result['heat'], 0.)
        prefactor = .1/(2*np.sqrt(np.pi)*1e9)
        np.testing.assert_allclose(result['rate'], 16*np.sqrt(np.pi)*prefactor*np.sqrt(2), rtol=1e-14)
        for threads in (2, 4):
            threaded = NativeGaussianEnskog(threads=threads)(*args)
            for key in result:
                np.testing.assert_array_equal(result[key], threaded[key])
