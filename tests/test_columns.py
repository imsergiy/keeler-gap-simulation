"""Analytic controls for Gaussian source and fixed detector integration."""
import unittest
from observation.columns.checks import run as pencil_checks
from observation.columns.adaptive_checks import run as detector_checks


class ColumnTests(unittest.TestCase):
    def test_pencil_source_analytic_controls(self):
        pencil_checks()

    def test_detector_area_and_source_failure_controls(self):
        detector_checks()


class OpticalDepthTests(unittest.TestCase):
    def test_nonuniform_crossings_axes_and_signed_tails(self):
        import numpy as np
        from scipy.special import ndtr
        from observation.columns.model import Columns, direction
        scene = Columns(np.array([-3., -.4, 2., 7.]), np.array([-5., .2, 4.]),
                        np.array([[0., 10., 60.], [30., 4., 0.]]),
                        np.array([[.1, 1., 4.], [3., .2, 2.]]),
                        np.array([[-2., 0., 3.], [1., -.5, 0.]]))
        rng = np.random.default_rng(947)
        rays = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, -1, 0),
                (1e-15, 0, 1), (1, 0, 1e-15), (1, 2, -3)]
        rays.extend(rng.normal(size=(40, 3)))
        for vector in rays:
            ray = direction(vector)
            for point in (np.array([0., 0., 0.]), np.array([2., .2, 12.]), np.array([-4., 1., -10.])):
                expected = sum(scene.segment_tau(point, ray, *segment)
                               for segment in scene.segments(point, ray, 0))
                actual = scene.optical_depth(point, ray)
                self.assertTrue(np.isfinite(actual))
                self.assertGreaterEqual(actual, 0.)
                self.assertAlmostEqual(actual, expected, delta=2e-14*max(1., abs(expected)))
        slab = Columns(np.array([-1., 1.]), np.array([-1., 1.]),
                       np.array([[100.]]), np.ones((1, 1)))
        for z, sign in ((8., 1.), (-8., -1.)):
            actual = slab.optical_depth(np.array([0., 0., z]), np.array([0., 0., sign]))
            self.assertAlmostEqual(actual/(.3*ndtr(-8.)), 1., delta=1e-13)
        self.assertEqual(slab.optical_depth(np.array([2., 0., 0.]), np.array([0., 0., 1.])), 0.)
