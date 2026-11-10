"""Native 3D gravity controls; no forced height or thermal interpretation."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from baseline.inclined import run


class InclinedTests(unittest.TestCase):
    def test_planar_zero_and_reflection_with_timestep_refinement(self):
        with tempfile.TemporaryDirectory() as directory:
            def sample(name, orientation=90., steps=1000, **options):
                path = Path(directory)/name
                metrics = run(0., orientation, steps, output=path, end_tau=1., **options)
                self.assertEqual(metrics['invalid_count'], 0)
                self.assertGreater(metrics['min_moon_interval_clearance_m'], 0)
                self.assertGreater(metrics['min_saturn_interval_clearance_m'], 0)
                with np.load(path/'trajectories.npz') as archive:
                    return archive['z_m'].copy(), archive['vz_m_s'].copy()
            planar, planar_v = sample('planar', inc=0.)
            np.testing.assert_array_equal(planar, 0.)
            np.testing.assert_array_equal(planar_v, 0.)
            positive, _ = sample('positive')
            reflected, _ = sample('reflected', orientation=270.)
            fine, _ = sample('fine', steps=2000)
            self.assertGreater(np.max(np.abs(positive)), .01)
            np.testing.assert_allclose(positive, -reflected, atol=1e-4, rtol=0)
            np.testing.assert_allclose(positive, fine[::2], atol=1e-3, rtol=0)
