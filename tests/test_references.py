"""Independent physical controls for the bounded reference components."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from baseline.run import Config, run as orbit
from validation.run import run as contact
from observation.light import render, unit


class ReferenceTests(unittest.TestCase):
    def test_no_moon_matches_analytic_circular_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            result = orbit(Config(radial_samples=2, phase_samples=3,
                                  periods=1, steps_per_orbit=4096,
                                  outputs_per_orbit=16, moon=False), directory)
            self.assertLess(result['maximum_delta_a_abs_m'], .001)
            self.assertLess(result['no_moon_analytic_max_position_error_m'], .01)
            self.assertEqual(result['invalid_surface_count'], 0)
            with np.load(Path(directory)/'trajectories.npz') as data:
                self.assertTrue(np.isnan(data['time_since_closest_s'][:, ~data['encountered']]).all())

    def test_isolated_pair_restitution_and_energy_partition(self):
        case = dict(x=[[-.501, 0, 0], [.501, 0, 0]],
                    v=[[.05, 0, 0], [-.05, 0, 0]], r=[.5, .5], e=.5, end=.1)
        result = contact(case)
        self.assertFalse(result['failures'])
        np.testing.assert_allclose(result['v'], [[-.025, 0, 0], [.025, 0, 0]], atol=1e-7)
        initial_energy = result['m'][0]*.05**2
        self.assertAlmostEqual(result['K']/initial_energy, .25, places=6)
        self.assertAlmostEqual(result['Q']/initial_energy, .75, places=6)

    def test_lambert_phase_and_occultation(self):
        for phase in (0, np.pi/2):
            image, area = render([([0, 0, 0], 1., .6)],
                                 (np.sin(phase), 0, np.cos(phase)), n=128, width=2.2)
            expected = 2*.6*15/3*(np.sin(phase)+(np.pi-phase)*np.cos(phase))/np.pi
            self.assertLess(abs(image.sum()*area/expected-1), .005)
        image, _ = render([([0, 0, 0], .5, .6), ([0, 0, 2], .5, 0)], (0, 0, 1), n=32)
        self.assertEqual(image.max(), 0)

    def test_invalid_numerical_inputs_fail(self):
        with self.assertRaises(ValueError):
            Config(steps_per_orbit=257).validate()
        for direction in ((0, 0, 0), (np.nan, 0, 1)):
            with self.assertRaises(ValueError):
                unit(direction)


if __name__ == '__main__':
    unittest.main()
