"""Offline epoch, frame, local-basis and immutable-response controls."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from data.geometry import derive


class EpochGeometryTests(unittest.TestCase):
    def test_fixed_utc_basis_phase_and_local_observer(self):
        result = derive()
        self.assertEqual(result['api_response_version'], '1.2')
        expected_times = ('2006-Oct-27 23:44:10.2160 UTC', '2009-May-24 09:28:39.9610 UTC')
        for epoch, expected in zip(result['epochs'], expected_times):
            self.assertEqual(epoch['epoch_utc'], expected)
            basis = np.array(epoch['basis_rows_radial_prograde_north'])
            np.testing.assert_allclose(basis@basis.T, np.eye(3), atol=3e-16)
            self.assertAlmostEqual(np.linalg.det(basis), 1., places=14)
            geometry = epoch['footpoint_geometry']
            sun = np.array(geometry['sun_unit_radial_prograde_north'])
            observer = np.array(geometry['observer_unit_radial_prograde_north'])
            self.assertAlmostEqual(np.linalg.norm(sun), 1., places=14)
            self.assertAlmostEqual(np.linalg.norm(observer), 1., places=14)
            self.assertAlmostEqual(sun@observer, np.cos(np.deg2rad(geometry['phase_deg'])), places=14)
            self.assertNotIn('longitude_from_node_deg', epoch)
        geometry = result['epochs'][1]['footpoint_geometry']
        self.assertAlmostEqual(geometry['observer_elevation_deg'], -52.557599, places=5)
        self.assertGreater(abs(geometry['observer_elevation_deg']-(-57.507)), 4.)
        self.assertAlmostEqual(result['epochs'][1]['outer_radius136522_minus_rho_km'], 17.904073, places=5)

    def test_changed_response_fails_offline_hash_check(self):
        source = Path(__file__).resolve().parents[1]/'data/epochs'
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            for name in ('daphnis.json', 'cassini.json', 'sun.json', 'queries.json'):
                shutil.copy(source/name, destination/name)
            path = destination/'daphnis.json'
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                derive(destination)
