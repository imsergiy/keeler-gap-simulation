"""SI transfer uses raw extensive moments and rejects unsupported input."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from continuum import pilot
from continuum.export import export_columns
from observation.columns.model import load_columns


class ExportTests(unittest.TestCase):
    def make_run(self, root):
        root.mkdir()
        c = pilot.Config(nx=3, ny=4, yhalf=200., y_center_km=-100.)
        x, y, _, _ = pilot.geometry(c)
        q = pilot.initial(x, y, c)
        config = asdict(c)
        source = Path(pilot.__file__).read_bytes()
        summary = dict(config=config, config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                       source_sha256=hashlib.sha256(source).hexdigest(), collision_closure='finite_size_enskog_control',
                       time_omega=.1, completion='wall_cap')
        (root/'executed_source.py').write_bytes(source)
        for name, value in (('config', config), ('summary', summary)):
            (root/(name+'.json')).write_text(json.dumps(value))
        np.savez(root/'checkpoint.npz', state=q, x=x, y=y)
        return q, x, y

    def test_exact_transfer_units_original_bytes_and_partial_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'run'
            q, x, y = self.make_run(source)
            original = (source/'checkpoint.npz').read_bytes()
            validity = export_columns(source, root/'export')
            with np.load(root/'export/columns.npz') as fields:
                np.testing.assert_array_equal(fields['x_m'], x*1000)
                np.testing.assert_array_equal(fields['y_m'], y*1000)
                np.testing.assert_allclose(fields['Sigma_kg_m2'], (4*np.pi*.5**3*500/3)*pilot.N0*q[0], rtol=1e-15)
                np.testing.assert_allclose(fields['H_m'], np.sqrt(q[6]/q[0])*1000, rtol=1e-15)
                np.testing.assert_array_equal(fields['meanZ_m'], 0.)
            self.assertEqual((source/'checkpoint.npz').read_bytes(), original)
            self.assertEqual(validity['completion'], 'wall_cap')
            self.assertFalse(validity['physical_regime_accepted'])
            load_columns(root/'export/columns.npz')

    def test_provenance_and_undefined_vacuum_rejected_before_output(self):
        for change in ('config', 'source', 'vacuum'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root/'run'
                q, x, y = self.make_run(source)
                if change == 'config':
                    path = source/'config.json'; value = json.loads(path.read_text()); value['width'] = 4.; path.write_text(json.dumps(value))
                elif change == 'source':
                    path = source/'executed_source.py'; path.write_bytes(path.read_bytes()+b'\n')
                else:
                    q[:, 1, 1] = 0.; np.savez(source/'checkpoint.npz', state=q, x=x, y=y)
                with self.assertRaises(ValueError):
                    export_columns(source, root/'export')
                self.assertFalse((root/'export').exists())
