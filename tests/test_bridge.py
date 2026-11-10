import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from observation.bridge import render_artifact,complete_area,digest
from observation.columns.model import geometry,lambert_phase


class BridgeTest(unittest.TestCase):
    def test_actual_fields_profiles_and_failure_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sigma=np.array([[10.,20.],[30.,40.]])
            np.savez(root/'columns.npz',x_m=np.array([[-5e5,5e5]]),y_m=np.array([[-5e5],[5e5]]),Sigma_kg_m2=sigma,H_m=np.array([[1.,5.],[500.,1000.]]),meanZ_m=np.zeros((2,2)))
            for name,data in [('summary',{'time_omega':.35}),('config',{'ring_gravity':False}),('validity',{'height_or_brightness_calibration':False})]:
                (root/(name+'.json')).write_text(json.dumps(data))
            def run(out,**options):
                with contextlib.redirect_stdout(io.StringIO()):
                    return render_artifact(root/'columns.npz',root/out,root/'summary.json',root/'config.json',root/'validity.json',nx=2,ny=2,**options)
            report=run('valid');sun,obs=geometry(75,15,90)
            expected=.5*lambert_phase(np.deg2rad(75))/4*sun[2]/(sun[2]+obs[2])*(-np.expm1(-.003*sigma*(1/sun[2]+1/obs[2])))
            with np.load(root/'valid/maps_profiles.npz') as raw:
                np.testing.assert_allclose(raw['sub4_I_over_F'],expected,atol=1e-10,rtol=0)
                np.testing.assert_allclose(raw['sub4_profile_I_over_F'],expected.mean(axis=0),atol=1e-10,rtol=0)
                self.assertTrue(raw['profile_source_converged'].all())
            self.assertEqual(digest(root/'columns.npz'),digest(root/'valid/input_columns.npz'))
            self.assertEqual(report['input_field_stats']['H_max_m'],1000.)
            self.assertEqual(report['source_status'],'converged')
            frozen={name:digest(root/'valid'/name) for name in ('report.json','maps_profiles.npz')}
            for name in ('input_columns.npz','config.json','preview_bridge_source.py'):
                path=root/'valid'/name;original=path.read_bytes()
                with self.subTest(modified=name):
                    if name=='input_columns.npz':
                        with np.load(path) as f:changed={k:f[k] for k in f.files}
                        changed['Sigma_kg_m2']*=2
                        np.savez(path,**changed)
                    elif name=='config.json':
                        changed=json.loads(path.read_text());changed['omega']=.9;path.write_text(json.dumps(changed))
                    else:path.write_bytes(original+b'\n')
                    with patch('observation.bridge.detector_pixels') as compute:
                        with self.assertRaisesRegex(RuntimeError,'identity mismatch'):complete_area(root/'valid')
                        compute.assert_not_called()
                    path.write_bytes(original)
                    for file,saved in frozen.items():self.assertEqual(digest(root/'valid'/file),saved)
            from observation import bridge
            import observation.columns.model as renderer
            for changed_path in (Path(bridge.__file__).resolve(),Path(renderer.__file__).resolve()):
                with self.subTest(modified=str(changed_path)):
                    with patch('observation.bridge.digest',side_effect=lambda p: '0'*64 if Path(p).resolve()==changed_path else digest(p)):
                        with patch('observation.bridge.detector_pixels') as compute:
                            with self.assertRaisesRegex(RuntimeError,'identity mismatch'):complete_area(root/'valid')
                            compute.assert_not_called()
                    for file,saved in frozen.items():self.assertEqual(digest(root/'valid'/file),saved)

            failed=run('failed',source_epsabs=1e-30)
            self.assertTrue(failed['source_status'].startswith('NONCONVERGENCE'))
            self.assertIn('NONCONVERGENCE',failed['status'])
            with np.load(root/'failed/maps_profiles.npz') as raw:
                self.assertGreater(raw['sub4_failed_subrays'].sum(),0)
                self.assertFalse(raw['profile_source_converged'].all())


    def test_area_completion_keeps_same_footprint_and_budget_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sigma=np.tile(np.array([10.,20.,50.,100.]),(2,1))
            np.savez(root/'columns.npz',x_m=np.array([[-1.5e6,-.5e6,.5e6,1.5e6]]),y_m=np.array([[-5e5],[5e5]]),Sigma_kg_m2=sigma,H_m=np.ones((2,4)),meanZ_m=np.zeros((2,4)))
            for name,data in [('summary',{'time_omega':.35}),('config',{'ring_gravity':False}),('validity',{'height_or_brightness_calibration':False})]:
                (root/(name+'.json')).write_text(json.dumps(data))
            with contextlib.redirect_stdout(io.StringIO()):
                render_artifact(root/'columns.npz',root/'out',root/'summary.json',root/'config.json',root/'validity.json',nx=1,ny=1,subray_pair=(1,2),area_target=1e-8)
                stopped=complete_area(root/'out',max_seconds=1e-12)
                self.assertTrue(stopped['adaptive_area_completion']['budget_stopped'])
                self.assertFalse(stopped['adaptive_area_completion']['target_met_by_successive_estimate'])
                finished=complete_area(root/'out')
            self.assertTrue(finished['adaptive_area_completion']['target_met_by_successive_estimate'])
            sun,obs=geometry(75,15,90)
            expected=np.mean(.5*lambert_phase(np.deg2rad(75))/4*sun[2]/(sun[2]+obs[2])*(-np.expm1(-.003*sigma*(1/sun[2]+1/obs[2]))))
            with np.load(root/'out/maps_profiles.npz') as raw:
                self.assertAlmostEqual(raw['final_I_over_F'][0,0],expected,places=10)
                self.assertEqual(raw['final_subrays_per_axis'][0,0],16)
                self.assertIn('sub1_I_over_F',raw.files)
                self.assertIn('sub2_I_over_F',raw.files)
                self.assertTrue(raw['final_profile_source_converged'].all())


if __name__=='__main__':unittest.main()
