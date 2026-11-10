"""Export one final continuum checkpoint to SI Gaussian columns without floors."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from . import pilot
from observation.columns.model import Columns


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_columns(run_directory, output):
    source = Path(run_directory)
    paths = {name: source/name for name in ('checkpoint.npz', 'config.json', 'summary.json', 'executed_source.py')}
    hashes = {name: digest(path) for name, path in paths.items()}
    config = json.loads(paths['config.json'].read_text())
    summary = json.loads(paths['summary.json'].read_text())
    expected_config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if summary['config'] != config or summary['config_sha256'] != expected_config_hash:
        raise ValueError('Checkpoint configuration provenance mismatch')
    if summary['source_sha256'] != hashes['executed_source.py']:
        raise ValueError('Executed source provenance mismatch')
    if hashes['executed_source.py'] != digest(pilot.__file__):
        raise ValueError('Executed source is incompatible with the current pilot interpreter')
    c = pilot.Config(**config)
    if c.ring_gravity or summary['collision_closure'] != 'finite_size_enskog_control':
        raise ValueError('Only the declared finite-size no-ring-gravity control is supported')
    with np.load(paths['checkpoint.npz'], allow_pickle=False) as checkpoint:
        q, x, y = (checkpoint[key].copy() for key in ('state', 'x', 'y'))
    gx, gy, dx, dy = pilot.geometry(c)
    if q.shape != (10, c.ny, c.nx) or not np.array_equal(x, gx) or not np.array_equal(y, gy):
        raise ValueError('Checkpoint state/grid does not match configuration')
    if not pilot.valid(q, x)[0]:
        raise ValueError('Checkpoint is not numerically realizable')
    if np.any(q[0] <= 0):
        raise ValueError('Vacuum export is unsupported: Gaussian H is undefined; no floor is supplied')
    n, _, _, _, _, _, _, h, _, _ = pilot.primitive(q, x)
    radius, density = .5, 500.
    sigma = (4*np.pi*radius**3*density/3)*pilot.N0*n
    height = np.sqrt(h)*pilot.L
    mean_z = np.zeros_like(n)
    Columns(np.r_[x.ravel()-dx/2, x.ravel()[-1]+dx/2]*pilot.L,
            np.r_[y.ravel()-dy/2, y.ravel()[-1]+dy/2]*pilot.L, sigma, height, mean_z)
    if any(digest(paths[name]) != value for name, value in hashes.items()):
        raise RuntimeError('Input changed during SI export')
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination/'columns.npz', x_m=x*pilot.L, y_m=y*pilot.L,
                        Sigma_kg_m2=sigma, H_m=height, meanZ_m=mean_z)
    validity = dict(numerical_realizability=True, physical_regime_accepted=False,
                    height_or_brightness_calibration=False, coherent_bending=False,
                    ring_gravity=False, meanZ_policy='Exactly zero: coplanar centered-column model',
                    vacuum_policy='Rejected; no undefined H replacement or density floor',
                    source_input_sha256=hashes, source_time_omega=summary['time_omega'],
                    completion=summary['completion'], export_source_sha256=digest(__file__),
                    constants=dict(N0_per_m2=pilot.N0, length_unit_m=pilot.L,
                                   particle_radius_m=radius, material_density_kg_m3=density),
                    limitations='Numerical transient control; not validated Keeler height or calibrated brightness')
    (destination/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (destination/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (destination/'validity.json').write_text(json.dumps(validity, indent=2)+'\n')
    return validity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(export_columns(args.run, args.out), indent=2))
