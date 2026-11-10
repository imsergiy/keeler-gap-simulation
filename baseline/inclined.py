"""Physical-amplitude eccentric/inclined native global control; no observed-phase fit."""
from pathlib import Path
import time, json, hashlib
import numpy as np
from . import run as b
INC = np.deg2rad(0.0036)
ECC = 3.31e-05
OMEGA = 2 * np.pi / b.PERIOD
END = 21.85

def run(fdeg, udeg, steps=12000, inc=INC, moon=True, *, output, end_tau=END):
    if type(steps) is not int or not 256 <= steps <= 24000:
        raise ValueError('steps must be an integer in [256,24000]')
    if not np.all(np.isfinite([fdeg, udeg, inc, end_tau])) or not 0 < end_tau <= END or type(moon) is not bool:
        raise ValueError('require finite angles, Boolean moon and 0 < end_tau <= 21.85')
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    x, y = np.meshgrid(np.linspace(20000, 21000, 3), np.linspace(60000, 90000, 21), indexing='ij')
    x = x.ravel()
    y = y.ravel()
    w = 1 / (1 + np.exp(-(x - 20000) / 1000))
    w[(x == 20000) | (x == 21000)] *= 0.5
    w[(y == 60000) | (y == 90000)] *= 0.5
    w /= w.sum()
    s = b.rebound.Simulation()
    s.G = b.G
    s.integrator = 'whfast'
    s.integrator.coordinates = 'JACOBI'
    s.integrator.safe_mode = 1
    s.integrator.corrector = 0
    s.add(m=b.M_SATURN)
    s.add(m=b.M_DAPHNIS if moon else 0, a=b.A_MOON, e=ECC, inc=inc, omega=np.deg2rad(udeg - fdeg), f=np.deg2rad(fdeg), primary=s.particles[0])
    theta = np.arctan2(s.particles[1].y, s.particles[1].x)
    for xx, yy in zip(x, y):
        s.add(m=0, a=b.A_MOON + xx, f=theta + yy / b.A_MOON, primary=s.particles[0])
    s.N_active = 2
    s.testparticle_type = 0
    s.move_to_com()
    s.dt = end_tau / OMEGA / steps
    state = np.empty((len(x) + 2, 6))
    out = {k: [] for k in ['tau', 'z_m', 'vz_m_s', 'a_m', 'radius_m', 'arc_m', 'moon_relative_xyz_m', 'pair_specific_energy', 'pair_specific_L']}
    closest = np.full(len(x), np.inf)
    valid = np.ones(len(x), bool)
    min_moon_interval_margin = np.inf
    min_saturn_interval_margin = np.inf
    mu = b.G * b.M_SATURN
    mum = b.G * (b.M_DAPHNIS if moon else 0)
    accbound = 2 * mu / b.R_SATURN ** 2 + mum / b.R_MOON ** 2
    for i in range(steps + 1):
        s.serialize_particle_data(xyzvxvyvz=state)
        r = state[2:, :3] - state[0, :3]
        v = state[2:, 3:] - state[0, 3:]
        rm = state[1, :3] - state[0, :3]
        vm = state[1, 3:] - state[0, 3:]
        d = state[2:, :3] - state[1, :3]
        dv = state[2:, 3:] - state[1, 3:]
        rad = np.linalg.norm(r, axis=1)
        dist = np.linalg.norm(d, axis=1)
        closest = np.minimum(closest, dist)
        valid &= (dist > b.R_MOON) & (rad > b.R_SATURN)
        if i:
            moon_margin = dist - np.linalg.norm(dv, axis=1) * s.dt - 0.5 * accbound * s.dt * s.dt - b.R_MOON
            saturn_margin = rad - np.linalg.norm(v, axis=1) * s.dt - 0.5 * accbound * s.dt * s.dt - b.R_SATURN
            min_moon_interval_margin = min(min_moon_interval_margin, float(moon_margin.min()))
            min_saturn_interval_margin = min(min_saturn_interval_margin, float(saturn_margin.min()))
            valid &= (moon_margin > 0) & (saturn_margin > 0)
        if i % 10 == 0 or i == steps:
            angle = np.arctan2(r[:, 1], r[:, 0]) - np.arctan2(rm[1], rm[0])
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            vals = [i * end_tau / steps, r[:, 2].copy(), v[:, 2].copy(), 1 / (2 / rad - (v * v).sum(axis=1) / mu), rad, angle * b.A_MOON, rm.copy(), 0.5 * (vm @ vm) - (mu + mum) / np.linalg.norm(rm), np.cross(rm, vm)]
            for k, value in zip(out, vals):
                out[k].append(value)
        if i < steps:
            s.steps(1)
    arr = {k: np.array(v) for k, v in out.items()}
    arr.update(x0_m=x, y0_m=y, weights=w, closest_m=closest, valid=valid)
    z = arr['z_m']
    mean = z @ w
    rms = np.sqrt(z * z @ w)
    spread = np.sqrt(np.maximum(z * z @ w - mean ** 2, 0))
    idx = np.unravel_index(np.argmax(abs(z)), z.shape)
    ir = int(np.argmax(rms))
    ic = int(np.argmax(abs(mean)))
    iss = int(np.argmax(spread))
    report = dict(
        config=dict(f_deg=fdeg, u_deg=udeg, inc_rad=inc, e=ECC, moon=moon, steps=steps, end_tau=end_tau, initial_reference='Ring plane is z=0. Tracers circular around Saturn, a=A+x; initial azimuth relative to projected moon = y/A. Moon node longitude 0; omega=u-f. No epoch mapping.'),
        wall_s=time.perf_counter() - start,
        max_abs_z_m=float(abs(z[idx])),
        max_abs_z_tau=float(arr['tau'][idx[0]]),
        max_abs_z_seconds=float(arr['tau'][idx[0]] / OMEGA),
        max_weighted_rms_z_m=float(rms[ir]),
        max_rms_tau=float(arr['tau'][ir]),
        max_abs_cohort_mean_z_m=float(abs(mean[ic])),
        max_abs_mean_tau=float(arr['tau'][ic]),
        max_centered_cohort_spread_m=float(spread[iss]),
        max_spread_tau=float(arr['tau'][iss]),
        final_mean_z_m=float(mean[-1]),
        final_rms_z_m=float(rms[-1]),
        moon_max_abs_z_m=float(abs(arr['moon_relative_xyz_m'][:, 2]).max()),
        minimum_moon_distance_m=float(closest.min()),
        invalid_count=int((~valid).sum()),
        min_moon_interval_clearance_m=min_moon_interval_margin,
        min_saturn_interval_clearance_m=min_saturn_interval_margin,
        surface_contract='Backward per-step travel bound uses current relative speed and current distance, with acceleration bounded outside both enclosing surfaces.',
        max_abs_delta_a_m=float(abs(arr['a_m'] - arr['a_m'][0]).max()),
        pair_energy_relative_drift=float(abs(arr['pair_specific_energy'] / arr['pair_specific_energy'][0] - 1).max()),
        pair_L_vector_relative_drift=float((np.linalg.norm(arr['pair_specific_L'] - arr['pair_specific_L'][0], axis=1) / np.linalg.norm(arr['pair_specific_L'][0])).max()),
        jacobi_check='Not valid: eccentric pair has no single rigid rotation rate; no Jacobi conservation claim.',
    )
    report.update(scope='Collision-free spherical-Saturn orientation control; coherent bending and cohort spread are not local thermal thickness.', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), baseline_sha256=hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest(), versions=dict(rebound=b.rebound.__version__, numpy=np.__version__))
    np.savez_compressed(destination / 'trajectories.npz', **arr)
    (destination / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    return report

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--f-deg', type=float, default=0.)
    parser.add_argument('--u-deg', type=float, default=90.)
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--end-tau', type=float, default=END)
    parser.add_argument('--coplanar', action='store_true')
    parser.add_argument('--no-moon', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.f_deg, args.u_deg, args.steps,
                         0. if args.coplanar else INC, not args.no_moon,
                         output=args.out, end_tau=args.end_tau), indent=2))
