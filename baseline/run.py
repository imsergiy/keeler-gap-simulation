"""Collision-free global Daphnis control; SI units, fixed quadrature tracers."""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, json, time, resource, platform
import numpy as np
import rebound

G = 6.67430e-11
M_SATURN = 3.793120623e16 / G
M_DAPHNIS = 6.8e13
A_MOON = 136506000.0
R_SATURN = 60268000.0
R_MOON = 4300.0
PERIOD = 2*np.pi*np.sqrt(A_MOON**3/(G*(M_SATURN+M_DAPHNIS)))

@dataclass(frozen=True)
class Config:
    radial_samples: int = 5
    phase_samples: int = 101
    steps_per_orbit: int = 4096
    periods: int = 3
    outputs_per_orbit: int = 256
    moon: bool = True
    seed: int = 0
    wall_limit_s: float = 3500.0

    def validate(self):
        integer_fields = (self.radial_samples, self.phase_samples, self.steps_per_orbit,
                          self.periods, self.outputs_per_orbit, self.seed)
        if any(type(value) is not int for value in integer_fields):
            raise ValueError('Sample, step, output, period and seed values must be integers')
        if self.radial_samples < 2 or self.phase_samples < 2 or self.periods < 1:
            raise ValueError('At least two samples per axis and one period are required')
        if self.steps_per_orbit < 256 or self.outputs_per_orbit < 1:
            raise ValueError('Require at least 256 steps and one output per orbit')
        if self.steps_per_orbit % self.outputs_per_orbit:
            raise ValueError('Output cadence must divide the step count')
        if not 0 < self.wall_limit_s <= 3600 or type(self.moon) is not bool:
            raise ValueError('Require a Boolean moon switch and a wall cap of at most one hour')
        if self.radial_samples*self.phase_samples > 20000:
            raise ValueError('Tracer count exceeds this pilot contract')
        if self.periods*self.steps_per_orbit > 1000000:
            raise ValueError('Step count exceeds this pilot contract')
        if self.periods*self.outputs_per_orbit*self.radial_samples*self.phase_samples > 2000000:
            raise ValueError('Saved sample count exceeds this pilot memory contract')


def build(config):
    config.validate()
    x, y = np.meshgrid(np.linspace(15000, 25000, config.radial_samples),
                       np.linspace(-300000, 200000, config.phase_samples), indexing='ij')
    x, y = x.ravel(), y.ravel()
    s = rebound.Simulation(); s.G = G
    s.integrator = 'whfast'; s.integrator.safe_mode = 1
    s.integrator.coordinates = 'JACOBI'; s.integrator.corrector = 0
    s.add(m=M_SATURN)
    s.add(m=M_DAPHNIS if config.moon else 0, a=A_MOON, primary=s.particles[0])
    for xi, yi in zip(x,y):
        s.add(m=0, a=A_MOON+xi, f=yi/A_MOON, primary=s.particles[0])
    s.N_active = 2; s.testparticle_type = 0; s.move_to_com()
    s.dt = PERIOD/config.steps_per_orbit
    return s, x, y


def run(config, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); cpu = time.process_time()
    s, x0, y0 = build(config); n = len(x0)
    dt = s.dt; stride = config.steps_per_orbit//config.outputs_per_orbit
    total = config.periods*config.steps_per_orbit
    state = np.empty((n+2,6)); valid = np.ones(n,dtype=bool)
    closest = np.full(n,np.inf); tclosest = np.zeros(n)
    data = {k:[] for k in ['t','a_m','radius_m','along_m','distance_m','jacobi_h','valid','pair_energy','pair_lz']}
    mu = G*M_SATURN; mum = G*(M_DAPHNIS if config.moon else 0)
    omega = np.sqrt((mu+mum)/A_MOON**3)
    surface_accel_bound = 2*mu/R_SATURN**2 + mum/R_MOON**2
    have_previous = False; max_analytic = 0.; crossing_count = np.zeros(n,dtype=int)
    previous_y = None
    for step in range(total+1):
        s.serialize_particle_data(xyzvxvyvz=state)
        pos, vel = state[:,:3], state[:,3:]
        r = pos[2:]-pos[0]; v = vel[2:]-vel[0]
        d = pos[2:]-pos[1]; dv = vel[2:]-vel[1]
        radius = np.linalg.norm(r,axis=1); distance=np.linalg.norm(d,axis=1)
        if have_previous:
            # Before the first crossing, acceleration is bounded outside both enclosing surfaces.
            valid &= distance - np.linalg.norm(last_dv,axis=1)*dt - .5*surface_accel_bound*dt**2 > R_MOON
            valid &= radius - np.linalg.norm(last_v,axis=1)*dt - .5*surface_accel_bound*dt**2 > R_SATURN
        valid &= (distance>R_MOON) & (radius>R_SATURN)
        improve = distance < closest; closest[improve]=distance[improve]; tclosest[improve]=s.t
        angle=np.arctan2(pos[1,1]-pos[0,1],pos[1,0]-pos[0,0])
        along=A_MOON*np.arctan2(r[:,1]*np.cos(angle)-r[:,0]*np.sin(angle),r[:,0]*np.cos(angle)+r[:,1]*np.sin(angle))
        if previous_y is not None: crossing_count += (previous_y>0)&(along<=0)
        previous_y=along.copy()
        if step % stride == 0:
            a=1/(2/radius-np.sum(v*v,axis=1)/mu)
            h=.5*np.sum(vel[2:]**2,axis=1)-mu/radius-mum/distance-omega*np.cross(pos[2:],vel[2:])[:,2]
            pair_r=np.linalg.norm(pos[1]-pos[0]); pair_v=vel[1]-vel[0]
            values=[s.t,a,radius,along,distance,h,valid.copy(),.5*np.dot(pair_v,pair_v)-(mu+mum)/pair_r,np.cross(pos[1]-pos[0],pair_v)[2]]
            for k,value in zip(data,values):data[k].append(value)
            if not config.moon:
                phi=y0/A_MOON+np.sqrt(mu/(A_MOON+x0)**3)*s.t
                analytic=np.column_stack(((A_MOON+x0)*np.cos(phi),(A_MOON+x0)*np.sin(phi),np.zeros(n)))
                max_analytic=max(max_analytic,float(np.max(np.linalg.norm(r-analytic,axis=1))))
        if time.perf_counter()-start > config.wall_limit_s:
            s.save_to_file(str(out/'checkpoint.bin'),delete_file=True)
            raise TimeoutError('Wall cap reached; partial checkpoint retained')
        if step % config.steps_per_orbit == 0:
            s.save_to_file(str(out/'checkpoint.bin'),delete_file=True)
        if step==total:break
        have_previous=True;last_dv=dv.copy();last_v=v.copy()
        s.steps(1)
    arr={k:np.asarray(v) for k,v in data.items()}
    arr.update(x0_m=x0,y0_m=y0,closest_m=closest,tclosest_s=tclosest,crossing_count=crossing_count)
    arr['encountered']=(crossing_count==1)&(tclosest>0)&(tclosest<s.t)
    arr['time_since_closest_s']=np.where(arr['encountered'][None,:],arr['t'][:,None]-tclosest[None,:],np.nan)
    arr['delta_a_m']=arr['a_m']-arr['a_m'][0]
    arr['radial_epicycle_m']=arr['radius_m']-arr['a_m']
    incoming=(y0>=60000)&valid&arr['encountered']&(tclosest < (config.periods-1)*PERIOD)
    weights=np.ones(n)
    weights[(x0==x0.min())|(x0==x0.max())]*=.5
    weights[(y0==60000)|(y0==200000)]*=.5
    arr['incoming_cohort']=incoming; arr['cohort_weights']=weights
    np.savez_compressed(out/'trajectories.npz',**arr)
    final_da=arr['delta_a_m'][-1]; radial=arr['radial_epicycle_m'][-1]
    hill=A_MOON*(M_DAPHNIS/(3*M_SATURN))**(1/3)
    metrics={'config':asdict(config),'particles':n,'period_s':PERIOD,'duration_s':s.t,'dt_s':dt,
      'wall_s':time.perf_counter()-start,'cpu_s':time.process_time()-cpu,
      'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      'invalid_surface_count':int(np.sum(~valid)),'minimum_distance_m':float(closest.min()),
      'incoming_count':int(incoming.sum()),'crossing_count_max':int(crossing_count.max()),
      'incoming_final_delta_a_mean_m':float(np.average(final_da[incoming],weights=weights[incoming])) if incoming.any() else None,
      'incoming_final_delta_a_range_m':[float(final_da[incoming].min()),float(final_da[incoming].max())] if incoming.any() else None,
      'incoming_final_epicycle_range_m':[float(radial[incoming].min()),float(radial[incoming].max())] if incoming.any() else None,
      'maximum_delta_a_abs_m':float(np.max(np.abs(arr['delta_a_m']))),
      'jacobi_h_max_drift_m2_s2':float(np.max(np.abs(arr['jacobi_h']-arr['jacobi_h'][0]))),
      'jacobi_h_drift_over_hill_velocity_squared':float(np.max(np.abs(arr['jacobi_h']-arr['jacobi_h'][0]))/(omega*hill)**2),
      'pair_energy_relative_drift':float(np.max(np.abs(arr['pair_energy']/arr['pair_energy'][0]-1))),
      'pair_lz_relative_drift':float(np.max(np.abs(arr['pair_lz']/arr['pair_lz'][0]-1))),
      'no_moon_analytic_max_position_error_m':max_analytic if not config.moon else None,
      'versions':{'python':platform.python_version(),'rebound':rebound.__version__,'numpy':np.__version__}}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--out',required=True);p.add_argument('--radial',type=int,default=5);p.add_argument('--phases',type=int,default=101);p.add_argument('--steps',type=int,default=4096);p.add_argument('--periods',type=int,default=3);p.add_argument('--no-moon',action='store_true')
    a=p.parse_args()
    config=Config(**json.loads(Path(a.config).read_text())) if a.config else Config(radial_samples=a.radial,phase_samples=a.phases,steps_per_orbit=a.steps,periods=a.periods,moon=not a.no_moon)
    print(json.dumps(run(config,a.out),indent=2))
