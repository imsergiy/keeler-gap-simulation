"""Run the bounded comparison matrix, assert numerical controls, and plot results."""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import Config, PERIOD

ROOT=Path(__file__).resolve().parent

def main(execute=False):
    configs=json.loads((ROOT/'configs.json').read_text())
    if execute:
        for name,config in configs.items():
            out=ROOT/'results'/name;out.mkdir(parents=True,exist_ok=True)
            (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
            with (out/'stdout.json').open('w') as stream:
                subprocess.run([sys.executable,str(ROOT/'run.py'),'--config',str(out/'config.json'),'--out',str(out)],stdout=stream,check=True,timeout=3600)
    metrics={name:json.loads((ROOT/'results'/name/'metrics.json').read_text()) for name in configs}
    raw={name:np.load(ROOT/'results'/name/'trajectories.npz') for name in configs}
    c,f,s=raw['coarse'],raw['time-fine'],raw['sample-fine']
    nested=np.arange(9*201).reshape(9,201)[::2,::2].ravel()
    checks={
      'timestep_max_final_delta_a_difference_m':float(np.max(np.abs(c['delta_a_m'][-1]-f['delta_a_m'][-1]))),
      'timestep_max_radius_difference_m':float(np.max(np.abs(c['radius_m']-f['radius_m']))),
      'sampling_common_trajectory_max_delta_a_difference_m':float(np.max(np.abs(f['delta_a_m']-s['delta_a_m'][:,nested]))),
      'sampling_cohort_mean_delta_a_difference_m':abs(metrics['time-fine']['incoming_final_delta_a_mean_m']-metrics['sample-fine']['incoming_final_delta_a_mean_m'])}
    for name in ('no-moon','no-moon-fine'):
        assert metrics[name]['maximum_delta_a_abs_m'] < .001
        assert metrics[name]['no_moon_analytic_max_position_error_m'] < .01
    assert metrics['sample-fine']['incoming_final_delta_a_range_m'][0] > 50
    assert metrics['sample-fine']['incoming_final_epicycle_range_m'][1] > 1000
    assert checks['timestep_max_final_delta_a_difference_m'] < .1
    assert checks['timestep_max_radius_difference_m'] < .1
    assert checks['sampling_common_trajectory_max_delta_a_difference_m'] < .001
    assert checks['sampling_cohort_mean_delta_a_difference_m'] < 20
    assert all(m['invalid_surface_count']==0 and m['crossing_count_max']<=1 for m in metrics.values())
    assert all(m['jacobi_h_drift_over_hill_velocity_squared']<1e-4 for m in metrics.values())
    assert all(m['peak_rss_MiB']<750 and m['wall_s']<3500 for m in metrics.values())
    assert all(np.all(np.isnan(a['time_since_closest_s'][:,~a['encountered']])) for a in raw.values())
    summary={'verdict':'Bounded collision-free numerical control passes; physical edge retreat is not established.',
      'checks':checks,'runs':metrics,'total_matrix_wall_s':sum(m['wall_s'] for m in metrics.values()),
      'scope':'Global native circular/coplanar pair, spherical Saturn gravity, massless collision-free quadrature tracers; no steady-state wake, collisional retreat, vertical splashing/damping or calibrated brightness.'}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    rows=['case,N,dt_s,wall_s,peak_rss_MiB,cohort_delta_a_mean_m,jacobi_drift_m2_s2']
    for name,m in metrics.items():rows.append(','.join(map(str,[name,m['particles'],m['dt_s'],m['wall_s'],m['peak_rss_MiB'],m['incoming_final_delta_a_mean_m'],m['jacobi_h_max_drift_m2_s2']])))
    (ROOT/'metrics.csv').write_text('\n'.join(rows)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for k,ax in enumerate(axes.ravel()):
        j=k*256
        points=ax.scatter(s['along_m'][j]/1000,(s['radius_m'][j]-136506000)/1000,s=2,c=s['x0_m']/1000,cmap='viridis')
        ax.scatter([0],[0],c='black',s=20)
        ax.set(xlabel='Moon-frame arc distance (km; upstream +)',ylabel='Saturn radius minus moon orbit (km)',title=f'{k} orbital periods; fixed initial sample')
        ax.set_xlim(-1500,250);ax.set_ylim(-5,45);ax.grid(alpha=.2)
    fig.colorbar(points,ax=axes,label='Initial radial offset (km)')
    fig.suptitle('Collision-free control: geometry, not a steady-state density map')
    fig.savefig(ROOT/'geometry.png',dpi=180);fig.savefig(ROOT/'geometry.pdf');plt.close(fig)
    i=np.argmin((s['x0_m']-20000)**2+(s['y0_m']-100000)**2)
    fig,axes=plt.subplots(2,1,figsize=(9,7),constrained_layout=True)
    t=s['time_since_closest_s'][:,i]/PERIOD
    axes[0].plot(t,(s['radius_m'][:,i]-s['radius_m'][0,i])/1000,label='Instantaneous radius change')
    axes[0].plot(t,s['delta_a_m'][:,i]/1000,label='Osculating semimajor-axis change')
    axes[0].set(xlabel='Periods from sampled closest encounter',ylabel='Change (km)',title='Tracer initially +20 km radially, +100 km upstream');axes[0].legend()
    for a in axes:a.grid(alpha=.3)
    cohort=s['incoming_cohort'];points=axes[1].scatter(s['x0_m'][cohort]/1000,s['delta_a_m'][-1,cohort],s=5,c=s['y0_m'][cohort]/1000,cmap='plasma')
    axes[1].set(xlabel='Initial radial offset (km)',ylabel='Final osculating Δa (m)',title='Incoming cohort only: +60 to +200 km initially; color = initial arc distance')
    fig.colorbar(points,ax=axes[1],label='Initial upstream arc distance (km)')
    fig.suptitle('Radial return is distinct from mean orbit change; no collision model')
    fig.savefig(ROOT/'orbit-change.png',dpi=180);fig.savefig(ROOT/'orbit-change.pdf');plt.close(fig)
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');main(p.parse_args().run)
