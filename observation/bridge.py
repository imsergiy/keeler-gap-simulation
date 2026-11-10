"""Artifact-only actual pilot columns to reviewed fixed detector radiance."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from observation.columns.model import load_columns,opacity,geometry
from observation.columns.adaptive import detector_pixels


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_artifact(columns_path, output, summary_path, config_path, validity_path,
                    nx=8, ny=4, subray_pair=(2,4), radius_m=.5,
                    material_density_kg_m3=500., omega=.5, source_epsabs=1e-7,
                    area_target=2e-4):
    """Return actual fixed-pixel maps/profiles; keep all failures and estimates."""
    if (not np.all(np.isfinite([radius_m,material_density_kg_m3,omega,source_epsabs,area_target]))
        or radius_m<=0 or material_density_kg_m3<=0 or not 0<=omega<=1
        or source_epsabs<=0 or area_target<=0 or not isinstance(nx,int) or not isinstance(ny,int) or min(nx,ny)<1
        or len(subray_pair)!=2 or not all(isinstance(n,int) and n>=1 for n in subray_pair) or subray_pair[1]<=subray_pair[0]):
        raise ValueError('invalid grain parameters, detector size, refinement pair or targets')
    import observation.columns.model as model_module
    import observation.columns.adaptive as adaptive_module
    input_paths=[columns_path,summary_path,config_path,validity_path,model_module.__file__,adaptive_module.__file__,__file__]
    input_hashes={str(Path(p).resolve()):digest(p) for p in input_paths}
    start=time.time();output=Path(output);output.mkdir(exist_ok=True,parents=True)
    columns_path=Path(columns_path);columns=load_columns(columns_path)
    columns.kappa=opacity(radius_m,material_density_kg_m3)
    source_summary=json.loads(Path(summary_path).read_text());source_config=json.loads(Path(config_path).read_text());validity=json.loads(Path(validity_path).read_text())
    sun,observer=geometry(75,15,90)
    ue=np.linspace(columns.x_edges[0],columns.x_edges[-1],nx+1)
    ve=np.linspace(columns.y_edges[0],columns.y_edges[-1],ny+1)
    config={'input_columns':str(columns_path.resolve()),'input_summary':str(Path(summary_path).resolve()),
        'input_config':str(Path(config_path).resolve()),'input_validity':str(Path(validity_path).resolve()),
        'geometry_label':'representative, not measured or fitted','phase_deg':75,'sun_elevation_deg':15,'observer_elevation_deg':90,
        'center_m':[0,0,0],'right':[1,0,0],'up':[0,1,0],'sun':sun.tolist(),'observer':observer.tolist(),
        'u_edges_m':ue.tolist(),'v_edges_m':ve.tolist(),'detector_shape':[ny,nx],'subrays_per_axis':list(subray_pair),
        'grain_radius_m':radius_m,'material_density_kg_m3':material_density_kg_m3,'kappa_m2_kg':columns.kappa,'omega':omega,
        'source_epsabs_I_over_F':source_epsabs,'source_epsrel':0.,'area_absolute_target_I_over_F':area_target,
        'horizontal_boundary':'vacuum outside full input cell edges','vertical_boundary':'infinite Gaussian tails',
        'profile_definition':'mean I/F across full detector v extent, weighted by physical pixel width; local radial x, not Saturn radius'}
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    raw={'u_edges_m':ue,'v_edges_m':ve};levels=[];previous=None
    for n in subray_pair:
        t=time.time();r=detector_pixels(columns,[0,0,0],[1,0,0],[0,1,0],ue,ve,sun,observer,subrays=n,
            omega=omega,epsabs=source_epsabs,epsrel=0)
        for key,value in r.items():raw[f'sub{n}_{key}']=value
        profile=np.average(r['I_over_F'],axis=0,weights=np.diff(ve));raw[f'sub{n}_profile_I_over_F']=profile
        levels.append({'subrays_per_axis':n,'rays':nx*ny*n*n,'seconds':time.time()-t,
            'failed_pixels':int(np.count_nonzero(~r['source_converged'])),'failed_subrays':int(r['failed_subrays'].sum()),
            'max_mean_source_error_estimate':float(r['source_error_estimate'].max()),'source_evaluations':int(r['evaluations'].sum())})
        print(levels[-1],flush=True)
        if previous is not None:
            raw['same_pixel_area_change']=abs(r['I_over_F']-previous['I_over_F'])
            raw['same_profile_area_change']=abs(profile-raw[f'sub{subray_pair[0]}_profile_I_over_F'])
        previous=r
    raw['profile_x_m']=(ue[:-1]+ue[1:])/2
    raw['profile_source_converged']=np.all(r['source_converged'],axis=0)
    raw['profile_source_error_estimate']=np.average(r['source_error_estimate'],axis=0,weights=np.diff(ve))
    source_ok=all(level['failed_pixels']==0 for level in levels)
    area_ok=bool(np.all(raw['same_pixel_area_change']<=area_target))
    report={'status':('numerical preview; SOURCE NONCONVERGENCE' if not source_ok else ('numerical preview; unresolved detector area' if not area_ok else 'numerical control; successive detector area estimate below target')),
        'source_status':'converged' if source_ok else 'NONCONVERGENCE: retain flags, do not infer brightness',
        'source_levels':levels,'max_same_pixel_area_change':float(raw['same_pixel_area_change'].max()),
        'mean_same_pixel_area_change':float(raw['same_pixel_area_change'].mean()),'max_same_profile_area_change':float(raw['same_profile_area_change'].max()),
        'area_target_met_by_successive_estimate':area_ok,'area_estimate_is_rigorous_bound':False,
        'input_field_stats':{'Sigma_min_kg_m2':float(columns.sigma.min()),'Sigma_max_kg_m2':float(columns.sigma.max()),
            'H_min_m':float(columns.height.min()),'H_max_m':float(columns.height.max()),'meanZ_min_m':float(columns.mean_z.min()),'meanZ_max_m':float(columns.mean_z.max())},
        'source_dynamics_summary':source_summary,'source_dynamics_config':source_config,'source_validity':validity,
        'limitations':['actual dynamics snapshot, no synthetic height prescription','source dynamics validity is retained; this adapter does not validate dynamics',
            'exported meanZ is used directly; a zero-mean coplanar field cannot represent coherent bending',
            'single scattering; no calibrated ice phase law or observed brightness inference','no physical height validation, PSF or multiple scattering',
            'two detector-area levels can miss unsampled structure; no dynamics convergence claim'],
        'seconds':time.time()-start,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    if any(digest(path)!=saved for path,saved in input_hashes.items()):
        raise RuntimeError('an input or renderer source changed during rendering; output is not an immutable snapshot')
    report['input_sha256']=input_hashes
    report['completion_identity']={
        'input_columns':input_hashes[str(columns_path.resolve())],
        'config':digest(output/'config.json'),
        'generator':input_hashes[str(Path(__file__).resolve())],
        'model':input_hashes[str(Path(model_module.__file__).resolve())],
        'adaptive':input_hashes[str(Path(adaptive_module.__file__).resolve())]}
    np.savez_compressed(output/'maps_profiles.npz',**raw)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    from shutil import copyfile
    copyfile(columns_path,output/'input_columns.npz')
    copyfile(__file__,output/'preview_bridge_source.py')
    rows=np.column_stack([raw['profile_x_m'],raw[f'sub{subray_pair[0]}_profile_I_over_F'],raw[f'sub{subray_pair[-1]}_profile_I_over_F'],raw['same_profile_area_change'],raw['profile_source_error_estimate'],raw['profile_source_converged']])
    np.savetxt(output/'radial_profile.csv',rows,delimiter=',',header=f'local_radial_x_m,I_over_F_sub{subray_pair[0]},I_over_F_sub{subray_pair[-1]},successive_area_change,source_error_estimate,source_converged',comments='')
    return report


def complete_area(output, max_subrays=32, max_seconds=600.):
    """Refine unresolved fixed footprints only; retain the initial preview pair."""
    if max_subrays not in (8,16,32) or not np.isfinite(max_seconds) or max_seconds<=0:
        raise ValueError('max_subrays must be8,16,32 and time budget positive')
    import observation.columns.model as model_module
    import observation.columns.adaptive as adaptive_module
    output=Path(output);report=json.loads((output/'report.json').read_text())
    if 'completion_identity' not in report:
        raise RuntimeError('preview has no completion identity; render a new output directory')
    identity=report['completion_identity']
    preview_snapshot=output/'preview_bridge_source.py'
    expected={output/'input_columns.npz':identity['input_columns'],output/'config.json':identity['config'],
        preview_snapshot:identity['generator'],Path(__file__):identity['generator'],
        Path(model_module.__file__):identity['model'],Path(adaptive_module.__file__):identity['adaptive']}
    for path,saved in expected.items():
        if digest(path)!=saved:
            raise RuntimeError(f'preview identity mismatch: {path}; render a new output directory')
    config=json.loads((output/'config.json').read_text())
    with np.load(output/'maps_profiles.npz') as saved:raw={key:saved[key] for key in saved.files}
    columns=load_columns(output/'input_columns.npz');columns.kappa=config['kappa_m2_kg']
    completion_hashes={str(path.resolve()):saved for path,saved in expected.items()}
    preview_hash=identity['generator']
    before_hash=digest(output/'input_columns.npz');ue=np.array(config['u_edges_m']);ve=np.array(config['v_edges_m'])
    base=config['subrays_per_axis'][-1]
    keys=('I_over_F','tau_camera','sun_transmission_weighted','shadow_fraction','source_error_estimate','source_converged','failed_subrays','evaluations')
    maps={key:raw[f'sub{base}_{key}'].copy() for key in keys}
    area=raw['same_pixel_area_change'].copy();sampling=np.full(area.shape,base,int)
    records=[];start=time.time();budget_used=report['seconds'];stopped=False
    for n in (8,16,32):
        if n<=base:continue
        if n>max_subrays:break
        active=np.argwhere(area>config['area_absolute_target_I_over_F'])
        if len(active)==0:break
        for j,i in active:
            if budget_used+time.time()-start>=max_seconds:
                stopped=True;break
            t=time.time();r=detector_pixels(columns,config['center_m'],config['right'],config['up'],ue[i:i+2],ve[j:j+2],config['sun'],config['observer'],subrays=n,
                omega=config['omega'],epsabs=config['source_epsabs_I_over_F'],epsrel=config['source_epsrel'])
            area[j,i]=abs(r['I_over_F'][0,0]-maps['I_over_F'][j,i]);sampling[j,i]=n
            for key in keys:maps[key][j,i]=r[key][0,0]
            records.append({'row':int(j),'column':int(i),'subrays_per_axis':n,'seconds':time.time()-t,'same_pixel_change':float(area[j,i]),
                'source_converged':bool(r['source_converged'][0,0]),'failed_subrays':int(r['failed_subrays'][0,0])})
            print(records[-1],flush=True)
        if stopped:break
    if any(digest(path)!=saved for path,saved in completion_hashes.items()):
        raise RuntimeError('an input or renderer changed during area completion')
    for key,value in maps.items():raw['final_'+key]=value
    raw['final_area_change']=area;raw['final_subrays_per_axis']=sampling
    raw['final_profile_I_over_F']=np.average(maps['I_over_F'],axis=0,weights=np.diff(ve))
    raw['final_profile_source_error_estimate']=np.average(maps['source_error_estimate'],axis=0,weights=np.diff(ve))
    raw['final_profile_source_converged']=np.all(maps['source_converged'],axis=0)
    raw['final_profile_mean_pixel_area_change']=np.average(area,axis=0,weights=np.diff(ve))
    source_ok=report['source_status']=='converged' and bool(maps['source_converged'].all())
    area_ok=bool(np.all(area<=config['area_absolute_target_I_over_F']))
    if 'preview_status' not in report:report['preview_status']=report['status']
    report['status']='numerical control; successive same-pixel estimate below target' if source_ok and area_ok else 'NUMERICAL PREVIEW; source or area unresolved'
    report['source_status']='converged' if source_ok else 'NONCONVERGENCE: retain all flags'
    report['adaptive_area_completion']={'records':records,'seconds':time.time()-start,'total_seconds':budget_used+time.time()-start,
        'max_subrays':max_subrays,'budget_seconds':max_seconds,'budget_stopped':stopped,'target':config['area_absolute_target_I_over_F'],
        'target_met_by_successive_estimate':area_ok,'unresolved_pixels':int(np.count_nonzero(area>config['area_absolute_target_I_over_F'])),
        'max_final_same_pixel_change':float(area.max()),'max_final_mean_profile_pixel_change':float(raw['final_profile_mean_pixel_area_change'].max()),
        'max_final_source_error_estimate':float(maps['source_error_estimate'].max()),'failed_final_pixels':int(np.count_nonzero(~maps['source_converged'])),
        'source_columns_copy_sha256':before_hash,'preview_adapter_snapshot_sha256':preview_hash,'completion_adapter_sha256':digest(__file__),
        'input_and_renderer_sha256':completion_hashes,
        'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'area_estimate_is_rigorous_bound':False}
    np.savez_compressed(output/'maps_profiles.npz',**raw);(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    rows=np.column_stack([raw['profile_x_m'],raw['final_profile_I_over_F'],raw['final_profile_mean_pixel_area_change'],raw['final_profile_source_error_estimate'],raw['final_profile_source_converged']])
    np.savetxt(output/'radial_profile_completed.csv',rows,delimiter=',',header='local_radial_x_m,I_over_F,mean_successive_pixel_area_change,source_error_estimate,source_converged',comments='')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--columns',type=Path,required=True);parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--validity',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--complete-area',action='store_true')
    args=parser.parse_args();report=render_artifact(args.columns,args.output,args.summary,args.config,args.validity)
    if args.complete_area:
        (args.output/'preview_report.json').write_text(json.dumps(report,indent=2)+'\n')
        report=complete_area(args.output)
    print(json.dumps({k:report[k] for k in ['status','source_status','seconds','peak_RSS_MiB','max_same_pixel_area_change','max_same_profile_area_change']},indent=2))
