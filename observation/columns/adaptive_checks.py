import json,time
import numpy as np
from .adaptive import adaptive_rays,detector_pixels
from .model import Columns,geometry,lambert_phase
from .checks import slab,analytic


def run():
    start=time.time();summary={};s,o=geometry(35,35,70);p=np.zeros((1,3))
    target=analytic(.3,s[2],o[2],.5,lambert_phase(np.deg2rad(35)))
    vals=[]
    for h in (1.,5.,1000.):
        result=adaptive_rays(slab(height=h),p,s,o,epsabs=1e-9,epsrel=0)
        assert result['converged'].all();vals.append(float(result['I_over_F'][0]))
    assert np.max(abs(np.array(vals)-target))<1e-10
    summary['adaptive_slab_H_1_5_1000']={'analytic':target,'values':vals}
    s=o=np.array([0.,0.,1.]);scene=Columns(np.array([-1.,0.,1.]),np.array([-1.,0.,1.]),np.array([[10.,50.],[100.,300.]]),np.ones((2,2)))
    gs,go=geometry(0,90,90)
    face_ray=adaptive_rays(scene,np.zeros((1,3)),gs,go,epsabs=1e-10,epsrel=0)
    exact_face=analytic(.003*300,1,1,.5,lambert_phase(0))
    assert abs(face_ray['I_over_F'][0]-exact_face)<1e-10
    summary['exact_vertical_on_cell_face']={'computed':float(face_ray['I_over_F'][0]),'analytic':exact_face}
    expected=np.mean([analytic(.003*sig,1,1,.5,lambert_phase(0)) for sig in (10,50,100,300)])
    records=[]
    for n in (1,2,4):
        r=detector_pixels(scene,[0,0,0],[1,0,0],[0,1,0],[-1,1],[-1,1],s,o,subrays=n,epsabs=1e-10,epsrel=0)
        assert r['source_converged'].all();records.append(float(r['I_over_F'][0,0]))
    assert abs(records[1]-expected)<1e-12 and abs(records[2]-expected)<1e-12
    summary['fixed_pixel_2D_step']={'expected_area_mean':expected,'subrays_per_axis':[1,2,4],'same_pixel_means':records}
    boundary=detector_pixels(scene,[0,0,0],[1,0,0],[0,1,0],[-2,2],[-2,2],s,o,subrays=4,epsabs=1e-10,epsrel=0)
    assert abs(boundary['I_over_F'][0,0]-expected/4)<1e-12
    summary['vacuum_boundary_pixel']={'computed':float(boundary['I_over_F'][0,0]),'analytic_quarter_area':expected/4}
    dark=adaptive_rays(scene,np.array([[.5,.5,0.]]),s,o,omega=0,epsabs=1e-10,epsrel=0)
    lit=adaptive_rays(scene,np.array([[.5,.5,0.]]),s,o,omega=.5,epsabs=1e-10,epsrel=0)
    assert dark['I_over_F'][0]==0 and abs(dark['shadow_fraction'][0]-lit['shadow_fraction'][0])<1e-8
    summary['zero_albedo_keeps_geometric_attenuation']=True
    failure=adaptive_rays(slab(),p,s,o,epsabs=1e-30,epsrel=0,limit=2)
    assert not failure['converged'][0] and failure['status'][0]!='ok'
    summary['explicit_failure']={'converged':bool(failure['converged'][0]),'status':failure['status'][0]}
    detector_failure=detector_pixels(slab(),[0,0,0],[1,0,0],[0,1,0],[-1,1],[-1,1],s,o,subrays=1,epsabs=1e-30,epsrel=0,limit=2)
    assert not detector_failure['source_converged'][0,0] and detector_failure['failed_subrays'][0,0]==1
    summary['pixel_preserves_source_failure']=True
    try:detector_pixels(scene,[0,0,0],[1,0,1],[0,1,0],[-1,1],[-1,1],s,o)
    except ValueError:summary['invalid_detector_basis_rejected']=True
    else:raise AssertionError('invalid detector basis accepted')
    scene=Columns(np.array([-2000.,-100.,100.,2000.]),np.array([-2000.,2000.]),np.full((1,3),100.),np.array([[1000.,1.,1.]]))
    s,o=geometry(75,15,90);points=np.array([[-200.,0,0],[-400.,0,0],[-800.,0,0]])
    coarse=adaptive_rays(scene,points,s,o,epsabs=1e-6,epsrel=0);fine=adaptive_rays(scene,points,s,o,epsabs=1e-9,epsrel=0)
    delta=abs(coarse['I_over_F']-fine['I_over_F'])
    assert coarse['converged'].all() and fine['converged'].all() and np.max(delta)<1e-6
    summary['cold_sun_layer_hot_camera']={'same_ray_x_m':points[:,0].tolist(),'epsabs':[1e-6,1e-9],'coarse':coarse['I_over_F'].tolist(),
        'fine':fine['I_over_F'].tolist(),'absolute_change':delta.tolist(),'fine_estimated_error':fine['source_error_estimate'].tolist(),'fine_evaluations':fine['evaluations'].tolist()}
    xe=np.linspace(-8000,8000,17);ye=np.linspace(-4000,4000,9)
    xx,yy=np.meshgrid((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2)
    scene=Columns(xe,ye,np.full_like(xx,100.),1+999*np.exp(-.5*((xx/1800)**2+(yy/2000)**2)))
    regression=adaptive_rays(scene,np.array([[-5000.,-3000.,0.]]),s,o,epsabs=1e-7,epsrel=0)
    assert regression['converged'].all() and regression['source_error_estimate'][0]<1e-7
    summary['roundoff_sized_feature_intervals']={'I_over_F':float(regression['I_over_F'][0]),'error_estimate':float(regression['source_error_estimate'][0]),'converged':True}
    summary['seconds']=time.time()-start;return summary

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
