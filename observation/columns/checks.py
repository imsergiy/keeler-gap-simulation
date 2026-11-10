import json,time
import numpy as np
from scipy.integrate import quad
from .model import Columns,geometry,lambert_phase,opacity


def slab(height=1,sigma=100,extent=1e8):
    return Columns(np.array([-extent,extent]),np.array([-extent,extent]),np.array([[sigma]]),np.array([[height]]))


def analytic(tau,mu0,mu,omega,phase):
    return omega*phase/4*mu0/(mu+mu0)*(-np.expm1(-tau*(1/mu+1/mu0)))


def run():
    start=time.time(); results={}
    point=np.zeros((1,3)); sun,obs=geometry(35,35,70)
    target=analytic(.3,sun[2],obs[2],.5,lambert_phase(np.deg2rad(35)))
    values=[float(slab(h).render(point,sun,obs,order=32)['I_over_F'][0]) for h in (1,5,1000)]
    results['slab']={'analytic':target,'computed_H_1_5_1000m':values,'relative_error':max(abs(np.array(values)/target-1))}
    assert results['slab']['relative_error']<1e-10
    thick=float(slab(sigma=1e5).render(point,sun,obs,order=64)['I_over_F'][0])
    thick_exact=analytic(300,sun[2],obs[2],.5,lambert_phase(np.deg2rad(35)))
    results['thick_slab_relative_error']=abs(thick/thick_exact-1); assert results['thick_slab_relative_error']<1e-10
    for name,column,omega in [('vacuum',slab(sigma=0),.5),('zero_albedo',slab(),0)]:
        value=float(column.render(point,sun,obs,omega=omega)['I_over_F'][0]); assert value==0; results[name]=value
    thin=[float(slab(sigma=s).render(point,sun,obs)['I_over_F'][0]) for s in (1e-6,2e-6)]
    results['thin_ratio']=thin[1]/thin[0]; assert abs(results['thin_ratio']-2)<1e-7
    integral=quad(lambda a:lambert_phase(a)*2*np.pi*np.sin(a),0,np.pi)[0]
    results['phase_integral_over_4pi']=integral/(4*np.pi); assert abs(integral/(4*np.pi)-1)<1e-12
    rho=.6; radius=.5
    scattered=rho*lambert_phase(0)/(4*np.pi)*np.pi*radius**2
    exact=2*rho*radius**2/3
    results['isolated_Lambert_backscatter']={'model_radiance_area_over_F':scattered,'analytic':exact}
    assert abs(scattered-exact)<1e-14
    # An infinite-half-plane slab ends at x=0; an upper-face ray exits at z=-x tan B.
    edge=Columns(np.array([-1e8,0.]),np.array([-1e8,1e8]),np.array([[100.]]),np.array([[2.]]))
    edge_sun=np.array([np.cos(.2),0,np.sin(.2)])
    p=np.array([-1.,0.,0.])
    from scipy.special import ndtr
    expected=.3/edge_sun[2]*(ndtr(np.tan(.2)/2)-.5)
    measured=edge.optical_depth(p,edge_sun)
    results['edge_tau']={'computed':measured,'analytic':expected}; assert abs(measured-expected)<1e-13
    vertical=np.array([0.,0.,1.]); phase=lambert_phase(np.arccos(edge_sun@vertical))
    def edge_source(z):
        chi=.3/(np.sqrt(2*np.pi)*2)*np.exp(-.5*(z/2)**2)
        tc=.3*ndtr(-z/2)
        ts=.3/edge_sun[2]*(ndtr((z+np.tan(.2))/2)-ndtr(z/2))
        return .5*phase/4*chi*np.exp(-tc-ts)
    edge_if=quad(edge_source,-np.inf,np.inf,epsabs=1e-12)[0]
    rendered_edge=float(edge.render(p[None],edge_sun,vertical,order=128)['I_over_F'][0])
    results['edge_I_over_F']={'computed':rendered_edge,'independent_z_integral':edge_if,'relative_error':abs(rendered_edge/edge_if-1)}
    assert abs(rendered_edge/edge_if-1)<1e-6
    # Raised cold ridge casts a shadow into neighboring low columns with the same albedo.
    scene=Columns(np.array([-100.,-10.,10.,100.]),np.array([-100.,100.]),np.array([[100.,100.,100.]]),np.array([[1.,1.,1.]]),np.array([[0.,6.,0.]]))
    pts=np.array([[-25.,0,0],[25.,0,0]])
    edge_values=[scene.render(pts,edge_sun,np.array([0.,0,1.]),order=n)['I_over_F'] for n in (16,32,64,128)]
    results['ridge_shadow']={'x_m':pts[:,0].tolist(),'I_over_F_orders_16_32_64_128':[v.tolist() for v in edge_values]}
    assert edge_values[-1][0]<edge_values[-1][1]*.8
    results['ridge_refinement_absolute_64_to_128']=float(np.max(abs(edge_values[-1]-edge_values[-2])))
    assert results['ridge_refinement_absolute_64_to_128']<1e-5
    try:geometry(150,30,30)
    except ValueError:results['inconsistent_geometry_rejected']=True
    else:raise AssertionError('inconsistent geometry accepted')
    results['seconds']=time.time()-start
    return results

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
