"""Experimental converged-source and fixed detector-footprint integration."""
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,ndtri
from .model import direction,lambert_phase


def solar_features(columns, point, observer, sun):
    """Camera distances where solar cell-face crossings pass Gaussian features."""
    levels=columns.mean_z[...,None]+columns.height[...,None]*np.array([-6.,-3.,-1.,0.,1.,3.,6.])
    features=[]
    for axis,edges in enumerate((columns.x_edges,columns.y_edges)):
        if abs(sun[axis])<1e-14:
            continue
        denominator=observer[2]-observer[axis]*sun[2]/sun[axis]
        if abs(denominator)<1e-14:
            continue
        other=1-axis
        for side in (0,1):
            face=edges[side:len(edges)-1+side]
            face=face[None,:,None] if axis==0 else face[:,None,None]
            t=(levels-point[2]-(face-point[axis])*sun[2]/sun[axis])/denominator
            u=(face-point[axis]-t*observer[axis])/sun[axis]
            transverse=point[other]+t*observer[other]+u*sun[other]
            lower=columns.y_edges[:-1,None,None] if axis==0 else columns.x_edges[None,:-1,None]
            upper=columns.y_edges[1:,None,None] if axis==0 else columns.x_edges[None,1:,None]
            valid=(u>=0)&(transverse>=lower)&(transverse<=upper)&(columns.sigma[...,None]>0)
            features.extend(t[valid].tolist())
    return np.unique(features)


def adaptive_rays(columns, points, sun, observer, omega=.5, epsabs=1e-7,
                  epsrel=1e-5, tau_cut=36., limit=200):
    """Integrate pencil rays; return per-ray quadrature error and failure status.

    epsabs is absolute I/F error target for the complete ray. Solar Gaussian
    transition brackets force narrow cold shadows into the adaptive partition.
    Error is QUADPACK's estimate plus the analytic omitted camera-tail bound.
    A false convergence flag must not be treated as a quantitative result.
    """
    sun,observer=direction(sun),direction(observer);points=np.asarray(points,float)
    # Match the existing cell-crossing convention for numerically zero axes.
    sun[abs(sun)<1e-14]=0.;observer[abs(observer)<1e-14]=0.
    if (sun[2]<=0 or observer[2]<=0 or not 0<=omega<=1 or epsabs<=0 or epsrel<0
        or not np.all(np.isfinite([epsabs,epsrel,tau_cut])) or tau_cut<=0 or not isinstance(limit,int) or limit<2
        or points.ndim<1 or points.shape[-1]!=3 or not np.all(np.isfinite(points))):
        raise ValueError('invalid upper-face geometry, accuracy, or points')
    factor=omega*lambert_phase(np.arccos(np.clip(sun@observer,-1,1)))/4
    shape=points.shape[:-1]
    result={name:np.zeros(shape) for name in ('I_over_F','tau_camera','sun_transmission_weighted','shadow_fraction','source_error_estimate')}
    result['converged']=np.ones(shape,bool);result['evaluations']=np.zeros(shape,int)
    result['status']=np.full(shape,'ok',dtype=object)
    for idx in np.ndindex(shape):
        point=points[idx];segments=columns.segments(point,observer)
        taus=[columns.segment_tau(point,observer,*segment) for segment in segments]
        total=sum(taus);result['tau_camera'][idx]=total
        if total==0:
            result['sun_transmission_weighted'][idx]=1
            continue
        features=solar_features(columns,point,observer,sun)
        value=error=accumulated=0.;neval=0;messages=[]
        for segment,dt in zip(reversed(segments),reversed(taus)):
            a,b,iy,ix=segment
            if dt==0:continue
            amount=min(dt,tau_cut-accumulated)
            if amount<=0:break
            h,z=columns.height[iy,ix],columns.mean_z[iy,ix];s=columns.kappa*columns.sigma[iy,ix]
            ua=(point[2]+a*observer[2]-z)/h
            ub=(point[2]+b*observer[2]-z)/h
            def source(q):
                lower=ndtr(ua)+(dt-q)*observer[2]/s
                upper=ndtr(-ub)+q*observer[2]/s
                if min(lower,upper)==0:
                    # Exact infinite Gaussian endpoint limits avoid inf times zero.
                    tau_sun=s/sun[2] if lower==0 and np.all(abs(sun[:2])<1e-14) else 0.
                else:
                    zq=z+h*ndtri(lower) if lower<upper else z-h*ndtri(upper)
                    tq=(zq-point[2])/observer[2]
                    tau_sun=columns.optical_depth(point+tq*observer,sun)
                return np.exp(-accumulated-q-tau_sun)
            interior=features[(features>a)&(features<b)]
            cuts=np.unique([columns.segment_tau(point,observer,t,b,iy,ix) for t in interior])
            cuts=cuts[(cuts>0)&(cuts<amount)]
            partition=np.r_[0.,cuts,amount]
            budget=epsabs/((factor if factor>0 else 1.)*max(1,len(segments))*len(partition))
            for lo,hi in zip(partition[:-1],partition[1:]):
                if hi-lo<=budget/2:
                    # A bounded midpoint rule avoids roundoff-sized QUADPACK intervals.
                    value+=(hi-lo)*source(lo+(hi-lo)/2);error+=hi-lo;neval+=1
                else:
                    output=quad(source,lo,hi,epsabs=budget/2,epsrel=epsrel,limit=limit,full_output=1)
                    val,err,info=output[:3];value+=val;error+=err;neval+=info['neval']
                    if len(output)>3:messages.append(output[3])
            accumulated+=dt
        tail=factor*np.exp(-tau_cut) if total>tau_cut else 0.
        result['I_over_F'][idx]=factor*value;result['source_error_estimate'][idx]=factor*error+tail
        clear=-np.expm1(-min(total,tau_cut))
        transmission=value/clear if clear>0 else 1.
        result['sun_transmission_weighted'][idx]=transmission;result['shadow_fraction'][idx]=1-transmission
        converged=not messages and result['source_error_estimate'][idx]<=max(epsabs,epsrel*abs(factor*value))
        result['converged'][idx]=converged;result['evaluations'][idx]=neval
        result['status'][idx]='; '.join(messages) if messages else ('ok' if converged else 'estimated error exceeds target')
    return result


def detector_pixels(columns, center, right, up, u_edges, v_edges, sun, observer,
                    subrays=2, **source_options):
    """Average equal-area midpoint subrays over fixed orthographic pixel faces.

    right/up must be orthonormal and perpendicular to observer. Edges specify
    fixed projected physical footprints, in meters. Increasing subrays keeps
    those same footprints. Returns source error separately from area refinement.
    Tau and shadow maps are averages of per-ray maps; I/F is mean radiance.
    source_converged concerns only source integration, never pixel-area accuracy.
    """
    observer=direction(observer);right=np.asarray(right,float);up=np.asarray(up,float);center=np.asarray(center,float)
    ue,ve=np.asarray(u_edges,float),np.asarray(v_edges,float)
    if (center.shape!=(3,) or right.shape!=(3,) or up.shape!=(3,) or not all(np.all(np.isfinite(v)) for v in (center,right,up,ue,ve))
        or not np.allclose(np.array([right@right,up@up,right@up,right@observer,up@observer]),[1,1,0,0,0],atol=1e-12,rtol=0)
        or ue.ndim!=1 or ve.ndim!=1 or len(ue)<2 or len(ve)<2 or np.any(np.diff(ue)<=0) or np.any(np.diff(ve)<=0)
        or not isinstance(subrays,int) or subrays<1):
        raise ValueError('finite detector basis must be orthonormal and perpendicular to observer; ordered edges and positive integer subrays required')
    result={name:np.zeros((len(ve)-1,len(ue)-1)) for name in ('I_over_F','tau_camera','sun_transmission_weighted','shadow_fraction','source_error_estimate')}
    result['source_converged']=np.ones(result['I_over_F'].shape,bool);result['evaluations']=np.zeros(result['I_over_F'].shape,int)
    result['failed_subrays']=np.zeros(result['I_over_F'].shape,int)
    for j in range(len(ve)-1):
        for i in range(len(ue)-1):
            f=(np.arange(subrays)+.5)/subrays
            u,v=np.meshgrid(ue[i]+f*(ue[i+1]-ue[i]),ve[j]+f*(ve[j+1]-ve[j]))
            points=center+u[...,None]*right+v[...,None]*up
            rays=adaptive_rays(columns,points,sun,observer,**source_options)
            for name in ('I_over_F','tau_camera','sun_transmission_weighted','shadow_fraction','source_error_estimate'):
                result[name][j,i]=np.mean(rays[name])
            result['source_converged'][j,i]=np.all(rays['converged'])
            result['failed_subrays'][j,i]=np.count_nonzero(~rays['converged'])
            result['evaluations'][j,i]=np.sum(rays['evaluations'])
    return result
