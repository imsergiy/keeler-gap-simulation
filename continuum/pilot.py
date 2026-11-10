"""Dilute Gaussian raw-moment Hill control. Units: km, Omega^-1, N/N0."""
from dataclasses import dataclass, asdict
import argparse, json, time, resource, hashlib
from .contact_operator import ContactOperator
from pathlib import Path
import numpy as np
G=6.67430e-11
MD=6.8e13
A=136506000.
OMEGA=np.sqrt((3.793120623e16+G*MD)/A**3)
L=1000.
MU=G*MD/(OMEGA**2*L**3)
N0=(np.pi/np.sqrt(2)*np.sqrt(.75)/8)/(np.pi*.5**2)
# n, n ux, n w, n ux², n ux w, n w², n H², n S, n W; w=uy+1.5x.
@dataclass
class Config:
    nx:int=64
    ny:int=256
    end:float=.15
    cfl:float=.15
    moon:bool=True
    collisions:bool=True
    xlo:float=5.
    xhi:float=45.
    yhalf:float=100.
    y_center_km:float=0.
    width:float=2.
    restitution:float=.5
    ring_gravity:bool=False
    output:str='result'
    max_wall_seconds:float=3300.
    edge_km:float=20.
    tag_width_km:float=1.
    tag_ymin_km:float=60.
    tag_ymax_km:float=90.

def initial(x,y,c):
    x,y=np.broadcast_arrays(x,y)
    dx=(c.xhi-c.xlo)/c.nx;dy=2*c.yhalf/c.ny
    integral=lambda z:.5*c.width*np.logaddexp(0,2*(z-c.edge_km)/c.width)
    n=(integral(x+dx/2)-integral(x-dx/2))/dx
    xl=np.maximum(x-dx/2,c.edge_km);xr=np.minimum(x+dx/2,c.edge_km+c.tag_width_km)
    tagged=np.where(xr>xl,(integral(xr)-integral(xl))/dx,0.)
    yf=np.clip((np.minimum(y+dy/2,c.tag_ymax_km)-np.maximum(y-dy/2,c.tag_ymin_km))/dy,0,1)
    tag=tagged*yf
    h=.005; cc=h*h/.75
    # Eq16 at omega/Omega=1/sqrt(2): xx=1.75c², yy=.5c², zz=.75c².
    xx=1.75*cc; yy=.5*cc; xy=cc/(4*np.sqrt(2))
    u=np.zeros_like(x); v=np.zeros_like(x)
    return np.stack([n,n*u,n*v,n*(u*u+xx),n*(u*v+xy),n*(v*v+yy),n*h*h,np.zeros_like(n),n*.75*cc,tag])

def primitive(q,x):
    n=q[0]; safe=np.where(n>0,n,1)
    u=q[1]/safe; v=q[2]/safe
    return n,u,v,v-1.5*x,q[3]/safe-u*u,q[4]/safe-u*v,q[5]/safe-v*v,q[6]/safe,q[7]/safe,q[8]/safe

def valid_cells(q):
    n=q[0];safe=np.where(n>0,n,1)
    u=q[1]/safe;v=q[2]/safe
    xx=q[3]/safe-u*u;xy=q[4]/safe-u*v;yy=q[5]/safe-v*v
    h=q[6]/safe;ss=q[7]/safe;w=q[8]/safe
    occupied=n>0
    cone=(xx>0)&(yy>0)&(xx*yy-xy*xy>0)&(h>0)&(h*w-ss*ss>0)
    vacuum=(n==0)&np.all(q==0,axis=0)
    tag_ok=True if len(q)==9 else (q[9]>=0)&(q[9]<=n)
    return np.all(np.isfinite(q),axis=0)&((occupied&cone)|vacuum)&tag_ok

def valid(q,x):
    n,u,v,uy,xx,xy,yy,h,ss,w=primitive(q,x)
    occupied=n>0
    det=xx*yy-xy*xy;vd=h*w-ss*ss
    return bool(np.all(valid_cells(q))),dict(min_density=float(n.min()),min_horizontal_determinant=float(det[occupied].min()) if occupied.any() else 0.,min_vertical_determinant=float(vd[occupied].min()) if occupied.any() else 0.)

def limited_faces_fluid(q,left,right,lo,hi):
    def mc(a,b):
        same=a*b>0
        return np.where(same,np.sign(a)*np.minimum(np.minimum(2*np.abs(a),2*np.abs(b)),.5*np.abs(a+b)),0.)
    sx=mc(q-np.concatenate([left,q[:,:,:-1]],2),np.concatenate([q[:,:,1:],right],2)-q)
    sy=mc(q-np.concatenate([lo,q[:,:-1,:]],1),np.concatenate([q[:,1:,:],hi],1)-q)
    def good(theta):
        return valid_cells(q+.5*theta*sx)&valid_cells(q-.5*theta*sx)&valid_cells(q+.5*theta*sy)&valid_cells(q-.5*theta*sy)
    full=good(np.ones_like(q[0]));theta=np.ones_like(q[0])
    if not np.all(full):
        bad=~full;qb=q[:,bad];xb=sx[:,bad];yb=sy[:,bad]
        low=np.zeros(np.count_nonzero(bad));high=np.ones_like(low)
        for _ in range(28):
            mid=.5*(low+high)
            ok=valid_cells(qb+.5*mid*xb)&valid_cells(qb-.5*mid*xb)&valid_cells(qb+.5*mid*yb)&valid_cells(qb-.5*mid*yb)
            low=np.where(ok,mid,low);high=np.where(ok,high,mid)
        theta[bad]=low
    # The contraction changes face states only; opposite slopes retain the cell average.
    return q-.5*theta*sx,q+.5*theta*sx,q-.5*theta*sy,q+.5*theta*sy,dict(limited_cells=int(np.count_nonzero(~full)),min_theta=float(theta.min()))

def limited_faces(q,left,right,lo,hi):
    fluid=limited_faces_fluid(q[:9],left[:9],right[:9],lo[:9],hi[:9])
    if len(q)==9:return fluid
    def mc(a,b):
        return np.where(a*b>0,np.sign(a)*np.minimum(np.minimum(2*np.abs(a),2*np.abs(b)),.5*np.abs(a+b)),0.)
    tag=q[9];xm,xp,ym,yp,stats=fluid
    def fraction(a):return np.divide(a[9],a[0],out=np.zeros_like(a[0]),where=a[0]>0)
    chi=fraction(q)
    cx=mc(chi-np.concatenate([fraction(left),chi[:,:-1]],1),np.concatenate([chi[:,1:],fraction(right)],1)-chi)
    cy=mc(chi-np.concatenate([fraction(lo),chi[:-1,:]],0),np.concatenate([chi[1:,:],fraction(hi)],0)-chi)
    tx=chi*(xp[0]-xm[0])+q[0]*cx
    ty=chi*(yp[0]-ym[0])+q[0]*cy
    # Project only tag slopes; a passive label cannot change a fluid reconstruction.
    bx=np.maximum(-2*tag,2*(tag-xm[0]));ex=np.minimum(2*tag,2*(xp[0]-tag))
    by=np.maximum(-2*tag,2*(tag-ym[0]));ey=np.minimum(2*tag,2*(yp[0]-tag))
    sx=np.minimum(np.maximum(tx,bx),ex);sy=np.minimum(np.maximum(ty,by),ey)
    stats['tag_limited_slopes']=int(np.count_nonzero(tx!=sx)+np.count_nonzero(ty!=sy))
    def attach(f,t):return np.concatenate([f,np.minimum(np.maximum(t,0),f[0])[None,...]],axis=0)
    return attach(xm,tag-.5*sx),attach(xp,tag+.5*sx),attach(ym,tag-.5*sy),attach(yp,tag+.5*sy),stats

def vertical_energy(q,c,x,y,dx,dy):
    mu=MU if c.moon else 0
    nu=1+mu/(x*x+y*y)**1.5
    return float(np.sum(.5*(q[8]+nu*q[6]))*dx*dy)

def forcing(q,c,x,y):
    n,u,v,uy,xx,xy,yy,h,ss,w=primitive(q,x)
    mu=MU if c.moon else 0;r2=x*x+y*y;r=np.sqrt(r2)
    fx=-mu*x/r**3+1.5*mu*x*h/r**5;fy=-mu*y/r**3+1.5*mu*y*h/r**5
    physical=n*(u*fx+uy*fy-mu/r**3*ss)
    wave=n*(u*fx+v*fy-mu/r**3*ss)
    spatial=-1.5*mu*q[6]*(u*x+uy*y)/r**5
    shear=1.5*n*(u*v+xy)
    return np.array([physical.sum(),wave.sum(),spatial.sum(),shear.sum()])

def flux(q,x,k):
    n,u,v,uy,xx,xy,yy,h,s,w=primitive(q,x)
    speed=u if k==0 else uy
    cx=xx if k==0 else xy; cy=xy if k==0 else yy
    f=q*speed
    f[1]+=n*cx; f[2]+=n*cy
    f[3]+=2*n*u*cx
    f[4]+=n*(u*cy+v*cx)
    f[5]+=2*n*v*cy
    a=np.abs(speed)+np.sqrt(3*np.maximum(xx if k==0 else yy,0))
    return f,a

def face(left,right,x,k):
    fl,al=flux(left,x,k); fr,ar=flux(right,x,k)
    return .5*(fl+fr)-.5*np.maximum(al,ar)*(right-left)

def geometry(c):
    dx=(c.xhi-c.xlo)/c.nx; dy=2*c.yhalf/c.ny
    x=c.xlo+(np.arange(c.nx)+.5)*dx
    y=c.y_center_km-c.yhalf+(np.arange(c.ny)+.5)*dy
    return x[None,:],y[:,None],dx,dy

def transport(q,c,x,y,dx,dy):
    # Reservoir boundaries: outward supersonic faces copy interior; inward use reservoir;
    # mixed faces use Rusanov interior/reservoir. No periodic encounter or sponge.
    nx,ny=c.nx,c.ny
    xf=np.linspace(c.xlo,c.xhi,nx+1)[None,:]
    left=initial(np.full((ny,1),c.xlo),y,c)*0
    right=initial(np.full((ny,1),c.xhi),y,c)
    lo=initial(x,np.full((1,nx),c.y_center_km-c.yhalf),c)*0
    hi=initial(x,np.full((1,nx),c.y_center_km+c.yhalf),c)
    def outside(interior,reservoir,xb,k,sign):
        _,u,v,uy,xx,xy,yy,*_=primitive(interior,xb)
        vel=u if k==0 else uy; a=np.sqrt(3*np.maximum(xx if k==0 else yy,0))
        return np.where((sign*vel>a)[None,...],interior,reservoir)
    left=outside(q[:,:,:1],left,c.xlo,0,-1)
    right=outside(q[:,:,-1:],right,c.xhi,0,1)
    lo=outside(q[:,:1,:],lo,x,1,-1)
    hi=outside(q[:,-1:,:],hi,x,1,1)
    xm,xp,ym,yp,stats=limited_faces(q,left,right,lo,hi)
    xl=np.concatenate([left,xp],2);xr=np.concatenate([xm,right],2)
    yl=np.concatenate([lo,yp],1);yr=np.concatenate([ym,hi],1)
    fx=face(xl,xr,xf,0);fy=face(yl,yr,x,1)
    stats['max_ax']=float(np.maximum(flux(xl,xf,0)[1],flux(xr,xf,0)[1]).max())
    stats['max_ay']=float(np.maximum(flux(yl,x,1)[1],flux(yr,x,1)[1]).max())
    rhs=-(fx[:,:,1:]-fx[:,:,:-1])/dx-(fy[:,1:,:]-fy[:,:-1,:])/dy
    b=dy*np.sum(fx[:,:,-1]-fx[:,:,0],axis=1)+dx*np.sum(fy[:,-1,:]-fy[:,0,:],axis=1)
    # Flux of total expanded potential energy; consistent with raw moment flux.
    def ef(f,xx,yy):
        nu=1+(MU/(xx*xx+yy*yy)**1.5 if c.moon else 0)
        phi=-1.5*xx*xx-(MU/np.sqrt(xx*xx+yy*yy) if c.moon else 0)
        return .5*(f[3]+f[5]-3*xx*f[2]+2.25*xx*xx*f[0]+f[8]+nu*f[6])+phi*f[0]
    be=dy*np.sum(ef(fx[:,:,-1],c.xhi,y[:,0])-ef(fx[:,:,0],c.xlo,y[:,0]))+dx*np.sum(ef(fy[:,-1,:],x[0],c.y_center_km+c.yhalf)-ef(fy[:,0,:],x[0],c.y_center_km-c.yhalf))
    jfx=fx[2]+.5*xf*fx[0];jfy=fy[2]+.5*x*fy[0]
    jr=-(jfx[:,1:]-jfx[:,:-1])/dx-(jfy[1:,:]-jfy[:-1,:])/dy
    bj=dy*np.sum(jfx[:,-1]-jfx[:,0])+dx*np.sum(jfy[-1,:]-jfy[0,:])
    def vef(f,xx,yy):return .5*(f[8]+(1+(MU/(xx*xx+yy*yy)**1.5 if c.moon else 0))*f[6])
    stats['vertical_boundary']=float(dy*np.sum(vef(fx[:,:,-1],c.xhi,y[:,0])-vef(fx[:,:,0],c.xlo,y[:,0]))+dx*np.sum(vef(fy[:,-1,:],x[0],c.y_center_km+c.yhalf)-vef(fy[:,0,:],x[0],c.y_center_km-c.yhalf)))
    stats['vertical_spatial_work']=float(forcing(q,c,x,y)[2]*dx*dy)
    stats['tag_y_crossing_flux']=float(fy[9,int(round((c.yhalf-c.y_center_km)/dy)),:].sum()*dx) if len(q)>9 else 0.
    def gc(f,xx):return xx+2*np.divide(f[2],f[0],out=np.zeros_like(f[0]),where=f[0]>0)
    outward=[-fx[9,:,0]*dy,fx[9,:,-1]*dy,-fy[9,0,:]*dx,fy[9,-1,:]*dx]
    g=[gc(xm[:,:,0],c.xlo),gc(xp[:,:,-1],c.xhi),gc(ym[:,0,:],x[0]),gc(yp[:,-1,:],x[0])]
    stats['tag_out_mass_flux']=float(sum(np.maximum(f,0).sum() for f in outward))
    stats['tag_in_mass_flux']=float(-sum(np.minimum(f,0).sum() for f in outward))
    stats['tag_out_gc_flux']=float(sum(np.sum(np.maximum(f,0)*gg) for f,gg in zip(outward,g)))
    return rhs,b,float(be),jr,float(bj),stats

def source_step(q,c,x,y,dt):
    n,u,v,uy,xx,xy,yy,h,ss,w=primitive(q,x)
    mu=MU if c.moon else 0
    r=np.sqrt(x*x+y*y);nu=1+mu/r**3
    om=np.sqrt(nu);ch=np.cos(om*dt/2);sh=np.sin(om*dt/2)
    hmid=ch*ch*h+2*ch*sh/om*ss+sh*sh/nu*w
    ax=-mu*x/r**3+1.5*mu*x*hmid/r**5
    ay=-mu*y/r**3+1.5*mu*y*hmid/r**5
    co=np.cos(dt);si=np.sin(dt)
    un=co*u+2*si*v+si*ax+2*(1-co)*ay
    vn=-.5*si*u+co*v+.5*(co-1)*ax+si*ay
    z=q.copy();z[1]=n*un;z[2]=n*vn
    z[3]=n*(un*un+co*co*xx+4*co*si*xy+4*si*si*yy)
    z[4]=n*(un*vn-.5*co*si*xx+(co*co-si*si)*xy+2*co*si*yy)
    z[5]=n*(vn*vn+.25*si*si*xx-co*si*xy+co*co*yy)
    om=np.sqrt(nu);co=np.cos(om*dt);si=np.sin(om*dt)
    z[6]=n*(co*co*h+2*co*si/om*ss+si*si/nu*w)
    z[7]=n*(-om*co*si*h+(co*co-si*si)*ss+co*si/om*w)
    z[8]=n*(nu*si*si*h-2*om*co*si*ss+co*co*w)
    return z,n*ay*dt

def energy(q,c,x,y,dx,dy):
    r=np.sqrt(x*x+y*y); mu=MU if c.moon else 0
    nu=1+mu/r**3; phi=-1.5*x*x-mu/r
    return float(np.sum(.5*(q[3]+q[5]-3*x*q[2]+2.25*x*x*q[0]+q[8]+nu*q[6])+phi*q[0])*dx*dy)

def run(c,collision_step):
    zero_face=(c.yhalf-c.y_center_km)/(2*c.yhalf/c.ny)
    if abs(zero_face-round(zero_face))>1e-10 or not 0<zero_face<c.ny:raise ValueError('y=0 must be an internal face')
    if c.tag_width_km<=0 or not 0<c.tag_ymin_km<c.tag_ymax_km<c.y_center_km+c.yhalf:raise ValueError('invalid material/conjunction geometry')
    if not isinstance(collision_step,ContactOperator):raise ValueError('finite-size model requires its complete contact operator')
    if not isinstance(c.nx,(int,np.integer)) or not isinstance(c.ny,(int,np.integer)) or not np.all(np.isfinite([c.end,c.cfl,c.width,c.xlo,c.xhi,c.yhalf,c.y_center_km,c.max_wall_seconds])) or c.max_wall_seconds<=0 or c.nx<3 or c.ny<3 or c.end<=0 or not 0<c.cfl<=.2 or c.width<=0 or c.xlo<=4.3 or c.xhi<=c.xlo or c.yhalf<=0:
        raise ValueError('invalid grid/time/CFL/domain; domain must stay outside physical 4.3km moon surface')
    out=Path(c.output)
    out.mkdir(parents=True,exist_ok=False)
    source_bytes=Path(__file__).read_bytes();source_hash=hashlib.sha256(source_bytes).hexdigest()
    (out/'executed_source.py').write_bytes(source_bytes)
    if collision_step.provenance['physics_source_sha256']!=source_hash:raise RuntimeError('physics source changed after operator construction')
    dep=out/'executed_dependencies';dep.mkdir()
    for name,data in collision_step.dependency_bytes.items():(dep/name).write_bytes(data)
    if collision_step.angular_rule is not None:np.savez_compressed(dep/'angular_rule.npz',normals=collision_step.angular_rule[0],weights=collision_step.angular_rule[1])
    config_hash=hashlib.sha256(json.dumps(asdict(c),sort_keys=True).encode()).hexdigest()
    (out/'config.json').write_text(json.dumps(asdict(c),indent=2))
    if not 0<=c.restitution<=1:raise ValueError('restitution must be in [0,1]')
    if c.ring_gravity:raise ValueError('Ring gravity is absent; this flag cannot enable a hydrostatic repair.')
    x,y,dx,dy=geometry(c); q=initial(x,y,c); t=0.; step=0; rejects=[]; history=[]
    bsum=np.zeros(len(q)); torque=0.; eb=0.; loss=0.; trans_error=0.; source_error=0.
    compat_work=0.;jboundary=0.;compat_l1=0.;compat_max=0.;limited_count=0;min_theta=1.
    init=q.copy(); e0=energy(q,c,x,y,dx,dy); m0=q[0].sum()*dx*dy;j0=np.sum(q[2]+.5*x*q[0])*dx*dy
    v0=vertical_energy(q,c,x,y,dx,dy);vboundary=0.;vcollision=0.;vspatial=0.;vsource_error=0.
    contact_energy_boundary=0.;contact_energy_residual=0.;contact_substeps=0
    work=np.zeros(4);tag_crossed=0.;tag_cross_time=0.;tag_limited=0;cohort_crossed_999_time=None;tag_exported=0.;tag_export_gc=0.;tag_imported=0.
    physical_extrema=dict(max_volume_fraction=0.,max_a_over_H=0.,max_a_abs_b_over_c=0.,max_H_over_distance=0.,max_a_planar_strain_over_c=0.,max_a_total_strain_over_c=0.)
    cohort_extrema={key:0. for key in physical_extrema};cohort_extrema_any={key:0. for key in physical_extrema};cohort_history_samples=0
    tag0=float(q[9].sum()*dx*dy);tag_gc0=float(np.sum(q[9]*(x+2*np.divide(q[2],q[0],out=np.zeros_like(q[0]),where=q[0]>0)))/q[9].sum()) if q[9].sum()>0 else None
    start=time.perf_counter();next_frame=np.pi/2;frame_index=1
    def save_state(filename):
        np.savez_compressed(out/filename,state=q,initial=init,x=x,y=y,time_omega=t,time_seconds=t/OMEGA,history=np.array(history),boundary=bsum,canonical_boundary=jboundary,moon_torque=torque,energy_boundary=eb+contact_energy_boundary,kinetic_energy_boundary=eb,collision_loss=loss,compatibility_work=compat_work,tag_exported_mass=tag_exported,tag_exported_gc_mass=tag_export_gc)
    save_state('frame_00.npz')
    def collision_call(state,h):
        try:return collision_step(state,c,x,h)
        except (ArithmeticError,ValueError) as error:
            save_state('checkpoint_latest.npz')
            candidate=collision_step.failed_candidate if collision_step.failed_candidate is not None else np.array([])
            np.savez_compressed(out/'failure.npz',last_accepted=q,collision_input=state,failed_candidate=candidate,x=x,y=y)
            (out/'failure.json').write_text(json.dumps(dict(reason=str(error),phase='collision/contact',time_omega=t,dt=h,config=asdict(c),source_sha256=source_hash),indent=2))
            raise

    while t<c.end-1e-14:
        _,ax=flux(q,x,0);_,ay=flux(q,x,1)
        dt=min(c.cfl/(ax.max()/dx+ay.max()/dy),.02,c.end-t)
        dt=min(dt,collision_step.timestep_bound(q,c,x))
        while True:
            qa,dl=collision_call(q,dt/2)
            cj1=qa[2]-q[2];cd1=collision_step.last.copy()
            vcol1=vertical_energy(qa,c,x,y,dx,dy)-vertical_energy(q,c,x,y,dx,dy)
            vbefore1=vertical_energy(qa,c,x,y,dx,dy);fw1=forcing(qa,c,x,y)
            qa,tq1=source_step(qa,c,x,y,dt/2)
            fw1=dt/4*(fw1+forcing(qa,c,x,y))
            vs1=vertical_energy(qa,c,x,y,dx,dy)-vbefore1
            epre=energy(qa,c,x,y,dx,dy)
            tr1,b1,be1,jr1,bj1,ls1=transport(qa,c,x,y,dx,dy)
            stage=qa+dt*tr1
            tested=stage
            ok,_=valid(stage,x)
            ok=ok and dt*(ls1['max_ax']/dx+ls1['max_ay']/dy)<=.2
            if ok:
                tr2,b2,be2,jr2,bj2,ls2=transport(stage,c,x,y,dx,dy)
                qb=.5*qa+.5*(stage+dt*tr2)
                tested=qb
                ok,detail=valid(qb,x)
                ok=ok and dt*(ls2['max_ax']/dx+ls2['max_ay']/dy)<=.2
            if ok:
                vbefore2=vertical_energy(qb,c,x,y,dx,dy);fw2=forcing(qb,c,x,y)
                qb,tq2=source_step(qb,c,x,y,dt/2)
                fw2=dt/4*(fw2+forcing(qb,c,x,y))
                vs2=vertical_energy(qb,c,x,y,dx,dy)-vbefore2
                qc,dl2=collision_call(qb,dt/2)
                cj2=qc[2]-qb[2];cd2=collision_step.last.copy()
                vcol2=vertical_energy(qc,c,x,y,dx,dy)-vertical_energy(qb,c,x,y,dx,dy)
                ecompat_before=energy(qc,c,x,y,dx,dy)
                target=q[2]+.5*x*q[0]+dt*.5*(jr1+jr2)+tq1+tq2+cj1+cj2
                dp=target-(qc[2]+.5*x*qc[0])
                dw=np.divide(dp,qc[0],out=np.zeros_like(dp),where=qc[0]>0)
                qc[5]+=2*qc[2]*dw+qc[0]*dw*dw
                qc[4]+=qc[1]*dw
                qc[2]+=dp
                dcompat=energy(qc,c,x,y,dx,dy)-ecompat_before
                tested=qc;ok,detail=valid(qc,x)
                ok=ok and dt<=min(cd1['global_dt_bound'],cd2['global_dt_bound'])*(1+1e-12)
            if ok:break
            rejects.append(dict(t=t,dt=dt,stage='full covariance'))
            dt*=.5
            if dt<1e-10:
                out=Path(c.output);out.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(out/'failure.npz',last_valid=q,failed_stage=tested,x=x,y=y)
                (out/'failure.json').write_text(json.dumps(dict(config_sha256=config_hash,config=asdict(c),time=float(t),steps=step,rejections=rejects,reason='full covariance realizability; no floor or state reset',realizability=valid(tested,x)[1],source_sha256=source_hash,wall_seconds=time.perf_counter()-start),indent=2))
                raise RuntimeError(f'positivity failure at {t}; checkpoint: {out}/failure.npz')
        et=energy(dt*.5*(tr1+tr2),c,x,y,dx,dy)
        trans_error+=et+dt*.5*(be1+be2)
        es=energy(qc,c,x,y,dx,dy)-energy(q,c,x,y,dx,dy)-et+(dl+dl2)*dx*dy
        source_error+=es-dcompat-(cd1['energy_change']+cd2['energy_change']+(dl+dl2)*dx*dy)
        bsum+=dt*.5*(b1+b2);torque+=np.sum(tq1+tq2)*dx*dy
        jboundary+=dt*.5*(bj1+bj2)+cd1['canonical_boundary']+cd2['canonical_boundary'];compat_work+=dcompat
        contact_energy_boundary+=cd1['energy_boundary']+cd2['energy_boundary'];contact_energy_residual+=cd1['energy_residual']+cd2['energy_residual'];contact_substeps+=cd1['substeps']+cd2['substeps']
        compat_l1+=float(np.abs(dp).sum()*dx*dy);compat_max=max(compat_max,float(np.abs(dw).max()))
        limited_count+=ls1['limited_cells']+ls2['limited_cells'];min_theta=min(min_theta,ls1['min_theta'],ls2['min_theta'])
        eb+=dt*.5*(be1+be2);loss+=(dl+dl2)*dx*dy
        work+=(fw1+fw2)*dx*dy
        vboundary+=dt*.5*(ls1['vertical_boundary']+ls2['vertical_boundary'])
        vspatial+=dt*.5*(ls1['vertical_spatial_work']+ls2['vertical_spatial_work'])
        vcollision+=vcol1+vcol2;vsource_error+=vs1+vs2
        crossed=-dt*.5*(ls1['tag_y_crossing_flux']+ls2['tag_y_crossing_flux'])
        tag_crossed+=crossed;tag_cross_time+=(t+dt/2)*crossed
        if cohort_crossed_999_time is None and tag_crossed>=.999*tag0:cohort_crossed_999_time=t+dt
        tag_limited+=ls1.get('tag_limited_slopes',0)+ls2.get('tag_limited_slopes',0)
        tag_exported+=dt*.5*(ls1['tag_out_mass_flux']+ls2['tag_out_mass_flux'])
        tag_imported+=dt*.5*(ls1['tag_in_mass_flux']+ls2['tag_in_mass_flux'])
        tag_export_gc+=dt*.5*(ls1['tag_out_gc_flux']+ls2['tag_out_gc_flux'])
        q=qc;t+=dt;step+=1
        pn,pu,pv,puy,pxx,pxy,pyy,ph,ps,pw=primitive(q,x)
        pcz=pw-np.divide(ps*ps,ph,out=np.zeros_like(ph),where=ph>0);pc=(pxx+pyy+pcz)/3;occupied=pn>0
        phi=np.divide((4*np.pi*.5**3/3)*N0*pn,np.sqrt(2*np.pi)*np.sqrt(ph)*L,out=np.zeros_like(pn),where=occupied)
        ah=np.divide(.5,L*np.sqrt(ph),out=np.zeros_like(ph),where=occupied)
        bs=np.divide(.5*np.abs(ps),L*ph*np.sqrt(np.maximum(pc,0)),out=np.zeros_like(ph),where=occupied)
        physical_extrema['max_volume_fraction']=max(physical_extrema['max_volume_fraction'],float(phi.max()))
        physical_extrema['max_a_over_H']=max(physical_extrema['max_a_over_H'],float(ah.max()))
        physical_extrema['max_a_abs_b_over_c']=max(physical_extrema['max_a_abs_b_over_c'],float(bs.max()))
        ux_y,ux_x=np.gradient(pu,dy,dx);uy_y,uy_x=np.gradient(puy,dy,dx)
        planar=np.sqrt(ux_x**2+ux_y**2+uy_x**2+uy_y**2)
        pshear=np.divide(.5*planar,L*np.sqrt(pc),out=np.zeros_like(pc),where=occupied)
        totalstrain=np.sqrt(pshear*pshear+bs*bs);hd=np.sqrt(ph/(x*x+y*y))
        diag=dict(max_volume_fraction=phi,max_a_over_H=ah,max_a_abs_b_over_c=bs,max_H_over_distance=hd,max_a_planar_strain_over_c=pshear,max_a_total_strain_over_c=totalstrain)
        for key,value in diag.items():physical_extrema[key]=max(physical_extrema[key],float(value.max()))
        tag_weight=float(q[9].sum())
        if tag_weight>0:
            for key,value in diag.items():cohort_extrema_any[key]=max(cohort_extrema_any[key],float(np.sum(q[9]*value)/tag_weight))
            if tag_weight*dx*dy>=.01*tag0:
                cohort_history_samples+=1
                for key,value in diag.items():cohort_extrema[key]=max(cohort_extrema[key],float(np.sum(q[9]*value)/tag_weight))
        if step%20==0 or t>=c.end-1e-14:
            retained=float(q[9].sum()*dx*dy)
            retained_gc=float(np.sum(q[9]*(x+2*pv))*dx*dy/retained) if retained>0 else np.nan
            history.append([t,energy(q,c,x,y,dx,dy),q[0].sum()*dx*dy,np.sum(q[2]+.5*x*q[0])*dx*dy,retained,retained_gc,tag_exported,tag_export_gc/tag_exported if tag_exported>0 else np.nan,vertical_energy(q,c,x,y,dx,dy)])
        if step%100==0:
            (out/'progress.json').write_text(json.dumps(dict(time_omega=t,time_seconds=t/OMEGA,orbital_periods=t/(2*np.pi),steps=step,wall_seconds=time.perf_counter()-start,physical_extrema=physical_extrema),indent=2))
        if step%500==0:save_state('checkpoint_latest.npz')
        if t>=next_frame:
            save_state(f'frame_{frame_index:02d}.npz');frame_index+=1;next_frame+=np.pi/2
        if time.perf_counter()-start>=c.max_wall_seconds:
            save_state('checkpoint_latest.npz');break
    n,u,v,uy,xx,xy,yy,h,s,w=primitive(q,x)
    sig=4*np.pi*.5**3*500/3*N0*n
    cz=w-s*s/h;cc=(xx+yy+cz)/3
    _,detail=valid(q,x)
    summ=dict(cohort_history_extrema_mass_weighted=cohort_extrema,cohort_history_extrema_any_retained_mass=cohort_extrema_any,cohort_history_valid_samples=cohort_history_samples,cohort_metric_window='retained tagged mass >=1percent initial; all occupied global maxima are never masked',applicability_limits='Ybar1,smooth monodisperse Gaussian molecular chaos; no responding SG/wakes/friction/spin; diagnostics are not physical acceptance',cohort_crossed_999_time_omega=cohort_crossed_999_time,post_conjunction_orbits=(t-cohort_crossed_999_time)/(2*np.pi) if cohort_crossed_999_time is not None else None,coordinate_contract='x is local Hill radial coordinate; y is unwrapped azimuth distance; x+2w is not exact global semimajor axis',max_abs_y_over_orbit_radius=float(np.max(np.abs(y))*L/A),contact_provenance=collision_step.provenance,contact_source_evaluations=collision_step.evaluations,completion='horizon_reached' if t>=c.end-1e-12 else 'wall_cap',physical_history_extrema=physical_extrema,tag_exported_mass=tag_exported,tag_imported_mass=tag_imported,tag_exported_guiding_center_km=tag_export_gc/tag_exported if tag_exported>0 else None,tag_retained_guiding_center_valid=bool(q[9].sum()*dx*dy>1e-8*tag0),collision_closure='finite_size_enskog_control',contact_energy_boundary=contact_energy_boundary,contact_energy_residual=contact_energy_residual,contact_substeps=contact_substeps,reconstruction='MC raw peculiar moments with common full-cone face contraction',face_limiter_cell_energy_work=0.,limited_cell_stages=limited_count,min_face_theta=min_theta,config_sha256=config_hash,source_sha256=source_hash,config=asdict(c),steps=step,time_omega=t,time_seconds=t/OMEGA,wall_seconds=time.perf_counter()-start,peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,positivity_rejections=rejects,realizability=detail,
        mass_initial=m0,mass_final=float(q[0].sum()*dx*dy),mass_boundary=float(bsum[0]),mass_residual=float(q[0].sum()*dx*dy-m0+bsum[0]),canonical_initial=j0,canonical_boundary=jboundary,moon_torque=torque,canonical_residual=float(np.sum(q[2]+.5*x*q[0])*dx*dy-j0+jboundary-torque),canonical_compatibility_energy_work=compat_work,canonical_compatibility_momentum_l1=compat_l1,canonical_compatibility_max_dw=compat_max,energy_initial=e0,energy_final=energy(q,c,x,y,dx,dy),energy_boundary=eb+contact_energy_boundary,kinetic_energy_boundary=eb,collision_loss=loss,source_energy_work=source_error,transport_energy_work_with_boundary=trans_error,energy_residual=energy(q,c,x,y,dx,dy)-e0+eb+contact_energy_boundary+loss,
        max_H_over_distance=float(np.sqrt(h/(x*x+y*y)).max()),max_filling=float((4*np.pi*.5**3/3*N0*n/(np.sqrt(2*np.pi)*np.sqrt(h)*L)).max()),max_vertical_finite_size_strain=float((.5*np.abs(s/h)/(L*np.sqrt(cc))).max()),min_Q=float((np.sqrt(xx)*OMEGA*L*OMEGA/(np.pi*G*sig)).min()),max_omitted_vertical_ring_gradient_ratio=float((4*np.pi*G*sig/(np.sqrt(2*np.pi)*np.sqrt(h)*L)/OMEGA**2).max()),claim='no-ring-gravity dilute Gaussian numerical control; not a validated Keeler closure',
        moon_physical_kinetic_work=float(work[0]),moon_peculiar_wave_work=float(work[1]),background_shear_work=float(work[3]),vertical_energy_initial=v0,vertical_energy_final=vertical_energy(q,c,x,y,dx,dy),vertical_boundary=vboundary,vertical_collision_gain=vcollision,vertical_spatial_work=vspatial,vertical_source_error=vsource_error,vertical_energy_residual=vertical_energy(q,c,x,y,dx,dy)-v0+vboundary-vcollision-vspatial,
        tag_initial_mass=tag0,tag_final_mass=float(q[9].sum()*dx*dy),tag_boundary_mass=float(bsum[9]),tag_mass_residual=float(q[9].sum()*dx*dy-tag0+bsum[9]),tag_min=float(q[9].min()),tag_max_excess_over_N=float(np.max(q[9]-q[0])),tag_initial_band_km=[c.edge_km,c.edge_km+c.tag_width_km,c.tag_ymin_km,c.tag_ymax_km],tag_limited_slopes=tag_limited,tag_initial_guiding_center_km=tag_gc0,tag_final_guiding_center_km=float(np.sum(q[9]*(x+2*v))/q[9].sum()) if q[9].sum()>0 else None,tag_downstream_mass=float(np.sum(q[9]*(y<0))*dx*dy),tag_net_y_crossed_mass=tag_crossed,tag_crossing_plane_km=c.y_center_km-c.yhalf+round((c.yhalf-c.y_center_km)/dy)*dy,tag_flux_weighted_mean_crossing_time_omega=tag_cross_time/tag_crossed if tag_crossed>0 else None,age_tracking='omitted: no individual encounter-age model is included',tag_closure='fractional exact initial profile mass; shares cell mean velocity/height; unresolved tag-position/velocity covariance omitted; net face crossing is not individual encounter age')
    out=Path(c.output);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summ,indent=2));(out/'config.json').write_text(json.dumps(asdict(c),indent=2))
    np.savez_compressed(out/'checkpoint.npz',state=q,initial=init,x=x,y=y,history=np.array(history))
    return summ

def main():
    import sys
    from collision.native.backend import NativeGaussianEnskog
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx',type=int,default=64)
    parser.add_argument('--ny',type=int,default=256)
    parser.add_argument('--end',type=float,default=.15)
    parser.add_argument('--cfl',type=float,default=.15)
    parser.add_argument('--width',type=float,default=2.)
    parser.add_argument('--xlo',type=float,default=5.)
    parser.add_argument('--xhi',type=float,default=45.)
    parser.add_argument('--ylo',type=float,default=-100.)
    parser.add_argument('--yhi',type=float,default=100.)
    parser.add_argument('--restitution',type=float,default=.5)
    parser.add_argument('--edge-km',type=float,default=20.)
    parser.add_argument('--tag-width-km',type=float,default=1.)
    parser.add_argument('--tag-ymin-km',type=float,default=60.)
    parser.add_argument('--tag-ymax-km',type=float,default=90.)
    parser.add_argument('--no-moon',action='store_true')
    parser.add_argument('--no-collisions',action='store_true')
    parser.add_argument('--ring-gravity',action='store_true')
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-wall-seconds',type=float,default=3300.)
    parser.add_argument('--threads',type=int,choices=[1,2,4],default=1)
    args=parser.parse_args()
    config=Config(nx=args.nx,ny=args.ny,end=args.end,cfl=args.cfl,width=args.width,
                  xlo=args.xlo,xhi=args.xhi,yhalf=(args.yhi-args.ylo)/2,
                  y_center_km=(args.yhi+args.ylo)/2,restitution=args.restitution,
                  edge_km=args.edge_km,tag_width_km=args.tag_width_km,
                  tag_ymin_km=args.tag_ymin_km,tag_ymax_km=args.tag_ymax_km,
                  moon=not args.no_moon,collisions=not args.no_collisions,
                  ring_gravity=args.ring_gravity,output=args.output,
                  max_wall_seconds=args.max_wall_seconds)
    operator=ContactOperator(NativeGaussianEnskog(order=17,threads=args.threads).loss_gain,
                             physics=sys.modules[__name__])
    result=run(config,operator)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
