"""Experimental positive finite-size contact step with vacuum-safe traction.

This is a chosen boundary discretization. Its spatial and physical accuracy
must be checked; the gain/loss map proves covariance positivity only.
"""
import numpy as np
import hashlib,sys
from pathlib import Path

def cayley(C,loss,gain,dt):
 eye=np.broadcast_to(np.eye(3),C.shape)
 R=np.linalg.solve(eye+.5*dt*loss,eye)
 T=R@(eye-.5*dt*loss)
 value=T@C@np.swapaxes(T,-1,-2)+dt*R@gain@np.swapaxes(R,-1,-2)
 return .5*(value+np.swapaxes(value,-1,-2))

def overlap(a,b):
 scale=np.maximum(a,b)
 aa=np.divide(a,scale,out=np.zeros_like(a),where=scale>0);bb=np.divide(b,scale,out=np.zeros_like(b),where=scale>0)
 denom=aa*aa+bb*bb
 return np.divide(2*aa*bb,denom,out=np.zeros_like(denom),where=denom>0)

def weighted_gradient(N,U,dx,dy):
 """Input has2 ghost layers; output includes1 ghost layer."""
 n=N[1:-1,1:-1];u=U[1:-1,1:-1]
 wr=overlap(n,N[1:-1,2:]);wl=overlap(n,N[1:-1,:-2])
 wt=overlap(n,N[2:,1:-1]);wb=overlap(n,N[:-2,1:-1])
 A=np.zeros(n.shape+(3,3))
 A[...,0:2,0]=(wr[...,None]*(U[1:-1,2:]-u)+wl[...,None]*(u-U[1:-1,:-2]))/(2*dx)
 A[...,0:2,1]=(wt[...,None]*(U[2:,1:-1]-u)+wb[...,None]*(u-U[:-2,1:-1]))/(2*dy)
 return A

def stress_force(N,T,U,dx,dy):
 """Input has1 ghost layer. Return force and explicit boundary work."""
 wx=overlap(N[1:-1,:-1],N[1:-1,1:]);wy=overlap(N[:-1,1:-1],N[1:,1:-1])
 tx=.5*wx[...,None,None]*(T[1:-1,:-1]+T[1:-1,1:])
 ty=.5*wy[...,None,None]*(T[:-1,1:-1]+T[1:,1:-1])
 force=-(tx[:,1:,:,0]-tx[:,:-1,:,0])/dx-(ty[1:,:,:,1]-ty[:-1,:,:,1])/dy
 bj=float(dy*np.sum(tx[:,-1,1,0]-tx[:,0,1,0])+dx*np.sum(ty[-1,:,1,1]-ty[0,:,1,1]))
 # Half-pair boundary work is the exact partner of the weighted cell gradient.
 def cross(w,Ti,Tg,ui,ug,k):return .5*w*(np.sum(ui*Tg[...,0:2,k],axis=-1)+np.sum(ug*Ti[...,0:2,k],axis=-1))
 right=cross(wx[:,-1],T[1:-1,-2],T[1:-1,-1],U[1:-1,-2],U[1:-1,-1],0)
 left=cross(wx[:,0],T[1:-1,1],T[1:-1,0],U[1:-1,1],U[1:-1,0],0)
 top=cross(wy[-1,:],T[-2,1:-1],T[-1,1:-1],U[-2,1:-1],U[-1,1:-1],1)
 bottom=cross(wy[0,:],T[1,1:-1],T[0,1:-1],U[1,1:-1],U[0,1:-1],1)
 be=float(dy*np.sum(right-left)+dx*np.sum(top-bottom))
 return force,bj,be

class ContactOperator:
 def __init__(self,evaluator,physics):
  self.evaluator=evaluator;self.physics=physics;self.last={};self.evaluations=0;self.failed_candidate=None
  owner=getattr(evaluator,'__self__',None);module=sys.modules[evaluator.__module__]
  self.provenance={'contact_operator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'physics_source_path':physics.__file__,'physics_source_sha256':hashlib.sha256(Path(physics.__file__).read_bytes()).hexdigest(),'backend_module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'backend_module':evaluator.__module__,'evaluator':evaluator.__name__,'threads':getattr(owner,'threads',None),'directions':len(owner.weights) if hasattr(owner,'weights') else None,'library_sha256':hashlib.sha256(Path(owner.library._name).read_bytes()).hexdigest() if hasattr(owner,'library') else None}
  self.dependency_bytes={'contact_operator.py':Path(__file__).read_bytes(),'physics.py':Path(physics.__file__).read_bytes(),'backend.py':Path(module.__file__).read_bytes()}
  if hasattr(owner,'library'):self.dependency_bytes['libgaussian.so']=Path(owner.library._name).read_bytes()
  self.angular_rule=(owner.normals.copy(),owner.weights.copy()) if hasattr(owner,'normals') else None

 def padded(self,q,c):
  x,y,dx,dy=self.physics.geometry(c);xe=x[0,0]+(np.arange(c.nx+4)-2)*dx;ye=y[0,0]+(np.arange(c.ny+4)-2)*dy
  ext=self.physics.initial(xe[None,:],ye[:,None],c);ext[:,2:-2,2:-2]=q
  n,u,w,uy,xx,xy,yy,*_=self.physics.primitive(q,x)
  for side,normal,variance,sign in [('left',u[:,:1],xx[:,:1],-1),('right',u[:,-1:],xx[:,-1:],1)]:
   outgoing=sign*normal>np.sqrt(3*np.maximum(variance,0))
   if side=='left':ext[:,2:-2,:2]=np.where(outgoing[None,...],q[:,:,:1],0.)
   else:ext[:,2:-2,-2:]=np.where(outgoing[None,...],q[:,:,-1:],ext[:,2:-2,-2:])
  for side,row,sign in [('bottom',0,-1),('top',-1,1)]:
   outgoing=sign*uy[row,:]>np.sqrt(3*np.maximum(yy[row,:],0))
   if side=='bottom':ext[:,:2,2:-2]=np.where(outgoing[None,None,:],q[:,:1,:],0.)
   else:ext[:,-2:,2:-2]=np.where(outgoing[None,None,:],q[:,-1:,:],ext[:,-2:,2:-2])
  # Vacuum incoming corners have no contact traction; top/right are reservoirs.
  ext[:,:2,:2]=0.;ext[:,:2,-2:]=0.;ext[:,-2:,:2]=0.
  return ext,xe,ye,dx,dy
 def evaluate(self,q,c):
  self.evaluations+=1
  ext,xe,ye,dx,dy=self.padded(q,c);n,u,w,uy,xx,xy,yy,h,s,W=self.physics.primitive(ext,xe[None,:])
  U=np.stack([u,uy],axis=-1);A=weighted_gradient(n,U,dx,dy)
  I=h[1:-1,1:-1];S=s[1:-1,1:-1];A[...,2,2]=np.divide(S,I,out=np.zeros_like(S),where=I>0)
  C=np.zeros(I.shape+(3,3));C[...,0,0]=xx[1:-1,1:-1];C[...,0,1]=C[...,1,0]=xy[1:-1,1:-1];C[...,1,1]=yy[1:-1,1:-1];C[...,2,2]=W[1:-1,1:-1]-np.divide(S*S,I,out=np.zeros_like(S),where=I>0)
  nc=n[1:-1,1:-1];occ=nc>0
  out={key:np.zeros_like(C) for key in ['Q','P','L','B']};out.update(heat=np.zeros_like(I),rate=np.zeros_like(I))
  if occ.any():
   value=self.evaluator(C[occ],A[occ],.5/self.physics.L,self.physics.N0*self.physics.L**2*nc[occ],np.sqrt(I[occ]),c.restitution,enhancement=1.)
   for key in out:out[key][occ]=value[key]
  T=nc[...,None,None]*out['P'];force,bj,be=stress_force(nc,T,U[1:-1,1:-1],dx,dy)
  interior=(slice(1,-1),slice(1,-1));v={key:value[interior] for key,value in out.items()}
  return v,force,bj,be
 def timestep_bound(self,q,c,x):
  if not c.collisions:return np.inf
  v,_,_,_=self.evaluate(q,c);n,u,w,uy,xx,xy,yy,*_=self.physics.primitive(q,x)
  dx=(c.xhi-c.xlo)/c.nx;dy=2*c.yhalf/c.ny
  ax=np.abs(u)+np.sqrt(3*np.maximum(xx+2*v['P'][...,0,0],0));ay=np.abs(uy)+np.sqrt(3*np.maximum(yy+2*v['P'][...,1,1],0))
  nu=2*(.5/self.physics.L)**2*float(np.trace(v['L'],axis1=-2,axis2=-1).max())
  visc=.2/(8*nu*(1/dx**2+1/dy**2)) if nu>0 else np.inf
  I=self.physics.primitive(q,x)[7];S=self.physics.primitive(q,x)[8]
  frequency=np.sqrt(np.divide(v['P'][...,2,2],I,out=np.zeros_like(I),where=I>0)).max()
  strain=np.abs(np.divide(S,I,out=np.zeros_like(I),where=I>0)).max()
  return min(c.cfl/(ax.max()/dx+ay.max()/dy),visc,.2/frequency if frequency>0 else np.inf,.2/strain if strain>0 else np.inf)
 def __call__(self,q,c,x,dt):
  if not c.collisions:
   self.last={'canonical_increment':np.zeros_like(q[0]),'canonical_boundary':0.,'energy_boundary':0.,'energy_change':0.,'energy_residual':0.,'global_dt_bound':np.inf,'substeps':0}
   return q.copy(),0.
  y=self.physics.geometry(c)[1];dx=(c.xhi-c.xlo)/c.nx;dy=2*c.yhalf/c.ny
  initial_energy=self.physics.energy(q,c,x,y,dx,dy);q0=q.copy();loss=0.;bjtotal=0.;betotal=0.;substeps=0;time=0.;bound=np.inf
  def centered(a):
   n,u,w,uy,xx,xy,yy,I,S,W=self.physics.primitive(a,x);b=np.divide(S,I,out=np.zeros_like(S),where=I>0)
   C=np.zeros(n.shape+(3,3));C[...,0,0]=xx;C[...,0,1]=C[...,1,0]=xy;C[...,1,1]=yy;C[...,2,2]=W-b*b*I
   return n,u,w,I,b,C
  def restore(a,n,u,w,I,b,C):
   z=a.copy();z[1]=n*u;z[2]=n*w;z[3]=n*(u*u+C[...,0,0]);z[4]=n*(u*w+C[...,0,1]);z[5]=n*(w*w+C[...,1,1]);z[7]=n*b*I;z[8]=n*(C[...,2,2]+b*b*I)
   return z
  while time<dt-1e-16:
   v,F,bj,be=self.evaluate(q,c);n,u,w,I,b,C=centered(q);occ=n>0
   spectral=float(np.max(np.linalg.eigvalsh(v['L'])))
   nu=2*(.5/self.physics.L)**2*float(np.trace(v['L'],axis1=-2,axis2=-1).max())
   h=min(dt-time,.25/spectral if spectral>0 else dt-time,.2/(8*nu*(1/dx**2+1/dy**2)) if nu>0 else dt-time)
   invn=np.divide(1,n,out=np.zeros_like(n),where=occ)
   while True:
    half=cayley(C,v['L'],v['B'],h/2)
    bh=b+.5*h*np.divide(v['P'][...,2,2],I,out=np.zeros_like(I),where=I>0)
    middle=restore(q,n,u+.5*h*F[...,0]*invn,w+.5*h*F[...,1]*invn,I,bh,half)
    vm,Fm,bjm,bem=self.evaluate(middle,c)
    if h*float(np.max(np.linalg.eigvalsh(vm['L'])))<=.25*(1+1e-12):break
    h*=.5
    if h<1e-12:raise ArithmeticError('finite-size source accuracy substep underflow')
   newC=cayley(C,vm['L'],vm['B'],h)
   newb=b+h*np.divide(vm['P'][...,2,2],I,out=np.zeros_like(I),where=I>0)
   new=restore(q,n,u+h*Fm[...,0]*invn,w+h*Fm[...,1]*invn,I,newb,newC)
   self.failed_candidate=new
   if not self.physics.valid(new,x)[0]:raise ArithmeticError('contact step covariance/column realizability failed; no floor')
   loss+=h*float(np.sum(n*vm['heat']));bjtotal+=h*bjm;betotal+=h*bem
   soundx=np.abs(u)+np.sqrt(3*np.maximum(C[...,0,0]+2*vm['P'][...,0,0],0));soundy=np.abs(w-1.5*x)+np.sqrt(3*np.maximum(C[...,1,1]+2*vm['P'][...,1,1],0))
   frequency=np.sqrt(np.divide(vm['P'][...,2,2],I,out=np.zeros_like(I),where=I>0)).max()
   strain=np.abs(newb).max()
   bound=min(bound,.2/(soundx.max()/dx+soundy.max()/dy),.2/frequency if frequency>0 else np.inf,.2/strain if strain>0 else np.inf)
   q=new;time+=h;substeps+=1
  final_energy=self.physics.energy(q,c,x,y,dx,dy)
  self.last={'canonical_increment':q[2]-q0[2],'canonical_boundary':bjtotal,'energy_boundary':betotal,'energy_change':final_energy-initial_energy,'energy_residual':final_energy-initial_energy+betotal+loss*dx*dy,'global_dt_bound':bound,'substeps':substeps}
  return q,loss
