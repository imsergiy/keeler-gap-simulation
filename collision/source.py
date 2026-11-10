"""Gaussian finite-separation binary source, column averaged in Gaussian z."""
import math
import numpy as np
from scipy.integrate import lebedev_rule
from scipy.special import ndtr,pbdv
from numpy.polynomial.legendre import leggauss


def truncated_moments(mean,variance):
    """Return E[X^j 1(X>0)], j=1,2,3, including deterministic variance=0."""
    h,s2=np.broadcast_arrays(mean,variance)
    if np.any(s2<0):raise ValueError('negative normal variance')
    s=np.sqrt(s2);t=np.divide(h,s,out=np.zeros_like(s),where=s>0)
    p=ndtr(t);phi=np.exp(-t*t/2)/np.sqrt(2*np.pi)
    moments=[h*p+s*phi,(h*h+s2)*p+h*s*phi,(h**3+3*h*s2)*p+(h*h+2*s2)*s*phi]
    moments=[np.array(value,copy=True) for value in moments]
    # Negative tails use the parabolic-cylinder integral to avoid cancellation.
    tail=(t < -4)&(s>0)
    tail_prefactor=np.exp(-t*t/4)
    resolved=tail & (tail_prefactor>0)
    for j in [1,2,3]:
        moments[j-1][tail]=0.
        if np.any(resolved):
            moments[j-1][resolved]=math.factorial(j)*s[resolved]**j*tail_prefactor[resolved]*pbdv(-j-1,-t[resolved])[0]/np.sqrt(2*np.pi)
        moments[j-1]=np.where(s==0,np.maximum(h,0)**j,moments[j-1])
    return moments


class GaussianEnskog:
    def __init__(self,order=17,scheme='lebedev'):
        """Lebedev orders7/11/17 have26/50/110 directions; or use Legendre."""
        if scheme=='lebedev':
            n,w=lebedev_rule(order);self.normals=n.T;self.weights=w
        elif scheme=='legendre':
            z,w=leggauss(order);phi=np.arange(2*order)*np.pi/order
            self.normals=np.stack(np.broadcast_arrays(np.sqrt(1-z[:,None]**2)*np.cos(phi),np.sqrt(1-z[:,None]**2)*np.sin(phi),z[:,None]),axis=-1).reshape(-1,3)
            self.weights=np.repeat(w*np.pi/order,2*order)
        else:raise ValueError('scheme must be lebedev or legendre')
        self.dyad=np.einsum('ni,nj->nij',self.normals,self.normals)

    def __call__(self,C,A,a,N,H,e,enhancement=1.,chunk=1024):
        return self._evaluate(C,A,a,N,H,e,enhancement,chunk)

    def loss_gain(self,C,A,a,N,H,e,enhancement=1.,chunk=1024):
        if np.any(self.weights<0):raise ValueError('loss/gain requires nonnegative angular weights')
        return self._evaluate(C,A,a,N,H,e,enhancement,chunk,True)

    def _evaluate(self,C,A,a,N,H,e,enhancement=1.,chunk=1024,loss_gain=False):
        """Return dict Q,P,heat,rate; consistent length,time units, per mass.

        C[...,3,3] is covariance, A[...,3,3]=partial_j u_i, a is scalar
        radius, N[...,] surface number density, H[...] rms column height.
        enhancement is explicit positive column-averaged Enskog Y, not fitted.
        """
        C=np.asarray(C,float);A=np.asarray(A,float)
        if C.shape[-2:]!=(3,3) or A.shape[-2:]!=(3,3):raise ValueError('C,A require final axes3,3')
        shape=np.broadcast_shapes(C.shape[:-2],A.shape[:-2],np.shape(N),np.shape(H),np.shape(enhancement))
        C=np.broadcast_to(C,shape+(3,3)).reshape(-1,3,3);A=np.broadcast_to(A,shape+(3,3)).reshape(-1,3,3)
        N=np.broadcast_to(N,shape).reshape(-1);H=np.broadcast_to(H,shape).reshape(-1);Y=np.broadcast_to(enhancement,shape).reshape(-1)
        if not np.all(np.isfinite(C)) or not np.allclose(C,C.swapaxes(-1,-2),atol=0,rtol=1e-12) or np.any(np.linalg.eigvalsh(C)<0):raise ValueError('C must be finite symmetric positive semidefinite')
        if not np.all(np.isfinite(A)) or not np.isfinite(a) or a<=0 or not 0<=e<=1 or np.any(~np.isfinite(N+H+Y)) or np.any(N<0) or np.any(H<=0) or np.any(Y<=0):raise ValueError('invalid physical input')
        Q=np.empty_like(C);P=np.empty_like(C);heat=np.empty(len(C));rate=np.empty(len(C));n=self.normals
        L=np.empty_like(C) if loss_gain else None;B=np.empty_like(C) if loss_gain else None
        for lo in range(0,len(C),chunk):
            sl=slice(lo,lo+chunk);Cn=np.einsum('bij,nj->bni',C[sl],n)
            normal_variance=2*np.einsum('ni,bni->bn',n,Cn)
            h=-2*a*np.einsum('ni,bij,nj->bn',n,A[sl],n)
            m1,m2,m3=truncated_moments(h,normal_variance)
            weight=self.weights[None,:]*np.exp(-(a*n[None,:,2]/H[sl,None])**2)
            pref=N[sl]/(2*np.sqrt(np.pi)*H[sl])*Y[sl]
            mixed=np.einsum('bn,ni,bnj->bij',weight*m1,n,Cn)
            Q[sl]=2*pref[:,None,None]*a*a*(-2*(1+e)*(mixed+mixed.swapaxes(-1,-2))+.5*(1+e)**2*np.einsum('bn,nij->bij',weight*m3,self.dyad))
            P[sl]=2*(1+e)*pref[:,None,None]*a**3*np.einsum('bn,nij->bij',weight*m2,self.dyad)
            heat[sl]=.5*(1-e*e)*pref*a*a*np.sum(weight*m3,axis=-1)
            rate[sl]=4*pref*a*a*np.sum(weight*m1,axis=-1)
            if loss_gain:
                L[sl]=4*(1+e)*pref[:,None,None]*a*a*np.einsum('bn,nij->bij',weight*m1,self.dyad)
                B[sl]=(1+e)**2*pref[:,None,None]*a*a*np.einsum('bn,nij->bij',weight*m3,self.dyad)
        result={'Q':Q.reshape(shape+(3,3)),'P':P.reshape(shape+(3,3)),'heat':heat.reshape(shape),'rate':rate.reshape(shape)}
        if loss_gain:result.update(L=L.reshape(shape+(3,3)),B=B.reshape(shape+(3,3)))
        return result
