"""Explicit ctypes backend; CDLL calls release the Python GIL."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import lebedev_rule

_D=ctypes.POINTER(ctypes.c_double)
_Z=ctypes.c_size_t

def _pointer(array):
    return array.ctypes.data_as(_D)


class NativeGaussianEnskog:
    def __init__(self,order=17,scheme='lebedev',threads=1):
        if threads not in (1,2,4):raise ValueError('threads must be1,2,4')
        self.threads=threads
        if scheme=='lebedev':
            n,w=lebedev_rule(order);n=n.T
        elif scheme=='legendre':
            z,w=leggauss(order);phi=np.arange(2*order)*np.pi/order
            n=np.stack(np.broadcast_arrays(np.sqrt(1-z[:,None]**2)*np.cos(phi),np.sqrt(1-z[:,None]**2)*np.sin(phi),z[:,None]),axis=-1).reshape(-1,3)
            w=np.repeat(w*np.pi/order,2*order)
        else:raise ValueError('scheme must be lebedev or legendre')
        self.normals=np.ascontiguousarray(n);self.weights=np.ascontiguousarray(w)
        self.library=ctypes.CDLL(str(Path(__file__).with_name('libgaussian.so')))
        self.function=self.library.gaussian_batch
        self.function.argtypes=[_Z,_D,_D,_D,_D,_D,ctypes.c_double,ctypes.c_double,_Z,_D,_D,_D,_D,_D,_D,ctypes.POINTER(_Z)]
        self.function.restype=ctypes.c_int
        self.loss_function=self.library.gaussian_batch_loss_gain
        self.loss_function.argtypes=self.function.argtypes+[_D,_D]
        self.loss_function.restype=ctypes.c_int
        self.normal_function=self.library.gaussian_normal_moments
        self.normal_function.argtypes=[ctypes.c_double,ctypes.c_double,_D]
        self.normal_function.restype=ctypes.c_int

    def normal_moments(self,mean,variance):
        result=np.empty(3)
        if self.normal_function(mean,variance,_pointer(result)):
            raise ValueError('invalid normal input')
        return result

    def __call__(self,C,A,a,N,H,e,enhancement=1.):
        return self._evaluate(C,A,a,N,H,e,enhancement)

    def loss_gain(self,C,A,a,N,H,e,enhancement=1.):
        if np.any(self.weights<0):raise ValueError('loss/gain requires nonnegative angular weights')
        return self._evaluate(C,A,a,N,H,e,enhancement,True)

    def _evaluate(self,C,A,a,N,H,e,enhancement=1.,loss_gain=False):
        C=np.asarray(C,float);A=np.asarray(A,float)
        if C.shape[-2:]!=(3,3) or A.shape[-2:]!=(3,3):raise ValueError('C,A require final axes3,3')
        shape=np.broadcast_shapes(C.shape[:-2],A.shape[:-2],np.shape(N),np.shape(H),np.shape(enhancement))
        C=np.ascontiguousarray(np.broadcast_to(C,shape+(3,3)).reshape(-1,3,3));A=np.ascontiguousarray(np.broadcast_to(A,shape+(3,3)).reshape(-1,3,3))
        N=np.ascontiguousarray(np.broadcast_to(N,shape).reshape(-1),dtype=float);H=np.ascontiguousarray(np.broadcast_to(H,shape).reshape(-1),dtype=float);Y=np.ascontiguousarray(np.broadcast_to(enhancement,shape).reshape(-1),dtype=float)
        if not np.all(np.isfinite(C)) or not np.allclose(C,C.swapaxes(-1,-2),atol=0,rtol=1e-12) or np.any(np.linalg.eigvalsh(C)<0):raise ValueError('C must be finite symmetric positive semidefinite')
        if not np.all(np.isfinite(A)) or not np.isfinite(a) or a<=0 or not 0<=e<=1 or np.any(~np.isfinite(N+H+Y)) or np.any(N<0) or np.any(H<=0) or np.any(Y<=0):raise ValueError('invalid physical input')
        Q=np.empty_like(C);P=np.empty_like(C);heat=np.empty(len(C));rate=np.empty(len(C))
        L=np.empty_like(C) if loss_gain else None;B=np.empty_like(C) if loss_gain else None
        def run(bounds):
            lo,hi=bounds;bad=_Z()
            function=self.loss_function if loss_gain else self.function
            extra=(_pointer(L[lo:hi]),_pointer(B[lo:hi])) if loss_gain else ()
            error=function(hi-lo,_pointer(C[lo:hi]),_pointer(A[lo:hi]),_pointer(N[lo:hi]),_pointer(H[lo:hi]),_pointer(Y[lo:hi]),a,e,len(self.normals),_pointer(self.normals),_pointer(self.weights),_pointer(Q[lo:hi]),_pointer(P[lo:hi]),_pointer(heat[lo:hi]),_pointer(rate[lo:hi]),ctypes.byref(bad),*extra)
            return (lo+bad.value) if error else None
        if len(C):
            if self.threads==1:errors=[run((0,len(C)))]
            else:
                edges=np.linspace(0,len(C),min(self.threads,len(C))+1,dtype=int)
                with ThreadPoolExecutor(max_workers=self.threads) as pool:
                    errors=list(pool.map(run,zip(edges[:-1],edges[1:])))
            if any(x is not None for x in errors):raise ValueError('negative normal variance')
        result={'Q':Q.reshape(shape+(3,3)),'P':P.reshape(shape+(3,3)),'heat':heat.reshape(shape),'rate':rate.reshape(shape)}
        if loss_gain:result.update(L=L.reshape(shape+(3,3)),B=B.reshape(shape+(3,3)))
        return result
