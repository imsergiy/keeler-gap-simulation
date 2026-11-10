"""Single scattering in piecewise constant horizontal Gaussian columns, SI units."""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr, ndtri
from numpy.polynomial.legendre import leggauss


def direction(v):
    v = np.asarray(v, float)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0:
        raise ValueError('direction must be a finite nonzero vector')
    return v / np.linalg.norm(v)


def geometry(alpha_deg, sun_elevation_deg, observer_elevation_deg):
    """Build upper-face directions; reject incompatible common-reference angles."""
    a, bs, bo = np.deg2rad([alpha_deg, sun_elevation_deg, observer_elevation_deg])
    if not (0 <= a <= np.pi and 0 < bs <= np.pi/2 and 0 < bo <= np.pi/2):
        raise ValueError('require alpha in [0,180], elevations in (0,90]')
    numerator = np.cos(a)-np.sin(bs)*np.sin(bo)
    denominator = np.cos(bs)*np.cos(bo)
    if abs(denominator) < 1e-14:
        if abs(numerator) > 1e-12:
            raise ValueError('inconsistent phase and elevations')
        azimuth = 0.0
    else:
        cosine = numerator / denominator
        if abs(cosine) > 1 + 1e-12:
            raise ValueError('inconsistent phase and elevations')
        azimuth = np.arccos(np.clip(cosine, -1, 1))
    return np.array([np.cos(bs), 0, np.sin(bs)]), np.array([
        np.cos(bo)*np.cos(azimuth), np.cos(bo)*np.sin(azimuth), np.sin(bo)])


def lambert_phase(alpha):
    """P integrates to 4 pi; alpha is Sun-target-observer phase in radians."""
    return 8/(3*np.pi)*(np.sin(alpha)+(np.pi-alpha)*np.cos(alpha))


def opacity(radius_m=0.5, material_density_kg_m3=500):
    """Geometric extinction area / grain mass, no diffraction extinction factor."""
    if radius_m <= 0 or material_density_kg_m3 <= 0:
        raise ValueError('positive radius and material density required')
    return 3/(4*radius_m*material_density_kg_m3)


@dataclass
class Columns:
    x_edges: np.ndarray
    y_edges: np.ndarray
    sigma: np.ndarray
    height: np.ndarray
    mean_z: np.ndarray | None = None
    kappa: float = opacity()

    def __post_init__(self):
        self.x_edges, self.y_edges = np.asarray(self.x_edges,float), np.asarray(self.y_edges,float)
        self.sigma, self.height = np.asarray(self.sigma,float), np.asarray(self.height,float)
        self.mean_z = np.zeros_like(self.sigma) if self.mean_z is None else np.asarray(self.mean_z,float)
        shape = (len(self.y_edges)-1, len(self.x_edges)-1)
        if (self.sigma.shape != shape or self.height.shape != shape or self.mean_z.shape != shape
            or not all(np.all(np.isfinite(a)) for a in [self.x_edges,self.y_edges,self.sigma,self.height,self.mean_z])
            or np.any(np.diff(self.x_edges)<=0) or np.any(np.diff(self.y_edges)<=0)
            or np.any(self.sigma<0) or np.any(self.height<=0) or not np.isfinite(self.kappa) or self.kappa<0):
            raise ValueError('invalid columns: finite ordered edges, Sigma>=0, H>0, kappa>=0 required')

    def segments(self, point, ray, minimum=-np.inf):
        """Exact horizontal cell crossings; Gaussian extends to infinite z."""
        low, high = minimum, np.inf
        crossings = []
        for axis, edges in enumerate([self.x_edges,self.y_edges]):
            if abs(ray[axis]) < 1e-14:
                if not edges[0] <= point[axis] < edges[-1]:
                    return []
            else:
                t = (edges-point[axis])/ray[axis]
                low, high = max(low,min(t[0],t[-1])), min(high,max(t[0],t[-1]))
                crossings.extend(t.tolist())
        if low >= high:
            return []
        ts = np.array([low]+sorted(set(t for t in crossings if low<t<high))+[high])
        result=[]
        for a,b in zip(ts[:-1],ts[1:]):
            mid = (a+b)/2 if np.isfinite(a) and np.isfinite(b) else (b-1 if np.isfinite(b) else (a+1 if np.isfinite(a) else 0))
            xy = point[:2]+mid*ray[:2]
            ix,iy = np.searchsorted(self.x_edges,xy[0],side='right')-1,np.searchsorted(self.y_edges,xy[1],side='right')-1
            result.append((a,b,iy,ix))
        return result

    def segment_tau(self, point, ray, a,b,iy,ix):
        h,z,s = self.height[iy,ix],self.mean_z[iy,ix],self.kappa*self.sigma[iy,ix]
        if s == 0:
            return 0.0
        if abs(ray[2]) < 1e-14:
            return s/(np.sqrt(2*np.pi)*h)*np.exp(-0.5*((point[2]-z)/h)**2)*(b-a)
        u,v=(point[2]+a*ray[2]-z)/h,(point[2]+b*ray[2]-z)/h
        # Reflect upper tails before subtracting to avoid cancellation near CDF=1.
        delta = ndtr(-u)-ndtr(-v) if u>0 and v>0 else ndtr(v)-ndtr(u)
        return float(s*delta/ray[2])

    def optical_depth(self, point, ray):
        return sum(self.segment_tau(point,ray,*seg) for seg in self.segments(point,ray,0))

    def render(self, points, sun, observer, omega=0.5, order=16, tau_cut=36):
        """I/F=pi I/Fsun at orthographic rays through points; exterior is vacuum.

        Gaussian segments are exact; scattering uses Gauss-Legendre in camera
        optical depth. tau_cut bounds omitted I/F by omega P exp(-tau_cut)/4.
        """
        sun,observer=direction(sun),direction(observer)
        if sun[2]<=0 or observer[2]<=0 or not 0<=omega<=1 or order<2 or tau_cut<=0:
            raise ValueError('upper-face Sun/observer, omega in [0,1], order>=2 required')
        points=np.asarray(points,float)
        if points.shape[-1]!=3 or not np.all(np.isfinite(points)):
            raise ValueError('finite ray reference points (...,3) required')
        alpha=np.arccos(np.clip(sun@observer,-1,1)); factor=omega*lambert_phase(alpha)/4
        nodes,weights=leggauss(order)
        output=np.zeros(points.shape[:-1]); total=np.zeros_like(output); transmission=np.zeros_like(output)
        for index in np.ndindex(output.shape):
            p=points[index]; segments=self.segments(p,observer); accumulated=0.; value=0.; clear=0.
            total[index]=sum(self.segment_tau(p,observer,*seg) for seg in segments)
            for a,b,iy,ix in reversed(segments):
                dt=self.segment_tau(p,observer,a,b,iy,ix)
                if dt==0:
                    continue
                limit=min(dt,tau_cut-accumulated)
                if limit<=0:
                    break
                q=0.5*limit*(nodes+1); w=0.5*limit*weights
                h,z=self.height[iy,ix],self.mean_z[iy,ix]; s=self.kappa*self.sigma[iy,ix]
                ub=(p[2]+b*observer[2]-z)/h
                # Invert the upper-tail CDF, which increases going away from camera.
                if ub < 0:
                    zz=z+h*ndtri(ndtr(ub)-q*observer[2]/s)
                else:
                    zz=z-h*ndtri(ndtr(-ub)+q*observer[2]/s)
                tt=(zz-p[2])/observer[2]
                sunlight=np.array([self.optical_depth(p+t*observer,sun) for t in tt])
                attenuation=np.exp(-accumulated-q)
                value+=np.sum(w*attenuation*np.exp(-sunlight))
                clear+=np.sum(w*attenuation)
                accumulated+=dt
            output[index]=factor*value
            transmission[index]=value/clear if clear>0 else 1
        return {'I_over_F':output,'tau_camera':total,'sun_transmission_weighted':transmission,
                'shadow_fraction':1-transmission}


def load_columns(path):
    """Load pilot SI center grids; infer uniform cell edges from center spacing."""
    with np.load(path) as raw:
        x,y=raw['x_m'].ravel(),raw['y_m'].ravel()
        def edges(c):
            if len(c)<2 or not np.allclose(np.diff(c),np.diff(c)[0],rtol=1e-8,atol=1e-8):
                raise ValueError('at least two uniform cell centers required')
            return np.r_[c[0]-(c[1]-c[0])/2,(c[:-1]+c[1:])/2,c[-1]+(c[-1]-c[-2])/2]
        return Columns(edges(x),edges(y),raw['Sigma_kg_m2'],raw['H_m'],raw['meanZ_m'] if 'meanZ_m' in raw else None)
