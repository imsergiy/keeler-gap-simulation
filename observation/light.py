"""Opaque spheres, parallel Sun and orthographic observer; SI meters and watts."""
import numpy as np


def unit(v):
    v = np.asarray(v, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or not np.any(v):
        raise ValueError("direction must be a finite nonzero three-vector")
    scaled = v / np.max(np.abs(v))
    return scaled / np.linalg.norm(scaled)


def hit_sphere(origins, direction, center, radius):
    q = origins - center
    b = q @ direction
    c = np.sum(q*q, axis=-1) - radius*radius
    disc = b*b-c
    root = np.sqrt(np.maximum(disc, 0))
    near, far = -b-root, -b+root
    return np.where((disc >= 0) & (far > 1e-9), np.where(near > 1e-9, near, far), np.inf)


def illumination(points, normals, sun, spheres, irradiance=15.0, ray_offset_m=1e-8):
    sun = unit(sun)
    blocked = np.zeros(points.shape[:-1], dtype=bool)
    for center, radius, _ in spheres:
        blocked |= np.isfinite(hit_sphere(points+normals*ray_offset_m, sun, np.array(center), radius))
    return irradiance*np.maximum(normals @ sun, 0)*~blocked


def render(spheres, sun, observer=(0,0,1), width=4.0, n=256,
           target=(0,0,0), plane_albedo=None, irradiance=15.0, ray_offset_m=1e-8):
    """Return radiance W m^-2 sr^-1 and projected pixel area m².

    Camera is 100 m from target. Scene must fit in front of it.
    """
    observer = unit(observer)
    axes = np.eye(3)[[1,0,2]]
    reference = axes[np.argmin(np.abs(axes @ observer))]
    right = unit(np.cross(reference, observer))
    up = np.cross(observer, right)
    a = (np.arange(n)+0.5)*width/n-width/2
    x,y = np.meshgrid(a,a)
    origins = np.asarray(target)+100*observer+x[...,None]*right+y[...,None]*up
    direction = -observer
    distance = np.full((n,n), np.inf)
    normals = np.zeros((n,n,3)); rho = np.zeros((n,n))
    if plane_albedo is not None and direction[2] < 0:
        t = -origins[...,2]/direction[2]
        ok = t > 0
        distance[ok] = t[ok]; normals[ok] = (0,0,1); rho[ok] = plane_albedo
    for center, radius, albedo in spheres:
        t = hit_sphere(origins, direction, np.array(center), radius)
        ok = t < distance
        points = origins[ok]+t[ok,None]*direction
        normals[ok] = (points-center)/radius
        rho[ok] = albedo; distance[ok] = t[ok]
    valid = np.isfinite(distance)
    radiance = np.zeros((n,n))
    points = origins[valid]+distance[valid,None]*direction
    radiance[valid] = rho[valid]/np.pi*illumination(points,normals[valid],sun,spheres,irradiance,ray_offset_m)
    return radiance, (width/n)**2


def population(radii, density=500):
    radii = np.asarray(radii)
    return dict(number=len(radii), cross_section_m2=float(np.pi*np.sum(radii**2)),
                volume_m3=float(4*np.pi/3*np.sum(radii**3)),
                mass_kg=float(density*4*np.pi/3*np.sum(radii**3)))
