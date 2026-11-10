"""Independent closed-form checks; run with Python to write raw results."""
import json
from pathlib import Path
import numpy as np
from light import render, illumination, population

ROOT=Path(__file__).resolve().parent
E=15.; RHO=.6
rows=[]
for n in (128,256,512):
    for phase in (0,45,90,135):
        a=np.deg2rad(phase)
        expected=2/3*RHO*E*(np.sin(a)+(np.pi-a)*np.cos(a))/np.pi
        image,area=render([([0,0,0],1.,RHO)],(np.sin(a),0,np.cos(a)),n=n,width=2.2)
        measured=float(image.sum()*area)
        rows.append(dict(test='sphere_phase',n=n,phase_deg=phase,expected_W_sr=expected,
                         measured_W_sr=measured,relative_error=(measured-expected)/expected))
        assert abs(measured/expected-1)<.005

# These shadow measurements call lighting on the plane directly: the camera
# must not hide the shadow behind the sphere in a geometric footprint test.
for angle in (3.,10.):
    b=np.deg2rad(angle); radius=.5; h=1.5
    sun=np.array([np.cos(b),0,np.sin(b)])
    # Eliminating t from |(x,y,0)+t*s-(0,0,h)|²=r² gives this ellipse.
    cx=-h/np.tan(b); ax=radius/np.sin(b); ay=radius
    for n in (128,256,512):
        xs=cx+(np.arange(n)+.5)/n*2.2*ax-1.1*ax
        ys=(np.arange(n)+.5)/n*2.2*ay-1.1*ay
        x,y=np.meshgrid(xs,ys)
        p=np.stack((x,y,np.zeros_like(x)),axis=-1)
        normal=np.broadcast_to([0.,0.,1.],p.shape)
        dark=illumination(p,normal,sun,[([0,0,h],radius,RHO)],E)==0
        analytic=((x-cx)/ax)**2+(y/ay)**2<=1
        pixel=2.2*ax/n*2.2*ay/n
        measured=float(dark.sum()*pixel); expected=np.pi*ax*ay
        center=float(x[dark].mean()); semi_x=float((x[dark].max()-x[dark].min())/2)
        semi_y=float((y[dark].max()-y[dark].min())/2)
        rows.append(dict(test='plane_shadow',angle_deg=angle,n=n,center_x_m=center,
                         expected_center_x_m=cx,semiaxis_x_m=semi_x,expected_semiaxis_x_m=ax,
                         semiaxis_y_m=semi_y,expected_semiaxis_y_m=ay,
                         expected_area_m2=expected,measured_area_m2=measured,
                         relative_error=(measured-expected)/expected,
                         analytic_pixel_disagreements=int(np.count_nonzero(dark!=analytic))))
        assert np.array_equal(dark,analytic)
        assert abs(measured/expected-1)<.003
        assert abs(center-cx)<1e-12
        assert abs(semi_x/ax-1)<.02 and abs(semi_y/ay-1)<.02

# Ray-offset sensitivity isolates self-shadow numerical choices from image error.
for offset in (1e-10,1e-8,1e-6):
    image,pixel=render([([0,0,0],1.,RHO)],(1,0,0),n=512,width=2.2,ray_offset_m=offset)
    measured=float(image.sum()*pixel); expected=2*RHO*E/(3*np.pi)
    rows.append(dict(test='ray_offset_sensitivity',ray_offset_m=offset,phase_deg=90,
                     measured_W_sr=measured,expected_W_sr=expected,relative_error=measured/expected-1))
    assert abs(measured/expected-1)<.001
    b=np.deg2rad(3); cx=-1.5/np.tan(b); ax=.5/np.sin(b)
    x,y=np.meshgrid(cx+(np.arange(256)+.5)/256*2.2*ax-1.1*ax,
                     (np.arange(256)+.5)/256*1.1-.55)
    points=np.stack((x,y,np.zeros_like(x)),axis=-1)
    dark=illumination(points,np.broadcast_to([0.,0.,1.],points.shape),
                      (np.cos(b),0,np.sin(b)),[([0,0,1.5],.5,RHO)],E,offset)==0
    expected_mask=((x-cx)/ax)**2+(y/.5)**2<=1
    disagreement=int(np.count_nonzero(dark!=expected_mask))
    rows.append(dict(test='shadow_offset_sensitivity',ray_offset_m=offset,n=256,
                     analytic_pixel_disagreements=disagreement))
    assert disagreement==0

# Axis-Y is a valid observer and must not make the camera basis singular.
image,pixel=render([([0,0,0],1.,RHO)],(0,1,0),observer=(0,1,0),n=512,width=2.2)
measured=float(image.sum()*pixel); expected=2*RHO*E/3
assert np.isfinite(image).all() and abs(measured/expected-1)<.001
rows.append(dict(test='axis_y_camera',expected_W_sr=expected,measured_W_sr=measured,
                 relative_error=measured/expected-1))
for direction in ((0,0,0),(np.nan,0,1),(0,np.inf,1)):
    for field in ('sun','observer'):
        kwargs=dict(sun=(0,0,1),observer=(0,0,1))
        kwargs[field]=direction
        try:
            render([([0,0,0],1.,RHO)],n=8,**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {field} direction accepted")
rows.append(dict(test='invalid_directions',rejected_cases=6))

# Rotating observer and Sun together preserves phase and flux for a sphere.
v=np.array([np.sin(np.deg2rad(35)),0,np.cos(np.deg2rad(35))])
sun=np.array([np.sin(np.deg2rad(80)),0,np.cos(np.deg2rad(80))])
image,pixel=render([([0,0,0],1.,RHO)],sun,observer=v,n=512,width=2.2)
a=np.pi/4
expected=2/3*RHO*E*(np.sin(a)+(np.pi-a)*np.cos(a))/np.pi
measured=float(image.sum()*pixel)
rows.append(dict(test='tilted_camera',observer_elevation_deg=55,phase_deg=45,
                 expected_W_sr=expected,measured_W_sr=measured,relative_error=measured/expected-1))
assert abs(measured/expected-1)<.001
# Numerical integral over outgoing Lambert directions recovers rho times input.
mu,w=np.polynomial.legendre.leggauss(32)
hemisphere_integral=2*np.pi*np.sum(w/2*((mu+1)/2))/np.pi
assert abs(hemisphere_integral-1)<1e-14
rows.append(dict(test='lambert_energy_integral',hemisphere_factor=float(hemisphere_integral),expected=1.))

# Equal scaled grids remove sampling differences from radius² and albedo controls.
base,area=render([([0,0,0],1.,RHO)],(0,0,1),n=256,width=2.2)
base_flux=base.sum()*area
for radius,albedo in ((.5,.6),(1.,.3),(1.,0.)):
    image,pixel=render([([0,0,0],radius,albedo)],(0,0,1),n=256,width=2.2*radius)
    ratio=float(image.sum()*pixel/base_flux); expected=(radius**2)*(albedo/RHO)
    rows.append(dict(test='scaling',radius_m=radius,albedo=albedo,ratio=ratio,expected_ratio=expected))
    assert abs(ratio-expected)<1e-12

# Front black sphere covers the rear reflecting sphere in the observer direction.
image,_=render([([0,0,0],.5,RHO),([0,0,2],.5,0.)],(0,0,1),width=2.,n=256)
rows.append(dict(test='camera_occultation',max_radiance=float(image.max())))
assert image.max()==0
# Sunward blocker is outside the camera image; the visible target becomes dark.
image,_=render([([0,0,0],.5,RHO),([3,0,0],1.,RHO)],(1,0,0),width=1.2,n=256)
assert image.max()==0
rows.append(dict(test='rendered_full_shadow',max_radiance=float(image.max())))
# A larger sunward sphere blocks all sunlight on a smaller, separately sampled sphere.
p=np.array([[0.,0.,.5],[.2,0.,np.sqrt(.25-.04)]])
norm=p/.5
light=illumination(p,norm,(0,0,1),[([0,0,0],.5,RHO),([0,0,3],1.,RHO)],E)
assert np.all(light==0)
rows.append(dict(test='sun_occlusion',sample_irradiances_W_m2=light.tolist()))
# A flat surface receives E sin(B), with reflected hemispheric flux rho E sin(B).
for angle in (3.,10.):
    b=np.deg2rad(angle)
    measured=float(illumination(np.array([[0.,0.,0.]]),np.array([[0.,0.,1.]]),
                                (np.cos(b),0,np.sin(b)),[],E)[0])
    assert abs(measured-E*np.sin(b))<1e-14
    rows.append(dict(test='plane_energy',angle_deg=angle,incident_W_m2=measured,
                     reflected_W_m2=RHO*measured,absorbed_W_m2=(1-RHO)*measured))
# Same total solid volume: eight half-meter spheres versus one meter sphere.
pops={'one_1m':population([1.]),'eight_0.5m':population([.5]*8)}
assert pops['eight_0.5m']['mass_kg']==pops['one_1m']['mass_kg']
assert pops['eight_0.5m']['cross_section_m2']==2*pops['one_1m']['cross_section_m2']
params=dict(units='SI',solar_irradiance_W_m2=E,assumed_diffuse_reflectance=RHO,
            resolutions=[128,256,512],shadow_sphere_center_m=[0,0,1.5],shadow_radius_m=.5,
            shadow_sun_elevations_deg=[3,10],phase_sphere_center_m=[0,0,0],phase_radius_m=1.,
            orthographic_observer_direction=[0,0,1],camera_distance_m=100.,
            ray_start_offset_m=1e-8,ray_start_sensitivity_m=[1e-10,1e-8,1e-6],
            minimum_positive_ray_hit_m=1e-9,
            shadow_extent='2.2 times analytic semiaxes',phase_image_width_m=2.2)
(ROOT/'parameters.json').write_text(json.dumps(params,indent=2)+'\n')
(ROOT/'results.json').write_text(json.dumps(dict(checks=rows,populations=pops),indent=2)+'\n')
fine=[r for r in rows if r.get('n')==512]
summary=dict(status='all checks passed',checks=len(rows),
 max_finest_sphere_relative_error=max(abs(r['relative_error']) for r in fine if r['test']=='sphere_phase'),
 max_finest_shadow_area_relative_error=max(abs(r['relative_error']) for r in fine if r['test']=='plane_shadow'),
 shadow_pixel_disagreements=sum(r.get('analytic_pixel_disagreements',0) for r in rows),
 scope='Numerical observation reference only; not calibrated Saturn photometry or a Keeler brightness explanation.',
 omitted=['multiple scattering','diffraction','finite Sun angular size','ice phase function',
          'transmission','surface roughness','polarization','instrument PSF','dynamics'])
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
