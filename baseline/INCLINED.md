# Collision-free coherent bending control

```sh
python -m baseline.inclined --out results/inclined
python -m baseline.inclined --out results/inclined-fine --steps 24000
python -m baseline.inclined --out results/inclined-planar --coplanar
```

This separate native global control uses the existing baseline constants and WHFast settings. Tracers start circular in Saturn's equatorial plane. Daphnis starts with eccentricity 0.0000331 and inclination 0.0036 degrees, measured early-Cassini mean-element amplitudes inserted as idealized Kepler initial values. Node longitude is zero; CLI true anomaly `--f-deg` and argument of latitude `--u-deg` are declared orientation choices, not an observed epoch fit. Defaults are 0 and 90 degrees. No artificial height force is added.

The fixed incoming grid is 3 radial samples from +20 to +21 km and 21 arc samples from +60 to +90 km, with declared sigmoid-profile/trapezoid weights. Default horizon is `tau=21.85`, where `tau=Omega*t`. `--end-tau` permits a shorter control. The step count is bounded to 256–24000 and the maximum horizon to 21.85. A coarse step can fail the conservative surface clearance test; inspect `invalid_count` and margins before using trajectories.

`trajectories.npz` retains signed Saturn-relative z and vz, osculating semimajor axis, radius, arc coordinate, moon position, pair energy/angular momentum, weights, sampled closest distances and persistent validity flags. `metrics.json` reports sampled maxima, their times, source hashes and strict interval margins. Output is sampled every ten steps and at the final step; extrema are not root solved. Halving dt also halves output spacing, so timestep checks compare matched times.

Signed cohort mean height measures coherent bending. RMS about that mean is spread across distinct trajectories; it is not local Gaussian thermal H. Total RMS is also reported separately. The eccentric pair has no single rigid rotation rate, so no circular Jacobi conservation claim is made. Pair energy and angular-momentum vector provide appropriate native controls. Current-endpoint distance/speed and an outside-surface acceleration bound provide conservative backward interval clearance checks for both enclosing spheres.

Tests check exact planar z=vz=0, reflected orientation sign reversal, and matched-time timestep refinement. The source audit's full ten-case orientation study found sampled maximum tracer |z| of about 165–586 m; this is a sensitivity result, not a calibrated height prediction. No collisions, damping, ring self-gravity, Saturn J2/precession, other moons, real-image phase reconstruction, or brightness fit is present. This gravity control does not establish localized collisional walls.

Primary parameter source: [Jacobson et al. (2008), Revised Orbits of Saturn's Small Inner Satellites, Table 2](https://doi.org/10.1088/0004-6256/135/1/261). `inclined-sources.json` records uncertainties and reference-frame limits. [JPL mean-element guidance](https://ssd.jpl.nasa.gov/sats/elem/sep.html) warns against using mean elements as ephemerides. [Torii et al. (2025)](https://arxiv.org/abs/2504.10989) supplies context for coherent bending versus localized wall behavior. Photometric phase is not orbital true anomaly or argument of latitude.
