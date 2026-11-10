DAPHNIS GLOBAL GRAVITY CONTROL — FIRST RUNNABLE PILOT

Purpose
This is a collision-free gravitational control. It measures a single encounter
and later radial oscillation separately from osculating semimajor-axis change.
It does not establish an explanation of observed Keeler edge retreat.

Run (from this directory)
  python analyze.py --run
  python analyze.py
Small independent case (63 tracers, all three orbital periods):
  python run.py --out results/small --radial 3 --phases 21
No-moon small case:
  python run.py --out results/small-no-moon --radial 3 --phases 21 --no-moon
Import contract:
  from run import Config, build, run
  metrics = run(Config(radial_samples=3, phase_samples=21), 'results/custom')
Config is a frozen dataclass. configs.json defines the comparison matrix.
CLI --config FILE loads one Config object; it replaces CLI numerical defaults.
Seed 0 is recorded; the grid is deterministic and uses no random sampling.
Each run has a 3500-second wall cap checked every step and a subprocess
3600-second cap in the suite. A partial checkpoint is saved on an internal cap.
checkpoint.bin stores the native REBOUND state at each complete orbit and end.
It can be opened with rebound.Simulation(path). It is not a full diagnostic
resume file: the previous closest-approach history is in trajectories.npz.

Physical setup and sources (checked 2026-11-09)
SI units. G=6.67430e-11. Saturn GM=3.793120623e16 m^3/s^2 (SAT441)
from https://ssd.jpl.nasa.gov/sats/phys_par/ ; mass is this GM divided by G.
Saturn enclosing spherical surface radius=60268 km, its equatorial radius:
https://ssd.jpl.nasa.gov/planets/phys_par.html .
Daphnis adopted mass=6.8e13 kg; a=136506 km. This is an estimate within the
1.0–1.6e-13 mass-ratio interval in Table 1 of Torii et al. (2025),
https://arxiv.org/abs/2504.10989 ; it is not a claim of exact measured mass.

Daphnis has an enclosing radius=4.3 km (largest tabulated semiaxis):
https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html .
NASA gives a mean radius 3.8 km and orbital distance about136500 km:
https://science.nasa.gov/saturn/moons/daphnis/ .
No moon/rock mass or orbital radius is artificially enlarged.

The native massive pair starts circular and coplanar. In the no-moon control,
the second body has zero mass and follows the matching circular reference orbit.
All tracers start on exactly circular Saturn-relative orbits: radial offsets
15–25 km and initial moon-frame arc distances -300 to +200 km, inclusive.
Positive arc distance is upstream: outer circular orbits lag the moon and
move toward decreasing arc distance. Tracers at +60 to +200 km form the incoming
cohort. They have one conjunction and >=1 orbital period after closest approach
in this three-period calculation. The initially downstream population is cold
initial data, not a previously encountered or statistically steady wake.
All initial positions are populated once. No boundary reinjection, clipping,
removal, periodic wrapping, or externally prescribed wave occurs. The displayed
arc coordinate is angle wrapped into [-pi,pi]; no sample approaches this cut.
Particles remain global Newtonian orbits, not tangent-plane Hill coordinates.

Massless tracers are quadrature samples of initial conditions. They are not
large physical rocks. No particle count, surface density, or optical depth is
inferred from their number. Cohort means use trapezoid endpoint weights on
the radial and +60 to +200 km phase intervals. Configurations whose grid does
not hit the cohort endpoints should use individual trajectories, not treat
those means as the same quadrature integral. Refinement uses nested grids.

Integrator and cost
REBOUND5.2.2 WHFast, Jacobi coordinates, safe_mode=1, corrector=0.
Saturn is added first; all elements explicitly name Saturn as the primary.
N_active=2 and testparticle_type=0: only the massive pair acts on tracers;
tracers do not act on each other or on the pair. Gravity work is O(N), with
no massless-pair loop or collision search. The default basic gravity solver
and native Kepler drift handle global curvature. The installed Python API
uses sim.integrator.coordinates/safe_mode and sim.steps(1).
Official documentation: https://rebound.hanno-rein.de/integrators/whfast/
https://rebound.hanno-rein.de/ipython_examples/Testparticles/
https://rebound.hanno-rein.de/simulation/variables/
Results are generated locally and are not included in this repository. All final runs
cover3 periods=154358.45s (42.88h physical time), with dt=P/4096 or P/8192.
Outputs are everyP/256; closest approach and surfaces are checked every step.
Costs and installed versions are measured per process in metrics.json;
summary.json and metrics.csv collect the matrix. Peak RSS is the OS ru_maxrss
high-water mark, which can include inherited high-water memory at process exec.
The final artifact excludes old microtest reruns; earlier numerical/contact
controls remain separate evidence and do not validate new collision physics.

Saved diagnostics and validation
trajectories.npz retains every tracer and every output, including initial
x0_m/y0_m, Saturn-relative osculating a_m, radius_m, along_m, distance_m,
delta_a_m, radius-minus-a radial_epicycle_m, closest_m/tclosest_s,
encountered, crossing_count, incoming_cohort, quadrature weights, valid flags,
Jacobi H, massive-pair relative specific energy and angular momentum.
An encounter requires one positive-to-negative conjunction and an interior
closest point. time_since_closest_s is NaN for non-encounters; before an
encounter it is negative. Closest time has at most one timestep sampling
uncertainty; it is not an event-root solve. Initially downstream boundary
minima are not labeled encounters. A no-moon conjunction is geometric only.
Invalidity persists after a possible surface approach. Before a first surface
crossing, a conservative relative-acceleration bound outside both enclosing
spheres and the previous speed bound the per-step possible travel. A possible
intersection is invalid even if sampled endpoints remain outside. Invalid
trajectories are retained, but excluded from cohort statistics. No actual
or possible surface crossing is accepted in the original reference matrix.

The rotating invariant is H=|v_bary|^2/2-GM_S/r_S-GM_D/r_D-n*Lz_bary.
Its absolute drift and drift divided by (n*r_Hill)^2 are reported to avoid
normalizing numerical errors by the much larger Saturn binding energy.
Native two-body specific energy and Lz are also checked. The no-moon control
compares every saved position with an analytic circular orbit and demands
maximum delta-a below1mm and position error below1cm. analyze.py asserts
timestep, sampling, invariant, surface, encounter-label and cost controls.
An osculating a shift at3 periods is a finite-time value, not an asymptotic
scattering integral. No damping is present; radius may come back toward its
initial value as an epicycle while its mean orbit remains shifted.

Boundary of physical claims
No collisions, ring self-gravity, physical particle radii/masses, Saturn J2,
other moons, moon eccentricity/inclination, vertical excitation, material
law, restitution, thermal/radiative forces, light scattering or observation
operator. Starting exactly coplanar means vertical height remains exactlyzero.
This has not fitted Cassini geometry or calibrated brightness. A gravity-only
radial return does not reproduce collisional pullback or explain the observed
edge-retreat profile. Adding a measured reduced collision/transport model and
comparing edge position, height/damping and brightness against observations
are subsequent work, with independent convergence and budget measurements.

Historical reference result (rerun the matrix to reproduce)
The three-period pilot matrix costs62.40s total. The largest run (1809 tracers)
costs30.66s, peak221MiB; the 505-tracer runs cost5.56–10.31s each.
Halving dt changes final delta-a by at most0.241mm and any saved radius by
0.452mm. Nested common trajectories agree exactly. Cohort quadrature mean
delta-a changes324.443 to312.477m with sampling refinement:11.966m or3.83%
of the fine result. This is a measured sampling difference, not an error bound
or evidence of physical completeness. The 20m regression ceiling in analyze.py
is a pilot stability check, not an observational accuracy requirement.
Incoming final delta-a spans83.07–1000.68m; radius-minus-a spans about±4.07km.
The two no-moon controls give maximum delta-a13.03micrometers and analytic
position error1.954mm. Maximum rotating-H drift2.981e-7m^2/s^2 is9.19e-7
of the Hill velocity squared. Massive-pair energy relative drift is below5e-14.
No possible surface crossings occur; minimum moon distance is13.309km.
