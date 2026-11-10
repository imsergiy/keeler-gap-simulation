DOUBLE CONTACT-LAW NUMERICAL REFERENCE (not SoftIS or a ring solver)

Replay (Python 3.12):
  python -m pip install scipy==1.17.1 numpy==2.5.4
  python run.py
Import: from run import run, cases, coefficients, LIMITS
run(cases()[0], 32) returns one full JSON-compatible case result.
Only resolutions 16 and 32 are defined by the contract. They use max_step
T/16 and T/32, and (rtol,atol)=(1e-10,1e-13),(1e-12,1e-15).

All units SI. Smooth physical spheres, density=500 kg/m^3 is a test assumption.
No spin, tangential force, gravity, enlargement, calibration or scaling claim.
For pair normal n=(xj-xi)/distance, overlap=ri+rj-distance, rate=-(vj-vi).n:
force on j=(k*overlap+c*rate)*n during contact, opposite on i.
Q'=c*rate^2. Signed unloading is retained through zero overlap; no force clamp.
k=mu*((pi/T)^2+gamma^2), c=2*mu*gamma, gamma=-ln(e)/T, T=.01s.
T is a declared isolated head-on duration parameter, not measured ice data.
Different physical masses use pair reduced mass mu=mi*mj/(mi+mj).

Numerical method: float64 SciPy DOP853. Each active set defines a smooth ODE.
Entry/exit roots stop the solver; the force set changes and the ODE restarts.
When the active set is empty, exact quadratic ballistic roots find the first
entry. This prevents missing a complete grazing overlap in free flight.
At restart, an event evaluated at exactly the initial time is assigned the
known contact side (+/-1e-14 m), if abs(gap)<1e-12 m. Positions and time are
never moved by an epsilon. Grouping uses 1e-12 s for ballistic roots and
1e-12 m for coincident active roots with the required direction of motion.
The 1e-5 s near-arrival delays remain distinct. More than 100 restarts or
nonincreasing active-event time is an explicit unresolved failure, not a
fallback. All tested initial states are separated; all finish separated.

Contract and tests:
23 cases at both resolutions (46 runs): pair/triple e=.5 and 1 at three
start-time phases; oblique; unequal physical radii .35/.65m; three unequal
arrival delays; tangent/miss/near-grazing; full 3-D axis-angle rotation;
Galilean boost; an elastic chain with pair 0-1,1-2,0-1 repeated contacts.
Pair/triple analytic oscillator checks independently compare velocity,
contact duration, K and Q. For triple each outer effective mass is m;
pair effective mass is m/2. The center of a symmetric triple stays fixed.
Expected K=m*u_out^2; Q=m*u_in^2-K (u denotes individual outer speed).
These formulas independently explain why triple restitution is not e.
Declared absolute limits: velocity 1e-7 m/s (~1 ppm of .1 m/s), time 1e-9s
(0.1 ppm T), K and Q each 1e-6J (~1 ppm of 1J). Momentum and angular
momentum 1e-7 kg m/s and kg m^2/s are ~4e-9 of one equal body's test scale.
Both grids compare velocities, K and Q separately and exact contact sequence.
The rotated/boosted unequal case compares inverse-transformed velocities,
COM kinetic energy and Q; angular momentum uses the actual 3-D x cross mv.
Invariant energy includes spring U, and never substitutes for K/Q checks.

Scope limit: during active contact SciPy detects sign crossings per step.
This suite's concurrent contacts are collinear triples, not simultaneous
arbitrarily narrow grazing entries. No general completeness bound is claimed
for other active-contact geometries. This runner is a bounded oracle, not a
production event engine. The tangent control has zero force/contact duration.
No outgoing-state oracle is claimed for general oblique/multiple contacts;
convergence and invariants are necessary checks, not proof of correctness.
Two resolutions and a small phase set do not prove a universal error bound.

Provenance:
Prior exact signed model definition: ../softis-evaluation/run.py and README.txt.
Independent pair/triple oracle: ../contact-review/review.txt and audit.py.
Published model: https://academic.oup.com/mnras/article/521/1/638/7040577
SciPy event/precision API verified from primary current documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
The version-specific web page could not be retrieved. The installed 1.17.1
primary API docstring is saved as scipy-solve_ivp-doc.txt and checked as well.
SciPy documentation explicitly warns that multiple zero crossings in one
step can be missed. DOP853 is recommended there for high precision.
The executable source is run.py; summary.json includes its exact SHA256,
NumPy/SciPy versions, failures, analytic maxima and frame checks.
results.json retains every complete input, final state, root, and failure.

Timing/RSS: one serial run of main(), after Python/NumPy/SciPy imports,
including all simulations and checks but excluding JSON serialization.
RSS is process lifetime high-water RSS in Linux KiB, including imports and
inherited process high-water state; it is not incremental solver storage.
No production performance, particle count, or ring memory estimate follows.
Most important independent review: verify event-side restarts and grouping,
then derive pair/triple analytic outputs without using runner oracle code.
