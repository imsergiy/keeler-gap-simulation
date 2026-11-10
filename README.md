# Keeler gap simulation

This is a research prototype for Saturn's Keeler gap. It does not yet reproduce the complete observed ring response.

The three scientific goals are:

1. Measure the persistent ring orbit change after an encounter with Daphnis, separately from radial oscillation and observed edge retreat.
2. Explain vertical excitation and damping with a justified collision and transport model.
3. Compare predicted geometry and brightness with Cassini observations, with explicit illumination and viewing geometry.

The current components include independent numerical references and descriptive calibrated image analysis. They do not form a coupled solution.

| Component | Current status | Main limit |
| --- | --- | --- |
| `baseline/` | Global REBOUND gravity control, analytic no-moon control, timestep and sampling checks | Collision-free, massless, circular and coplanar; no vertical excitation or damping |
| `collision/` | Finite-grain Gaussian collision source with explicit native build and reference parity | Assumed molecular chaos and Gaussian columns; no spatial transport solver or dense-correlation calibration |
| `validation/` | Small signed spring-dashpot contact reference, analytic pair/triple checks | Assumed contact time and restitution; not calibrated ice or a production ring solver |
| `data/` | Verified archive download, restricted VICAR reader and descriptive native-pixel profiles | No crest isolation, calibrated uncertainty, ring-plane navigation or height inversion |
| `observation/` | Opaque-sphere Lambert controls and Gaussian-column single scattering with adaptive source and fixed detector checks | Assumed reflectance; source flags and pixel-area refinement are separate; no calibrated ice brightness or Cassini fit |

## Quick start

Use Python 3.12 and a virtual environment. The dependency versions below are the versions used for the reference checks.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-build.txt
python collision/native/build.py
python -m unittest discover -s tests -v
python baseline/run.py --out results/small --radial 3 --phases 21
python baseline/analyze.py --run
python validation/run.py
python observation/checks.py
python -m observation.columns.checks
python -m observation.columns.adaptive_checks
python data/fetch.py
python data/analyze.py --out data/results
```

Run all commands from the repository root. The observation command includes high-resolution checks and can take several minutes. `baseline/analyze.py --run` runs the five-case matrix and writes plots in `baseline/`. `baseline/run.py` writes trajectories, metrics and a native REBOUND checkpoint to the selected output folder. Contact and observation checks write JSON files next to their scripts. Generated data are ignored by Git. A failed assertion or exception must be investigated before results are used.

The runtime target is 2 GiB RAM, four CPU cores and less than one hour per run. These scripts use one process at a time. Baseline configuration limits bound saved samples and steps; its wall limit is checked at each step, with partial results saved before an exception. The matrix also sets a one-hour subprocess timeout. These are safeguards, not a claim that every allowed custom input meets the resource target. The historical baseline matrix used about 221 MiB peak RAM and 63 seconds in its original environment; rerun it to measure your host. Contact and ray checks are bounded numerical examples, not arbitrary production scene inputs.

## Model limits and next outputs

A tracer count is a quadrature choice, not a physical particle count or optical depth. A radial epicycle returning toward its initial radius does not establish collisional edge retreat. The contact reference uses smooth spheres with no friction or spin, density 500 kg/m³ and an assumed 0.01-second isolated contact duration. Lambert reflection uses a declared diffuse albedo and a parallel Sun. These assumptions require physical calibration.

Planned outputs are an orbit-change and retreat comparison with uncertainty, vertical height and damping profiles, and synthetic brightness profiles at observed geometry. A reduced transport model must be checked independently before it is coupled to these references. No goal is claimed complete.

## Provenance and license

The reference implementations were consolidated from the original Daphnis gravity, contact-validation and light-geometry experiments. Collision build, units and physical assumptions are in `collision/README.md`. Gaussian-column API, units, error flags and physical limits are in `observation/columns/README.md`. Archive URLs and checksums are in `data/products.json`; image limitations and outputs are in `data/README.md`. Detailed numerical contracts and parameter sources are in `baseline/README.txt` and `validation/README.txt`.

Primary sources:

- [Torii et al. (2025), Daphnis and the Keeler gap](https://arxiv.org/abs/2504.10989): context and adopted mass range.
- [JPL satellite physical parameters](https://ssd.jpl.nasa.gov/sats/phys_par/): Saturn GM; [planet physical parameters](https://ssd.jpl.nasa.gov/planets/phys_par.html): Saturn equatorial radius.
- [NASA satellite fact sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html): enclosing Daphnis radius.
- [REBOUND WHFast documentation](https://rebound.hanno-rein.de/integrators/whfast/) and [test particles](https://rebound.hanno-rein.de/ipython_examples/Testparticles/): native integration and inactive tracers.
- [SciPy solve_ivp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html): DOP853 and root events in the contact reference.

This project is licensed under GPL-3.0-or-later. `LICENSE` contains the standard GNU GPL version 3 text. Dependencies retain their own licenses.
