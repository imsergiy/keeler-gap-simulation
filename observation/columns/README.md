# Gaussian-column single scattering reference

This is a bounded synthetic observation model, separate from the opaque-sphere renderer in `observation/light.py`. It has analytic source and detector controls. It does not validate continuum dynamics, fit Cassini data, calibrate ice reflectance, or explain Keeler gap brightness. No solver outputs or earlier coarse render previews are included.

From the repository root:

```sh
python -m observation.columns.checks
python -m observation.columns.adaptive_checks
python -m unittest discover -s tests -v
```

Both check commands print JSON to standard output. They test analytic thin/thick slabs with Gaussian H of 1, 5 and 1000 m, vacuum, zero albedo, phase normalization, geometric shadows, fixed detector area means, vacuum-area denominators, invalid geometry, and explicit integration failure. The regression tests run both suites in CI.

## Physical and input contract

Import `Columns`, `geometry` and `opacity` from `observation.columns.model`; import `adaptive_rays` and `detector_pixels` from `observation.columns.adaptive`.

`Columns(x_edges, y_edges, sigma, height, mean_z=None, kappa=.003)` accepts ordered horizontal cell edges in meters and fields of shape `(ny,nx)`. `sigma` is surface mass density in kg/m²; `height` is positive Gaussian standard deviation H in meters; `mean_z` is the Gaussian center in meters, default zero. Within each horizontal cell:

```
rho(z) = Sigma exp[-(z-mean_z)^2/(2 H^2)] / (sqrt(2 pi) H)
chi = kappa rho
```

The horizontal exterior is vacuum; vertical Gaussian tails are infinite. Default opacity is geometric extinction area per grain mass for opaque spheres with radius 0.5 m and material density 500 kg/m³: `kappa=3/(4 a rho_material)=.003 m²/kg`. These are declared assumptions, not measured ice properties. `load_columns(path)` accepts uniform SI center grids `x_m`, `y_m`, fields `Sigma_kg_m2`, `H_m`, and optional `meanZ_m`; it does not validate the dynamics that produced them.

Sun and observer directions point from a scattering point toward the source and camera. Both must have positive z. `geometry(alpha_deg, sun_elevation_deg, observer_elevation_deg)` rejects incompatible phase/elevation angles. No planet or moon occultation is present.

Single scattering uses `I/F = omega P(alpha)/4 * integral exp(-tau_sun-tau_camera) d tau_camera`, with `I/F=pi I/Fsun` and solar flux per area perpendicular to the beam. No extra sine of solar elevation belongs in this volume source. Lambert-sphere `P=8/(3 pi)[sin(alpha)+(pi-alpha)cos(alpha)]` integrates to `4 pi`. For assumed opaque Lambert grains, `omega` equals hemispheric surface reflectance; geometric albedo is `2 omega/3`.

## Source and detector API

`adaptive_rays(columns, points, sun, observer, omega=.5, epsabs=1e-7, epsrel=1e-5, tau_cut=36, limit=200)` accepts physical ray reference points `(...,3)` in meters. Each ray is `point+t*observer`. Return maps include `I_over_F`, `tau_camera`, `sun_transmission_weighted`, `shadow_fraction`, `source_error_estimate`, `converged`, `evaluations`, and `status`. Consumers must retain failures and status; a finite result does not imply convergence.

Optical-depth segments use exact Gaussian CDF differences. Adaptive source quadrature splits at solar cell-face Gaussian features, including narrow cold layers. Tiny intervals use a bounded midpoint contribution. The error estimate combines QUADPACK estimates, bounded midpoint errors and the omitted-camera-tail bound `omega P exp(-tau_cut)/4`. Gaussian vertical tails remain untruncated. QUADPACK estimates are not universal rigorous bounds. Exceptions and failed estimates have no fixed-order fallback. `Columns.render` remains a fixed-order pencil-ray reference; its finite output alone does not establish source convergence.

`detector_pixels(columns, center, right, up, u_edges, v_edges, sun, observer, subrays=2, **source_options)` averages `subrays²` equal-area midpoint rays inside each fixed parallelogram footprint. `center` and ordered edges are in meters; `right/up` must be orthonormal and perpendicular to the observer. Nonuniform pixel widths are allowed. Vacuum area remains in the denominator. Output I/F is mean radiance, not summed flux; no extra projected-area cosine is applied.

Detector `source_converged`, `failed_subrays` and mean absolute source-error estimates concern source integration only. They do not certify pixel-area convergence. Refine subrays while holding footprint edges fixed, and report successive same-pixel changes separately. Such changes can miss an unsampled feature and are not rigorous area bounds. Averaged tau and shadow diagnostics are means of per-ray maps, not effective optical depth inferred from mean intensity. `shadow_fraction` includes in-cell attenuation and is not a binary occultation mask.

Single scattering omits multiple scattering, diffraction, coherent backscatter, a calibrated ice phase law, instrument PSF/noise, and finite solar size. No physical height or brightness inference follows without justified fields and observing geometry.

Primary numerical API sources: [SciPy quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html), [normal CDF](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtr.html), and [inverse CDF](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtri.html). Equations and independent slab/physical-distance controls are explicit in `model.py`, `checks.py` and `adaptive_checks.py`.
