# Continuum checkpoint to synthetic light

This bridge transfers actual centered-column fields to the Gaussian single-scattering reference. It does not prescribe height, fit an image, or validate the source dynamics. Observational comparison remains pending.

After a public continuum run, from the repository root:

```sh
python -m continuum.export --run results/continuum-smoke --out results/columns
python -m observation.bridge --columns results/columns/columns.npz --summary results/columns/summary.json --config results/columns/config.json --validity results/columns/validity.json --output results/light --complete-area
```

The exporter requires a new output directory. It validates final checkpoint shape/grid, numerical realizability, config/summary identity and executed-source hash. It preserves input bytes and records their hashes. A `wall_cap` snapshot stays labeled partial. This command supports the declared finite-size, no-ring-gravity public control; it is not a general checkpoint reader.

Conversion is `N=N0*q[0]`, `Sigma=(4 pi a³ rho/3)*N`, `H=sqrt(q[6]/q[0])*1000 m`, `x_m=1000*x`, `y_m=1000*y`, with a=0.5 m and material density 500 kg/m³. `meanZ_m` is exactly zero because the source model has no coherent signed displacement. These assumed grain values are shared with the dynamics and default opacity. The transfer uses the canonical owner conversion, not a new physical model. An empty cell has undefined H and is rejected explicitly; no density floor or arbitrary vacuum height is supplied. Supporting vacuum export requires a separate documented representation.

`columns.npz` contains SI centers, Sigma, Gaussian H and meanZ. `validity.json` contains the conversion constants, input/source hashes, snapshot time, completion and explicit physical limits. Complete dynamics summary and config are copied. Numerical realizability is not physical acceptance or calibrated height.

The reviewed bridge defaults to a fixed 8x4 detector over the complete input cell edges with 2x2 and 4x4 subrays per footprint. Its geometry is representative phase 75°, Sun elevation 15° and vertical observer, not measured Cassini geometry. Single-scattering omega=0.5 is spatially constant. The Gaussian radiative API and omitted physics are in `columns/README.md`.

`render_artifact(columns_path, output, summary_path, config_path, validity_path, ...)` retains maps, source failures, successive same-pixel changes and a radial profile weighted by physical detector-v widths. It copies the original columns and bridge source. `complete_area(output, max_subrays=32, max_seconds=600)` refines only unresolved fixed footprints. This budget applies to area completion, not the initial preview. Budget or subray stops remain unresolved; successive changes are estimates and can miss unsampled features.

Before and after completion, the copied input, preview config, saved/current bridge and both renderer modules must match the preview identity. Old outputs without identities fail. A mismatch raises before rendering and preserves existing maps/reports; do not bypass this guard to combine different snapshots. Original source paths and hashes remain in the preview report, while completion uses the immutable copied columns. New renderer code requires a new preview output.

Outputs include `input_columns.npz`, `preview_bridge_source.py`, `config.json`, `report.json`, `maps_profiles.npz`, `radial_profile.csv`, and, after completion, `radial_profile_completed.csv`. Reports distinguish source convergence from pixel-area estimates. Finite radiance is not a success flag. The local radial profile is not a Saturn-radius measurement, isolated crest brightness, density inversion or physical height validation. A zero-mean coplanar source cannot represent the coherent bending in the inclined global control.

The integrated bridge is the frozen guarded-v2 source SHA256 `1c81420b2015373cf502ba87ad1e754e134ad2ada2c59d718205accbb38c5054`. Experimental cropped/parallel drivers and their render artifacts are excluded. Regression tests cover SI units, original-byte preservation, partial-status transfer, invalid source/config and unsupported-vacuum rejection, analytic profiles, source failure propagation, area budgets, and pre-computation identity rejection.
