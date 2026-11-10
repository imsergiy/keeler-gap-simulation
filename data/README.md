# Calibrated Cassini native-pixel profiles

From the repository root:

```sh
python data/fetch.py
python data/analyze.py --out data/results
```

`products.json` records the two fixed RMS/PDS Cassini ISS calibrated products, original IMG/LBL URLs, byte lengths, SHA256 values, OPUS metadata URLs and geometry. Fetch downloads four original files to ignored `data/cache/`. Existing files must pass verification; a mismatch fails explicitly. Analysis verifies all four files before reading. `--cache PATH` selects another cache. No calibration or binary rewriting is performed.

The reader supports only single-band BSQ REAL/RIEEE little-endian float32, no binary prefix or end label, and exact declared file length. It includes `NLB` binary-header records: both products begin pixels at byte 8192. Detached label pointers disagree with that embedded VICAR layout. It does not implement a general PDS/VICAR reader. Labels state CISSCAL 4.0beta, stage 10, I/F. Negative calibrated noise is retained. A finite sample is not a detector quality mask; calibration uncertainty and an independent bad-pixel mask are unavailable here.

The profiles use zero-based native pixels. A three-pixel mean and descending gradient locate the visible gap boundary within fixed search windows. Six spatial bins and all nine strip-width/reference-offset choices are retained. The Daphnis neighborhood and visible narrow streak are excluded explicitly in output metadata. Red annotation strips are edge neighborhoods; cyan strips are nearby reference material. They are not isolated crests or proven undisturbed controls. No rotation or resampling enters the measurements. The PNG uses a display stretch only.

Outputs are `*-profiles.csv` (all cross-offset samples), `*-comparisons.csv` (strip means and ratios), `*-masks.npz`, `*-rois.png`, and `analysis-metadata.json`. Spatial standard deviation, percentiles and mask sensitivity are descriptive, correlated variation, not confidence intervals. These outputs do not establish crest brightening, heights, density, damping, or orbital retreat. Shadow origin/end assignment and per-pixel ring-plane navigation remain absent.

The proposed PIA11653/2009 and PIA08319/2006 frame associations are inferred from matching date, geometry and morphology. They are not proven archive cross-references. Primary caption context: [Tiny Moon, Looming Shadows](https://science.nasa.gov/resource/tiny-moon-looming-shadows/) and [Edge Waves](https://science.nasa.gov/resource/edge-waves/). Original data and metadata provenance is recorded in `products.json`; the archive retains its own data terms, separate from the project code license.
