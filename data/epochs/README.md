# Two image-epoch geometric inputs

`python -m data.geometry` replays the included responses offline and prints derived JSON. The three original JPL Horizons JSON responses each contain two epochs: six geometric states for Daphnis, Cassini and the Sun. Raw bytes are unchanged. `queries.json` records exact request parameters, public URLs and SHA256. No mutable API call runs in CI; a future live request can use a different ephemeris revision and must not silently replace these snapshots. No second maintained derived-data file is needed.

The mid-exposure UTCs are 2006-10-27T23:44:10.216 and 2009-05-24T09:28:39.961, from calibrated image detached labels identified and hashed in `../products.json`. Start times are respectively 23:44:10.086 and 09:28:39.681; half the 260/560 ms exposure is added. PIA associations remain inferred, not official archive cross-references.

The actual API response signature is version **1.2**; current API documentation can have another version. Requests use TIME_TYPE=UT, Saturn body-center origin, BODY EQUATOR, km/km/s and VEC_CORR=NONE. Responses explicitly identify IAU_SATURN body equator and node of date. Geometric simultaneous states are intentional; this is not a light-time-corrected apparent camera solution. Source ephemeris identifiers remain in the raw response text.

The declared local point is the equatorial footpoint F=(Dx,Dy,0). The basis is radial `(Dx,Dy,0)/rho`, prograde `(-ery,erx,0)` and Saturn north. Sun/observer directions normalize target minus F and project into this basis. **Position longitude** measured from the equator/ICRF-plane node is not orbital ascending-node longitude Ω. No orbital Ω or encounter age is inferred from that position angle.

At the 2009 footpoint, Sun elevation is −1.210384°, observer elevation −52.557599° and phase 121.326816°. Their unit-vector dot product is consistent. Saturn-center observer opening −57.507° describes a different location; combining it with the local phase creates inconsistent angles. These directions do not automatically satisfy the current renderer's upper-face-only contract, and are not installed into a renderer or solver by this data addition.

The PDS nominal outer Keeler radius 136522 km is +16 km relative to the existing 136506 km reference. The current +20 km control edge is therefore 4 km farther out. PDS itself lists Daphnis at 136504 km; do not silently mix these reference conventions. A nominal radius is not a navigated local wavy edge. The derived instantaneous radial differences are parameter audits, not material retreat or uncertainty bounds.

No pixel navigation, ring-plane ray mapping, crest identification, shadow height, brightness fit, stationary encounter initialization or ephemeris covariance follows. No current continuum physics, epoch or renderer field is changed.

Primary sources: [JPL Horizons API](https://ssd-api.jpl.nasa.gov/doc/horizons.html), [Horizons manual](https://ssd.jpl.nasa.gov/horizons/manual.html), [PDS Saturn tables](https://pds-rings.seti.org/saturn/saturn_tables.html). Raw request URLs are retained in `queries.json`.
