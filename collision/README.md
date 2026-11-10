# Finite-grain Gaussian collision source

This module evaluates binary collision moments for identical smooth spheres with constant normal restitution. It is a collision backend, not a spatial ring solver, covariance timestep, or demonstration that Keeler ring material satisfies the closure assumptions.

## Explicit Linux build

From the repository root, with Python 3.12:

```sh
python -m pip install -r requirements.txt -r requirements-build.txt
python collision/native/build.py
python -m unittest discover -s tests -v
```

The build uses pinned `ziglang==0.17.1`, C++17, `-O3 -shared -fPIC`, standard floating-point semantics and no fast-math. It writes ignored `collision/native/libgaussian.so` and prints build time separately. The native wrapper loads this exact file. A missing library is an error: there is no implicit build or Python fallback. The build and runtime contract is currently Linux; other platforms have not been tested. `source.py` is an independent Python mathematical reference used by tests, not an alternate production path. No compiled binary, point-source lookup table, Monte Carlo dataset or BGK model is included.

## API and units

```python
from collision.native.backend import NativeGaussianEnskog
source = NativeGaussianEnskog(order=17, scheme='lebedev', threads=1)
result = source.loss_gain(C, A, a, N, H, e, enhancement=1.)
```

`C[...,3,3]` is velocity covariance [m²/s²], symmetric positive semidefinite. `A[...,3,3]=partial_j u_i` is mean velocity gradient [s⁻¹]. Scalar radius `a` and height `H` are in meters; `N` is surface number density [m⁻²]. Restitution `e` is scalar in [0,1]. Leading C/A dimensions broadcast with N, H and enhancement. Require a>0, H>0, N>=0 and enhancement>0. Invalid states fail; no covariance or density floors are applied. Zero covariance and zero density are supported.

Ordinary `source(...)` returns `Q`, `P`, `heat`, and `rate`. `loss_gain(...)` also returns `L` and `B`:

| Output | Meaning | SI units |
| --- | --- | --- |
| Q | Collisional covariance production | m²/s³ |
| P | Positive contact stress per surface mass density | m²/s² |
| heat | Irreversible energy loss per mass | m²/s³ |
| rate | Encounters per particle per time | s⁻¹ |
| L | Covariance loss coefficient | s⁻¹ |
| B | Covariance gain tensor | m²/s³ |

Multiply P and heat by surface mass density for column stress and dissipation. Pair events per area per time are `N*rate/2`. With nonnegative angular weights, L and B are positive semidefinite and `Q=-L C-C L+B`. Negative-weight rules are rejected by `loss_gain`; no eigenvalue clipping is used. This identity does not itself specify a positive time update.

Lebedev orders 7/11/17 use 26/50/110 directions. `scheme='legendre'` supplies a tensor angular rule instead. The caller selects the rule explicitly; low angular order requires its own convergence check. Threads may explicitly be 1, 2 or 4; default is 1. There is no automatic timing-based switch. Native calls release the Python GIL and operate on disjoint array ranges.

## Closure assumptions and identities

Both contact centers share covariance C. Their relative mean velocity follows the local linear gradient over a diameter, `d=-2a A k`. Random velocities are independent Gaussians; the vertical number density is Gaussian with RMS H. The pair overlap includes `N/(2 sqrt(pi) H) exp[-(a kz/H)²]` inside the angular integral, retaining finite grain separation. Constant restitution is assumed. Enduring contacts, spin, friction, adhesion, size mixtures, correlated encounters, wakes and non-Gaussian distributions are absent.

`enhancement` is an explicit positive column-averaged Enskog factor. Using 1 means finite-size Boltzmann molecular chaos, not a validated dense-correlation prescription. No enhancement, viscosity or height is fitted to observations.

The source obeys `.5 trace(Q)=-P:A-heat`; contact work can heat compression even when restitution is elastic. Finite contact positions also require the vertical virial source `d<z vz>/dt=Pzz`. These moments do not add contact-stress divergence, its energy flux, vertical dynamics, self-gravity or source subcycling to any spatial equations. Those remain separate integration requirements.

The negative normal-velocity tail uses a stable continued-fraction evaluation of the same truncated-Gaussian moments; zero variance uses exact deterministic moments. The reference evaluates its independent special-function representation without calling the native kernel. Neither adds a physical velocity cutoff.

Tests check angular integrals, independent scalar Gaussian integrals, selected anisotropic/cold/expanding/vacuum native-reference states, energy partition, loss/gain reconstruction and PSD, elastic isotropic limits, explicit thread parity, and invalid/negative-weight inputs. Passing these controls does not prove closure accuracy for real ring material or a complete simulation.

Primary model source: [Latter and Ogilvie (2008), *Dense planetary rings and the viscous overstability*](https://arxiv.org/html/0803.4123v1), equations 20–39. The implementation uses its constant-restitution Gaussian binary-collision setting. Numerical angular API: [SciPy lebedev_rule](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.lebedev_rule.html).
