# Terrain routing: tutorial and study method

The version 2 terrain API describes a fixed, shared problem in SI units. A
`TerrainScenario` contains metre coordinates, observed or synthetic elevation,
the natural log of positive slowness (seconds per horizontal metre), and
separate GeoJSON barriers. Every planner receives the same scenario source
grid; only its route discretisation and numerical method vary.

## A first experiment

```bash
uv sync
uv run python examples/terrain_experiment.py
```

The equivalent Python workflow is:

```python
from path_planning_ode import PlannerConfig, evaluate_route, plan, validation_terrain

scenario = validation_terrain("uniform")
result = plan(scenario, PlannerConfig(method="slsqp", initialization="straight"))
evaluation = evaluate_route(scenario, result.route_m)
print(evaluation.feasible, evaluation.cost_s, evaluation.length_m)
```

`plan` returns solver diagnostics and a candidate route. `evaluate_route` is a
separate end-to-end check: it integrates the interpolated field along every
segment, checks endpoints and domain coverage, and reports barrier collisions.
Domain boundary contact is allowed; contact with an impassable barrier is a
collision. A colliding route still receives an independent cost measurement.

The three methods are `euler_lagrange`, constrained `slsqp`, and first-order
`fast_marching`. Local methods can use `straight`, `arc_left`, `arc_right`, or
`barrier` initialisation; the barrier seed and FMM warm start are shared where
applicable. For a v1 Gaussian `Scene`, use
`scene_to_terrain_scenario(scene)` explicitly: it is a sampled bicubic
approximation and does not silently claim to be the original analytic field.

## CLI and browser

```bash
uv run run-terrain-benchmarks --profile smoke --no-figures \
  --output-dir terrain-study-results/smoke
uv run run-terrain-benchmarks --profile full --workers 4 \
  --output-dir terrain-study-results/full
```

Smoke runs exercise the complete data path quickly. The frozen full protocol
contains 80 baseline cases (four synthetic families with 20 seeds each) plus
16 declared contrast-sweep cases. Its method, initialisation, and resolution
combinations produce 2,816 runs. A full run is capped at 60 seconds per run and
writes JSONL records, CSV summaries, figures, hashes, environment metadata,
and a report. A reduced or interrupted run cannot support full-study claims.
Use `--resume` only to continue an interrupted archive in the same output
directory. The checked-in `experiments/published` archive is a publication
artifact, not a resume target for a fresh reproduction.

Synthetic fields regenerated on different math libraries can differ in their
last floating-point bits. The exact problems used by the published records are
therefore bundled as deterministic gzip files in
`experiments/published/scenarios`. Load one without regenerating its field:

```python
import gzip
import json

from path_planning_ode.terrain import TerrainScenario

with gzip.open("experiments/published/scenarios/<scenario-hash>.json.gz", "rt") as stream:
    scenario = TerrainScenario.from_dict(json.load(stream))
```

The static browser demo is available at
[`web/terrain.html`](../web/terrain.html). It uses the same pinned Python
runtime through Pyodide and loads SciPy/Shapely lazily. Build and serve it with
`npm run build` and `npm run preview`. Version-2 imports validate terrain,
configuration, route, and profile structure before drawing. Imported result
records are cached evidence and remain labelled unverified; selecting **Run
comparison** evaluates the selected routes in the browser. Published study
records retain their native provenance and explicit reference-resolution
qualification.

## Method and interpretation

Synthetic terrain is generated from four deterministic families (`ridge_pass`,
`competing_corridors`, `dead_ends`, and `correlated_roughness`) on one fixed
source grid. Five analytic fixtures cover uniform cost, layered refraction,
symmetry, a detour barrier, and a disconnected domain. The bundled Mount
Tamalpais crop uses observed elevation from [AWS Terrain
Tiles](https://registry.opendata.aws/terrain-tiles/) and visible [Joerd
attribution](https://github.com/tilezen/joerd/blob/master/docs/attribution.md).
Its local WGS84 metre projection records a maximum horizontal scale residual of
about 0.0117%; source URLs, checksums, and processing metadata are stored with
the data.

Elevation is observed data. The isotropic slowness transformation and optional
Mount Tamalpais rectangle are illustrative modelling assumptions, not a
calibrated hiking or vehicle model. Results describe numerical route candidates,
not travel-time truth or algorithmic novelty.

FMM is a first-order grid method and a numerical reference, not a certified
optimum. Reference differences are signed: a finer-grid cost can be higher or
lower. The reference is resolved only when both finest declared grids, 513 and
1025, are feasible and their evaluated costs differ by no more than 0.5%.
Missing either level leaves the reference unresolved. These signed differences
must not be reported as optimality gaps.

Paired method differences include a case only when both routes are feasible and
share the same initial-route hash. The published native timings use eight
worker processes. The runner requested one thread through the OMP, OpenBLAS,
and MKL environment variables, but the native NumPy/SciPy build uses Apple
Accelerate and its actual thread count was not measured. Process concurrency
and possible Accelerate contention limit how far these wall times and timeout
rates generalize to another machine. Bootstrap intervals
resample family-and-seed clusters, so resolution and initialisation variants
are not treated as independent terrain cases. The fixed residual tolerance is not
swept; any relationship between residual and route cost is descriptive and
does not establish a causal effect. A small residual or solver success flag
does not prove global optimality.

## Methods appendix

### Continuous objective and field derivatives

Let a route be $q:[0,1]\to\mathbb{R}^2$, with fixed endpoints, and let
$c(q)>0$ be slowness in seconds per horizontal metre. The reported objective
is weighted length,

\[
J[q]=\int_0^1 c(q(t))\,\lVert q'(t)\rVert\,dt.
\]

This quantity depends on the geometric curve and not on its parameter speed.
The Euler–Lagrange solver uses the smoother energy

\[
E[q]=\int_0^1 c(q(t))^2\,\lVert q'(t)\rVert^2\,dt.
\]

For a fixed geometric curve, Cauchy–Schwarz gives $E\geq J^2$, with equality
when weighted speed $c(q)\lVert q'\rVert$ is constant. Thus minimizing over
both curves and their parameterisations gives the same minimizing geometry.
The stationary equation used in the implementation is

\[
q''=\frac{\lVert q'\rVert^2\nabla c
       -2q'(\nabla c\mathbin{\cdot}q')}{c}.
\]

The code applies centred differences to this equation and a sparse analytic
Jacobian; its residual measures stationarity of that mesh equation. It is not a
geometric feasibility measure and does not prove that the stationary curve is
a minimum. See [`local_planners.py`](../src/path_planning_ode/local_planners.py)
and the independent checks in
[`test_terrain_audit.py`](../tests/test_terrain_audit.py).

The source grid stores ℓ = log c and fits an interpolating bicubic rectangular
spline (`s=0`). Positivity follows from $c=\exp(\ell)$, while the derivatives
used by both local methods follow the chain rule:

\[
\nabla c=c\nabla\ell,\qquad
\nabla^2c=c\left(\nabla^2\ell+\nabla\ell\,\nabla\ell^\mathsf{T}\right).
\]

The strict field wrapper rejects evaluation outside the scenario bounds,
because the underlying spline library otherwise extrapolates. The implementation
is in [`terrain.py`](../src/path_planning_ode/terrain.py); SciPy documents the
rectangular spline, interpolation setting, derivatives, and extrapolation
behaviour in
[`RectBivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html).

### Constrained polyline method

SLSQP optimizes only the interior vertices. Coordinates are normalized to the
unit square for conditioning, then mapped back to physical metres before cost
and constraint evaluation. The objective is the Gauss–Legendre approximation
of weighted length over every segment, with its analytic vertex gradient.
Equality constraints make adjacent physical segment lengths equal. Bounds keep
free vertices within the inclusive terrain domain; an eight-epsilon inward
guard prevents an optimizer iterate on a floating-point boundary from creating
an accidental extrapolation query.

For a segment with midpoint $m$, length $L$, requested clearance $r$, and
signed distance $d(m)$ from the barrier boundary, the implemented sufficient
constraint is

\[
d(m)-L/2-r\geq0.
\]

Distance to a closed set is 1-Lipschitz, and every point of the segment lies at
most $L/2$ from its midpoint. The inequality therefore certifies clearance
along the whole segment. It is conservative: failure of the midpoint
certificate does not prove collision. The implementation can refine the route
representation and otherwise returns `resolution_limited`, rather than claiming
that no continuous route exists. Solver termination, the independently computed
KKT stationarity residual, and final geometric feasibility remain separate
fields. SciPy describes SLSQP's bound, equality, inequality, and Jacobian
interface in [`minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

### Fast marching reference

The reference solves the isotropic Eikonal equation

\[
\lVert\nabla T\rVert=c
\]

on a uniform Cartesian grid. If $a$ and $b$ are the smallest accepted
horizontal and vertical neighbour times, the first-order update solves

\[
\left(\frac{T-a}{\Delta x}\right)^2+
\left(\frac{T-b}{\Delta y}\right)^2=c^2,
\qquad T\geq\max(a,b),
\]

and falls back to the smaller causal one-sided update when that quadratic root
is inadmissible. A heap accepts nodes in increasing arrival time. Any grid node
whose clipped Voronoi control cell touches a barrier is masked, which prevents
grid edges from cutting through a barrier but can disconnect a coarse grid even
when a continuous route exists.

Route extraction descends a continuous piecewise-affine interpolation over a
fixed triangulation of each grid cell. Near a conservative mask boundary, where
the adjacent triangle gradients may all point into unavailable cells, it tries
progressively shorter directions on the same interpolant. Every accepted step
must strictly reduce arrival time and have a barrier-clear connector. Failure is
reported as `extraction_failed`, `unreachable_on_grid`, or a resolution limit;
none proves continuous infeasibility. See
[`fast_marching.py`](../src/path_planning_ode/fast_marching.py) and Sethian's
original [fast marching paper](https://math.berkeley.edu/~sethian/2006/Papers/sethian.fastmarching.pdf)
and later [review](https://math.berkeley.edu/~sethian/2006/Papers/sethian.siam_fast.pdf).

### Independent route evaluation

All methods finish with the same evaluator. It splits every polyline segment at
source-grid knots and applies Gauss–Legendre quadrature on each subinterval.
The numerical audit compares those costs with adaptive quadrature on analytic
fields; see [`test_terrain_audit.py`](../tests/test_terrain_audit.py). Profile
sampling uses convex interpolation within submitted segments, so floating-point
cancellation cannot move an otherwise valid sample beyond a boundary. The
submitted route itself is never clipped or changed.

The domain boundary is inclusive. Any contact with an impassable barrier,
including tangency, is a collision. A colliding in-domain route retains its
measured cost so solver behavior remains inspectable; an out-of-domain route has
`cost_s=None` because the field is deliberately undefined there. Endpoint,
domain, and barrier violations are reported separately. These checks distinguish
geometry from mesh diagnostics: a small Euler–Lagrange or KKT residual says
nothing by itself about barrier contact, and a feasible route may come from a
solver that did not satisfy its stationarity test.

For the original v1 derivation and conventions, see
[`docs/mathematics.md`](mathematics.md) and the project [`README.md`](../README.md).
