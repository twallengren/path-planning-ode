# Numerical audit

This document records checks that are independent of the numerical routines they
test. It distinguishes a passing check from an assumption or an untested claim.

## Audit rules

- Terrain derivatives are compared with centered finite differences of the
  public field evaluator on a nonsymmetric cubic log-slowness surface.
- Route costs are compared with an independently sampled numerical integral,
  with relative agreement required at `1e-5`.
- Geometry checks use complete line segments. Vertex-only sampling is not
  evidence of collision freedom.
- Domain-boundary contact is allowed. Contact with any barrier boundary is
  infeasible.
- An out-of-domain route keeps its original geometry and length, receives no
  cost or field profile, and is never silently clamped to the domain.
- Solver diagnostics are audited separately from route feasibility and cost.
  A successful optimizer status does not establish feasibility, stationarity,
  or global optimality.
- Empirical route convergence is assessed with the two finest planner/reference
  grid resolutions applied to the same fixed continuous field. A cost change
  above 0.5% is reported as unresolved rather than extrapolated into a
  convergence claim.

## Foundation checks

The independent foundation suite covers:

- bicubic interpolation of cubic log-slowness and finite-difference checks of
  the gradient and Hessian after exponentiation;
- independent integration of weighted polyline length;
- constant-field, straight-line, orientation-reversal, and reflection
  invariants;
- endpoint and domain validation; and
- full-segment collision checks for corner cutting, thin barriers, barrier
  contact, domain-boundary contact, domain exit, and a barrier that disconnects
  the rectangular domain.

These checks pass against the shared implementation. The command below now also
includes the solver checks described in the next section:

```text
$ .venv/bin/pytest tests/test_terrain_audit.py -q
.......................                                                  [100%]
23 passed in 3.13s
```

The run used SciPy 1.18.1 and Shapely 2.1.2. The independent route-cost oracle
uses SciPy adaptive quadrature, whereas the evaluator uses fixed-order
Gauss-Legendre quadrature split at the source-field knots. The comparison
requires relative agreement within `1e-5`.

## Numerical solver checks

The independent solver gate covers:

- direct recomputation of the first-order isotropic Godunov equation on every
  reachable non-seed node of an asymmetric field;
- exact comparison of the conservative grid mask with independently constructed
  Shapely control cells;
- strict decrease of independently evaluated piecewise-linear arrival values
  along an extracted obstacle detour, plus an explicit `extraction_failed`
  result when a flat arrival field offers no descent direction;
- refinement toward the analytic straight-line cost on uniform terrain and the
  Snell invariant across the layered fixture;
- the SLSQP objective against adaptive quadrature and its normalized-coordinate
  gradient against centered finite differences;
- equal-segment constraints and their Jacobian, plus the conservative midpoint
  clearance bound, against independent formulas;
- the Euler--Lagrange Jacobian against centered finite differences on a
  nonsymmetric cubic log-slowness field;
- independent feasibility, equality-residual, and KKT-stationarity checks on an
  asymmetric terrain result;
- identical configured seed hashes for both local methods;
- infeasibility introduced by coarse equal-arclength resampling and the
  corresponding `resolution_limited` distinction; and
- retained route and cost data after an optimizer failure, along with
  serializable timing whose total includes initialization.

All 23 audit tests pass. No numerical correction was required after this gate.

## Interpretation and limits

The tests support the following bounded conclusions:

- Fast marching solves its stated first-order discrete Eikonal equation and is
  a useful grid reference. It is not a certified continuous optimum.
- The obstacle detour is collision-free and strictly descends the triangulated
  arrival field. The public result has no branch-use telemetry, so the audit
  does not claim how often the directional fallback was selected.
- The layered source is represented by a bicubic transition rather than an
  ideal discontinuity. Its cost varies only with `y`, so approximate
  conservation of the horizontal Snell quantity is the appropriate check.
- SLSQP success requires independently checked feasibility, constraints, and
  stationarity. The independent KKT case has inactive bounds and no barrier;
  barrier clearance and the conservative inequality formula are checked
  separately.
- Euler--Lagrange convergence establishes discrete stationarity only. The audit
  includes a stationary route that crosses a barrier, demonstrating why
  stationarity cannot replace feasibility or establish minimality.
- `resolution_limited` means the conservative polyline certificate was not met
  at the permitted representation resolution. It does not mean that no
  continuous route exists.

The completed benchmark contains feasible 513- and 1025-grid reference results
for all 80 baseline cases on the same fixed continuous source field. Seventy-nine
cases meet the declared 0.5% refinement threshold. The remaining case,
`correlated_roughness-s09`, is explicitly unresolved at a 0.533% absolute
change. These empirical comparisons are not continuous error bounds; the full
counts and qualified method comparisons are in the
[study report](study-report.md). Native
and browser agreement was measured on two deterministic parity fixtures. On
the uniform fixture, SLSQP, Euler--Lagrange, and fast marching had the same
feasibility classifications and absolute evaluated-cost differences below
`5e-8` seconds. On the heterogeneous symmetry fixture, the two local methods
had the same feasibility classifications and absolute differences below
`5e-7` seconds. These checks do not establish route-coordinate equality or an
all-case browser tolerance.

The browser lock uses Python 3.14.2, NumPy 2.4.6, SciPy 1.18.0, and Shapely
2.1.2. The native audit environment uses Python 3.13.12, NumPy 2.5.3, SciPy
1.18.1, and Shapely 2.1.2. Runtime measurements from these environments are
reported separately and are not pooled.

No audit result supports a blanket global-optimality claim.

## Experiment and statistics gate

The independent experiment audit covers:

- exact expansion of the frozen full matrix into `80 * 33 + 16 * 11 = 2816`
  run specifications, with one scenario hash per case across planner
  resolutions and a separately fixed 65 by 65 source field;
- incomplete-profile wording and the rule that missing declared 1025-grid data
  cannot be replaced by a successful 257/513 comparison;
- resume rejection when an existing run ID, configuration hash, protocol hash,
  execution profile, or numerical source hash is incompatible;
- finite-sample paired summaries over cases shared by both methods, including
  matching initial-route hashes and family/seed cluster resampling;
- omission of bootstrap bounds when only one cluster is available;
- separate family and contrast-sweep strata;
- visibility of feasible routes from nonconverged runs without counting those
  runs as solver successes;
- direct recomputation of within-case initialization sensitivity;
- local refinement using only the two finest declared levels (64 and 128), with
  missing or failed 128-point runs explicitly unresolved; and
- a real subprocess timeout showing that the hard wall clock covers the whole
  child run, including scenario construction and preprocessing.

The combined independent gate passes:

```text
$ .venv/bin/pytest tests/test_terrain_audit.py tests/test_experiment_audit.py -q
...............................                                          [100%]
31 passed in 8.74s
```

The audit identified and regression-tested three corrections before the full
run: exact requested-configuration validation on resume, declared-finest local
refinement accounting, and declared-1025/reference plus incomplete-full wording
for limited executions.

A 20-record smoke preflight completed all subprocesses under a reduced 15-second
cap. Independent inspection found one protocol digest, one numerical-source
digest, unique run IDs, one scenario hash per case, and matching EL/SLSQP seed
hashes for all eight paired starts. Sixteen candidates were geometrically
feasible. This smoke profile tests the data path only and provides no evidence
for or against the study hypotheses.

The final archive contains all 2,816 unique run IDs under one numerical
execution hash: 2,345 returned records and 471 hard subprocess timeouts. Each
returned record has a JSON-valid deterministic gzip detail file, and the
published index matches the current analysis hash. The report keeps feasible
nonconverged candidates visible and labelled, excludes infeasible candidates
and unresolved references from quality summaries, and reports the timeout
count separately from returned-run timing medians.

## Boundary correction review

An independent integration review followed the first full-run boundary failure.
It checks a constant-cost oracle along every side and diagonal of the inclusive
domain, verifies that profile sampling and quadrature leave the submitted route
unchanged, and confirms that a point genuinely outside by `1e-12` metres still
returns `outside_domain` with no cost. The exact frozen ridge-pass seed-0,
arc-left SLSQP configuration now returns a classified candidate without an
escaping evaluator exception. A separately forced evaluator exception returns
the structured `evaluation_failed` outcome with both `feasible` and
`solver_success` false.
