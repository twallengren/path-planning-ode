# Terrain energy-descent numerical audit

This is a focused, reproducible audit of the terrain energy-descent solver. It
is separate from the frozen terrain study and does not support a claim that one
planner is generally superior.

## Quantity being minimized

For a piecewise-linear route with equal parameter intervals, the solver
minimizes

\[
E[q] = \int_0^1 c(q(t))^2\lvert q'(t)\rvert^2\,dt.
\]

The public energy function reports this physical value in seconds squared and
its derivative with respect to vertex coordinates in seconds squared per
metre. The optimizer uses a dimensionless form. With domain diagonal \(L\),
source-grid median slowness \(c_\mathrm{ref}\), and
\(E_\mathrm{scale}=(Lc_\mathrm{ref})^2\), it sets

\[
z=(q-q_0)/L, \qquad F=E/E_\mathrm{scale}, \qquad
\nabla_zF=L\nabla_qE/E_\mathrm{scale}.
\]

This auxiliary energy differs from the independently evaluated travel cost
\(\int c\,ds\). A small discrete energy gradient also differs from a small
continuous Euler–Lagrange residual at finite mesh resolution. The result keeps
all three diagnostics separate.

## Independent checks

The tests in `tests/test_terrain_energy_audit.py` cover:

- physical energy against adaptive SciPy integration split at every source-grid
  knot, including a narrow off-grid high-cost feature;
- the analytic vertex gradient against centered numerical differences while
  holding quadrature panels fixed;
- invariance of energy and normalized diagnostics under metre-to-millimetre and
  inverse-slowness unit conversion, plus the expected quadratic scaling when
  slowness is multiplied;
- strict decrease of accepted before/after energies evaluated on a common
  quadrature panel set, and exact preservation of endpoints;
- hard barrier behavior for both an infeasible direct seed and a feasible seed
  whose trial steps are blocked;
- identical shared-seed hashes between energy descent and Euler–Lagrange,
  followed by an independent full-route cost and feasibility evaluation;
- structured iteration-limit and wall-clock-limit results, including the case
  where time expires immediately after an accepted step. In that case stale
  gradient diagnostics must be null and `gradient_diagnostics_current` false;
- FMM warm-start timing and the distinction between discrete stationarity and
  the continuous ODE residual.

The focused audit command is:

```bash
.venv/bin/pytest -q tests/test_terrain_energy_audit.py
```

## Cold-start comparison

The predeclared matrix used four synthetic softened-wall cases, 32 and 64
interior points, and both `straight` and `arc_left` shared seeds. Each cell ran
energy descent and Euler–Lagrange with tolerance \(10^{-6}\), at most 1,000
iterations, an 8-second wall-clock budget, 65 profile samples, and order-8
quadrature. This produced 32 planner runs. Every returned route was feasible
and independently evaluated; energy descent reported success in 8 of 16 runs,
and Euler–Lagrange in 3 of 16. Failures remain in the table.

The cost column is the independent travel cost in seconds. “Stationary” and
other text in parentheses are termination reasons. The signed difference is
energy descent minus Euler–Lagrange for that cell.

| Case | N | Initialization | Energy descent | Euler–Lagrange | Signed cost difference (s) | Same seed |
|---|---:|---|---|---|---:|:---:|
| ridge_pass | 32 | straight | 726.777 (stationary) | 813.922 (stagnated) | -87.145 | yes |
| ridge_pass | 32 | arc_left | 9108.914 (time_limit) | 18396.519 (stagnated) | -9287.606 | yes |
| ridge_pass | 64 | straight | 726.766 (stationary) | 731.665 (stationary) | -4.899 | yes |
| ridge_pass | 64 | arc_left | 9110.653 (time_limit) | 18075.188 (stagnated) | -8964.536 | yes |
| competing_corridors | 32 | straight | 1192.890 (stationary) | 89837.713 (stagnated) | -88644.823 | yes |
| competing_corridors | 32 | arc_left | 1255.454 (time_limit) | 1598.921 (stagnated) | -343.467 | yes |
| competing_corridors | 64 | straight | 1192.599 (stationary) | 82583.110 (stagnated) | -81390.511 | yes |
| competing_corridors | 64 | arc_left | 1254.853 (time_limit) | 1575.826 (stagnated) | -320.972 | yes |
| dead_ends | 32 | straight | 23093.010 (time_limit) | 24645.606 (stagnated) | -1552.596 | yes |
| dead_ends | 32 | arc_left | 23112.505 (time_limit) | 143853.218 (stagnated) | -120740.713 | yes |
| dead_ends | 64 | straight | 23093.010 (time_limit) | 23901.715 (stagnated) | -808.706 | yes |
| dead_ends | 64 | arc_left | 23121.501 (time_limit) | 119447.537 (stagnated) | -96326.036 | yes |
| correlated_roughness | 32 | straight | 609.735 (stationary) | 748.100 (stagnated) | -138.365 | yes |
| correlated_roughness | 32 | arc_left | 609.735 (stationary) | 609.740 (stationary) | -0.005 | yes |
| correlated_roughness | 64 | straight | 609.593 (stationary) | 747.888 (stagnated) | -138.295 | yes |
| correlated_roughness | 64 | arc_left | 609.593 (stationary) | 609.593 (stationary) | -0.000 | yes |

These local solutions depend strongly on initialization. A lower cost attached
to a failed termination is still only the cost of the returned feasible route;
it is not a stationary solution or proof of route quality.

## Exact hybrid-default check

The interface default is explicitly hybrid: energy descent starts from an FMM
route. A separate check used `initialization="fast_marching"`, 32 interior
points, a 129-by-129 reference grid, tolerance \(10^{-5}\), at most 400
iterations, and the 60-second interface budget. The FMM reference was also run
and evaluated separately.

| Case | Energy cost (s) | FMM reference (s) | Difference (s) | Termination | Feasible | ODE residual (m) | Initialization (s) | Total (s) |
|---|---:|---:|---:|---|:---:|---:|---:|---:|
| ridge_pass | 726.777 | 738.388 | -11.612 | stationary | yes | 174.7 | 0.311 | 6.663 |
| competing_corridors | 1192.873 | 1198.843 | -5.970 | stationary | yes | 1034.3 | 0.319 | 6.515 |
| dead_ends | 935.131 | 936.817 | -1.686 | stationary | yes | 85.6 | 0.331 | 2.351 |
| correlated_roughness | 609.735 | 613.602 | -3.867 | stationary | yes | 162.7 | 0.324 | 1.312 |

All four scaled discrete energy-gradient norms were below \(10^{-5}\). The ODE
residuals in the table are much larger than zero, so these outcomes must not be
described as continuous ODE convergence.

## Reproduction and provenance

Run both the cold matrix and hybrid check, writing JSON and CSV to a chosen
temporary directory:

```bash
.venv/bin/python examples/terrain_energy_comparison.py \
  --output-dir /tmp/terrain-energy-comparison
```

Run only the four hybrid-default cells:

```bash
.venv/bin/python examples/terrain_energy_comparison.py \
  --warm-only --output-dir /tmp/terrain-energy-comparison
```

The JSON includes every complete planner config and hash, scenario and initial
route hashes, full planner results, package versions, hardware/platform data,
the Git commit, and whether the worktree was dirty. The measurements above were
made at commit `de210b1b719bce19baabf396db754be476e76766` in a dirty development
worktree on macOS 15.7.9 x86-64, Python 3.13.12, NumPy 2.5.3, SciPy 1.18.1, and
Shapely 2.1.2. Timings are machine-specific.
