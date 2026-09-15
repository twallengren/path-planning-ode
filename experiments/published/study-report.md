# Terrain Routing: A Reproducible Computational Study

## Abstract

This computational study compares an Euler–Lagrange boundary-value solver, direct constrained SLSQP optimization, and an isotropic fast-marching grid reference on a frozen heterogeneous-terrain protocol. Complete frozen study matrix. The artifact contains 2816 run records: 2345 completed, 471 hit the hard subprocess timeout, 0 failed in a worker, and 1610 yielded routes that passed independent whole-segment validation.

## Scope and model

Static isotropic travel cost in seconds per horizontal metre with separate impassable barriers; elevation-driven costs are illustrative, not calibrated travel dynamics.
Every planner queries one fixed bicubic spline of log-slowness, exponentiated to keep cost positive. Barriers remain separate geometric constraints. The shared evaluator integrates each submitted polyline and tests every complete segment; infeasible routes never enter route-quality summaries.
Fast marching is a grid-based reference rather than a certified continuous optimum. Differences below are signed `(candidate - reference) / reference`; negative values are retained. A case is unresolved when either of the two finest declared grids is unavailable or their evaluated costs differ by more than 0.5%.
See the [methods and reproduction appendix](https://github.com/twallengren/path-planning-ode/blob/master/docs/terrain-study.md) and the [independent numerical audit](https://github.com/twallengren/path-planning-ode/blob/master/docs/numerical-audit.md) for derivations, validation cases, collision semantics, and cross-implementation checks.

## Frozen design

| Component | Frozen value |
| --- | --- |
| Baseline terrain cases | 80 = 4 families × 20 seeds |
| Difficulty assignment | ('easy', 'medium', 'hard')[seed % 3] |
| Local methods | euler_lagrange, slsqp |
| Shared initializations | straight, arc_left, arc_right, barrier, fast_marching |
| Local interior points | 32, 64, 128 |
| Reference grids | 257, 513, 1025 |
| Contrast sweep | 0.0, 1.0, 2.0, 4.0 |
| Per-run budget | 60 s including initialization/preprocessing |
| Bootstrap | 2000 resamples; cluster=family and seed; seed=20260915 |

## Aggregate baseline results

Feasibility and cost come from the independent evaluator. Solver success records termination separately. Cost medians omit unavailable or infeasible routes. Time medians use end-to-end child wall time, including scenario construction, for returned planner results; hard subprocess timeouts are right-censored at 60 seconds and shown in a separate column.

| Method | Runs | Feasible | Solver success | Median signed ref. diff. (%) | Resolved quality n | Median clearance (m) | Median returned end-to-end wall time (s) | Hard timeouts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| euler_lagrange | 1200 | 650/1200 (54.2%) | 708/1200 (59.0%) | -0.005 | 638 | 63.60 | 1.09 | 0 |
| fast_marching | 240 | 240/240 (100.0%) | 240/240 (100.0%) | 0.043 | 237 | 70.54 | 5.56 | 0 |
| slsqp | 1200 | 603/1200 (50.2%) | 45/1200 (3.8%) | -0.005 | 593 | 73.35 | 18.29 | 456 |

### Configuration-level results

| Method | Initialization | Resolution | Runs | Feasible rate | Solver success rate | Median signed ref. diff. (%) | Resolved quality n | Median clearance (m) | Median returned end-to-end wall time (s) | Hard timeouts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| euler_lagrange | arc_left | N=128 | 80 | 45.0% | 41.2% | 17.468 | 36 | 60.56 | 1.15 | 0 |
| euler_lagrange | arc_left | N=32 | 80 | 42.5% | 32.5% | 25.322 | 34 | 60.51 | 0.92 | 0 |
| euler_lagrange | arc_left | N=64 | 80 | 42.5% | 38.8% | 18.635 | 34 | 59.89 | 0.99 | 0 |
| euler_lagrange | arc_right | N=128 | 80 | 41.2% | 50.0% | 5.294 | 32 | 55.68 | 1.17 | 0 |
| euler_lagrange | arc_right | N=32 | 80 | 40.0% | 43.8% | 4.552 | 31 | 59.14 | 0.93 | 0 |
| euler_lagrange | arc_right | N=64 | 80 | 41.2% | 48.8% | 3.930 | 32 | 55.70 | 1.02 | 0 |
| euler_lagrange | barrier | N=128 | 80 | 58.8% | 43.8% | 9.555 | 46 | 40.13 | 1.27 | 0 |
| euler_lagrange | barrier | N=32 | 80 | 73.8% | 57.5% | 0.016 | 58 | 63.93 | 0.92 | 0 |
| euler_lagrange | barrier | N=64 | 80 | 66.2% | 38.8% | 17.821 | 52 | 34.29 | 1.07 | 0 |
| euler_lagrange | fast_marching | N=128 | 80 | 91.2% | 100.0% | -0.037 | 72 | 75.52 | 5.21 | 0 |
| euler_lagrange | fast_marching | N=32 | 80 | 91.2% | 95.0% | 0.001 | 72 | 75.46 | 5.09 | 0 |
| euler_lagrange | fast_marching | N=64 | 80 | 91.2% | 100.0% | -0.031 | 72 | 75.51 | 5.08 | 0 |
| euler_lagrange | straight | N=128 | 80 | 28.7% | 65.0% | 3.655 | 22 | 65.04 | 1.11 | 0 |
| euler_lagrange | straight | N=32 | 80 | 30.0% | 65.0% | 4.860 | 23 | 64.48 | 0.88 | 0 |
| euler_lagrange | straight | N=64 | 80 | 28.7% | 65.0% | 3.659 | 22 | 65.05 | 0.97 | 0 |
| fast_marching | reference | grid 1025 | 80 | 100.0% | 100.0% | 0.000 | 79 | 69.80 | 18.00 | 0 |
| fast_marching | reference | grid 257 | 80 | 100.0% | 100.0% | 0.148 | 79 | 71.53 | 2.25 | 0 |
| fast_marching | reference | grid 513 | 80 | 100.0% | 100.0% | 0.050 | 79 | 71.21 | 5.56 | 0 |
| slsqp | arc_left | N=128 | 80 | 0.0% | 0.0% | — | 0 | — | 27.43 | 63 |
| slsqp | arc_left | N=32 | 80 | 83.8% | 7.5% | 1.505 | 66 | 76.62 | 6.01 | 1 |
| slsqp | arc_left | N=64 | 80 | 57.5% | 10.0% | 1.101 | 45 | 76.66 | 29.41 | 18 |
| slsqp | arc_right | N=128 | 80 | 0.0% | 0.0% | — | 0 | — | 29.19 | 60 |
| slsqp | arc_right | N=32 | 80 | 80.0% | 12.5% | 2.013 | 63 | 62.75 | 6.77 | 4 |
| slsqp | arc_right | N=64 | 80 | 50.0% | 7.5% | 2.971 | 39 | 46.13 | 28.33 | 22 |
| slsqp | barrier | N=128 | 80 | 1.2% | 0.0% | -0.064 | 1 | 69.97 | 52.53 | 79 |
| slsqp | barrier | N=32 | 80 | 95.0% | 3.8% | 0.011 | 75 | 77.38 | 5.14 | 4 |
| slsqp | barrier | N=64 | 80 | 95.0% | 3.8% | -0.012 | 75 | 77.66 | 30.65 | 4 |
| slsqp | fast_marching | N=128 | 80 | 0.0% | 0.0% | — | 0 | — | — | 80 |
| slsqp | fast_marching | N=32 | 80 | 90.0% | 0.0% | -0.001 | 71 | 75.55 | 8.31 | 8 |
| slsqp | fast_marching | N=64 | 80 | 90.0% | 0.0% | -0.031 | 71 | 75.59 | 26.45 | 8 |
| slsqp | straight | N=128 | 80 | 1.2% | 0.0% | -0.064 | 1 | 69.97 | 45.30 | 71 |
| slsqp | straight | N=32 | 80 | 70.0% | 6.2% | 1.311 | 55 | 64.33 | 17.36 | 0 |
| slsqp | straight | N=64 | 80 | 40.0% | 5.0% | -0.055 | 31 | 63.96 | 27.80 | 34 |

### Termination and failure modes

| Method | Termination/status | Count |
| --- | --- | --- |
| euler_lagrange | iteration_limit | 2 |
| euler_lagrange | stagnated | 524 |
| euler_lagrange | stationary | 754 |
| fast_marching | converged | 256 |
| slsqp | converged | 55 |
| slsqp | nonstationary | 559 |
| slsqp | optimization_failed | 195 |
| slsqp | timeout | 471 |

## Paired local-method comparison

Each pair uses the same case, initialization name, discretization, and verified initial-route hash. The effect is `SLSQP - Euler–Lagrange` as a percentage of the Euler–Lagrange evaluated cost. Paired feasible denominators expose survivorship; 95% intervals resample case/seed clusters.

| Initialization | N | Declared pairs | Hash matched | Both feasible / matched | Mean difference (%) | Bootstrap 95% interval (%) |
| --- | --- | --- | --- | --- | --- | --- |
| arc_left | 32 | 80 | 79 | 34/79 | -14.626 | [-19.732, -9.900] |
| arc_left | 64 | 80 | 62 | 33/62 | -13.734 | [-18.469, -9.248] |
| arc_left | 128 | 80 | 17 | 0/17 | — | insufficient clusters |
| arc_right | 32 | 80 | 76 | 32/76 | -7.214 | [-10.988, -3.544] |
| arc_right | 64 | 80 | 58 | 32/58 | -7.909 | [-11.863, -4.009] |
| arc_right | 128 | 80 | 20 | 0/20 | — | insufficient clusters |
| barrier | 32 | 80 | 76 | 57/76 | -7.062 | [-9.674, -4.494] |
| barrier | 64 | 80 | 76 | 51/76 | -15.814 | [-20.598, -11.533] |
| barrier | 128 | 80 | 1 | 1/1 | 0.000 | insufficient clusters |
| fast_marching | 32 | 80 | 72 | 72/72 | -0.007 | [-0.015, -0.002] |
| fast_marching | 64 | 80 | 72 | 72/72 | -0.000 | [-0.001, 0.000] |
| fast_marching | 128 | 80 | 0 | 0/0 | — | insufficient clusters |
| straight | 32 | 80 | 80 | 24/80 | -8.070 | [-11.742, -4.824] |
| straight | 64 | 80 | 46 | 23/46 | -7.934 | [-12.025, -4.521] |
| straight | 128 | 80 | 9 | 1/9 | 0.000 | insufficient clusters |

## Reference refinement

Across 80 baseline cases, 79 met the empirical refinement check and 1 were unresolved, including 0 with a missing or infeasible declared 513/1025 result. This is an empirical check, not an error bound.
Unresolved baseline cases and signed 513→1025 changes: correlated_roughness-s09 (-0.533%).

| Family | Cases | Resolved | Unresolved | Missing finest pair | Median 513→1025 change (%) | Maximum absolute change (%) |
| --- | --- | --- | --- | --- | --- | --- |
| competing_corridors | 20 | 20 | 0 | 0 | -0.042 | 0.153 |
| correlated_roughness | 20 | 19 | 1 | 0 | -0.088 | 0.533 |
| dead_ends | 20 | 20 | 0 | 0 | -0.027 | 0.036 |
| ridge_pass | 20 | 20 | 0 | 0 | -0.058 | 0.069 |

## Initialization and local refinement

Initialization sensitivity is the within-case percentage range across feasible starts. Cases with fewer than two feasible starts are reported in the denominator but cannot contribute a range.

| Method | N | Cases with ≥2 feasible starts / cases | Median range (%) | 90th percentile (%) | Maximum (%) |
| --- | --- | --- | --- | --- | --- |
| euler_lagrange | 32 | 77/80 | 3.034 | 45.866 | 98.838 |
| euler_lagrange | 64 | 73/80 | 29.321 | 63.257 | 131.179 |
| euler_lagrange | 128 | 69/80 | 29.340 | 87.751 | 108.248 |
| slsqp | 32 | 80/80 | 7.548 | 65.560 | 413.215 |
| slsqp | 64 | 80/80 | 0.217 | 18.134 | 39.973 |
| slsqp | 128 | 1/80 | 0.000 | 0.000 | 0.000 |

The next table compares only the two finest declared local levels, 64 and 128 interior points. Missing levels remain unresolved rather than being replaced by coarser successful runs.

| Family | Method | Initialization | Levels | Both feasible / cases | Unresolved | Median signed change (%) | Median absolute change (%) | Absolute change >0.5% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| competing_corridors | euler_lagrange | arc_left | 64→128 | 15/20 | 5 | -0.009 | 0.282 | 6 |
| competing_corridors | euler_lagrange | arc_right | 64→128 | 20/20 | 0 | -0.005 | 0.005 | 0 |
| competing_corridors | euler_lagrange | barrier | 64→128 | 0/20 | 20 | — | — | 0 |
| competing_corridors | euler_lagrange | fast_marching | 64→128 | 20/20 | 0 | -0.008 | 0.008 | 0 |
| competing_corridors | euler_lagrange | straight | 64→128 | 0/20 | 20 | — | — | 0 |
| competing_corridors | slsqp | arc_left | 64→128 | 0/20 | 20 | — | — | 0 |
| competing_corridors | slsqp | arc_right | 64→128 | 0/20 | 20 | — | — | 0 |
| competing_corridors | slsqp | barrier | 64→128 | 0/20 | 20 | — | — | 0 |
| competing_corridors | slsqp | fast_marching | 64→128 | 0/20 | 20 | — | — | 0 |
| competing_corridors | slsqp | straight | 64→128 | 0/20 | 20 | — | — | 0 |
| correlated_roughness | euler_lagrange | arc_left | 64→128 | 10/20 | 10 | -0.021 | 0.098 | 2 |
| correlated_roughness | euler_lagrange | arc_right | 64→128 | 10/20 | 10 | -0.014 | 0.035 | 2 |
| correlated_roughness | euler_lagrange | barrier | 64→128 | 16/20 | 4 | -0.012 | 0.114 | 4 |
| correlated_roughness | euler_lagrange | fast_marching | 64→128 | 13/20 | 7 | -0.007 | 0.007 | 0 |
| correlated_roughness | euler_lagrange | straight | 64→128 | 14/20 | 6 | -0.012 | 0.016 | 1 |
| correlated_roughness | slsqp | arc_left | 64→128 | 0/20 | 20 | — | — | 0 |
| correlated_roughness | slsqp | arc_right | 64→128 | 0/20 | 20 | — | — | 0 |
| correlated_roughness | slsqp | barrier | 64→128 | 0/20 | 20 | — | — | 0 |
| correlated_roughness | slsqp | fast_marching | 64→128 | 0/20 | 20 | — | — | 0 |
| correlated_roughness | slsqp | straight | 64→128 | 0/20 | 20 | — | — | 0 |
| dead_ends | euler_lagrange | arc_left | 64→128 | 0/20 | 20 | — | — | 0 |
| dead_ends | euler_lagrange | arc_right | 64→128 | 0/20 | 20 | — | — | 0 |
| dead_ends | euler_lagrange | barrier | 64→128 | 8/20 | 12 | 18.432 | 18.432 | 8 |
| dead_ends | euler_lagrange | fast_marching | 64→128 | 20/20 | 0 | -0.006 | 0.006 | 0 |
| dead_ends | euler_lagrange | straight | 64→128 | 0/20 | 20 | — | — | 0 |
| dead_ends | slsqp | arc_left | 64→128 | 0/20 | 20 | — | — | 0 |
| dead_ends | slsqp | arc_right | 64→128 | 0/20 | 20 | — | — | 0 |
| dead_ends | slsqp | barrier | 64→128 | 0/20 | 20 | — | — | 0 |
| dead_ends | slsqp | fast_marching | 64→128 | 0/20 | 20 | — | — | 0 |
| dead_ends | slsqp | straight | 64→128 | 0/20 | 20 | — | — | 0 |
| ridge_pass | euler_lagrange | arc_left | 64→128 | 7/20 | 13 | -0.001 | 0.001 | 0 |
| ridge_pass | euler_lagrange | arc_right | 64→128 | 1/20 | 19 | -0.001 | 0.001 | 0 |
| ridge_pass | euler_lagrange | barrier | 64→128 | 18/20 | 2 | -0.000 | 0.000 | 0 |
| ridge_pass | euler_lagrange | fast_marching | 64→128 | 20/20 | 0 | -0.000 | 0.000 | 0 |
| ridge_pass | euler_lagrange | straight | 64→128 | 9/20 | 11 | -0.000 | 0.000 | 0 |
| ridge_pass | slsqp | arc_left | 64→128 | 0/20 | 20 | — | — | 0 |
| ridge_pass | slsqp | arc_right | 64→128 | 0/20 | 20 | — | — | 0 |
| ridge_pass | slsqp | barrier | 64→128 | 1/20 | 19 | -0.000 | 0.000 | 0 |
| ridge_pass | slsqp | fast_marching | 64→128 | 0/20 | 20 | — | — | 0 |
| ridge_pass | slsqp | straight | 64→128 | 1/20 | 19 | -0.000 | 0.000 | 0 |

## Controlled cost-contrast sweep

One predeclared seed per family is evaluated at each contrast. The table keeps family and contrast separate; it is descriptive because each point contains one terrain seed.

| Family | Contrast | Method | Feasible / runs | Median evaluated cost (s) | Resolved quality n | Median signed ref. diff. (%) |
| --- | --- | --- | --- | --- | --- | --- |
| competing_corridors | 0.0 | euler_lagrange | 0/5 | — | 0 | — |
| competing_corridors | 0.0 | fast_marching | 1/1 | 719.47 | 0 | — |
| competing_corridors | 0.0 | slsqp | 3/5 | 720.20 | 0 | — |
| competing_corridors | 1.0 | euler_lagrange | 3/5 | 1209.35 | 0 | — |
| competing_corridors | 1.0 | fast_marching | 1/1 | 1211.32 | 0 | — |
| competing_corridors | 1.0 | slsqp | 4/5 | 1209.34 | 0 | — |
| competing_corridors | 2.0 | euler_lagrange | 4/5 | 2501.38 | 0 | — |
| competing_corridors | 2.0 | fast_marching | 1/1 | 1952.54 | 0 | — |
| competing_corridors | 2.0 | slsqp | 4/5 | 1944.08 | 0 | — |
| competing_corridors | 4.0 | euler_lagrange | 4/5 | 13559.14 | 0 | — |
| competing_corridors | 4.0 | fast_marching | 1/1 | 7267.25 | 0 | — |
| competing_corridors | 4.0 | slsqp | 4/5 | 7834.76 | 0 | — |
| correlated_roughness | 0.0 | euler_lagrange | 5/5 | 704.00 | 0 | — |
| correlated_roughness | 0.0 | fast_marching | 1/1 | 704.83 | 0 | — |
| correlated_roughness | 0.0 | slsqp | 5/5 | 704.00 | 0 | — |
| correlated_roughness | 1.0 | euler_lagrange | 4/5 | 630.23 | 0 | — |
| correlated_roughness | 1.0 | fast_marching | 1/1 | 601.41 | 0 | — |
| correlated_roughness | 1.0 | slsqp | 5/5 | 600.94 | 0 | — |
| correlated_roughness | 2.0 | euler_lagrange | 5/5 | 588.85 | 0 | — |
| correlated_roughness | 2.0 | fast_marching | 1/1 | 521.01 | 0 | — |
| correlated_roughness | 2.0 | slsqp | 5/5 | 520.74 | 0 | — |
| correlated_roughness | 4.0 | euler_lagrange | 4/5 | 752.71 | 0 | — |
| correlated_roughness | 4.0 | fast_marching | 1/1 | 423.79 | 0 | — |
| correlated_roughness | 4.0 | slsqp | 5/5 | 423.59 | 0 | — |
| dead_ends | 0.0 | euler_lagrange | 0/5 | — | 0 | — |
| dead_ends | 0.0 | fast_marching | 1/1 | 815.84 | 0 | — |
| dead_ends | 0.0 | slsqp | 1/5 | 820.80 | 0 | — |
| dead_ends | 1.0 | euler_lagrange | 1/5 | 947.57 | 0 | — |
| dead_ends | 1.0 | fast_marching | 1/1 | 947.92 | 0 | — |
| dead_ends | 1.0 | slsqp | 2/5 | 947.58 | 0 | — |
| dead_ends | 2.0 | euler_lagrange | 2/5 | 2925.74 | 0 | — |
| dead_ends | 2.0 | fast_marching | 1/1 | 962.34 | 0 | — |
| dead_ends | 2.0 | slsqp | 2/5 | 961.92 | 0 | — |
| dead_ends | 4.0 | euler_lagrange | 2/5 | 20981.31 | 0 | — |
| dead_ends | 4.0 | fast_marching | 1/1 | 975.79 | 0 | — |
| dead_ends | 4.0 | slsqp | 2/5 | 10438.77 | 0 | — |
| ridge_pass | 0.0 | euler_lagrange | 5/5 | 704.00 | 0 | — |
| ridge_pass | 0.0 | fast_marching | 1/1 | 704.83 | 0 | — |
| ridge_pass | 0.0 | slsqp | 3/5 | 704.00 | 0 | — |
| ridge_pass | 1.0 | euler_lagrange | 3/5 | 726.93 | 0 | — |
| ridge_pass | 1.0 | fast_marching | 1/1 | 727.78 | 0 | — |
| ridge_pass | 1.0 | slsqp | 4/5 | 726.93 | 0 | — |
| ridge_pass | 2.0 | euler_lagrange | 1/5 | 744.86 | 0 | — |
| ridge_pass | 2.0 | fast_marching | 1/1 | 745.66 | 0 | — |
| ridge_pass | 2.0 | slsqp | 4/5 | 744.86 | 0 | — |
| ridge_pass | 4.0 | euler_lagrange | 1/5 | 783.97 | 0 | — |
| ridge_pass | 4.0 | fast_marching | 1/1 | 784.85 | 0 | — |
| ridge_pass | 4.0 | slsqp | 4/5 | 783.97 | 0 | — |

## Evidence for the hypotheses

### 1. Initialization, topology, and residual tolerance

Among 380 case–method–resolution cells with at least two feasible starts, the median within-case cost range was 6.179% and the maximum was 413.215%.

The initialization-range and family-level feasibility tables quantify sensitivity to the starting geometry and terrain topology. The frozen study uses one residual tolerance, so it cannot estimate a causal effect of residual tolerance or establish that initialization matters *more* than tolerance.

### 2. Global initialization followed by local refinement

euler lagrange at N=128: cold 92/240 feasible (38.3%), median resolved difference 8.077% (n=90); barrier-only 47/80 feasible (58.8%), median resolved difference 9.555% (n=46); fast-marching warm 73/80 feasible (91.2%), median resolved difference -0.037% (n=72). slsqp at N=128: cold 1/240 feasible (0.4%), median resolved difference -0.064% (n=1); barrier-only 1/80 feasible (1.2%), median resolved difference -0.064% (n=1); fast-marching warm 0/80 feasible (0.0%), median resolved difference —% (n=0).

For a local method, `fast_marching` in the Initialization column identifies a warm start; `fast_marching` in the Method column identifies the grid reference itself. Warm starts can be compared with straight/arc cold starts and the barrier-only start using feasibility, solver termination, evaluated cost, and end-to-end time. The 64→128 table shows whether additional local resolution changed route quality among cases feasible at both levels.
In this run, fast-marching initialization strongly increased Euler–Lagrange feasibility at N=128. It did not rescue SLSQP at N=128 within the fixed budget, so the evidence does not support a universal warm-start benefit.

### 3. Residual, route quality, and resolution

The lowest-residual quartile contained 160 candidates (cutoff 8.645e-10); its maximum absolute signed reference difference was 75.243%.

The following Euler–Lagrange candidates were selected by large absolute signed reference difference while retaining stationarity and solver-success fields. They are diagnostic candidates; solver stationarity alone does not establish route quality, feasibility, or adequate spatial resolution.
The low-residual subset above contains large route-quality differences, directly supporting the claim that a small numerical residual can coexist with a poor route.

| Run | Case | Initialization | N | Cost (s) | Stationarity norm | Signed ref. diff. (%) | Solver success |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [3705f9971a5cd56d4cfcde02](runs/3705f9971a5cd56d4cfcde02.json.gz) | dead_ends-s05 | barrier | 64 | 2136.38 | 1.027e+04 | 131.169 | no |
| [8d855eddac08f011ca7bca9c](runs/8d855eddac08f011ca7bca9c.json.gz) | dead_ends-s07 | barrier | 128 | 1981.99 | 7.491e+03 | 108.224 | no |
| [d16a36daf701183c42b3a94d](runs/d16a36daf701183c42b3a94d.json.gz) | dead_ends-s01 | barrier | 128 | 2016.43 | 7.646e+03 | 107.671 | no |
| [9b18fbcc2c19f186ed637e24](runs/9b18fbcc2c19f186ed637e24.json.gz) | dead_ends-s16 | barrier | 128 | 1950.08 | 7.808e+03 | 102.789 | no |
| [60f25ff8fef5e98ee11a0b49](runs/60f25ff8fef5e98ee11a0b49.json.gz) | competing_corridors-s05 | arc_left | 32 | 2673.45 | 4.546e+03 | 98.840 | no |
| [6498d287238acb7d34f219a8](runs/6498d287238acb7d34f219a8.json.gz) | correlated_roughness-s11 | arc_left | 32 | 1156.34 | 2.969e+02 | 95.844 | no |
| [d38ef92ed3b71d5074372178](runs/d38ef92ed3b71d5074372178.json.gz) | correlated_roughness-s11 | arc_left | 64 | 1154.21 | 2.592e+01 | 95.483 | no |
| [0ee2ef60310dda3920932876](runs/0ee2ef60310dda3920932876.json.gz) | correlated_roughness-s11 | arc_left | 128 | 1153.80 | 1.787e-09 | 95.413 | yes |

## Representative resolved route-quality outcomes

These records show the lowest, highest, and closest-to-reference signed differences for each local method among feasible candidates with a resolved reference.

| Selection | Run | Case | Initialization | N | Cost (s) | Signed ref. diff. (%) | Stationarity | Solver success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| euler_lagrange: lowest signed difference | [3eb92fe412e72cb5e3c8a58f](runs/3eb92fe412e72cb5e3c8a58f.json.gz) | correlated_roughness-s05 | fast_marching | 128 | 596.97 | -0.183 | 1.080e-08 | yes |
| euler_lagrange: largest signed difference | [3705f9971a5cd56d4cfcde02](runs/3705f9971a5cd56d4cfcde02.json.gz) | dead_ends-s05 | barrier | 64 | 2136.38 | 131.169 | 1.027e+04 | no |
| euler_lagrange: closest to reference | [878204ed7f96f3ad50eaf8b6](runs/878204ed7f96f3ad50eaf8b6.json.gz) | competing_corridors-s14 | arc_right | 32 | 1517.02 | 0.001 | 5.126e-11 | yes |
| slsqp: lowest signed difference | [d9481e9836a50cb3d2d85ede](runs/d9481e9836a50cb3d2d85ede.json.gz) | correlated_roughness-s05 | fast_marching | 64 | 597.02 | -0.175 | 5.252e-03 | no |
| slsqp: largest signed difference | [34fd13c6ea9a85f0107e35be](runs/34fd13c6ea9a85f0107e35be.json.gz) | dead_ends-s11 | arc_right | 32 | 4688.80 | 413.265 | 5.225e-01 | no |
| slsqp: closest to reference | [92c504e76afe320e9a63b630](runs/92c504e76afe320e9a63b630.json.gz) | competing_corridors-s14 | fast_marching | 32 | 1517.00 | -0.001 | 4.366e-02 | no |

## Representative unsuccessful runs

These concrete records span observed family/reason combinations. Full routes, collision diagnostics, profiles, and timings are retained in the linked compressed detail records; all other failures remain in the raw archive and dashboard.

| Run | Family | Method | Initialization | Resolution | Reason | Feasible | Cost (s) | Stationarity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [bdc1f026d1153bf586680ca5](runs/bdc1f026d1153bf586680ca5.json.gz) | competing_corridors | euler_lagrange | straight | 64 | iteration_limit | no | 2029.04 | 3.923e+02 |
| [0022732879d00f44f188d08b](runs/0022732879d00f44f188d08b.json.gz) | competing_corridors | slsqp | arc_right | 32 | nonstationary | yes | 1281.58 | 1.948e-02 |
| [00269ec7824069c608cc2e5d](runs/00269ec7824069c608cc2e5d.json.gz) | competing_corridors | slsqp | straight | 64 | optimization_failed | no | 2637.68 | 2.015e-01 |
| [0397f4f7b7fb5a8f87d51f25](runs/0397f4f7b7fb5a8f87d51f25.json.gz) | competing_corridors | euler_lagrange | barrier | 32 | stagnated | no | 1515.06 | 3.442e+03 |
| [0a3043c8c4ac29ef9308f441](runs/0a3043c8c4ac29ef9308f441.json.gz) | competing_corridors | euler_lagrange | straight | 64 | stationary | no | 5710.97 | 3.574e-10 |
| [013471d137f60e532067bf4a](runs.jsonl.gz) | competing_corridors | slsqp | straight | 128 | timeout | — | — | — |
| [009c2fdf56d843e07ba377e7](runs/009c2fdf56d843e07ba377e7.json.gz) | correlated_roughness | slsqp | straight | 64 | nonstationary | yes | 861.51 | 3.316e-02 |
| [41c3698210a61abd02caba39](runs/41c3698210a61abd02caba39.json.gz) | correlated_roughness | slsqp | arc_left | 32 | optimization_failed | yes | 1136.50 | 1.642e-01 |
| [00ab8d2d8d46099debe11c06](runs/00ab8d2d8d46099debe11c06.json.gz) | correlated_roughness | euler_lagrange | barrier | 64 | stagnated | yes | 752.71 | 9.098e+03 |
| [01d0f0c5ab406916c3b0fdb3](runs/01d0f0c5ab406916c3b0fdb3.json.gz) | correlated_roughness | euler_lagrange | arc_left | 128 | stationary | no | 645.79 | 3.581e-09 |
| [077448beeaf07b8e87cee8bc](runs.jsonl.gz) | correlated_roughness | slsqp | straight | 128 | timeout | — | — | — |
| [01541cd98334c0e3283d4295](runs/01541cd98334c0e3283d4295.json.gz) | dead_ends | slsqp | barrier | 64 | nonstationary | yes | 947.58 | 3.619e-02 |

## Figures and machine-readable artifacts

- [Feasibility by initialization](feasibility-by-initialization.png)
- [Signed reference-difference distributions](reference-difference-distributions.png)
- [Reference-grid refinement](reference-refinement.png)
- [Contrast sweep](contrast-sweep.png)
- [Stationarity versus route quality](stationarity-vs-reference-difference.png)
- [Compact dashboard index](index.json)
- [Aggregate CSV](summary.csv)
- [Paired-comparison CSV](paired.csv)
- [Seed-level paired outcomes](paired-outcomes.csv)
- [Local-refinement CSV](local-refinement.csv)
- [Initialization-sensitivity CSV](initialization-sensitivity.csv)
- [Authoritative run archive](runs.jsonl.gz)

## Reproduction and provenance

```bash
uv sync --extra plot
uv run run-terrain-benchmarks --profile full --workers 8 --output-dir terrain-study-results/full
```

Add `--resume` only to continue that same interrupted output archive after verifying its protocol and numerical-source provenance.

Protocol SHA-256: `b4005505357ff6a0f5045f2207a28aada37d29286bf0013c6306aa5e778dfca5`.
Numerical execution source SHA-256: `3e088a6c8404c0d644c2456ddc42397405848d03e937a73bf71b6e12fd08b1cd`.
Analysis source SHA-256: `e55478cb1b0b73e3710bfa904b84ac6ff8fcfa4ff7415fc11515ade9a073dc3a`.
Reviewed implementation commit: `250fe184b283886f5d8d77fadb0747b18f42ee1f`.
The published execution used 8 concurrent worker processes. The launcher requested OPENBLAS/OMP/MKL limits {'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}. Apple Accelerate was the reported backend and VECLIB_MAXIMUM_THREADS was not set; actual library thread counts were not measured. Every record stores dependency versions, hardware/platform data, commit identity, working-tree numerical hash, scenario/configuration hashes, and timing components.

## Limitations

The cost law is illustrative, all costs are static and direction-independent, and modeled barriers are not observed traversability. The 80 cases cover four designed families rather than a population of real landscapes. First-order fast marching has discretization error, and its extracted polyline may fail even when its grid arrival field reaches the endpoint. Bootstrap intervals describe variation across the frozen seeds; they do not correct model misspecification. Browser timings are reported separately in the interface and are not pooled with these native runs. 8-process scheduling and unmeasured library threads may have contended for CPU, so wall times and timeout frequencies should not be generalized to other machines. Supplemental NumPy/SciPy build metadata is stored in index.json.

The bundled Mount Tamalpais crop is an attributed visualization and transfer example; it is not part of the frozen 80 synthetic-case protocol.
