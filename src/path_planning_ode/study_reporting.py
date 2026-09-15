"""Tables, figures, and narrative for the published terrain-routing study.

This module is deliberately separate from benchmark execution.  Raw run records
remain authoritative, while reports can be regenerated as the presentation is
reviewed without changing the numerical execution provenance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def capture_numerical_build_metadata() -> dict[str, Any]:
    """Capture structured BLAS/LAPACK build information for publication."""
    import scipy

    return {
        "numpy_version": np.__version__,
        "numpy_build": np.show_config(mode="dicts"),
        "scipy_version": scipy.__version__,
        "scipy_build": scipy.show_config(mode="dicts"),
    }


def _result(run: Mapping[str, Any]) -> Mapping[str, Any]:
    return run.get("result") or {}


def _cost(run: Mapping[str, Any]) -> float | None:
    result = _result(run)
    value = result.get("evaluated_cost_s")
    if run.get("record_status") != "completed" or not result.get("feasible") or value is None:
        return None
    return float(value)


def _resolved_difference(run: Mapping[str, Any]) -> float | None:
    value = run.get("reference_difference_percent")
    if value is None or run.get("reference_unresolved", True):
        return None
    return float(value)


def _returned_wall_time(run: Mapping[str, Any]) -> float | None:
    if run.get("record_status") != "completed":
        return None
    value = run.get("child_wall_time_s")
    if value is None:
        value = _result(run).get("timing_s", {}).get("total")
    return None if value is None else float(value)


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(np.median(values))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    number = float(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _fmt_scientific(value: Any) -> str:
    if value is None:
        return "—"
    number = float(value)
    return "—" if not np.isfinite(number) else f"{number:.3e}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    escaped = [[str(value).replace("|", "\\|").replace("\n", " ") for value in row] for row in rows]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in escaped),
    ]


def _link(prefix: str, relative: str) -> str:
    return f"{prefix.rstrip('/')}/{relative}" if prefix else relative


def _bootstrap_interval(
    values: Sequence[tuple[str, float]], samples: int, seed: int
) -> tuple[float | None, float | None]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for cluster, value in values:
        clusters[cluster].append(value)
    names = sorted(clusters)
    if len(names) < 2 or samples < 1:
        return None, None
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for index in range(samples):
        chosen = rng.choice(names, len(names), replace=True)
        estimates[index] = np.mean([value for name in chosen for value in clusters[name]])
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def _baseline_runs(index: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [run for run in index["runs"] if not run["case_id"].startswith("contrast-")]


def _method_rows(runs: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["config"]["method"]].append(run)
    rows = []
    for method, group in sorted(grouped.items()):
        feasible = [run for run in group if _cost(run) is not None]
        successful = [run for run in group if _result(run).get("solver_success")]
        differences = [
            difference for run in feasible if (difference := _resolved_difference(run)) is not None
        ]
        clearances = [
            float(_result(run)["evaluation"]["minimum_clearance_m"])
            for run in feasible
            if _result(run).get("evaluation", {}).get("minimum_clearance_m") is not None
        ]
        runtimes = [runtime for run in group if (runtime := _returned_wall_time(run)) is not None]
        rows.append(
            [
                method,
                str(len(group)),
                f"{len(feasible)}/{len(group)} ({len(feasible) / len(group):.1%})",
                f"{len(successful)}/{len(group)} ({len(successful) / len(group):.1%})",
                _fmt(_median(differences)),
                str(len(differences)),
                _fmt(_median(clearances), 2),
                _fmt(_median(runtimes), 2),
                str(sum(run["record_status"] == "timeout" for run in group)),
            ]
        )
    return rows


def _configuration_rows(runs: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        config = run["config"]
        if config["method"] == "fast_marching":
            key = ("fast_marching", "reference", f"grid {config['reference_grid_size']}")
        else:
            key = (
                config["method"],
                config["initialization"],
                f"N={config['interior_points']}",
            )
        grouped[key].append(run)
    rows = []
    for key, group in sorted(grouped.items()):
        feasible = [run for run in group if _cost(run) is not None]
        success = [run for run in group if _result(run).get("solver_success")]
        differences = [
            difference for run in feasible if (difference := _resolved_difference(run)) is not None
        ]
        clearances = [
            float(_result(run)["evaluation"]["minimum_clearance_m"])
            for run in feasible
            if _result(run).get("evaluation", {}).get("minimum_clearance_m") is not None
        ]
        runtimes = [runtime for run in group if (runtime := _returned_wall_time(run)) is not None]
        rows.append(
            [
                *map(str, key),
                str(len(group)),
                f"{len(feasible) / len(group):.1%}",
                f"{len(success) / len(group):.1%}",
                _fmt(_median(differences)),
                str(len(differences)),
                _fmt(_median(clearances), 2),
                _fmt(_median(runtimes), 2),
                str(sum(run["record_status"] == "timeout" for run in group)),
            ]
        )
    return rows


def _termination_rows(runs: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for run in runs:
        method = run["config"]["method"]
        reason = (
            run["record_status"]
            if run["record_status"] != "completed"
            else _result(run).get("termination_reason", "missing_result")
        )
        grouped[method][reason] += 1
    return [
        [method, reason, str(count)]
        for method, counts in sorted(grouped.items())
        for reason, count in sorted(counts.items())
    ]


def _paired_rows(runs: Sequence[Mapping[str, Any]], *, samples: int, seed: int) -> list[list[str]]:
    pairs: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    case_info: dict[str, tuple[str, int]] = {}
    for run in runs:
        config = run["config"]
        method = config["method"]
        if method not in {"euler_lagrange", "slsqp"}:
            continue
        key = (run["case_id"], config["initialization"], int(config["interior_points"]))
        pairs[key][method] = run
        case_info[run["case_id"]] = (run["family"], int(run["seed"]))
    grouped: dict[tuple[str, int], dict[str, Any]] = defaultdict(
        lambda: {"declared": 0, "matched": 0, "values": []}
    )
    for (case_id, initialization, interior), methods in pairs.items():
        if set(methods) != {"euler_lagrange", "slsqp"}:
            continue
        group = grouped[(initialization, interior)]
        group["declared"] += 1
        hashes = []
        for method in ("euler_lagrange", "slsqp"):
            hashes.append(
                _result(methods[method])
                .get("diagnostics", {})
                .get("initialization", {})
                .get("initial_route_hash")
            )
        if hashes[0] is None or hashes[0] != hashes[1]:
            continue
        group["matched"] += 1
        euler = _cost(methods["euler_lagrange"])
        slsqp = _cost(methods["slsqp"])
        if euler is None or slsqp is None:
            continue
        family, case_seed = case_info[case_id]
        group["values"].append((f"{family}:{case_seed}", 100 * (slsqp - euler) / euler))
    rows = []
    for (initialization, interior), group in sorted(grouped.items()):
        values = group["values"]
        low, high = _bootstrap_interval(values, samples, seed)
        rows.append(
            [
                initialization,
                str(interior),
                str(group["declared"]),
                str(group["matched"]),
                f"{len(values)}/{group['matched']}" if group["matched"] else "0/0",
                _fmt(None if not values else np.mean([value for _, value in values])),
                f"[{_fmt(low)}, {_fmt(high)}]" if low is not None else "insufficient clusters",
            ]
        )
    return rows


def _reference_rows(
    runs: Sequence[Mapping[str, Any]], declared_grids: Sequence[int], threshold: float
) -> tuple[list[list[str]], dict[str, int]]:
    required = tuple(sorted(int(value) for value in declared_grids)[-2:])
    grouped: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        if run["config"]["method"] != "fast_marching":
            continue
        cost = _cost(run)
        if cost is not None:
            grouped[run["family"]][run["case_id"]][int(run["config"]["reference_grid_size"])] = cost
    rows, totals = [], {"cases": 0, "resolved": 0, "unresolved": 0, "missing": 0}
    families = sorted({run["family"] for run in runs})
    for family in families:
        changes, missing = [], 0
        cases = {run["case_id"] for run in runs if run["family"] == family}
        for case_id in cases:
            values = grouped[family].get(case_id, {})
            if not all(grid in values for grid in required):
                missing += 1
                continue
            changes.append(100 * (values[required[1]] - values[required[0]]) / values[required[1]])
        unresolved = sum(abs(value) > threshold for value in changes) + missing
        resolved = len(cases) - unresolved
        totals["cases"] += len(cases)
        totals["resolved"] += resolved
        totals["unresolved"] += unresolved
        totals["missing"] += missing
        rows.append(
            [
                family,
                str(len(cases)),
                str(resolved),
                str(unresolved),
                str(missing),
                _fmt(_median(changes)),
                _fmt(None if not changes else max(abs(value) for value in changes)),
            ]
        )
    return rows, totals


def _initialization_rows(runs: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    cases: dict[tuple[str, int], int] = Counter()
    by_case: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    declared = {
        (run["case_id"], run["config"]["method"], int(run["config"]["interior_points"]))
        for run in runs
        if run["config"]["method"] in {"euler_lagrange", "slsqp"}
    }
    for run in runs:
        config = run["config"]
        if config["method"] not in {"euler_lagrange", "slsqp"}:
            continue
        cost = _cost(run)
        if cost is not None:
            by_case[(run["case_id"], config["method"], int(config["interior_points"]))].append(cost)
    for _, method, interior in declared:
        cases[(method, interior)] += 1
    for (_, method, interior), costs in by_case.items():
        if len(costs) >= 2:
            grouped[(method, interior)].append(100 * (max(costs) - min(costs)) / min(costs))
    return [
        [
            method,
            str(interior),
            f"{len(values)}/{cases[(method, interior)]}",
            _fmt(_median(values)),
            _fmt(None if not values else np.quantile(values, 0.9)),
            _fmt(None if not values else max(values)),
        ]
        for method, interior in sorted(cases)
        for values in [grouped[(method, interior)]]
    ]


def _local_refinement_rows(index: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    for row in index.get("local_refinement", []):
        if row["experiment"] != "baseline":
            continue
        rows.append(
            [
                row["family"],
                row["method"],
                row["initialization"],
                f"{row['declared_coarse_interior_points']}→{row['declared_fine_interior_points']}",
                f"{row['cases_with_two_feasible_finest_levels']}/{row['cases']}",
                str(row["unresolved_cases"]),
                _fmt(row["median_signed_change_percent"]),
                _fmt(row["median_absolute_change_percent"]),
                str(row["over_0p5_percent"]),
            ]
        )
    return rows


def _contrast_rows(index: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    grouped: dict[tuple[str, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    # Contrast belongs to the case and is not repeated in the compact run at
    # present. Recover it from the deterministic `-cXpY-` case-id component.
    for run in index["runs"]:
        if not run["case_id"].startswith("contrast-"):
            continue
        marker = run["case_id"].rsplit("-c", 1)[1]
        contrast = float(marker.replace("p", "."))
        grouped[(run["family"], contrast, run["config"]["method"])].append(run)
    for (family, contrast, method), group in sorted(grouped.items()):
        feasible = [run for run in group if _cost(run) is not None]
        differences = [
            difference for run in feasible if (difference := _resolved_difference(run)) is not None
        ]
        rows.append(
            [
                family,
                _fmt(contrast, 1),
                method,
                f"{len(feasible)}/{len(group)}",
                _fmt(_median([cost for run in feasible if (cost := _cost(run)) is not None]), 2),
                str(len(differences)),
                _fmt(_median(differences)),
            ]
        )
    return rows


def _diagnostic_examples(
    runs: Sequence[Mapping[str, Any]], artifact_prefix: str, limit: int = 12
) -> list[list[str]]:
    candidates = []
    for run in runs:
        result = _result(run)
        unsuccessful = run["record_status"] != "completed" or not (
            result.get("feasible") and result.get("solver_success")
        )
        if not unsuccessful:
            continue
        reason = (
            run["record_status"]
            if run["record_status"] != "completed"
            else result.get("termination_reason", "missing_result")
        )
        candidates.append((run["family"], reason, run))
    selected, seen = [], set()
    for family, reason, run in sorted(
        candidates, key=lambda item: (item[0], item[1], item[2]["run_id"])
    ):
        key = (family, reason)
        if key in seen:
            continue
        seen.add(key)
        selected.append(run)
        if len(selected) >= limit:
            break
    rows = []
    for run in selected:
        result = _result(run)
        config = run["config"]
        reason = (
            run["record_status"]
            if run["record_status"] != "completed"
            else result.get("termination_reason", "missing_result")
        )
        diagnostics = result.get("diagnostics", {})
        rows.append(
            [
                (
                    f"[{run['run_id']}]"
                    f"({_link(artifact_prefix, run.get('result_ref') or 'runs.jsonl.gz')})"
                ),
                run["family"],
                result.get("method", config["method"]),
                config["initialization"],
                str(
                    config.get("reference_grid_size")
                    if config["method"] == "fast_marching"
                    else config.get("interior_points")
                ),
                reason,
                _fmt(result.get("feasible")),
                _fmt(result.get("evaluated_cost_s"), 2),
                _fmt_scientific(diagnostics.get("stationarity_norm")),
            ]
        )
    return rows


def _residual_examples(
    index: Mapping[str, Any], artifact_prefix: str, limit: int = 8
) -> list[list[str]]:
    candidates = []
    for run in _baseline_runs(index):
        result = _result(run)
        if run["config"]["method"] != "euler_lagrange" or not result.get("feasible"):
            continue
        residual = result.get("diagnostics", {}).get("stationarity_norm")
        difference = _resolved_difference(run)
        if residual is None or difference is None:
            continue
        candidates.append((abs(difference), run, float(residual), difference))
    rows = []
    selected = sorted(candidates, reverse=True, key=lambda item: item[0])[:limit]
    for _, run, residual, difference in selected:
        result = _result(run)
        rows.append(
            [
                (
                    f"[{run['run_id']}]"
                    f"({_link(artifact_prefix, run.get('result_ref') or 'runs.jsonl.gz')})"
                ),
                run["case_id"],
                run["config"]["initialization"],
                str(run["config"]["interior_points"]),
                _fmt(result.get("evaluated_cost_s"), 2),
                _fmt_scientific(residual),
                _fmt(difference),
                _fmt(result["solver_success"]),
            ]
        )
    return rows


def _quality_examples(runs: Sequence[Mapping[str, Any]], artifact_prefix: str) -> list[list[str]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for method in ("euler_lagrange", "slsqp"):
        candidates = [
            run
            for run in runs
            if run["config"]["method"] == method
            and _cost(run) is not None
            and _resolved_difference(run) is not None
        ]
        if not candidates:
            continue
        for label, key in (
            ("lowest signed difference", lambda run: _resolved_difference(run)),
            ("largest signed difference", lambda run: -_resolved_difference(run)),
            ("closest to reference", lambda run: abs(_resolved_difference(run))),
        ):
            run = min(candidates, key=key)
            selected[f"{method}: {label}"] = run
    rows = []
    for label, run in selected.items():
        result = _result(run)
        rows.append(
            [
                label,
                (
                    f"[{run['run_id']}]"
                    f"({_link(artifact_prefix, run.get('result_ref') or 'runs.jsonl.gz')})"
                ),
                run["case_id"],
                run["config"]["initialization"],
                str(run["config"]["interior_points"]),
                _fmt(result.get("evaluated_cost_s"), 2),
                _fmt(_resolved_difference(run)),
                _fmt_scientific(result.get("diagnostics", {}).get("stationarity_norm")),
                _fmt(result.get("solver_success")),
            ]
        )
    return rows


def _initialization_evidence(runs: Sequence[Mapping[str, Any]]) -> str:
    spreads = []
    by_case: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for run in runs:
        config = run["config"]
        if config["method"] not in {"euler_lagrange", "slsqp"}:
            continue
        cost = _cost(run)
        if cost is not None:
            by_case[(run["case_id"], config["method"], int(config["interior_points"]))].append(cost)
    for costs in by_case.values():
        if len(costs) >= 2:
            spreads.append(100 * (max(costs) - min(costs)) / min(costs))
    if not spreads:
        return "No case had two feasible local initializations, so cost sensitivity is unavailable."
    return (
        f"Among {len(spreads)} case–method–resolution cells with at least two feasible starts, "
        f"the median within-case cost range was {_fmt(_median(spreads))}% and the maximum was "
        f"{_fmt(max(spreads))}%."
    )


def _warm_start_evidence(runs: Sequence[Mapping[str, Any]], finest: int) -> str:
    statements = []
    for method in ("euler_lagrange", "slsqp"):
        method_runs = [
            run
            for run in runs
            if run["config"]["method"] == method and int(run["config"]["interior_points"]) == finest
        ]
        parts = []
        for label, initializations in (
            ("cold", {"straight", "arc_left", "arc_right"}),
            ("barrier-only", {"barrier"}),
            ("fast-marching warm", {"fast_marching"}),
        ):
            group = [
                run for run in method_runs if run["config"]["initialization"] in initializations
            ]
            feasible = sum(_cost(run) is not None for run in group)
            resolved = [
                difference for run in group if (difference := _resolved_difference(run)) is not None
            ]
            rate = feasible / len(group) if group else 0.0
            parts.append(
                f"{label} {feasible}/{len(group)} feasible ({rate:.1%}), median resolved "
                f"difference {_fmt(_median(resolved))}% (n={len(resolved)})"
            )
        statements.append(f"{method.replace('_', ' ')} at N={finest}: " + "; ".join(parts) + ".")
    return " ".join(statements)


def _residual_evidence(index: Mapping[str, Any]) -> str:
    candidates = []
    for run in _baseline_runs(index):
        result = _result(run)
        if run["config"]["method"] != "euler_lagrange" or not result.get("feasible"):
            continue
        residual = result.get("diagnostics", {}).get("stationarity_norm")
        difference = _resolved_difference(run)
        if residual is not None and difference is not None and float(residual) >= 0:
            candidates.append((float(residual), abs(difference)))
    if not candidates:
        return "No feasible Euler–Lagrange candidate had both a residual and a resolved reference."
    cutoff = float(np.quantile([residual for residual, _ in candidates], 0.25))
    low_residual = [difference for residual, difference in candidates if residual <= cutoff]
    return (
        f"The lowest-residual quartile contained {len(low_residual)} candidates "
        f"(cutoff {_fmt_scientific(cutoff)}); its maximum absolute signed reference difference "
        "was "
        f"{_fmt(max(low_residual))}%."
    )


def write_study_report(
    index: Mapping[str, Any],
    path: str | Path,
    *,
    artifact_prefix: str = "../experiments/published",
    documentation_prefix: str = "",
) -> None:
    """Write a data-driven report from a compact published index."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    protocol, scope = index["protocol"], index["scope"]
    runs = index["runs"]
    baseline = _baseline_runs(index)
    completed = sum(run["record_status"] == "completed" for run in runs)
    feasible = sum(_cost(run) is not None for run in runs)
    worker_failures = sum(run["record_status"] == "worker_error" for run in runs)
    timeouts = sum(run["record_status"] == "timeout" for run in runs)
    reference_rows, reference_totals = _reference_rows(
        baseline,
        protocol["reference_grid_sizes"],
        float(protocol["reference_refinement_unresolved_percent"]),
    )
    unresolved_reference_cases = sorted(
        {
            run["case_id"]: run.get("reference_refinement_change_percent")
            for run in baseline
            if run["config"]["method"] == "fast_marching" and run["reference_unresolved"]
        }.items()
    )
    unresolved_reference_text = (
        ", ".join(f"{case} ({_fmt(change)}%)" for case, change in unresolved_reference_cases)
        or "none"
    )
    bootstrap = protocol["bootstrap"]
    paired_rows = _paired_rows(
        baseline, samples=int(bootstrap["samples"]), seed=int(bootstrap["seed"])
    )
    init_rows = _initialization_rows(baseline)
    refinement_rows = _local_refinement_rows(index)
    contrast_rows = _contrast_rows(index)
    environments = index.get("environments", [])
    environment = environments[0] if environments else {}
    execution_source = environment.get("numerical_execution_source_sha256", "unavailable")
    analysis_source = index.get("publication", {}).get(
        "analysis_source_sha256", environment.get("analysis_source_sha256", "unavailable")
    )
    worker_count = environment.get("benchmark_workers", "unknown")
    build_metadata = index.get("publication", {}).get("numerical_build", {})
    numpy_blas = (
        build_metadata.get("numpy_build", {})
        .get("Build Dependencies", {})
        .get("blas", {})
        .get("name", "unavailable")
    )
    accelerate_note = (
        "Apple Accelerate was the reported backend and VECLIB_MAXIMUM_THREADS was not set; "
        "actual library thread counts were not measured. "
        if str(numpy_blas).lower() == "accelerate"
        else f"The reported NumPy BLAS backend was {numpy_blas}; actual library thread counts "
        "were not measured. "
    )
    lines = [
        f"# {index['title']}",
        "",
        "## Abstract",
        "",
        (
            f"This computational study compares an Euler–Lagrange boundary-value solver, "
            f"direct constrained SLSQP optimization, and an isotropic fast-marching grid "
            f"reference on a frozen heterogeneous-terrain protocol. {scope['statement']} "
            f"The artifact contains {len(runs)} run records: {completed} completed, "
            f"{timeouts} hit the hard subprocess timeout, {worker_failures} failed in a worker, "
            f"and {feasible} yielded "
            "routes that passed independent whole-segment validation."
        ),
        "",
        "## Scope and model",
        "",
        protocol["model_boundary"],
        (
            "Every planner queries one fixed bicubic spline of log-slowness, exponentiated to "
            "keep cost positive. Barriers remain separate geometric constraints. The shared "
            "evaluator integrates each submitted polyline and tests every complete segment; "
            "infeasible routes never enter route-quality summaries."
        ),
        (
            "Fast marching is a grid-based reference rather than a certified continuous "
            "optimum. Differences below are signed `(candidate - reference) / reference`; "
            "negative values are retained. A case is unresolved when either of the two finest "
            "declared grids is unavailable or their evaluated costs differ by more than 0.5%."
        ),
        (
            "See the [methods and reproduction appendix]"
            f"({_link(documentation_prefix, 'terrain-study.md')}) and the "
            "[independent numerical audit]"
            f"({_link(documentation_prefix, 'numerical-audit.md')}) for derivations, "
            "validation cases, collision semantics, and cross-implementation checks."
        ),
        "",
        "## Frozen design",
        "",
        *_markdown_table(
            ("Component", "Frozen value"),
            (
                (
                    "Baseline terrain cases",
                    f"{protocol['baseline']['case_count']} = 4 families × 20 seeds",
                ),
                ("Difficulty assignment", protocol["difficulty_rule"]),
                ("Local methods", ", ".join(protocol["local_methods"])),
                ("Shared initializations", ", ".join(protocol["initializations"])),
                ("Local interior points", ", ".join(map(str, protocol["local_interior_points"]))),
                ("Reference grids", ", ".join(map(str, protocol["reference_grid_sizes"]))),
                ("Contrast sweep", ", ".join(map(str, protocol["contrast_sweep"]["values"]))),
                (
                    "Per-run budget",
                    f"{protocol['planner_time_limit_s']:.0f} s including "
                    "initialization/preprocessing",
                ),
                (
                    "Bootstrap",
                    f"{bootstrap['samples']} resamples; cluster={bootstrap['cluster']}; "
                    f"seed={bootstrap['seed']}",
                ),
            ),
        ),
        "",
        "## Aggregate baseline results",
        "",
        (
            "Feasibility and cost come from the independent evaluator. Solver success records "
            "termination separately. Cost medians omit unavailable or infeasible routes. Time "
            "medians use end-to-end child wall time, including scenario construction, for "
            "returned planner results; hard subprocess timeouts are right-censored at 60 "
            "seconds and shown in a separate column."
        ),
        "",
        *_markdown_table(
            (
                "Method",
                "Runs",
                "Feasible",
                "Solver success",
                "Median signed ref. diff. (%)",
                "Resolved quality n",
                "Median clearance (m)",
                "Median returned end-to-end wall time (s)",
                "Hard timeouts",
            ),
            _method_rows(baseline),
        ),
        "",
        "### Configuration-level results",
        "",
        *_markdown_table(
            (
                "Method",
                "Initialization",
                "Resolution",
                "Runs",
                "Feasible rate",
                "Solver success rate",
                "Median signed ref. diff. (%)",
                "Resolved quality n",
                "Median clearance (m)",
                "Median returned end-to-end wall time (s)",
                "Hard timeouts",
            ),
            _configuration_rows(baseline),
        ),
        "",
        "### Termination and failure modes",
        "",
        *_markdown_table(("Method", "Termination/status", "Count"), _termination_rows(runs)),
        "",
        "## Paired local-method comparison",
        "",
        (
            "Each pair uses the same case, initialization name, discretization, and verified "
            "initial-route hash. The effect is `SLSQP - Euler–Lagrange` as a percentage of the "
            "Euler–Lagrange evaluated cost. Paired feasible denominators expose survivorship; "
            "95% intervals resample case/seed clusters."
        ),
        "",
        *_markdown_table(
            (
                "Initialization",
                "N",
                "Declared pairs",
                "Hash matched",
                "Both feasible / matched",
                "Mean difference (%)",
                "Bootstrap 95% interval (%)",
            ),
            paired_rows,
        ),
        "",
        "## Reference refinement",
        "",
        (
            f"Across {reference_totals['cases']} baseline cases, {reference_totals['resolved']} "
            f"met the empirical refinement check and {reference_totals['unresolved']} were "
            f"unresolved, including {reference_totals['missing']} with a missing or infeasible "
            "declared 513/1025 result. This is an empirical check, not an error bound."
        ),
        f"Unresolved baseline cases and signed 513→1025 changes: {unresolved_reference_text}.",
        "",
        *_markdown_table(
            (
                "Family",
                "Cases",
                "Resolved",
                "Unresolved",
                "Missing finest pair",
                "Median 513→1025 change (%)",
                "Maximum absolute change (%)",
            ),
            reference_rows,
        ),
        "",
        "## Initialization and local refinement",
        "",
        (
            "Initialization sensitivity is the within-case percentage range across feasible "
            "starts. Cases with fewer than two feasible starts are reported in the denominator "
            "but cannot contribute a range."
        ),
        "",
        *_markdown_table(
            (
                "Method",
                "N",
                "Cases with ≥2 feasible starts / cases",
                "Median range (%)",
                "90th percentile (%)",
                "Maximum (%)",
            ),
            init_rows,
        ),
        "",
        (
            "The next table compares only the two finest declared local levels, 64 and 128 "
            "interior points. Missing levels remain unresolved rather than being replaced by "
            "coarser successful runs."
        ),
        "",
        *_markdown_table(
            (
                "Family",
                "Method",
                "Initialization",
                "Levels",
                "Both feasible / cases",
                "Unresolved",
                "Median signed change (%)",
                "Median absolute change (%)",
                "Absolute change >0.5%",
            ),
            refinement_rows,
        ),
        "",
        "## Controlled cost-contrast sweep",
        "",
        (
            "One predeclared seed per family is evaluated at each contrast. The table keeps "
            "family and contrast separate; it is descriptive because each point contains one "
            "terrain seed."
        ),
        "",
        *_markdown_table(
            (
                "Family",
                "Contrast",
                "Method",
                "Feasible / runs",
                "Median evaluated cost (s)",
                "Resolved quality n",
                "Median signed ref. diff. (%)",
            ),
            contrast_rows,
        ),
        "",
        "## Evidence for the hypotheses",
        "",
        "### 1. Initialization, topology, and residual tolerance",
        "",
        _initialization_evidence(baseline),
        "",
        (
            "The initialization-range and family-level feasibility tables quantify sensitivity "
            "to the starting geometry and terrain topology. The frozen study uses one residual "
            "tolerance, so it cannot estimate a causal effect of residual tolerance or establish "
            "that initialization matters *more* than tolerance."
        ),
        "",
        "### 2. Global initialization followed by local refinement",
        "",
        _warm_start_evidence(baseline, max(protocol["local_interior_points"])),
        "",
        (
            "For a local method, `fast_marching` in the Initialization column identifies a warm "
            "start; `fast_marching` in the Method column identifies the grid reference itself. "
            "Warm starts can be compared with straight/arc cold starts and the barrier-only "
            "start using feasibility, solver termination, evaluated cost, and end-to-end time. "
            "The 64→128 table shows whether additional local resolution changed route quality "
            "among cases feasible at both levels."
        ),
        (
            "In this run, fast-marching initialization strongly increased Euler–Lagrange "
            "feasibility at N=128. It did not rescue SLSQP at N=128 within the fixed budget, "
            "so the evidence does not support a universal warm-start benefit."
        ),
        "",
        "### 3. Residual, route quality, and resolution",
        "",
        _residual_evidence(index),
        "",
        (
            "The following Euler–Lagrange candidates were selected by large absolute signed "
            "reference difference while retaining stationarity and solver-success fields. They "
            "are diagnostic candidates; solver stationarity alone does not establish route "
            "quality, feasibility, or adequate spatial resolution."
        ),
        (
            "The low-residual subset above contains large route-quality differences, directly "
            "supporting the claim that a small numerical residual can coexist with a poor route."
        ),
        "",
        *_markdown_table(
            (
                "Run",
                "Case",
                "Initialization",
                "N",
                "Cost (s)",
                "Stationarity norm",
                "Signed ref. diff. (%)",
                "Solver success",
            ),
            _residual_examples(index, artifact_prefix),
        ),
        "",
        "## Representative resolved route-quality outcomes",
        "",
        (
            "These records show the lowest, highest, and closest-to-reference signed "
            "differences for each local method among feasible candidates with a resolved "
            "reference."
        ),
        "",
        *_markdown_table(
            (
                "Selection",
                "Run",
                "Case",
                "Initialization",
                "N",
                "Cost (s)",
                "Signed ref. diff. (%)",
                "Stationarity",
                "Solver success",
            ),
            _quality_examples(baseline, artifact_prefix),
        ),
        "",
        "## Representative unsuccessful runs",
        "",
        (
            "These concrete records span observed family/reason combinations. Full routes, "
            "collision diagnostics, profiles, and timings are retained in the linked compressed "
            "detail records; all other failures remain in the raw archive and dashboard."
        ),
        "",
        *_markdown_table(
            (
                "Run",
                "Family",
                "Method",
                "Initialization",
                "Resolution",
                "Reason",
                "Feasible",
                "Cost (s)",
                "Stationarity",
            ),
            _diagnostic_examples(runs, artifact_prefix),
        ),
        "",
        "## Figures and machine-readable artifacts",
        "",
        (
            "- [Feasibility by initialization]"
            f"({_link(artifact_prefix, 'feasibility-by-initialization.png')})"
        ),
        (
            "- [Signed reference-difference distributions]"
            f"({_link(artifact_prefix, 'reference-difference-distributions.png')})"
        ),
        f"- [Reference-grid refinement]({_link(artifact_prefix, 'reference-refinement.png')})",
        f"- [Contrast sweep]({_link(artifact_prefix, 'contrast-sweep.png')})",
        (
            "- [Stationarity versus route quality]"
            f"({_link(artifact_prefix, 'stationarity-vs-reference-difference.png')})"
        ),
        f"- [Compact dashboard index]({_link(artifact_prefix, 'index.json')})",
        f"- [Aggregate CSV]({_link(artifact_prefix, 'summary.csv')})",
        f"- [Paired-comparison CSV]({_link(artifact_prefix, 'paired.csv')})",
        f"- [Seed-level paired outcomes]({_link(artifact_prefix, 'paired-outcomes.csv')})",
        f"- [Local-refinement CSV]({_link(artifact_prefix, 'local-refinement.csv')})",
        (
            "- [Initialization-sensitivity CSV]"
            f"({_link(artifact_prefix, 'initialization-sensitivity.csv')})"
        ),
        f"- [Authoritative run archive]({_link(artifact_prefix, 'runs.jsonl.gz')})",
        "",
        "## Reproduction and provenance",
        "",
        "```bash",
        "uv sync --extra plot",
        (
            "uv run run-terrain-benchmarks --profile full --workers 8 "
            "--output-dir terrain-study-results/full"
        ),
        "```",
        "",
        (
            "Add `--resume` only to continue that same interrupted output archive after verifying "
            "its protocol and numerical-source provenance."
        ),
        "",
        f"Protocol SHA-256: `{protocol['protocol_sha256']}`.",
        (f"Numerical execution source SHA-256: `{execution_source}`."),
        f"Analysis source SHA-256: `{analysis_source}`.",
        (
            f"The published execution used {worker_count} "
            "concurrent worker processes. The launcher requested OPENBLAS/OMP/MKL limits "
            f"{environment.get('thread_limits', 'recorded in index.json')}. "
            f"{accelerate_note}"
            "Every record stores dependency versions, hardware/platform data, commit identity, "
            "working-tree numerical hash, scenario/configuration hashes, and timing components."
        ),
        "",
        "## Limitations",
        "",
        (
            "The cost law is illustrative, all costs are static and direction-independent, and "
            "modeled barriers are not observed traversability. The 80 cases cover four designed "
            "families rather than a population of real landscapes. First-order fast marching "
            "has discretization error, and its extracted polyline may fail even when its grid "
            "arrival field reaches the endpoint. Bootstrap intervals describe variation across "
            "the frozen seeds; they do not correct model misspecification. Browser timings are "
            "reported separately in the interface and are not pooled with these native runs. "
            f"{worker_count}-process scheduling and unmeasured library threads may have "
            "contended for "
            "CPU, so wall times and timeout frequencies should not be generalized to other "
            "machines. Supplemental NumPy/SciPy build metadata is stored in index.json."
        ),
        "",
        (
            "The bundled Mount Tamalpais crop is an attributed visualization and transfer "
            "example; it is not part of the frozen 80 synthetic-case protocol."
        ),
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_study_figures(index: Mapping[str, Any], output_dir: str | Path) -> list[str]:
    """Generate standalone descriptive figures from the compact index."""
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    runs = _baseline_runs(index)
    generated: list[str] = []
    finest_local = max(index["protocol"]["local_interior_points"])

    # Feasibility at the finest local discretization.
    local = [
        run
        for run in runs
        if run["config"]["method"] in {"euler_lagrange", "slsqp"}
        and int(run["config"]["interior_points"]) == finest_local
    ]
    labels = index["protocol"]["initializations"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for offset, method in ((-width / 2, "euler_lagrange"), (width / 2, "slsqp")):
        values, counts = [], []
        for initialization in labels:
            group = [
                run
                for run in local
                if run["config"]["method"] == method
                and run["config"]["initialization"] == initialization
            ]
            feasible_count = sum(_cost(run) is not None for run in group)
            values.append(feasible_count / len(group) if group else 0)
            counts.append((feasible_count, len(group)))
        bars = ax.bar(x + offset, values, width, label=method.replace("_", " "))
        for bar, (feasible_count, total) in zip(bars, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{feasible_count}/{total}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    ax.set_xticks(x, [label.replace("_", " ") for label in labels], rotation=20, ha="right")
    ax.set_ylabel("Independently feasible route fraction")
    ax.set_ylim(0, 1.16)
    ax.set_title(
        f"Feasibility at requested {finest_local} interior points\n"
        "labels are feasible/attempted; hard timeouts count as no returned feasible route"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    filename = "feasibility-by-initialization.png"
    fig.savefig(destination / filename, dpi=180)
    plt.close(fig)
    generated.append(filename)

    # Signed reference differences preserve negative values.
    groups, box_labels = [], []
    for method in ("euler_lagrange", "slsqp"):
        for initialization in labels:
            values = [
                difference
                for run in local
                if run["config"]["method"] == method
                and run["config"]["initialization"] == initialization
                and (difference := _resolved_difference(run)) is not None
            ]
            if values:
                groups.append(values)
                box_labels.append(
                    f"{method.split('_')[0]}\n{initialization.replace('_', ' ')}\nn={len(values)}"
                )
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    if groups:
        ax.boxplot(groups, tick_labels=box_labels, showfliers=True)
    else:
        ax.text(
            0.5,
            0.5,
            "No resolved reference comparisons in this profile",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Signed difference from finest-grid reference (%)")
    ax.set_title(
        f"Resolved route-quality distributions at requested {finest_local} interior points"
    )
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    filename = "reference-difference-distributions.png"
    fig.savefig(destination / filename, dpi=180)
    plt.close(fig)
    generated.append(filename)

    # Reference refinement by family.
    coarse, fine = sorted(index["protocol"]["reference_grid_sizes"])[-2:]
    by_case: dict[str, dict[int, float]] = defaultdict(dict)
    family_by_case = {}
    for run in runs:
        if run["config"]["method"] != "fast_marching":
            continue
        value = _cost(run)
        if value is not None:
            by_case[run["case_id"]][int(run["config"]["reference_grid_size"])] = value
            family_by_case[run["case_id"]] = run["family"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    families = index["protocol"]["families"]
    for position, family in enumerate(families):
        values = [
            100 * (levels[fine] - levels[coarse]) / levels[fine]
            for case, levels in by_case.items()
            if family_by_case.get(case) == family and coarse in levels and fine in levels
        ]
        if values:
            jitter = np.linspace(-0.12, 0.12, len(values))
            ax.scatter(np.full(len(values), position) + jitter, values, s=22, alpha=0.75)
    threshold = float(index["protocol"]["reference_refinement_unresolved_percent"])
    ax.axhline(threshold, color="firebrick", linestyle="--", linewidth=0.9)
    ax.axhline(-threshold, color="firebrick", linestyle="--", linewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(
        range(len(families)),
        [family.replace("_", " ") for family in families],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel(f"Signed {coarse}→{fine} cost change (%)")
    ax.set_title("Empirical fast-marching reference refinement")
    fig.tight_layout()
    filename = "reference-refinement.png"
    fig.savefig(destination / filename, dpi=180)
    plt.close(fig)
    generated.append(filename)

    # Contrast curves use evaluated cost even when the sweep's single
    # reference grid cannot pass the two-grid refinement check.
    contrast_groups: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in index["runs"]:
        if not run["case_id"].startswith("contrast-"):
            continue
        cost = _cost(run)
        if cost is None:
            continue
        marker = run["case_id"].rsplit("-c", 1)[1]
        contrast = float(marker.replace("p", "."))
        contrast_groups[(run["family"], run["config"]["method"])][contrast].append(cost)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), sharex=True)
    for ax, family in zip(axes.flat, families, strict=True):
        for method in ("euler_lagrange", "slsqp"):
            values = contrast_groups.get((family, method), {})
            levels = sorted(values)
            if levels:
                ax.plot(
                    levels,
                    [np.median(values[level]) for level in levels],
                    marker="o",
                    label=method.replace("_", " "),
                )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(family.replace("_", " "))
        ax.set_ylabel("Median evaluated route cost (s)")
        ax.set_xlabel("Cost contrast")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[0].legend(handles, legend_labels, frameon=False)
    fig.suptitle("Controlled contrast sweep (one fixed seed per family)")
    fig.tight_layout()
    filename = "contrast-sweep.png"
    fig.savefig(destination / filename, dpi=180)
    plt.close(fig)
    generated.append(filename)

    # EL stationarity and route-quality difference.
    residuals, differences, successes = [], [], []
    for run in runs:
        result = _result(run)
        if run["config"]["method"] != "euler_lagrange" or not result.get("feasible"):
            continue
        residual = result.get("diagnostics", {}).get("stationarity_norm")
        difference = _resolved_difference(run)
        if residual is None or difference is None or float(residual) <= 0:
            continue
        residuals.append(float(residual))
        differences.append(float(difference))
        successes.append(bool(result.get("solver_success")))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if residuals:
        for success, color, label in (
            (True, "#2563eb", "solver success"),
            (False, "#d97706", "other termination"),
        ):
            mask = np.asarray(successes) == success
            ax.scatter(
                np.asarray(residuals)[mask],
                np.asarray(differences)[mask],
                s=20,
                alpha=0.65,
                color=color,
                label=label,
            )
        ax.set_xscale("log")
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5,
            0.5,
            "No resolved residual/quality comparisons in this profile",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Euler–Lagrange stationarity norm")
    ax.set_ylabel("Signed difference from reference (%)")
    ax.set_title("Stationarity and evaluated route quality")
    fig.tight_layout()
    filename = "stationarity-vs-reference-difference.png"
    fig.savefig(destination / filename, dpi=180)
    plt.close(fig)
    generated.append(filename)
    return generated


__all__ = [
    "capture_numerical_build_metadata",
    "generate_study_figures",
    "write_study_report",
]
