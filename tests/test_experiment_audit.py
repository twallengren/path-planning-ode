"""Independent checks for benchmark design, aggregation, and interpretation."""

import json
from dataclasses import replace

import pytest

import path_planning_ode.experiments as experiments
from path_planning_ode.experiments import (
    build_published_index,
    build_run_specs,
    execute_run,
    initialization_sensitivity_summary,
    load_protocol,
    local_refinement_summary,
    paired_method_summary,
    run_experiments,
    summarize_records,
)


def _result(*, cost, feasible=True, success=True, reason="converged", seed_hash=None):
    return {
        "evaluated_cost_s": cost,
        "feasible": feasible,
        "solver_success": success,
        "termination_reason": reason,
        "diagnostics": {
            "initialization": {"initial_route_hash": seed_hash},
            "stationarity_norm": 1e-8,
        },
        "timing_s": {"total": 0.1},
        "evaluation": {"minimum_clearance_m": 3.0},
    }


def _record(
    run_id,
    *,
    case_id="ridge_pass-s00",
    family="ridge_pass",
    seed=0,
    experiment="baseline",
    contrast=1.0,
    method="slsqp",
    initialization="straight",
    interior=64,
    grid=513,
    cost=100.0,
    feasible=True,
    success=True,
    reason="converged",
    seed_hash="shared-seed",
    profile="full",
    protocol_sha256="protocol",
    source_hash="source",
):
    return {
        "schema_version": 1,
        "run_id": run_id,
        "profile": profile,
        "case": {
            "case_id": case_id,
            "family": family,
            "seed": seed,
            "contrast": contrast,
            "barriers": True,
            "difficulty": ("easy", "medium", "hard")[seed % 3],
            "scenario_ref": f"synthetic/{family}/{seed}",
            "scenario_hash": "a" * 64,
            "experiment": experiment,
        },
        "config": {
            "method": method,
            "initialization": initialization,
            "interior_points": interior,
            "reference_grid_size": grid,
        },
        "config_hash": "b" * 64,
        "protocol_sha256": protocol_sha256,
        "record_status": "completed",
        "child_wall_time_s": 0.2,
        "environment": {"working_tree_source_sha256": source_hash},
        "result": _result(
            cost=cost,
            feasible=feasible,
            success=success,
            reason=reason,
            seed_hash=seed_hash,
        ),
    }


def _record_from_spec(spec, *, cost):
    record = _record(
        spec.run_id,
        case_id=spec.case.case_id,
        family=spec.case.family,
        seed=spec.case.seed,
        experiment=spec.case.experiment,
        contrast=spec.case.contrast,
        method=spec.config.method,
        initialization=spec.config.initialization,
        interior=spec.config.interior_points,
        grid=spec.config.reference_grid_size,
        cost=cost,
        profile=spec.profile,
        protocol_sha256=spec.protocol_sha256,
    )
    record["case"] = spec.case.to_dict()
    record["config"] = spec.config.to_dict()
    record["config_hash"] = spec.config.config_hash
    return record


def test_full_matrix_factorization_and_fixed_scenario_sources():
    protocol = load_protocol()
    specs = build_run_specs(protocol, "full")
    baseline = [spec for spec in specs if spec.case.experiment == "baseline"]
    sweep = [spec for spec in specs if spec.case.experiment == "contrast_sweep"]
    assert len(baseline) == 80 * 33
    assert len(sweep) == 16 * 11
    assert len(specs) == 2816

    by_case = {}
    for spec in specs:
        by_case.setdefault(spec.case.case_id, []).append(spec)
    assert {len(group) for group in by_case.values() if group[0].case.experiment == "baseline"} == {
        33
    }
    sweep_sizes = {
        len(group) for group in by_case.values() if group[0].case.experiment == "contrast_sweep"
    }
    assert sweep_sizes == {11}
    assert all(len({spec.case.scenario_hash for spec in group}) == 1 for group in by_case.values())
    assert protocol["source_field_resolution"] == 65


def test_partial_full_subset_cannot_resolve_without_declared_1025_reference():
    protocol = load_protocol()
    full = build_run_specs(protocol, "full")
    case_id = full[0].case.case_id
    subset = [
        spec
        for spec in full
        if spec.case.case_id == case_id
        and spec.config.method == "fast_marching"
        and spec.config.reference_grid_size in {257, 513}
    ]
    records = [
        _record_from_spec(spec, cost=100.0 - index * 0.1) for index, spec in enumerate(subset)
    ]
    index = build_published_index(
        records,
        protocol,
        profile="full",
        bootstrap_samples=20,
        expected_specs=subset,
    )
    assert not index["scope"]["full_protocol"]
    assert index["scope"]["statement"].startswith("Incomplete full profile")
    assert all(run["reference_unresolved"] for run in index["runs"])
    assert all(run["reference_basis"] == "fallback_finest_available" for run in index["runs"])


def test_resume_rejects_an_existing_config_outside_requested_specs(tmp_path, monkeypatch):
    protocol = load_protocol()
    requested, incompatible = build_run_specs(protocol, "smoke")[:2]
    environment = {"working_tree_source_sha256": "same-source"}
    old = _record_from_spec(incompatible, cost=100.0)
    old["environment"] = environment
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(old) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        experiments,
        "execute_run",
        lambda *_: pytest.fail("incompatible resume should fail before executing a planner"),
    )
    with pytest.raises(ValueError, match="configuration|run ID|run_id"):
        run_experiments([requested], path, resume=True, workers=1, environment=environment)


def test_cluster_intervals_require_two_cases_and_paired_stats_use_shared_cases_only():
    one = _record("one", seed=0, cost=110.0)
    one["reference_difference_percent"] = 10.0
    one["reference_unresolved"] = False
    row = summarize_records([one], bootstrap_samples=200, bootstrap_seed=7)[0]
    assert row["mean_reference_difference_percent"] == 10.0
    assert row["bootstrap_95_low_percent"] is None
    assert row["bootstrap_95_high_percent"] is None

    records = []
    for seed, euler_cost, slsqp_cost in ((0, 100.0, 90.0), (1, 200.0, 220.0)):
        for method, cost in (("euler_lagrange", euler_cost), ("slsqp", slsqp_cost)):
            record = _record(
                f"{seed}-{method}",
                case_id=f"ridge_pass-s{seed:02d}",
                seed=seed,
                method=method,
                cost=cost,
            )
            record["reference_difference_percent"] = 0.0
            record["reference_unresolved"] = False
            records.append(record)
    # An unpaired case must not enter the paired denominator or mean.
    unpaired = _record("unpaired", case_id="ridge_pass-s02", seed=2, method="slsqp", cost=1.0)
    unpaired["reference_difference_percent"] = 0.0
    unpaired["reference_unresolved"] = False
    records.append(unpaired)

    paired = paired_method_summary(records, bootstrap_samples=500, bootstrap_seed=11)[0]
    assert paired["paired_cases"] == 2
    assert paired["matched_seed_pairs"] == 2
    assert paired["paired_feasible_cases"] == 2
    assert paired["mean_slsqp_minus_euler_percent"] == pytest.approx(0.0)
    assert paired["bootstrap_95_low_percent"] == pytest.approx(-10.0)
    assert paired["bootstrap_95_high_percent"] == pytest.approx(10.0)


def test_feasible_nonconverged_candidates_remain_visible_without_counting_as_solver_success():
    record = _record(
        "failed-feasible",
        cost=123.0,
        feasible=True,
        success=False,
        reason="optimization_failed",
    )
    record["reference_difference_percent"] = 2.0
    record["reference_unresolved"] = False
    row = summarize_records([record], bootstrap_samples=20)[0]
    assert row["feasible_runs"] == 1
    assert row["feasibility_rate"] == 1.0
    assert row["solver_success_rate"] == 0.0
    assert row["median_evaluated_cost_s"] == 123.0
    assert row["failure_modes"] == "optimization_failed:1"


def test_initialization_sensitivity_uses_within_case_percent_range():
    records = []
    for initialization, cost in (("straight", 100.0), ("arc_left", 110.0), ("arc_right", 105.0)):
        records.append(_record(initialization, initialization=initialization, cost=cost))
    row = initialization_sensitivity_summary(records)[0]
    assert row["cases_with_multiple_feasible_initializations"] == 1
    assert row["median_within_case_cost_range_percent"] == pytest.approx(10.0)
    assert row["maximum_within_case_cost_range_percent"] == pytest.approx(10.0)


def test_local_refinement_never_substitutes_a_coarser_available_level():
    records = []
    for seed in (0, 1):
        for interior, cost in ((32, 50.0), (64, 100.0)):
            records.append(
                _record(
                    f"{seed}-{interior}",
                    case_id=f"ridge_pass-s{seed:02d}",
                    seed=seed,
                    interior=interior,
                    cost=cost,
                )
            )
    records.append(_record("1-128", case_id="ridge_pass-s01", seed=1, interior=128, cost=99.0))

    row = local_refinement_summary(records, declared_levels=(32, 64, 128))[0]
    expected_change = 100 * (99.0 - 100.0) / 99.0
    assert row["declared_coarse_interior_points"] == 64
    assert row["declared_fine_interior_points"] == 128
    assert row["cases"] == 2
    assert row["cases_with_two_feasible_finest_levels"] == 1
    assert row["unresolved_cases"] == 1
    assert row["median_signed_change_percent"] == pytest.approx(expected_change)
    assert row["median_absolute_change_percent"] == pytest.approx(abs(expected_change))
    assert row["over_0p5_percent"] == 1


def test_subprocess_hard_timeout_covers_the_entire_child_run():
    protocol = load_protocol()
    original = build_run_specs(protocol, "smoke")[0]
    capped = replace(original, config=replace(original.config, time_limit_s=0.01))
    record = execute_run(capped, {"working_tree_source_sha256": "audit"})
    assert record["record_status"] == "timeout"
    assert record["result"] is None
    assert record["error"]["type"] == "HardTimeout"
    assert record["child_wall_time_s"] >= capped.config.time_limit_s
