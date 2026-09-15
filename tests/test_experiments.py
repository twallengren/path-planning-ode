import json
import shutil
from pathlib import Path

import pytest

import path_planning_ode.experiments as experiments
from path_planning_ode.experiments import (
    annotate_reference_differences,
    build_published_index,
    build_run_specs,
    cli_main,
    frozen_cases,
    load_protocol,
    paired_method_summary,
    publish_results,
    run_experiments,
    summarize_records,
)


def _record(
    *,
    run_id,
    case_id="ridge_pass-s00",
    family="ridge_pass",
    contrast=1.0,
    experiment="baseline",
    method="fast_marching",
    initialization="fast_marching",
    interior=32,
    grid=257,
    cost=100.0,
    feasible=True,
    success=True,
    reason="converged",
    seed_hash=None,
    status="completed",
):
    result = None
    if status == "completed":
        diagnostics = {}
        if method != "fast_marching":
            diagnostics = {
                "initialization": {"initial_route_hash": seed_hash},
                "stationarity_norm": 1e-8,
            }
        result = {
            "method": method,
            "initialization": initialization,
            "evaluated_cost_s": cost if feasible else None,
            "feasible": feasible,
            "solver_success": success,
            "termination_reason": reason,
            "diagnostics": diagnostics,
            "timing_s": {"total": 0.1},
            "evaluation": {
                "minimum_clearance_m": 2.0 if feasible else None,
            },
        }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "profile": "full",
        "case": {
            "case_id": case_id,
            "family": family,
            "seed": 0,
            "contrast": contrast,
            "barriers": True,
            "difficulty": "easy",
            "scenario_ref": f"synthetic/{family}/0",
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
        "record_status": status,
        "child_wall_time_s": 0.2,
        "environment": {"working_tree_source_sha256": "c" * 64},
        "result": result,
    }


def test_frozen_protocol_counts_and_difficulty_are_predeclared():
    protocol = load_protocol()
    cases = frozen_cases(protocol)
    baseline = [case for case in cases if case.experiment == "baseline"]
    assert len(baseline) == 80
    assert len(cases) == 96
    assert {case.difficulty for case in baseline} == {"easy", "medium", "hard"}
    assert len(build_run_specs(protocol, "full")) == 2816
    assert len(build_run_specs(protocol, "smoke")) == 20
    assert len(build_run_specs(protocol, "representative")) == 32


def test_failed_declared_finest_uses_labelled_fallback_and_remains_unresolved():
    records = [
        _record(run_id="257", grid=257, cost=100),
        _record(run_id="513", grid=513, cost=99),
        _record(run_id="1025", grid=1025, status="timeout"),
        _record(
            run_id="local",
            method="euler_lagrange",
            initialization="straight",
            cost=98,
            seed_hash="seed",
        ),
    ]
    annotated = annotate_reference_differences(
        records, expected_grids={"ridge_pass-s00": [257, 513, 1025]}
    )
    local = next(record for record in annotated if record["run_id"] == "local")
    assert local["reference_grid_size"] == 513
    assert local["reference_basis"] == "fallback_finest_available"
    assert local["comparison_basis"] == "fallback_finest_available"
    assert local["reference_unresolved_reason"] == "missing_declared_finest"
    assert local["reference_unresolved"]
    assert local["reference_difference_percent"] == pytest.approx(100 * (98 - 99) / 99)


def test_sweep_summaries_do_not_pool_families_or_contrasts():
    records = []
    for family in ("ridge_pass", "dead_ends"):
        for contrast in (0.0, 2.0):
            record = _record(
                run_id=f"{family}-{contrast}",
                family=family,
                contrast=contrast,
                experiment="contrast_sweep",
            )
            record["reference_difference_percent"] = 0.0
            record["reference_unresolved"] = True
            records.append(record)
    rows = summarize_records(records, bootstrap_samples=10)
    assert len(rows) == 4
    assert {(row["family"], row["contrast"]) for row in rows} == {
        ("ridge_pass", 0.0),
        ("ridge_pass", 2.0),
        ("dead_ends", 0.0),
        ("dead_ends", 2.0),
    }


def test_paired_summary_requires_matching_seed_hash_and_reports_denominator():
    records = []
    for case_id, hashes in (("matched", ("seed", "seed")), ("mismatch", ("a", "b"))):
        for method, seed_hash, cost in zip(
            ("euler_lagrange", "slsqp"), hashes, (105.0, 100.0), strict=True
        ):
            record = _record(
                run_id=f"{case_id}-{method}",
                case_id=case_id,
                method=method,
                initialization="straight",
                cost=cost,
                seed_hash=seed_hash,
            )
            record["reference_difference_percent"] = cost - 100
            record["reference_unresolved"] = False
            records.append(record)
    rows = paired_method_summary(records, bootstrap_samples=10, bootstrap_seed=1)
    assert len(rows) == 1
    assert rows[0]["paired_cases"] == 2
    assert rows[0]["matched_seed_pairs"] == 1
    assert rows[0]["paired_feasible_cases"] == 1
    assert rows[0]["bootstrap_95_low_percent"] is None


def test_partial_full_records_are_never_labelled_complete():
    protocol = load_protocol()
    spec = build_run_specs(protocol, "full")[0]
    record = _record(run_id=spec.run_id, family=spec.case.family, case_id=spec.case.case_id)
    index = build_published_index([record], protocol, profile="full", bootstrap_samples=10)
    assert not index["scope"]["full_protocol"]
    assert index["scope"]["expected_run_count"] == 2816
    assert index["scope"]["missing_run_count"] == 2815


def test_resume_rejects_changed_numerical_source(tmp_path):
    protocol = load_protocol()
    spec = build_run_specs(protocol, "smoke")[0]
    record = _record(run_id=spec.run_id)
    record.update(
        {
            "profile": "smoke",
            "protocol_sha256": protocol["protocol_sha256"],
            "config_hash": spec.config.config_hash,
            "environment": {"working_tree_source_sha256": "old"},
        }
    )
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        run_experiments(
            [spec],
            path,
            resume=True,
            environment={"working_tree_source_sha256": "new"},
        )


@pytest.mark.parametrize("workers", ["0", "9", "not-a-number"])
def test_cli_rejects_invalid_worker_counts(workers):
    with pytest.raises(SystemExit):
        cli_main(["--workers", workers, "--limit", "1"])


def test_cli_default_uses_profile_scoped_user_results(tmp_path, monkeypatch):
    captured = {}

    def fake_run(specs, output_path, **kwargs):
        captured["output"] = Path(output_path)
        return []

    def fake_publish(records, protocol, output_dir, **kwargs):
        captured["published"] = Path(output_dir)
        return {"scope": {"run_count": 0}}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiments, "run_experiments", fake_run)
    monkeypatch.setattr(experiments, "publish_results", fake_publish)

    assert cli_main(["--profile", "smoke", "--limit", "1", "--no-figures"]) == 0
    expected = tmp_path / "terrain-study-results" / "smoke"
    assert captured == {"output": expected / "runs.jsonl.gz", "published": expected}


def test_custom_publication_is_deterministic_and_does_not_replace_canonical_report(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    canonical = repository / "experiments" / "published"
    docs_report = repository / "docs" / "study-report.md"
    docs_report.parent.mkdir(parents=True)
    docs_report.write_text("canonical-full-report\n", encoding="utf-8")
    monkeypatch.setattr(experiments, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(experiments, "DEFAULT_PUBLISHED_DIR", canonical)
    monkeypatch.setattr(experiments, "_REPOSITORY_CHECKOUT", True)

    output = tmp_path / "custom-smoke"
    record = _record(run_id="deterministic-detail")
    record["profile"] = "smoke"
    publish_results(
        [record],
        load_protocol(),
        output,
        profile="smoke",
        bootstrap_samples=10,
        figures=False,
    )
    detail = output / "runs" / "deterministic-detail.json.gz"
    first = detail.read_bytes()
    publish_results(
        [record],
        load_protocol(),
        output,
        profile="smoke",
        bootstrap_samples=10,
        figures=False,
    )

    assert detail.read_bytes() == first
    assert int.from_bytes(first[4:8], "little") == 0
    output_report = (output / "study-report.md").read_text(encoding="utf-8")
    assert "[Authoritative run archive](runs.jsonl.gz)" in output_report
    assert "../experiments/published" not in output_report
    assert "github.com/twallengren/path-planning-ode/blob/master/docs" in output_report
    assert docs_report.read_text(encoding="utf-8") == "canonical-full-report\n"


def test_installed_source_hashes_read_package_and_protocol_bytes(tmp_path, monkeypatch):
    package = tmp_path / "path_planning_ode"
    package.mkdir()
    for name in (
        "terrain.py",
        "terrain_generators.py",
        "terrain_seeds.py",
        "fast_marching.py",
        "local_planners.py",
        "planners.py",
        "experiments.py",
        "study_reporting.py",
    ):
        shutil.copy2(Path(experiments.__file__).parent / name, package / name)
    (package / "data").mkdir()
    shutil.copy2(Path("experiments/protocol.json"), package / "data" / "protocol.json")
    monkeypatch.setattr(experiments, "PACKAGE_ROOT", package)
    monkeypatch.setattr(experiments, "REPOSITORY_ROOT", tmp_path / "not-a-checkout")
    monkeypatch.setattr(experiments, "_REPOSITORY_CHECKOUT", False)

    execution = experiments._execution_source_hash()
    analysis = experiments._analysis_source_hash()
    (package / "terrain.py").write_bytes((package / "terrain.py").read_bytes() + b"\n# changed\n")
    assert experiments._execution_source_hash() != execution

    (package / "terrain.py").write_bytes(
        (Path(experiments.__file__).parent / "terrain.py").read_bytes()
    )
    (package / "data" / "protocol.json").write_bytes(
        (package / "data" / "protocol.json").read_bytes() + b"\n"
    )
    assert experiments._execution_source_hash() != execution
    assert experiments._analysis_source_hash() != analysis
    assert analysis != experiments.hashlib.sha256().hexdigest()
