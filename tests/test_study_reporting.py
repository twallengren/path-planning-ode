from pathlib import Path

from path_planning_ode.study_reporting import generate_study_figures, write_study_report


def _run(
    run_id: str,
    case_id: str,
    method: str,
    *,
    initialization: str,
    resolution: int,
    cost: float | None,
    difference: float | None,
    solver_success: bool = True,
    status: str = "completed",
) -> dict:
    result = None
    if status == "completed":
        result = {
            "method": method,
            "initialization": initialization,
            "evaluated_cost_s": cost,
            "feasible": cost is not None,
            "solver_success": solver_success,
            "termination_reason": "converged" if solver_success else "stagnated",
            "diagnostics": {
                "initialization": {"initial_route_hash": f"seed-{case_id}-{initialization}"},
                "stationarity_norm": 1e-7,
            },
            "timing_s": {"total": 0.5},
            "evaluation": {
                "minimum_clearance_m": 4.0,
                "feasible": cost is not None,
            },
        }
    config = {
        "method": method,
        "initialization": initialization,
        "interior_points": resolution if method != "fast_marching" else 32,
        "reference_grid_size": resolution if method == "fast_marching" else 513,
        "options": {},
    }
    return {
        "run_id": run_id,
        "record_status": status,
        "case_id": case_id,
        "family": "ridge_pass",
        "seed": int(case_id[-2:]),
        "config": config,
        "result": result,
        "result_ref": None if result is None else f"runs/{run_id}.json.gz",
        "reference_difference_percent": difference,
        "reference_unresolved": False,
    }


def test_data_driven_report_includes_denominators_failures_and_provenance(tmp_path: Path):
    runs = []
    for seed in (0, 1):
        case_id = f"ridge_pass-s{seed:02d}"
        runs.extend(
            [
                _run(
                    f"ref-coarse-{seed}",
                    case_id,
                    "fast_marching",
                    initialization="fast_marching",
                    resolution=513,
                    cost=100.0,
                    difference=-0.2,
                ),
                _run(
                    f"ref-fine-{seed}",
                    case_id,
                    "fast_marching",
                    initialization="fast_marching",
                    resolution=1025,
                    cost=100.2,
                    difference=0.0,
                ),
                _run(
                    f"el-{seed}",
                    case_id,
                    "euler_lagrange",
                    initialization="straight",
                    resolution=128,
                    cost=101.0,
                    difference=0.8,
                ),
                _run(
                    f"slsqp-{seed}",
                    case_id,
                    "slsqp",
                    initialization="straight",
                    resolution=128,
                    cost=99.0,
                    difference=-1.2,
                    solver_success=seed == 0,
                ),
            ]
        )
    runs.append(
        _run(
            "timeout-run",
            "ridge_pass-s02",
            "slsqp",
            initialization="arc_left",
            resolution=128,
            cost=None,
            difference=None,
            status="timeout",
        )
    )
    protocol = {
        "protocol_sha256": "protocol-hash",
        "model_boundary": "Static illustrative model.",
        "baseline": {"case_count": 80},
        "difficulty_rule": "seed % 3",
        "local_methods": ["euler_lagrange", "slsqp"],
        "initializations": ["straight", "arc_left"],
        "local_interior_points": [32, 64, 128],
        "reference_grid_sizes": [257, 513, 1025],
        "contrast_sweep": {"values": [0, 1, 2, 4]},
        "planner_time_limit_s": 60.0,
        "reference_refinement_unresolved_percent": 0.5,
        "bootstrap": {"samples": 20, "cluster": "family and seed", "seed": 7},
        "families": ["ridge_pass"],
    }
    index = {
        "title": "Terrain study",
        "protocol": protocol,
        "scope": {"statement": "Incomplete full profile."},
        "runs": runs,
        "local_refinement": [],
        "residual_cost_examples": [],
        "environments": [
            {
                "benchmark_workers": 8,
                "thread_limits": {"OPENBLAS_NUM_THREADS": "1"},
                "numerical_execution_source_sha256": "execution-hash",
            }
        ],
        "publication": {"analysis_source_sha256": "analysis-hash"},
    }

    output = tmp_path / "study-report.md"
    write_study_report(index, output)
    report = output.read_text(encoding="utf-8")

    assert "## Paired local-method comparison" in report
    assert "Both feasible / matched" in report
    assert "timeout" in report
    assert "Negative values are retained" not in report  # sentence uses lowercase in prose
    assert "negative values are retained" in report
    assert "execution-hash" in report
    assert "analysis-hash" in report
    assert "Authoritative run archive" in report


def test_empty_resolved_quality_data_still_generates_all_figures(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    run = _run(
        "unresolved",
        "ridge_pass-s00",
        "euler_lagrange",
        initialization="straight",
        resolution=8,
        cost=100.0,
        difference=900.0,
    )
    run["reference_unresolved"] = True
    index = {
        "protocol": {
            "local_interior_points": [8],
            "initializations": ["straight"],
            "reference_grid_sizes": [17, 33, 65],
            "reference_refinement_unresolved_percent": 0.5,
            "families": [
                "ridge_pass",
                "competing_corridors",
                "dead_ends",
                "correlated_roughness",
            ],
        },
        "runs": [run],
    }

    generated = generate_study_figures(index, tmp_path)

    assert len(generated) == 5
    assert all((tmp_path / name).stat().st_size > 0 for name in generated)
