"""Independent claim and denominator checks for the generated study report."""

from path_planning_ode import study_reporting as report


def _run(
    run_id: str,
    *,
    method: str = "slsqp",
    feasible: bool = True,
    success: bool = True,
    unresolved: bool = False,
    difference: float | None = 1.0,
    initialization: str = "straight",
    case_id: str = "ridge_pass-s00",
):
    return {
        "run_id": run_id,
        "record_status": "completed",
        "case_id": case_id,
        "family": "ridge_pass",
        "seed": 0,
        "reference_difference_percent": difference,
        "reference_unresolved": unresolved,
        "config": {
            "method": method,
            "initialization": initialization,
            "interior_points": 128,
            "reference_grid_size": 1025,
        },
        "result": {
            "method": method,
            "feasible": feasible,
            "solver_success": success,
            "evaluated_cost_s": 100.0 if feasible else None,
            "termination_reason": "converged" if success else "time_limit",
            "timing_s": {"total": 2.0},
            "diagnostics": {"initialization": {"initial_route_hash": "same"}},
            "evaluation": {
                "minimum_clearance_m": 1.0,
                "length_m": 100.0,
                "violations": [],
            },
        },
    }


def test_unresolved_reference_differences_do_not_enter_claim_tables():
    resolved = _run("resolved", difference=-2.0)
    fallback = _run("fallback", unresolved=True, difference=900.0, case_id="ridge_pass-s01")

    method_row = report._method_rows([resolved, fallback])[0]
    configuration_row = report._configuration_rows([resolved, fallback])[0]

    assert method_row[4] == "-2.000"
    assert configuration_row[6] == "-2.000"


def test_initialization_denominator_includes_cases_with_no_feasible_start():
    feasible = _run("feasible")
    infeasible = _run("infeasible", feasible=False, success=False, case_id="ridge_pass-s01")

    rows = report._initialization_rows([feasible, infeasible])

    assert rows == [["slsqp", "128", "0/2", "—", "—", "—"]]


def test_one_cluster_pair_has_no_bootstrap_interval():
    euler = _run("euler", method="euler_lagrange")
    slsqp = _run("slsqp", method="slsqp")

    row = report._paired_rows([euler, slsqp], samples=100, seed=7)[0]

    assert row[2:5] == ["1", "1", "1/1"]
    assert row[-1] == "insufficient clusters"


def test_reference_summary_requires_both_declared_finest_grids():
    only_513 = _run("513", method="fast_marching")
    only_513["config"]["reference_grid_size"] = 513

    rows, totals = report._reference_rows([only_513], [257, 513, 1025], 0.5)

    assert rows[0][1:5] == ["1", "0", "1", "1"]
    assert totals == {"cases": 1, "resolved": 0, "unresolved": 1, "missing": 1}


def test_termination_table_accounts_for_every_record():
    runs = [
        _run("ok"),
        _run("soft-timeout", success=False),
        {**_run("hard-timeout"), "record_status": "timeout", "result": None},
        {**_run("worker"), "record_status": "worker_error", "result": None},
    ]

    rows = report._termination_rows(runs)

    assert sum(int(row[2]) for row in rows) == len(runs)
