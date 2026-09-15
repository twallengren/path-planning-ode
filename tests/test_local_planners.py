from dataclasses import replace

import numpy as np
import pytest

from path_planning_ode.core import Scene
from path_planning_ode.local_planners import (
    euler_lagrange_jacobian,
    euler_lagrange_residual,
    solve_euler_lagrange,
    solve_slsqp,
)
from path_planning_ode.planners import plan, scene_to_terrain_scenario
from path_planning_ode.terrain import PlannerConfig, TerrainField, evaluate_route
from path_planning_ode.terrain_generators import (
    obstacle_detour_fixture,
    ridge_pass_terrain,
    symmetry_fixture,
    uniform_terrain_fixture,
)
from path_planning_ode.terrain_seeds import make_seed


def _config(method, initialization="straight", **changes):
    return PlannerConfig(
        method=method,
        initialization=initialization,
        interior_points=changes.pop("interior_points", 16),
        tolerance=changes.pop("tolerance", 1e-6),
        max_iterations=changes.pop("max_iterations", 100),
        time_limit_s=changes.pop("time_limit_s", 10),
        **changes,
    )


@pytest.mark.parametrize(
    ("method", "solver"),
    [("euler_lagrange", solve_euler_lagrange), ("slsqp", solve_slsqp)],
)
def test_uniform_field_recovers_analytic_straight_route(method, solver):
    scenario = uniform_terrain_fixture()
    config = _config(method)
    result = solver(scenario, config)
    expected = np.linspace(scenario.start_m, scenario.goal_m, config.interior_points + 2)
    assert result.solver_success
    assert result.feasible
    np.testing.assert_allclose(result.route_m, expected, atol=1e-8)
    assert result.evaluated_cost_s == pytest.approx(
        0.8 * np.linalg.norm(np.asarray(scenario.goal_m) - scenario.start_m)
    )


def test_euler_lagrange_stationarity_does_not_override_barrier_infeasibility():
    scenario = obstacle_detour_fixture()
    result = solve_euler_lagrange(scenario, _config("euler_lagrange"))
    assert result.solver_success
    assert not result.feasible
    assert result.termination_reason == "stationary"
    assert result.evaluation.violations == ("barrier_collision",)


def test_slsqp_barrier_route_is_feasible_and_independently_checked():
    scenario = obstacle_detour_fixture()
    result = solve_slsqp(scenario, _config("slsqp", "barrier"))
    assert result.solver_success
    assert result.feasible
    assert result.diagnostics["constraint_violation"] <= 1e-5
    assert result.diagnostics["stationarity_norm"] <= result.diagnostics["stationarity_tolerance"]
    lengths = np.linalg.norm(np.diff(result.route_m, axis=0), axis=1)
    assert np.max(lengths) - np.min(lengths) <= 2e-5


def test_local_methods_receive_identical_shared_seeds():
    scenario = obstacle_detour_fixture()
    first = make_seed(scenario, "barrier", 16)
    second = make_seed(scenario, "barrier", 16)
    assert first.success and second.success
    assert first.diagnostics["route_hash"] == second.diagnostics["route_hash"]
    np.testing.assert_array_equal(first.route_m, second.route_m)
    assert evaluate_route(scenario, first.route_m).feasible
    euler = solve_euler_lagrange(scenario, _config("euler_lagrange", "barrier"))
    slsqp = solve_slsqp(scenario, _config("slsqp", "barrier"))
    assert (
        euler.diagnostics["initialization"]["initial_route_hash"]
        == slsqp.diagnostics["initialization"]["initial_route_hash"]
    )


def test_euler_lagrange_sparse_jacobian_matches_finite_difference():
    scenario = symmetry_fixture()
    field = TerrainField(scenario)
    path = make_seed(scenario, "arc_left", 4).route_m
    analytic = euler_lagrange_jacobian(path, field).toarray()
    numerical = np.empty_like(analytic)
    epsilon = 1e-5
    for index in range(analytic.shape[1]):
        plus, minus = path.copy(), path.copy()
        plus[1 + index // 2, index % 2] += epsilon
        minus[1 + index // 2, index % 2] -= epsilon
        numerical[:, index] = (
            euler_lagrange_residual(plus, field) - euler_lagrange_residual(minus, field)
        ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-5, atol=2e-5)


def test_initialization_failure_and_timeout_are_serializable_results():
    scenario = uniform_terrain_fixture()
    unknown = solve_slsqp(scenario, _config("slsqp", "does_not_exist"))
    assert not unknown.solver_success
    assert unknown.route_m is None
    assert unknown.termination_reason == "unknown_initialization"
    assert unknown.to_dict()

    timed = solve_euler_lagrange(
        scenario, _config("euler_lagrange", time_limit_s=np.finfo(float).tiny)
    )
    assert timed.termination_reason == "time_limit"
    assert not timed.solver_success
    assert timed.to_dict()


def test_common_plan_and_v1_adapter_record_approximation_provenance():
    scenario = uniform_terrain_fixture()
    config = _config("euler_lagrange")
    assert plan(scenario, config).solver_success

    adapted = scene_to_terrain_scenario(Scene(), samples=17)
    assert adapted.provenance["kind"] == "v1_scene_adapter"
    assert adapted.provenance["samples"] == [17, 17]
    adapted_result = plan(Scene(), replace(config, interior_points=8))
    assert adapted_result.scenario_hash != scenario.scenario_hash


def test_slsqp_domain_boundary_termination_returns_evaluated_result():
    """Regression for the frozen ridge-pass baseline's boundary-roundoff failure."""
    scenario = ridge_pass_terrain(seed=0, contrast=1.0, barriers=True)
    config = PlannerConfig(
        method="slsqp",
        initialization="arc_left",
        interior_points=32,
        reference_grid_size=513,
        tolerance=1e-6,
        max_iterations=200,
        time_limit_s=60,
        profile_samples=65,
        options={"quadrature_order": 8},
    )
    result = plan(scenario, config)
    assert result.evaluation is not None
    assert result.evaluated_cost_s is not None
    assert "evaluation_error" not in result.diagnostics
    assert result.to_dict()


def test_evaluation_numerical_failure_is_a_structured_result(monkeypatch):
    import path_planning_ode.local_planners as local_planners

    def fail_evaluation(*args, **kwargs):
        raise ValueError("synthetic evaluator failure")

    monkeypatch.setattr(local_planners, "evaluate_route", fail_evaluation)
    scenario = uniform_terrain_fixture()
    result = solve_euler_lagrange(scenario, _config("euler_lagrange"))
    assert result.termination_reason == "evaluation_failed"
    assert not result.solver_success
    assert result.route_m is not None
    assert result.evaluation is None
    assert result.diagnostics["evaluation_error"]["exception_type"] == "ValueError"
    assert result.to_dict()
