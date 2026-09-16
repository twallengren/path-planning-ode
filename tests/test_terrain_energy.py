import json
from dataclasses import replace

import numpy as np
import pytest

from path_planning_ode import PlannerConfig, plan
from path_planning_ode.terrain import TerrainField, TerrainScenario, evaluate_route
from path_planning_ode.terrain_energy import (
    solve_energy_descent,
    terrain_energy_and_gradient,
)
from path_planning_ode.terrain_generators import obstacle_detour_fixture, uniform_terrain_fixture
from path_planning_ode.terrain_seeds import make_seed


def _smooth_scenario(*, coordinate_factor=1.0, cost_factor=1.0):
    x = np.linspace(0, 100, 17) * coordinate_factor
    y = np.linspace(0, 80, 15) * coordinate_factor
    xx, yy = np.meshgrid(x / coordinate_factor, y / coordinate_factor)
    log_cost = (
        0.18
        + 0.00008 * (xx - 45) ** 2
        + 0.00005 * (yy - 30) ** 2
        + 0.00003 * xx * yy
        + np.log(cost_factor)
    )
    return TerrainScenario(
        name="smooth-energy-test",
        bounds_m=(0, 0, 100 * coordinate_factor, 80 * coordinate_factor),
        start_m=(8 * coordinate_factor, 18 * coordinate_factor),
        goal_m=(92 * coordinate_factor, 67 * coordinate_factor),
        field_x_m=x,
        field_y_m=y,
        elevation_m=np.zeros_like(log_cost),
        log_slowness=log_cost,
    )


def _config(initialization="straight", **changes):
    return PlannerConfig(
        method="energy_descent",
        initialization=initialization,
        interior_points=changes.pop("interior_points", 12),
        tolerance=changes.pop("tolerance", 1e-6),
        max_iterations=changes.pop("max_iterations", 200),
        time_limit_s=changes.pop("time_limit_s", 10),
        profile_samples=changes.pop("profile_samples", 33),
        **changes,
    )


def test_terrain_energy_gradient_matches_fixed_quadrature_finite_difference():
    scenario = _smooth_scenario()
    field = TerrainField(scenario)
    route = np.array([[8, 18], [22, 31], [43, 25], [61, 52], [79, 44], [92, 67]], dtype=float)
    panels = (3, 4, 2, 5, 3)
    _, analytic = terrain_energy_and_gradient(
        route, field, quadrature_order=8, quadrature_panels=panels
    )
    numerical = np.empty_like(route)
    epsilon = 2e-5
    for vertex in range(len(route)):
        for coordinate in range(2):
            plus, minus = route.copy(), route.copy()
            plus[vertex, coordinate] += epsilon
            minus[vertex, coordinate] -= epsilon
            numerical[vertex, coordinate] = (
                terrain_energy_and_gradient(
                    plus, field, quadrature_order=8, quadrature_panels=panels
                )[0]
                - terrain_energy_and_gradient(
                    minus, field, quadrature_order=8, quadrature_panels=panels
                )[0]
            ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-7, atol=5e-6)


def test_dimensionless_objective_and_gradient_are_unit_and_cost_scale_invariant():
    base = _smooth_scenario()
    transformed = _smooth_scenario(coordinate_factor=0.001, cost_factor=1000 * 7)
    base_field, transformed_field = TerrainField(base), TerrainField(transformed)
    route = np.linspace(base.start_m, base.goal_m, 9)
    route[2:-2, 1] += np.array([4, 7, 9, 6, 2])
    transformed_route = route * 0.001
    base_energy, base_gradient = terrain_energy_and_gradient(route, base_field)
    transformed_energy, transformed_gradient = terrain_energy_and_gradient(
        transformed_route, transformed_field
    )

    base_length = np.hypot(100, 80)
    transformed_length = base_length * 0.001
    base_cost = np.exp(np.median(np.asarray(base.log_slowness)))
    transformed_cost = np.exp(np.median(np.asarray(transformed.log_slowness)))
    base_scale = (base_length * base_cost) ** 2
    transformed_scale = (transformed_length * transformed_cost) ** 2
    assert transformed_energy / transformed_scale == pytest.approx(
        base_energy / base_scale, rel=2e-12
    )
    np.testing.assert_allclose(
        transformed_length * transformed_gradient / transformed_scale,
        base_length * base_gradient / base_scale,
        rtol=2e-10,
        atol=2e-11,
    )


def test_uniform_field_is_stationary_and_common_dispatch_is_serializable():
    scenario = uniform_terrain_fixture()
    config = _config(interior_points=16)
    result = plan(scenario, config)
    assert result.solver_success
    assert result.feasible
    assert result.termination_reason == "stationary"
    assert result.diagnostics["iterations"] == 0
    assert result.diagnostics["energy_s2"] == pytest.approx(640000)
    assert result.diagnostics["ode_residual_norm_m"] < 1e-9
    assert result.evaluated_cost_s == pytest.approx(800)
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()


def test_armijo_history_decreases_energy_and_independent_cost_is_separate():
    scenario = _smooth_scenario()
    result = solve_energy_descent(
        scenario,
        _config("arc_left", max_iterations=12, tolerance=1e-12),
    )
    before = np.asarray(result.diagnostics["accepted_baseline_energy_s2"])
    after = np.asarray(result.diagnostics["accepted_energy_s2"])
    assert len(before) == len(after) > 2
    assert np.all(after < before)
    assert result.diagnostics["energy_s2"] != pytest.approx(result.evaluated_cost_s)
    independent = evaluate_route(scenario, result.route_m)
    assert result.evaluated_cost_s == pytest.approx(independent.cost_s)
    np.testing.assert_array_equal(result.route_m[0], scenario.start_m)
    np.testing.assert_array_equal(result.route_m[-1], scenario.goal_m)


def test_hard_barrier_rejects_infeasible_seed_without_softening():
    scenario = obstacle_detour_fixture()
    result = solve_energy_descent(scenario, _config("straight", max_iterations=20))
    assert result.termination_reason == "infeasible_initialization"
    assert not result.solver_success
    assert not result.feasible
    assert result.evaluation.violations == ("barrier_collision",)
    assert result.diagnostics["constraint_violation"] == "barrier_collision"
    expected_seed = make_seed(scenario, "straight", 12)
    assert (
        result.diagnostics["initialization"]["initial_route_hash"]
        == expected_seed.diagnostics["route_hash"]
    )


def test_feasible_barrier_seed_and_every_accepted_route_remain_feasible():
    scenario = obstacle_detour_fixture()
    result = solve_energy_descent(
        scenario,
        _config("barrier", max_iterations=20, options={"clearance_m": 0.1}),
    )
    assert result.diagnostics["constraint_feasible"]
    assert result.feasible
    assert "barrier_collision" not in result.evaluation.violations
    assert result.diagnostics["requested_clearance_m"] == 0.1


def test_timeout_and_invalid_options_are_structured_or_rejected():
    scenario = _smooth_scenario()
    timed = solve_energy_descent(scenario, _config(time_limit_s=np.finfo(float).tiny))
    assert timed.termination_reason == "time_limit"
    assert not timed.solver_success
    assert timed.to_dict()

    with pytest.raises(ValueError, match="step_cap_cells"):
        solve_energy_descent(
            scenario,
            _config(options={"step_cap_cells": 0}),
        )


def test_v2_contract_accepts_new_method_without_changing_existing_methods():
    config = _config()
    restored = PlannerConfig.from_dict(config.to_dict())
    assert restored == config
    for method in ("euler_lagrange", "slsqp", "fast_marching"):
        assert replace(config, method=method).method == method
