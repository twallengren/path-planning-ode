"""Independent numerical audit for terrain energy descent."""

from __future__ import annotations

from math import log

import numpy as np
import pytest
from scipy.integrate import quad

from path_planning_ode.local_planners import solve_euler_lagrange
from path_planning_ode.soft_walls import soften_walls
from path_planning_ode.terrain import (
    PlannerConfig,
    TerrainField,
    TerrainScenario,
    evaluate_route,
)
from path_planning_ode.terrain_energy import (
    solve_energy_descent,
    terrain_energy_and_gradient,
)
from path_planning_ode.terrain_generators import (
    obstacle_detour_fixture,
    ridge_pass_terrain,
    symmetry_fixture,
    uniform_terrain_fixture,
)


def _adaptive_energy(route, field):
    """Integrate c²|q'|² independently, splitting at every source-grid knot."""
    route = np.asarray(route, dtype=float)
    intervals = len(route) - 1
    total = 0.0
    x_knots = np.asarray(field.scenario.field_x_m[1:-1])
    y_knots = np.asarray(field.scenario.field_y_m[1:-1])
    for start, end in zip(route[:-1], route[1:], strict=True):
        delta = end - start
        breaks = []
        if delta[0] != 0:
            breaks.extend((x_knots - start[0]) / delta[0])
        if delta[1] != 0:
            breaks.extend((y_knots - start[1]) / delta[1])
        points = sorted({float(value) for value in breaks if 0 < value < 1})

        def integrand(parameter):
            cost = float(field.cost(start + parameter * delta))
            return cost * cost

        integral = quad(
            integrand,
            0.0,
            1.0,
            points=points,
            epsabs=1e-10,
            epsrel=1e-10,
            limit=max(100, 2 * len(points)),
        )[0]
        total += intervals * float(delta @ delta) * integral
    return total


def _narrow_scenario():
    axis = np.linspace(0.0, 10.0, 65)
    x, y = np.meshgrid(axis, axis)
    cost = 1.0 + 18.0 * np.exp(-((x - 4.137) ** 2 + (y - 5.03) ** 2) / 0.09**2)
    return TerrainScenario(
        name="narrow interpolated field",
        bounds_m=(0.0, 0.0, 10.0, 10.0),
        start_m=(0.0, 5.0),
        goal_m=(10.0, 8.0),
        field_x_m=axis,
        field_y_m=axis,
        elevation_m=np.zeros_like(cost),
        log_slowness=np.log(cost),
    )


def _config(initialization="arc_left", **changes):
    return PlannerConfig(
        method="energy_descent",
        initialization=initialization,
        interior_points=changes.pop("interior_points", 16),
        tolerance=changes.pop("tolerance", 1e-6),
        max_iterations=changes.pop("max_iterations", 1000),
        time_limit_s=changes.pop("time_limit_s", 8.0),
        profile_samples=changes.pop("profile_samples", 33),
        options=changes.pop("options", {}),
        **changes,
    )


def _scaled_scenario(scenario, coordinate_factor=1.0, slowness_factor=1.0):
    values = scenario.to_dict()
    values["name"] = f"scaled {scenario.name}"
    values["bounds_m"] = [coordinate_factor * value for value in scenario.bounds_m]
    values["start_m"] = [coordinate_factor * value for value in scenario.start_m]
    values["goal_m"] = [coordinate_factor * value for value in scenario.goal_m]
    values["field_x_m"] = [coordinate_factor * value for value in scenario.field_x_m]
    values["field_y_m"] = [coordinate_factor * value for value in scenario.field_y_m]
    values["log_slowness"] = (np.asarray(scenario.log_slowness) + log(slowness_factor)).tolist()
    return TerrainScenario.from_dict(values)


def test_physical_energy_resolves_narrow_field_and_matches_adaptive_oracle():
    scenario = _narrow_scenario()
    field = TerrainField(scenario)
    route = np.array([[0.0, 5.0], [9.65, 4.92], [10.0, 8.0]])
    energy, _ = terrain_energy_and_gradient(route, field)
    assert energy == pytest.approx(_adaptive_energy(route, field), rel=2e-6)


def test_physical_vertex_gradient_matches_centered_finite_differences():
    scenario = _narrow_scenario()
    field = TerrainField(scenario)
    route = np.array([[0.5, 2.0], [2.3, 5.2], [5.1, 4.7], [8.0, 7.5]])
    panels = (15, 17, 16)
    _, analytic = terrain_energy_and_gradient(route, field, quadrature_panels=panels)
    numerical = np.empty_like(route)
    epsilon = 2e-6
    for vertex in range(len(route)):
        for axis in range(2):
            plus, minus = route.copy(), route.copy()
            plus[vertex, axis] += epsilon
            minus[vertex, axis] -= epsilon
            plus_energy = terrain_energy_and_gradient(plus, field, quadrature_panels=panels)[0]
            minus_energy = terrain_energy_and_gradient(minus, field, quadrature_panels=panels)[0]
            numerical[vertex, axis] = (plus_energy - minus_energy) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=3e-6, atol=3e-5)


def test_energy_and_dimensionless_diagnostics_are_unit_consistent():
    base = uniform_terrain_fixture()
    converted = _scaled_scenario(base, coordinate_factor=1000.0, slowness_factor=0.001)
    strengthened = _scaled_scenario(base, slowness_factor=7.0)
    base_route = np.linspace(base.start_m, base.goal_m, 10)
    converted_route = 1000.0 * base_route
    base_energy, base_gradient = terrain_energy_and_gradient(base_route, TerrainField(base))
    converted_energy, converted_gradient = terrain_energy_and_gradient(
        converted_route, TerrainField(converted)
    )
    strong_energy, strong_gradient = terrain_energy_and_gradient(
        base_route, TerrainField(strengthened)
    )
    assert converted_energy == pytest.approx(base_energy, rel=2e-13)
    np.testing.assert_allclose(converted_gradient, base_gradient / 1000.0, rtol=2e-12, atol=1e-14)
    assert strong_energy == pytest.approx(49.0 * base_energy, rel=2e-13)
    np.testing.assert_allclose(strong_gradient, 49.0 * base_gradient, rtol=2e-12, atol=5e-11)

    results = [
        solve_energy_descent(item, _config("straight", interior_points=8))
        for item in (base, converted, strengthened)
    ]
    assert all(result.solver_success for result in results)
    diagnostics = [result.diagnostics for result in results]
    assert diagnostics[1]["coordinate_scale_m"] == pytest.approx(
        1000.0 * diagnostics[0]["coordinate_scale_m"]
    )
    assert diagnostics[1]["slowness_scale_s_per_m"] == pytest.approx(
        diagnostics[0]["slowness_scale_s_per_m"] / 1000.0
    )
    assert diagnostics[2]["energy_scale_s2"] == pytest.approx(
        49.0 * diagnostics[0]["energy_scale_s2"]
    )
    np.testing.assert_allclose(
        [item["normalized_energy"] for item in diagnostics],
        diagnostics[0]["normalized_energy"],
        rtol=2e-13,
    )


def test_accepted_steps_lower_common_panel_energy_and_keep_endpoints():
    scenario = soften_walls(ridge_pass_terrain(seed=0), multiplier=100.0)
    config = _config("arc_left", interior_points=16)
    result = solve_energy_descent(scenario, config)
    assert result.route_m is not None
    np.testing.assert_array_equal(result.route_m[0], scenario.start_m)
    np.testing.assert_array_equal(result.route_m[-1], scenario.goal_m)
    baseline = np.asarray(result.diagnostics["accepted_baseline_energy_s2"])
    accepted = np.asarray(result.diagnostics["accepted_energy_s2"])
    steps = np.asarray(result.diagnostics["accepted_step_sizes"])
    assert len(baseline) == len(accepted) == len(steps) > 0
    assert np.all(np.isfinite(baseline)) and np.all(np.isfinite(accepted))
    assert np.all(accepted < baseline)
    assert np.all((steps > 0) & (steps <= 1))


def test_hard_barriers_reject_infeasible_seed_and_retain_feasible_blocked_route():
    scenario = obstacle_detour_fixture()
    direct = solve_energy_descent(scenario, _config("straight"))
    assert direct.termination_reason == "infeasible_initialization"
    assert not direct.solver_success and not direct.feasible
    assert direct.route_m is not None
    assert direct.evaluation.violations == ("barrier_collision",)
    assert direct.diagnostics["constraint_violation"] == "barrier_collision"

    detour = solve_energy_descent(scenario, _config("barrier"))
    assert detour.route_m is not None
    assert detour.feasible
    assert not detour.solver_success
    assert detour.termination_reason == "constraint_blocked"
    assert detour.diagnostics["infeasible_trial_rejections"] > 0
    assert detour.diagnostics["constraint_feasible"]
    assert detour.diagnostics["constraint_violation"] is None


def test_independent_cost_and_shared_seed_hash_are_kept_separate_from_stationarity():
    scenario = symmetry_fixture()
    common = dict(
        initialization="arc_left",
        interior_points=16,
        tolerance=1e-6,
        max_iterations=1000,
        time_limit_s=8.0,
        profile_samples=33,
    )
    energy = solve_energy_descent(scenario, PlannerConfig(method="energy_descent", **common))
    euler = solve_euler_lagrange(scenario, PlannerConfig(method="euler_lagrange", **common))
    assert (
        energy.diagnostics["initialization"]["initial_route_hash"]
        == euler.diagnostics["initialization"]["initial_route_hash"]
    )
    independent = evaluate_route(scenario, energy.route_m, profile_samples=33)
    assert energy.evaluated_cost_s == pytest.approx(independent.cost_s, rel=0, abs=1e-12)
    assert energy.feasible == independent.feasible
    assert energy.diagnostics["energy_s2"] != pytest.approx(energy.evaluated_cost_s)


def test_iteration_and_wall_clock_limits_return_serializable_failures():
    scenario = symmetry_fixture()
    limited = solve_energy_descent(scenario, _config("arc_left", max_iterations=1, tolerance=1e-12))
    assert limited.termination_reason == "iteration_limit"
    assert not limited.solver_success
    assert limited.route_m is not None and limited.evaluation is not None
    assert limited.to_dict()

    timed = solve_energy_descent(scenario, _config("arc_left", time_limit_s=np.finfo(float).tiny))
    assert timed.termination_reason == "time_limit"
    assert not timed.solver_success
    assert timed.to_dict()


def test_timeout_after_accepted_step_does_not_report_stale_gradient(monkeypatch):
    import path_planning_ode.terrain_energy as terrain_energy

    scenario = symmetry_fixture()
    candidate_evaluated = False
    original_energy = terrain_energy._terrain_energy

    def mark_candidate(*args, **kwargs):
        nonlocal candidate_evaluated
        value = original_energy(*args, **kwargs)
        candidate_evaluated = True
        return value

    def expire_after_candidate(deadline):
        if candidate_evaluated:
            raise terrain_energy._DeadlineExceeded

    monkeypatch.setattr(terrain_energy, "_terrain_energy", mark_candidate)
    monkeypatch.setattr(terrain_energy, "_check_deadline", expire_after_candidate)
    result = solve_energy_descent(scenario, _config("arc_left"))
    assert result.termination_reason == "time_limit"
    assert result.route_m is not None and result.evaluation is not None
    assert not result.diagnostics["gradient_diagnostics_current"]
    assert result.diagnostics["free_gradient_norm"] is None
    assert result.diagnostics["scaled_free_gradient_norm"] is None
    assert result.diagnostics["stationarity_norm"] is None
    assert result.to_dict()


def test_fmm_warm_start_includes_initialization_time_and_keeps_ode_diagnostic_separate():
    scenario = soften_walls(ridge_pass_terrain(seed=0), multiplier=100.0)
    config = PlannerConfig(
        method="energy_descent",
        initialization="fast_marching",
        interior_points=32,
        reference_grid_size=129,
        tolerance=1e-5,
        max_iterations=400,
        time_limit_s=60.0,
        profile_samples=65,
        options={"quadrature_order": 8},
    )
    result = solve_energy_descent(scenario, config)
    assert result.solver_success and result.feasible
    assert result.termination_reason == "stationary"
    assert result.timing_s["initialization"] > 0
    assert result.timing_s["total"] >= result.timing_s["initialization"]
    assert result.diagnostics["initialization"]["initial_route_hash"]
    assert result.diagnostics["stationarity_norm"] <= config.tolerance
    # Discrete energy-gradient convergence is not an ODE-convergence claim at
    # this mesh resolution; both diagnostics remain visible and distinct.
    assert result.diagnostics["ode_residual_norm_m"] > 1.0
    independent = evaluate_route(scenario, result.route_m, profile_samples=65)
    assert result.evaluated_cost_s == pytest.approx(independent.cost_s, abs=1e-12)
