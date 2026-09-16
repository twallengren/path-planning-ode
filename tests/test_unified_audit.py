"""Independent numerical audit for the consolidated field and path solver."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from path_planning_ode.core import Obstacle
from path_planning_ode.field_adapter import PlaygroundField, terrain_to_playground_field
from path_planning_ode.playground import (
    PlaygroundOptions,
    _field_jacobian,
    _field_residual,
    advance_playground,
    evaluate_playground,
    initialize_playground,
)
from path_planning_ode.terrain import TerrainScenario


def _terrain_scenario(*, coordinate_factor: float = 1.0, cost_factor: float = 1.0):
    x_unscaled = np.linspace(-5.0, 5.0, 21)
    y_unscaled = np.linspace(-3.0, 3.0, 17)
    xx, yy = np.meshgrid(x_unscaled, y_unscaled)
    log_cost = (
        np.log(0.8 * cost_factor)
        + 0.13 * np.sin(xx / 1.7)
        - 0.08 * np.cos(yy / 1.3)
        + 0.012 * xx * yy
    )
    return TerrainScenario(
        name="unified independent audit",
        bounds_m=tuple(coordinate_factor * np.array([-5.0, -3.0, 5.0, 3.0])),
        start_m=tuple(coordinate_factor * np.array([-4.0, 0.0])),
        goal_m=tuple(coordinate_factor * np.array([4.0, 0.0])),
        field_x_m=coordinate_factor * x_unscaled,
        field_y_m=coordinate_factor * y_unscaled,
        elevation_m=np.zeros_like(xx),
        log_slowness=log_cost,
    )


def _wobbly_path(interior_points: int, amplitude: float = 0.3) -> np.ndarray:
    parameter = np.linspace(0.0, 1.0, interior_points + 2)
    return np.column_stack(
        (
            -4.0 + 8.0 * parameter,
            0.5 * np.sin(np.pi * parameter) + amplitude * np.sin(8.0 * np.pi * parameter),
        )
    )


def _advance_until_terminal(state, limit: int = 100):
    history = [state]
    for _ in range(limit):
        if state.status != "running":
            break
        state = advance_playground(state)
        history.append(state)
    return state, history


def _javascript_number_shapes(value):
    """Model JSON.parse/stringify's loss of float spelling distinctions."""
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return value
    if isinstance(value, float):
        if value == 0:
            return 0
        return int(value) if value.is_integer() else value
    if isinstance(value, list):
        return [_javascript_number_shapes(item) for item in value]
    return {key: _javascript_number_shapes(item) for key, item in value.items()}


def _adaptive_route_cost(path: np.ndarray, field: PlaygroundField) -> float:
    """Integrate c ds independently, exposing grid and narrow-bump locations."""
    terrain = field._terrain
    x_knots = () if terrain is None else terrain.scenario.field_x_m[1:-1]
    y_knots = () if terrain is None else terrain.scenario.field_y_m[1:-1]
    total = 0.0
    for start, end in zip(path[:-1], path[1:], strict=True):
        delta = end - start
        breaks: list[float] = []
        if delta[0] != 0:
            breaks.extend((np.asarray(x_knots) - start[0]) / delta[0])
        if delta[1] != 0:
            breaks.extend((np.asarray(y_knots) - start[1]) / delta[1])
        length2 = float(delta @ delta)
        for bump in field.gaussians:
            center = np.array([bump.x, bump.y])
            closest = float(np.dot(center - start, delta) / length2)
            parameter_width = bump.width / np.sqrt(length2)
            breaks.extend(closest + parameter_width * np.arange(-6, 7))
        points = sorted({float(value) for value in breaks if 0 < value < 1})
        total += (
            np.sqrt(length2)
            * quad(
                lambda parameter: float(field.cost(start + parameter * delta)),
                0.0,
                1.0,
                points=points,
                epsabs=1e-11,
                epsrel=1e-11,
                limit=max(200, 2 * len(points)),
            )[0]
        )
    return total


def test_terrain_gaussian_product_derivatives_and_adaptive_route_cost():
    bump = Obstacle(0.37, -0.28, 7.0, 0.11)
    field = terrain_to_playground_field(_terrain_scenario(), (bump,))
    point = np.array([0.22, -0.19])
    cost, gradient, hessian = field.evaluate(point)
    epsilon = 8e-6
    numerical_gradient = np.empty(2)
    numerical_hessian = np.empty((2, 2))
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = epsilon
        numerical_gradient[axis] = (field.cost(point + offset) - field.cost(point - offset)) / (
            2 * epsilon
        )
        numerical_hessian[:, axis] = (
            field.gradient(point + offset) - field.gradient(point - offset)
        ) / (2 * epsilon)
    assert cost > 0
    np.testing.assert_allclose(gradient, numerical_gradient, rtol=3e-7, atol=3e-7)
    np.testing.assert_allclose(hessian, numerical_hessian, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=2e-13)

    path = np.array([[-4.0, -2.2], [-0.8, 1.5], [1.1, -1.3], [4.0, 2.1]])
    options = PlaygroundOptions(interior_points=2, method="descent")
    observed = evaluate_playground(path, field=field, options=options).route_cost
    assert observed == pytest.approx(_adaptive_route_cost(path, field), rel=1e-5)


def test_browser_stable_hashes_preserve_raw_v2_data_and_reject_tampering():
    raw = _terrain_scenario().to_dict()
    raw["metadata"] = {
        "integral_float": 1.0,
        "signed_zero": -0.0,
        "nested": [2.0, -0.0, 2.25],
    }
    raw["elevation_m"][0][0] = -0.0
    scenario = TerrainScenario.from_dict(raw)
    before_hash = deepcopy(scenario.to_dict())
    scenario_hash = scenario.scenario_hash
    assert scenario.to_dict() == before_hash  # hashing does not rewrite source data

    browser_raw = _javascript_number_shapes(scenario.to_dict())
    reconstructed = TerrainScenario.from_dict(browser_raw)  # raw version-2 import has no hash
    assert reconstructed.scenario_hash == scenario_hash
    np.testing.assert_array_equal(reconstructed.bounds_m, scenario.bounds_m)
    np.testing.assert_array_equal(reconstructed.elevation_m, scenario.elevation_m)
    np.testing.assert_array_equal(reconstructed.log_slowness, scenario.log_slowness)

    field = terrain_to_playground_field(scenario, (Obstacle(0.0, 0.5, 2.0, 0.4),))
    browser_spec = _javascript_number_shapes(field.to_spec())
    restored_field = PlaygroundField.from_spec(browser_spec)
    assert restored_field.field_hash == field.field_hash
    np.testing.assert_array_equal(
        restored_field.base_spec["scenario"]["log_slowness"], scenario.log_slowness
    )

    tampered = deepcopy(browser_spec)
    tampered["base_field"]["scenario"]["log_slowness"][3][4] += 1e-6
    with pytest.raises(ValueError, match="scenario_hash"):
        PlaygroundField.from_spec(tampered)


def test_ode_jacobian_matches_independent_centered_differences():
    field = terrain_to_playground_field(
        _terrain_scenario(),
        (Obstacle(-0.6, 0.4, 3.2, 0.55), Obstacle(1.3, -0.7, 1.7, 0.8)),
    )
    path = np.array([[-4.0, -1.2], [-2.4, 0.8], [-0.7, -0.3], [1.4, 1.1], [3.8, 0.4]])
    analytic = _field_jacobian(path, field)
    epsilon = 2e-6
    numerical = np.empty_like(analytic)
    for scalar in range(analytic.shape[1]):
        vertex, axis = divmod(scalar, 2)
        plus, minus = path.copy(), path.copy()
        plus[vertex + 1, axis] += epsilon
        minus[vertex + 1, axis] -= epsilon
        numerical[:, scalar] = (
            _field_residual(plus, field).ravel() - _field_residual(minus, field).ravel()
        ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=7e-6, atol=2e-5)


def test_unit_normalization_preserves_real_descent_trajectory_and_frozen_scale():
    factor = 1000.0
    path = _wobbly_path(16, amplitude=0.2)
    base = PlaygroundField(
        {"kind": "uniform", "cost": 2.0, "elevation": 0.0},
        (Obstacle(0.3, -0.4, 5.0, 0.7),),
        bounds=(-5.0, -3.0, 5.0, 3.0),
    )
    converted = PlaygroundField(
        {"kind": "uniform", "cost": 2.0 / factor, "elevation": 0.0},
        (Obstacle(0.3 * factor, -0.4 * factor, 5.0, 0.7 * factor),),
        bounds=(-5.0 * factor, -3.0 * factor, 5.0 * factor, 3.0 * factor),
    )
    options = PlaygroundOptions(interior_points=16, method="descent")
    first = initialize_playground(path[0], path[-1], field=base, path=path, options=options)
    second = initialize_playground(
        factor * path[0],
        factor * path[-1],
        field=converted,
        path=factor * path,
        options=options,
    )
    original_scales = first.coordinate_scale, second.coordinate_scale
    for _ in range(5):
        first = advance_playground(first)
        second = advance_playground(second)
        np.testing.assert_allclose(second.path / factor, first.path, rtol=0, atol=2e-13)
        assert second.step_size == pytest.approx(first.step_size, rel=2e-13)
        assert second.metrics.normalized_energy == pytest.approx(
            first.metrics.normalized_energy, rel=2e-13
        )
        assert second.metrics.normalized_free_gradient_norm == pytest.approx(
            first.metrics.normalized_free_gradient_norm, rel=2e-13
        )
        assert (first.coordinate_scale, second.coordinate_scale) == original_scales
        assert first.metrics.coordinate_scale == original_scales[0]
        assert second.metrics.coordinate_scale == original_scales[1]

    # For an unbounded field the live path bounding box can shrink, so this
    # separately distinguishes a frozen solve scale from per-step recomputation.
    loop = np.array([[-1.0, 0.0], [4.0, 1.0], [0.0, -1.0], [1.0, 0.0]])
    unbounded = PlaygroundField()
    frozen = initialize_playground(
        loop[0],
        loop[-1],
        field=unbounded,
        path=loop,
        options=PlaygroundOptions(interior_points=2, method="descent"),
    )
    initialized_scale = frozen.coordinate_scale
    frozen = advance_playground(frozen)
    assert unbounded.coordinate_scale(frozen.path) != pytest.approx(initialized_scale)
    assert frozen.coordinate_scale == initialized_scale
    assert frozen.metrics.coordinate_scale == initialized_scale


def test_auto_handoff_obeys_phase_wide_guards_and_avoids_false_convergence():
    path = _wobbly_path(32)
    obstacles = (Obstacle(0.0, -0.5, 8.0, 0.7),)
    options = PlaygroundOptions(interior_points=32, method="auto")
    final, history = _advance_until_terminal(
        initialize_playground(path[0], path[-1], obstacles, path=path, options=options)
    )
    newton_states = [state for state in history if state.phase == "newton"]
    assert final.status == "converged"
    assert final.metrics.scaled_ode_residual_norm <= options.auto_residual_tolerance
    assert newton_states and newton_states[0].accepted_descent_updates >= 8
    reference_energy = newton_states[0].phase_reference_energy
    reference_cost = newton_states[0].phase_reference_cost
    assert reference_energy is not None and reference_cost is not None
    for state in newton_states:
        assert state.metrics.energy <= reference_energy * (1 + options.newton_energy_guard)
        assert state.metrics.route_cost <= reference_cost * (1 + options.newton_cost_guard)

    descent_options = replace(options, method="descent", tolerance=1e-6)
    descent, _ = _advance_until_terminal(
        initialize_playground(path[0], path[-1], obstacles, path=path, options=descent_options)
    )
    assert descent.status == "converged"
    assert descent.metrics.scaled_free_gradient_norm <= descent_options.tolerance
    assert descent.metrics.normalized_ode_residual_norm > options.auto_residual_tolerance * 100
    restarted = initialize_playground(
        descent.path[0],
        descent.path[-1],
        obstacles,
        path=descent.path,
        options=options,
    )
    assert restarted.status == "running"


def test_real_newton_guard_rejection_and_eight_accepted_step_cooldown():
    path = _wobbly_path(32)
    obstacles = (Obstacle(0.0, -0.5, 8.0, 0.7),)
    options = PlaygroundOptions(
        interior_points=32,
        method="auto",
        auto_descent_updates=100,
        auto_cooldown=8,
    )
    state = initialize_playground(path[0], path[-1], obstacles, path=path, options=options)
    # The cost guard is deliberately impossible for this real Newton trial.
    state = replace(
        state,
        phase="newton",
        phase_reference_energy=state.metrics.energy * 0.5,
        phase_reference_cost=state.metrics.route_cost * 0.5,
    )
    state = advance_playground(state)
    assert state.phase == "descent" and state.phase_reason == "newton_failed"
    assert state.newton_cooldown == 8 and state.step_size == 0
    for expected in range(7, -1, -1):
        before = state.path.copy()
        state = advance_playground(state)
        assert state.status == "running"
        assert state.step_size > 0 and not np.array_equal(state.path, before)
        assert state.newton_cooldown == expected

    boundary_field = PlaygroundField(
        gaussians=(Obstacle(0.0, 0.0, 2.0, 0.5),), bounds=(-1.1, -1.1, 1.1, 1.1)
    )
    boundary_path = np.array([[-1.1, -1.1], [-0.5, -1.1], [0.5, -1.1], [1.1, -1.1]])
    blocked = initialize_playground(
        boundary_path[0],
        boundary_path[-1],
        field=boundary_field,
        path=boundary_path,
        options=PlaygroundOptions(interior_points=2, method="auto"),
    )
    blocked = replace(blocked, newton_cooldown=8)
    failed = advance_playground(blocked)
    assert failed.status == "backtracking_failed"
    assert failed.phase_reason == "descent_failed_during_cooldown"
    assert failed.newton_cooldown == 8
    np.testing.assert_array_equal(failed.path, boundary_path)


def test_fixed_endpoints_pin_and_coincident_loop_survive_both_phases():
    interior_points = 32
    parameter = np.linspace(0.0, 2 * np.pi, interior_points + 2)
    path = np.column_stack((np.cos(parameter), np.sin(parameter)))
    path[-1] = path[0]
    pin_index = 8
    field = PlaygroundField(
        gaussians=(Obstacle(0.0, 0.0, 2.0, 0.5),), bounds=(-1.1, -1.1, 1.1, 1.1)
    )
    state = initialize_playground(
        path[0],
        path[-1],
        field=field,
        path=path,
        options=PlaygroundOptions(interior_points=interior_points, method="auto"),
        pin_index=pin_index,
    )
    fixed = state.path[[0, pin_index, -1]].copy()
    saw_descent = saw_newton = False
    for _ in range(40):
        saw_descent |= state.phase == "descent"
        saw_newton |= state.phase == "newton"
        np.testing.assert_array_equal(state.path[[0, pin_index, -1]], fixed)
        if state.status != "running":
            break
        state = advance_playground(state)
    assert state.status == "converged"
    assert saw_descent and saw_newton
    np.testing.assert_array_equal(state.path[[0, pin_index, -1]], fixed)


@pytest.mark.parametrize("interior_points", [32, 64, 128])
def test_hard_wobble_converges_consistently_across_resolution(interior_points):
    path = _wobbly_path(interior_points, amplitude=1.0)
    state, _ = _advance_until_terminal(
        initialize_playground(
            path[0],
            path[-1],
            (Obstacle(0.0, -0.5, 8.0, 0.7),),
            path=path,
            options=PlaygroundOptions(interior_points=interior_points, method="auto"),
        ),
        limit=40,
    )
    assert state.status == "converged"
    assert state.iteration <= 20
    assert state.metrics.scaled_ode_residual_norm <= state.options.auto_residual_tolerance
    assert state.metrics.route_cost == pytest.approx(8.392, abs=8e-4)


def test_real_terrain_auto_case_converges_with_independent_cost_evaluation():
    scenario = _terrain_scenario()
    field = terrain_to_playground_field(scenario, (Obstacle(1.0, 0.6, 2.0, 0.35),))
    path = _wobbly_path(32)
    final, _ = _advance_until_terminal(
        initialize_playground(
            path[0],
            path[-1],
            field=field,
            path=path,
            options=PlaygroundOptions(interior_points=32, method="auto"),
        ),
        limit=60,
    )
    assert final.status == "converged"
    assert final.metrics.scaled_ode_residual_norm <= final.options.auto_residual_tolerance
    assert final.metrics.route_cost == pytest.approx(
        _adaptive_route_cost(final.path, field), rel=1e-5
    )
