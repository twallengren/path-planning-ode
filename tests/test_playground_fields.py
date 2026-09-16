import json
from copy import deepcopy

import numpy as np
import pytest
from scipy.integrate import quad

from path_planning_ode.core import Obstacle, Scene, _field, weighted_distance
from path_planning_ode.field_adapter import (
    PlaygroundField,
    build_playground_preset,
    terrain_to_playground_field,
)
from path_planning_ode.playground import (
    PlaygroundOptions,
    PlaygroundState,
    advance_playground,
    evaluate_playground,
    initialize_playground,
    playground_energy_gradient,
)
from path_planning_ode.terrain import TerrainScenario
from path_planning_ode.terrain_generators import obstacle_detour_fixture


def _terrain_scenario(*, coordinate_factor=1.0, cost_factor=1.0):
    x = np.linspace(0.0, 12.0, 9) * coordinate_factor
    y = np.linspace(-3.0, 7.0, 8) * coordinate_factor
    xx, yy = np.meshgrid(x / coordinate_factor, y / coordinate_factor)
    log_slowness = (
        np.log(0.7 * cost_factor)
        + 0.13 * np.sin(xx / 2.7)
        + 0.09 * np.cos(yy / 2.1)
        + 0.015 * xx * yy
    )
    elevation = 20 + 2 * xx - 0.4 * yy + 0.05 * xx * yy
    return TerrainScenario(
        name="playground field fixture",
        bounds_m=(0, -3 * coordinate_factor, 12 * coordinate_factor, 7 * coordinate_factor),
        start_m=(0.5 * coordinate_factor, -1.5 * coordinate_factor),
        goal_m=(11.5 * coordinate_factor, 5.5 * coordinate_factor),
        field_x_m=x,
        field_y_m=y,
        elevation_m=elevation,
        log_slowness=log_slowness,
        metadata={"fixture": True},
        provenance={"source": "analytic test grid"},
    )


def _terrain_field(*, coordinate_factor=1.0, cost_factor=1.0, gaussians=()):
    return terrain_to_playground_field(
        _terrain_scenario(coordinate_factor=coordinate_factor, cost_factor=cost_factor),
        gaussians,
    )


def _javascript_number_roundtrip(value):
    """Model JSON.parse/stringify changes relevant to Python-originated JSON."""
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return value
    if isinstance(value, float):
        if value == 0:
            return 0
        return int(value) if value.is_integer() else value
    if isinstance(value, list):
        return [_javascript_number_roundtrip(item) for item in value]
    return {key: _javascript_number_roundtrip(item) for key, item in value.items()}


def test_gaussian_adapter_preserves_core_derivatives_and_exact_route_cost():
    obstacles = (
        Obstacle(-0.4, 0.2, 3.0, 0.45),
        Obstacle(0.7, -0.1, 1.2, 0.8),
    )
    field = PlaygroundField(gaussians=obstacles)
    points = np.array([[-0.9, 0.1], [0.0, -0.3], [1.1, 0.7]])
    expected = _field(points, obstacles)
    actual = field.evaluate(points)
    for observed, wanted in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, wanted, rtol=0, atol=2e-14)
    np.testing.assert_array_equal(field.elevation(points), np.zeros(3))

    path = np.array([[-1.3, 0.2], [-0.2, 0.8], [0.9, -0.4], [1.6, 0.1]])
    metrics = evaluate_playground(
        path,
        obstacles,
        options=PlaygroundOptions(interior_points=2, method="descent"),
    )
    assert metrics.route_cost == pytest.approx(
        weighted_distance(
            path, Scene(start=tuple(path[0]), end=tuple(path[-1]), obstacles=obstacles)
        ),
        rel=0,
        abs=2e-14,
    )


def test_terrain_gaussian_product_gradient_and_hessian_match_finite_differences():
    field = _terrain_field(gaussians=(Obstacle(5.1, 1.2, 4.0, 0.7),))
    point = np.array([4.7, 1.6])
    cost, gradient, hessian = field.evaluate(point)
    epsilon = 2e-5
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
    np.testing.assert_allclose(gradient, numerical_gradient, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(hessian, numerical_hessian, rtol=3e-7, atol=3e-7)


def test_terrain_energy_gradient_and_independent_cost_match_oracles():
    field = _terrain_field(gaussians=(Obstacle(6.3, 0.1, 7.0, 0.45),))
    path = np.array([[0.5, -1.5], [3.0, 1.7], [7.6, -0.2], [11.5, 5.5]])
    panels = (8, 11, 10)
    energy, analytic = playground_energy_gradient(
        path,
        field=field,
        quadrature_panels=panels,
    )
    epsilon = 2e-5
    numerical = np.empty_like(path)
    for vertex in range(len(path)):
        for axis in range(2):
            plus, minus = path.copy(), path.copy()
            plus[vertex, axis] += epsilon
            minus[vertex, axis] -= epsilon
            numerical[vertex, axis] = (
                playground_energy_gradient(plus, field=field, quadrature_panels=panels)[0]
                - playground_energy_gradient(minus, field=field, quadrature_panels=panels)[0]
            ) / (2 * epsilon)
    assert energy > 0
    np.testing.assert_allclose(analytic, numerical, rtol=3e-7, atol=3e-6)

    options = PlaygroundOptions(interior_points=2, method="descent")
    route_cost = evaluate_playground(path, field=field, options=options).route_cost
    adaptive = 0.0
    for start, end in zip(path[:-1], path[1:], strict=True):
        delta = end - start
        adaptive += (
            np.linalg.norm(delta)
            * quad(
                lambda value: float(field.cost(start + value * delta)),
                0,
                1,
                epsabs=1e-10,
                epsrel=1e-10,
                limit=300,
            )[0]
        )
    assert route_cost == pytest.approx(adaptive, rel=1e-7)


def test_normalized_diagnostics_are_invariant_to_coordinate_and_cost_units():
    base = _terrain_field()
    converted = _terrain_field(coordinate_factor=1000.0, cost_factor=0.001)
    path = np.array([[0.5, -1.5], [3.2, 2.0], [7.0, 0.5], [11.5, 5.5]])
    options = PlaygroundOptions(interior_points=2, method="descent")
    first = initialize_playground(path[0], path[-1], field=base, path=path, options=options)
    second = initialize_playground(
        1000 * path[0],
        1000 * path[-1],
        field=converted,
        path=1000 * path,
        options=options,
    )
    assert second.metrics.normalized_energy == pytest.approx(first.metrics.normalized_energy)
    assert second.metrics.normalized_free_gradient_norm == pytest.approx(
        first.metrics.normalized_free_gradient_norm
    )
    assert second.metrics.normalized_ode_residual_norm == pytest.approx(
        first.metrics.normalized_ode_residual_norm
    )
    first_step = advance_playground(first)
    second_step = advance_playground(second)
    np.testing.assert_allclose(second_step.path / 1000, first_step.path, rtol=2e-12, atol=2e-12)
    assert first_step.coordinate_scale == first.coordinate_scale
    assert second_step.coordinate_scale == second.coordinate_scale


def test_auto_handoff_is_guarded_and_reaches_ode_criterion():
    parameter = np.linspace(0, 1, 34)
    path = np.column_stack(
        (-4 + 8 * parameter, 0.5 * np.sin(np.pi * parameter) + np.sin(8 * np.pi * parameter))
    )
    state = initialize_playground(
        path[0],
        path[-1],
        (Obstacle(0, -0.5, 8, 0.7),),
        path=path,
        options=PlaygroundOptions(interior_points=32, method="auto", tolerance=1e-5),
    )
    saw_newton = False
    reference = None
    for _ in range(80):
        if state.phase == "newton" and not saw_newton:
            saw_newton = True
            reference = (state.phase_reference_energy, state.phase_reference_cost)
            assert state.accepted_descent_updates >= 8
        if state.status != "running":
            break
        state = advance_playground(state)
    assert saw_newton
    assert state.status == "converged"
    assert state.metrics.scaled_ode_residual_norm <= state.options.auto_residual_tolerance
    assert state.metrics.energy <= reference[0] * (1 + state.options.newton_energy_guard)
    assert state.metrics.route_cost <= reference[1] * (1 + state.options.newton_cost_guard)


def test_auto_newton_failure_enters_real_cooldown(monkeypatch):
    import path_planning_ode.playground as playground

    parameter = np.linspace(0, 1, 10)
    path = np.column_stack((-2 + 4 * parameter, 0.4 * np.sin(3 * np.pi * parameter)))
    state = initialize_playground(
        path[0],
        path[-1],
        (Obstacle(0, 0.2, 5, 0.5),),
        path=path,
        options=PlaygroundOptions(interior_points=8, method="auto"),
    )
    state.phase = "newton"
    state.phase_reference_energy = state.metrics.energy
    state.phase_reference_cost = state.metrics.route_cost

    def singular(_state):
        raise np.linalg.LinAlgError

    monkeypatch.setattr(playground, "_newton_direction", singular)
    failed = advance_playground(state)
    assert failed.status == "running"
    assert failed.phase == "descent"
    assert failed.newton_cooldown == failed.options.auto_cooldown
    assert failed.phase_reason == "newton_failed"

    monkeypatch.setattr(
        playground, "_descent_direction", lambda state, gradient: np.zeros_like(state.path)
    )
    stopped = advance_playground(failed)
    assert stopped.status == "backtracking_failed"
    assert stopped.newton_cooldown == failed.options.auto_cooldown
    np.testing.assert_array_equal(stopped.path, failed.path)
    assert stopped.phase_reason == "descent_failed_during_cooldown"


def test_coincident_endpoints_loop_and_state_roundtrip_are_supported():
    parameter = np.linspace(0, 2 * np.pi, 14)
    path = np.column_stack((np.cos(parameter), np.sin(parameter)))
    options = PlaygroundOptions(interior_points=12, method="descent")
    state = initialize_playground(path[0], path[-1], path=path, options=options, pin_index=4)
    updated = advance_playground(state)
    np.testing.assert_array_equal(updated.path[[0, -1]], path[[0, -1]])
    np.testing.assert_array_equal(updated.path[4], path[4])
    encoded = json.loads(json.dumps(updated.to_dict(), allow_nan=False))
    restored = PlaygroundState.from_dict(encoded)
    np.testing.assert_array_equal(restored.path, updated.path)
    assert restored.field.field_hash == updated.field.field_hash

    collapsed = initialize_playground((1, 1), (1, 1), options=options)
    assert collapsed.status == "converged"
    assert collapsed.metrics.energy == 0


def test_hard_barriers_require_explicit_soft_conversion_and_preserve_metadata():
    scenario = obstacle_detour_fixture()
    with pytest.raises(ValueError, match="soften_barriers=True"):
        terrain_to_playground_field(scenario)
    field = terrain_to_playground_field(scenario, soften_barriers=True, wall_multiplier=25)
    soft = field.base_spec["scenario"]["metadata"]["soft_walls"]
    assert soft["multiplier"] == 25
    assert soft["geometry_geojson"] == list(scenario.barriers_geojson)
    assert field.base_spec["scenario"]["barriers_geojson"] == []


def test_presets_are_deterministic_and_keep_explicit_gaussians_separate():
    first = build_playground_preset("random_hills", seed=17)
    second = build_playground_preset("random_hills", seed=17)
    third = build_playground_preset("random_hills", seed=18)
    assert first == second
    assert first["gaussians"] != third["gaussians"]
    assert first["base_field"] == {"kind": "uniform", "cost": 1.0, "elevation": 0.0}
    assert first["strokes"] == []


def test_terrain_hash_survives_javascript_numbers_but_rejects_changed_field():
    preset = build_playground_preset("ridge_pass", seed=0)
    base = preset["base_field"]
    browser_base = _javascript_number_roundtrip(json.loads(json.dumps(base)))
    reconstructed = PlaygroundField.from_spec(browser_base)
    assert reconstructed.base_spec["scenario_hash"] == base["scenario_hash"]
    assert reconstructed.field_hash == PlaygroundField.from_spec(base).field_hash

    tampered = deepcopy(browser_base)
    tampered["scenario"]["log_slowness"][20][21] += 1e-6
    with pytest.raises(ValueError, match="scenario_hash"):
        PlaygroundField.from_spec(tampered)


def test_scenario_hash_normalizes_integral_float_and_signed_zero_metadata():
    original = _terrain_scenario()
    serialized = original.to_dict()
    serialized["metadata"] = {
        "integral_float": 1.0,
        "signed_zero": -0.0,
        "nested": [2.0, -0.0, 2.5],
    }
    scenario = TerrainScenario.from_dict(serialized)
    browser_value = _javascript_number_roundtrip(scenario.to_dict())
    restored = TerrainScenario.from_dict(browser_value)
    assert restored.scenario_hash == scenario.scenario_hash
