import json
from time import perf_counter

import numpy as np
import pytest
from scipy.integrate import quad

from path_planning_ode import Obstacle, cost_field
from path_planning_ode.playground import (
    MAX_GAUSSIANS,
    PlaygroundOptions,
    advance_playground,
    build_playground_field,
    initialize_playground,
    playground_energy_gradient,
    set_playground_pin,
    strokes_to_obstacles,
)


def _independent_energy(path, obstacles):
    intervals = len(path) - 1
    total = 0.0
    for start, end in zip(path[:-1], path[1:], strict=True):
        delta = end - start
        length2 = float(delta @ delta)
        integral = quad(
            lambda parameter: float(cost_field(start + parameter * delta, obstacles) ** 2),
            0,
            1,
            epsabs=1e-11,
            epsrel=1e-11,
            limit=300,
        )[0]
        total += intervals * length2 * integral
    return total


def test_discrete_energy_gradient_matches_finite_differences():
    obstacles = (
        Obstacle(-0.2, 0.4, weight=3.0, width=0.35),
        Obstacle(0.8, -0.1, weight=1.5, width=0.8),
    )
    path = np.array([[-1.3, -0.7], [-0.5, 0.3], [0.1, -0.2], [0.7, 0.6], [1.4, 0.2]])
    panels = (5, 4, 6, 5)
    _, analytic = playground_energy_gradient(path, obstacles, quadrature_panels=panels)
    numerical = np.empty_like(path)
    epsilon = 2e-6
    for vertex in range(len(path)):
        for coordinate in range(2):
            plus, minus = path.copy(), path.copy()
            plus[vertex, coordinate] += epsilon
            minus[vertex, coordinate] -= epsilon
            numerical[vertex, coordinate] = (
                playground_energy_gradient(plus, obstacles, quadrature_panels=panels)[0]
                - playground_energy_gradient(minus, obstacles, quadrature_panels=panels)[0]
            ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-7, atol=2e-7)


def test_composite_quadrature_resolves_narrow_high_cost_bump():
    obstacles = (Obstacle(0.17, 0.02, weight=80, width=0.025),)
    path = np.array([[-1.0, -0.12], [0.7, 0.08], [1.4, -0.2]])
    value, _ = playground_energy_gradient(path, obstacles)
    assert value == pytest.approx(_independent_energy(path, obstacles), rel=2e-11)


def test_descent_accepts_only_energy_decrease_and_preserves_endpoints():
    obstacles = (Obstacle(0.0, 0.25, weight=12, width=0.35),)
    path = np.array(
        [[-1.5, 0], [-1.0, 0.2], [-0.5, 0.35], [0, 0.4], [0.5, 0.35], [1, 0.2], [1.5, 0]]
    )
    state = initialize_playground(
        path[0],
        path[-1],
        obstacles,
        path=path,
        options=PlaygroundOptions(interior_points=5),
    )
    energies = [state.metrics.energy]
    endpoints = state.path[[0, -1]].copy()
    for _ in range(12):
        state = advance_playground(state)
        energies.append(state.metrics.energy)
        np.testing.assert_array_equal(state.path[[0, -1]], endpoints)
        if state.status != "running":
            break
    assert len(energies) > 2
    assert np.all(np.diff(energies) <= 1e-11)
    assert energies[-1] < energies[0]


def test_pin_is_exact_and_release_preserves_accepted_geometry():
    obstacles = (Obstacle(0, 0, weight=8, width=0.4),)
    state = initialize_playground(
        (-2, 0),
        (2, 0),
        obstacles,
        options=PlaygroundOptions(interior_points=7),
        pin_index=4,
        pin_position=(0, 0.8),
    )
    for _ in range(5):
        state = advance_playground(state)
        np.testing.assert_array_equal(state.path[4], [0, 0.8])
        if state.status != "running":
            break
    accepted = state.path.copy()
    released = set_playground_pin(state, None)
    assert released.pin_index is None
    assert released.pin_position is None
    np.testing.assert_array_equal(released.path, accepted)


def test_uniform_field_straight_path_is_stationary_for_both_methods():
    for method in ("descent", "newton"):
        state = initialize_playground(
            (-3, 2),
            (4, -1),
            options=PlaygroundOptions(interior_points=32, method=method),
        )
        assert state.status == "converged"
        assert state.metrics.free_gradient_norm < 1e-12
        assert state.metrics.ode_residual_norm < 1e-10


def test_newton_uses_reduced_pinned_system():
    path = np.array([[-2, 0], [-1.3, 0.4], [-0.4, 0.1], [0.2, 1.0], [1.1, 0.2], [2, 0]])
    state = initialize_playground(
        path[0],
        path[-1],
        path=path,
        options=PlaygroundOptions(interior_points=4, method="newton"),
        pin_index=3,
    )
    updated = advance_playground(state)
    assert updated.status == "converged"
    np.testing.assert_array_equal(updated.path[3], path[3])
    np.testing.assert_allclose(updated.path[:4], np.linspace(path[0], path[3], 4), atol=1e-13)
    np.testing.assert_allclose(updated.path[3:], np.linspace(path[3], path[-1], 3), atol=1e-13)


def test_panel_gate_tracks_large_segment_deformation_before_trial():
    obstacle = (Obstacle(0.0, 0.0, weight=30, width=0.03),)
    state = initialize_playground(
        (-1, 0.3),
        (1, 0.3),
        obstacle,
        options=PlaygroundOptions(interior_points=8),
    )
    original_panels = state.quadrature_panels
    # Model an interactive vertex drag while retaining an older incremental state.
    state.path[4] = [0, 2.5]
    state.status = "running"
    updated = advance_playground(state)
    assert max(updated.quadrature_panels) > max(original_panels)
    assert updated.metrics.energy == pytest.approx(
        _independent_energy(updated.path, obstacle), rel=2e-10
    )


def test_nonfinite_field_stops_without_moving_path():
    path = np.linspace((-1, 0), (1, 0), 6)
    state = initialize_playground(
        path[0],
        path[-1],
        (Obstacle(0, 0, weight=1e308, width=1),),
        path=path,
        options=PlaygroundOptions(interior_points=4),
    )
    assert state.status == "nonfinite"
    np.testing.assert_array_equal(state.path, path)
    assert advance_playground(state) is state


def test_stroke_conversion_is_sampling_invariant_and_uses_nominal_strength():
    sparse = [{"id": "stroke", "points": [[-3, 0], [3, 0]], "width": 0.5, "strength": 5}]
    dense = [
        {
            "id": "stroke",
            "points": [[value, 0] for value in np.linspace(-3, 3, 101)],
            "width": 0.5,
            "strength": 5,
        }
    ]
    first, second = strokes_to_obstacles(sparse), strokes_to_obstacles(dense)
    assert first == second
    center_cost = float(cost_field(np.array([0.0, 0.0]), first))
    assert center_cost == pytest.approx(5, rel=2e-6)
    click = strokes_to_obstacles([{"id": "click", "points": [[2, 3]], "width": 0.4, "strength": 7}])
    assert click == (Obstacle(2, 3, weight=6, width=0.4),)
    assert build_playground_field(sparse)["gaussian_count"] == len(first)


def test_stroke_cap_rejects_before_generating_unbounded_field():
    stroke = [{"id": "tiny", "points": [[0, 0], [1e6, 0]], "width": 1e-9, "strength": 2}]
    with pytest.raises(ValueError, match=f"maximum is {MAX_GAUSSIANS}"):
        strokes_to_obstacles(stroke)


def test_default_update_meets_interactive_timing_target():
    obstacles = (
        Obstacle(-0.4, 0.2, weight=8, width=0.3),
        Obstacle(0.5, -0.2, weight=5, width=0.25),
    )
    state = initialize_playground((-2, 0), (2, 0), obstacles)
    started = perf_counter()
    advance_playground(state)
    elapsed = perf_counter() - started
    # Generous CI allowance around the 100 ms reference-machine product target.
    assert elapsed < 0.5


def test_state_snapshot_is_strict_json_serializable():
    initial = initialize_playground(
        (-2, 0),
        (2, 0),
        (Obstacle(0, 0.2, weight=4, width=0.3),),
    )
    advanced = advance_playground(initial)
    pinned = set_playground_pin(advanced, 12, (advanced.path[12, 0], 0.7))
    for state in (initial, advanced, pinned):
        snapshot = state.to_dict()
        encoded = json.dumps(snapshot, allow_nan=False)
        assert json.loads(encoded)["quadrature_panels"] == list(state.quadrature_panels)

        pending = [snapshot]
        while pending:
            value = pending.pop()
            assert not isinstance(value, np.generic)
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, list | tuple):
                pending.extend(value)


@pytest.mark.parametrize(
    "call",
    [
        lambda: PlaygroundOptions(interior_points=0),
        lambda: PlaygroundOptions(method="other"),
        lambda: strokes_to_obstacles(
            [{"id": "bad", "points": [[0, 0]], "width": 1, "strength": 0.5}]
        ),
        lambda: initialize_playground((0, 0), (1, 0), path=np.zeros((3, 2))),
        lambda: initialize_playground((0, 0), (1, 0), pin_index=0),
    ],
)
def test_validation_rejects_malformed_inputs(call):
    with pytest.raises(ValueError):
        call()
