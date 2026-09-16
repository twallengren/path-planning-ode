"""Independent numerical audit of the incremental path playground."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from path_planning_ode.core import (
    Obstacle,
    Scene,
    SolverOptions,
    cost_field,
    jacobian,
    residual,
    weighted_distance,
)
from path_planning_ode.playground import (
    BrushStroke,
    PlaygroundOptions,
    advance_playground,
    evaluate_playground,
    initialize_playground,
    playground_energy_gradient,
    set_playground_pin,
    strokes_to_obstacles,
)


def _adaptive_energy(path, obstacles):
    """High-accuracy oracle derived directly from integral c²|q'|² dt."""
    path = np.asarray(path, dtype=float)
    intervals = len(path) - 1
    total = 0.0
    for start, end in zip(path[:-1], path[1:], strict=True):
        delta = end - start

        def integrand(parameter):
            value = cost_field(start + parameter * delta, obstacles)
            return float(value * value)

        critical = [
            min(
                1.0,
                max(
                    0.0,
                    float((np.asarray([obstacle.x, obstacle.y]) - start) @ delta / (delta @ delta)),
                ),
            )
            for obstacle in obstacles
        ]
        integral = quad(
            integrand,
            0.0,
            1.0,
            epsabs=1e-11,
            epsrel=1e-11,
            limit=300,
            points=critical,
        )[0]
        total += intervals * float(delta @ delta) * integral
    return total


def _scene(path, obstacles):
    return Scene(
        start=tuple(path[0]),
        end=tuple(path[-1]),
        obstacles=tuple(obstacles),
        options=SolverOptions(interior_points=len(path) - 2),
        guesses=("straight",),
    )


def test_stroke_conversion_is_sampling_invariant_and_cap_is_strict():
    sparse = BrushStroke("line", ((-2.0, 0.0), (2.0, 0.0)), 0.5, 3.0)
    dense = BrushStroke(
        "line",
        tuple((float(x), 0.0) for x in np.linspace(-2, 2, 101)),
        0.5,
        3.0,
    )
    assert strokes_to_obstacles([sparse]) == strokes_to_obstacles([dense])
    with pytest.raises(ValueError, match="requires .* maximum"):
        strokes_to_obstacles([sparse], max_gaussians=3)


def test_brush_strength_is_nominal_multiplicative_cost():
    click = BrushStroke("click", ((0.0, 0.0),), 0.4, 10.0)
    assert float(cost_field(np.array([0.0, 0.0]), strokes_to_obstacles([click]))) == pytest.approx(
        10.0
    )

    line = BrushStroke("line", ((-10.0, 0.0), (10.0, 0.0)), 1.0, 10.0)
    represented = strokes_to_obstacles([line])
    assert float(cost_field(np.array([0.0, 0.0]), represented)) == pytest.approx(10.0, rel=2e-10)

    with pytest.raises(ValueError, match="strength at least one"):
        BrushStroke("invalid", ((0.0, 0.0),), 1.0, 0.9)


def test_iteration_budget_has_a_hard_2000_step_ceiling():
    assert PlaygroundOptions(max_iterations=2000).max_iterations == 2000
    with pytest.raises(ValueError, match="at most 2000"):
        PlaygroundOptions(max_iterations=2001)


def test_narrow_gaussian_energy_and_cost_match_independent_oracles():
    path = np.array([[-5.0, 0.1], [4.7, -0.15], [5.0, 2.0]])
    obstacles = (Obstacle(0.137, 0.0, weight=12.0, width=0.035),)
    energy, _ = playground_energy_gradient(path, obstacles)
    metrics = evaluate_playground(path, obstacles)
    assert energy == pytest.approx(_adaptive_energy(path, obstacles), rel=2e-10)
    assert metrics.energy == pytest.approx(energy, rel=1e-14)
    assert metrics.route_cost == pytest.approx(weighted_distance(path, _scene(path, obstacles)))


def test_analytic_energy_gradient_matches_fixed_quadrature_finite_difference():
    path = np.array([[-2.0, -1.0], [-0.7, 0.8], [0.4, -0.2], [2.1, 1.3]])
    obstacles = (
        Obstacle(-0.1, 0.2, weight=2.5, width=0.45),
        Obstacle(1.2, 0.7, weight=1.2, width=0.8),
    )
    panels = (7, 5, 6)
    _, analytic = playground_energy_gradient(path, obstacles, quadrature_panels=panels)
    numerical = np.empty_like(path)
    epsilon = 2e-6
    for vertex in range(len(path)):
        for axis in range(2):
            plus, minus = path.copy(), path.copy()
            plus[vertex, axis] += epsilon
            minus[vertex, axis] -= epsilon
            plus_value = playground_energy_gradient(plus, obstacles, quadrature_panels=panels)[0]
            minus_value = playground_energy_gradient(minus, obstacles, quadrature_panels=panels)[0]
            numerical[vertex, axis] = (plus_value - minus_value) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-6)


def test_dragged_long_segments_get_new_panels_and_do_not_alias_narrow_field():
    obstacles = (Obstacle(0.11, 0.03, weight=15.0, width=0.025),)
    short = np.array([[-1.0, 0.0], [-0.2, 0.1], [0.2, -0.1], [1.0, 0.0]])
    dragged = short.copy()
    dragged[1] = [4.5, 0.04]
    options = PlaygroundOptions(interior_points=2, quadrature_order=8, panel_widths=1.0)
    initial = initialize_playground(short[0], short[-1], obstacles, path=short, options=options)
    updated = initialize_playground(
        dragged[0], dragged[-1], obstacles, path=dragged, options=options
    )
    assert updated.quadrature_panels[0] > initial.quadrature_panels[0]
    assert updated.metrics.energy == pytest.approx(_adaptive_energy(dragged, obstacles), rel=2e-10)


def test_pin_is_exactly_held_release_preserves_path_and_descent_lowers_energy():
    obstacles = (Obstacle(0.0, 0.0, weight=5.0, width=0.7),)
    path = np.column_stack((np.linspace(-3, 3, 10), 0.5 * np.sin(np.linspace(0, np.pi, 10))))
    pin_position = np.array([-0.25, 1.4])
    options = PlaygroundOptions(interior_points=8, method="descent", tolerance=1e-10)
    state = initialize_playground(
        path[0],
        path[-1],
        obstacles,
        path=path,
        options=options,
        pin_index=4,
        pin_position=pin_position,
    )
    previous = _adaptive_energy(state.path, obstacles)
    for _ in range(8):
        state = advance_playground(state)
        assert state.path[0].tolist() == path[0].tolist()
        assert state.path[-1].tolist() == path[-1].tolist()
        np.testing.assert_array_equal(state.path[4], pin_position)
        current = _adaptive_energy(state.path, obstacles)
        assert current <= previous + 2e-10 * max(1.0, previous)
        previous = current
        if state.status != "running":
            break

    held_path = state.path.copy()
    released = set_playground_pin(state, None)
    np.testing.assert_array_equal(released.path, held_path)
    assert released.pin_index is None
    advanced = advance_playground(released)
    assert not np.array_equal(advanced.path[4], held_path[4])


def test_newton_step_uses_core_ode_with_pin_rows_and_columns_removed():
    obstacles = (Obstacle(0.3, -0.2, weight=1.5, width=1.1),)
    path = np.column_stack((np.linspace(-2, 3, 8), 0.7 * np.sin(np.linspace(0, np.pi, 8))))
    options = PlaygroundOptions(interior_points=6, method="newton", tolerance=1e-12)
    state = initialize_playground(
        path[0], path[-1], obstacles, path=path, options=options, pin_index=3
    )
    scene = _scene(state.path, obstacles)
    free_vertices = np.array([1, 2, 4, 5, 6])
    scalar = np.ravel(np.column_stack((2 * (free_vertices - 1), 2 * (free_vertices - 1) + 1)))
    expected = np.linalg.solve(
        jacobian(state.path, scene)[np.ix_(scalar, scalar)],
        -residual(state.path, scene)[scalar],
    ).reshape(-1, 2)
    advanced = advance_playground(state)
    delta = advanced.path[free_vertices] - state.path[free_vertices]
    ratios = delta[np.abs(expected) > 1e-10] / expected[np.abs(expected) > 1e-10]
    np.testing.assert_allclose(ratios, ratios[0], rtol=2e-10, atol=2e-10)
    # Reconstructing delta from (path + direction) - path can exceed a unit
    # step by a few ulps on a different BLAS/platform combination.
    assert 0 < ratios[0] <= 1 + 2e-14
    np.testing.assert_array_equal(advanced.path[3], state.path[3])
    assert advanced.metrics.ode_residual_norm < state.metrics.ode_residual_norm


def test_reported_free_gradient_and_ode_residual_exclude_fixed_vertices():
    obstacles = (Obstacle(0.4, 0.1, weight=2.0, width=0.9),)
    path = np.array([[-2.0, 0.0], [-1.0, 0.8], [0.0, 1.1], [1.0, 0.4], [2.0, 0.0]])
    options = PlaygroundOptions(interior_points=3)
    state = initialize_playground(
        path[0], path[-1], obstacles, path=path, options=options, pin_index=2
    )
    _, gradient = playground_energy_gradient(
        state.path,
        obstacles,
        quadrature_panels=state.quadrature_panels,
        quadrature_order=options.quadrature_order,
    )
    free = np.array([1, 3])
    expected_gradient = np.linalg.norm(gradient[free]) / np.sqrt(2 * len(free))
    ode = residual(state.path, _scene(state.path, obstacles)).reshape(-1, 2)
    expected_ode = np.linalg.norm(ode[free - 1]) / np.sqrt(2 * len(free))
    assert state.metrics.free_gradient_norm == pytest.approx(expected_gradient)
    assert state.metrics.ode_residual_norm == pytest.approx(expected_ode)
    assert state.metrics.energy != pytest.approx(state.metrics.route_cost)


def test_symmetric_stationary_candidates_can_have_different_route_costs():
    obstacle = (Obstacle(0.0, 0.0, weight=6.0, width=1.0),)
    count = 32
    parameter = np.linspace(0, 1, count + 2)
    options = PlaygroundOptions(
        interior_points=count, method="newton", tolerance=1e-8, max_iterations=100
    )
    solutions = []
    for amplitude in (1.5, -1.5, 0.0):
        path = np.column_stack((-4 + 8 * parameter, amplitude * np.sin(np.pi * parameter)))
        state = initialize_playground(path[0], path[-1], obstacle, path=path, options=options)
        solutions.append(advance_playground(state, iterations=100))
    upper, lower, direct = solutions
    assert all(state.status == "converged" for state in solutions)
    np.testing.assert_allclose(upper.path[:, 0], -upper.path[::-1, 0], atol=2e-12)
    np.testing.assert_allclose(upper.path[:, 1], upper.path[::-1, 1], atol=2e-12)
    np.testing.assert_allclose(upper.path[:, 0], lower.path[:, 0], atol=2e-12)
    np.testing.assert_allclose(upper.path[:, 1], -lower.path[:, 1], atol=2e-12)
    assert upper.metrics.route_cost == pytest.approx(lower.metrics.route_cost, rel=1e-13)
    # Stationarity is not a global-minimum certificate: the direct stationary
    # route through the hill has a much larger independently measured cost.
    assert direct.metrics.route_cost > upper.metrics.route_cost * 1.5


def _run_newton(interior_points):
    t = np.linspace(0, 1, interior_points + 2)
    path = np.column_stack((-3 + 6 * t, 1.1 * np.sin(np.pi * t)))
    obstacles = (Obstacle(0.2, -0.15, weight=1.4, width=1.3),)
    options = PlaygroundOptions(
        interior_points=interior_points,
        method="newton",
        tolerance=2e-8,
        max_iterations=100,
    )
    state = initialize_playground(path[0], path[-1], obstacles, path=path, options=options)
    state = advance_playground(state, iterations=100)
    return state, obstacles


def _run_descent(interior_points):
    parameter = np.linspace(0, 1, interior_points + 2)
    path = np.column_stack((-3 + 6 * parameter, 1.1 * np.sin(np.pi * parameter)))
    obstacles = (Obstacle(0.2, -0.15, weight=1.4, width=1.3),)
    options = PlaygroundOptions(
        interior_points=interior_points,
        method="descent",
        tolerance=1e-6,
        max_iterations=2000,
    )
    state = initialize_playground(path[0], path[-1], obstacles, path=path, options=options)
    return advance_playground(state, iterations=2000), obstacles


def test_smooth_fixture_refines_ode_path_and_weighted_speed():
    solutions = [_run_newton(count) for count in (32, 64, 128)]
    common = np.linspace(0, 1, 401)
    interpolated = []
    speed_variation = []
    for state, obstacles in solutions:
        assert state.status == "converged"
        parameter = np.linspace(0, 1, len(state.path))
        interpolated.append(
            np.column_stack(
                [np.interp(common, parameter, state.path[:, axis]) for axis in range(2)]
            )
        )
        delta = np.diff(state.path, axis=0)
        midpoint = (state.path[:-1] + state.path[1:]) / 2
        weighted_speed = (
            cost_field(midpoint, obstacles) * np.linalg.norm(delta, axis=1) * (len(state.path) - 1)
        )
        speed_variation.append(np.std(weighted_speed) / np.mean(weighted_speed))
    coarse_difference = np.linalg.norm(interpolated[0] - interpolated[1])
    fine_difference = np.linalg.norm(interpolated[1] - interpolated[2])
    assert fine_difference < coarse_difference * 0.65
    assert speed_variation[2] < speed_variation[0]


def test_descent_stationarity_improves_ode_consistency_under_mesh_refinement():
    solutions = [_run_descent(count) for count in (32, 64, 128)]
    residuals = []
    speed_variation = []
    for state, obstacles in solutions:
        assert state.status == "converged"
        assert state.metrics.scaled_free_gradient_norm <= state.options.tolerance
        residuals.append(state.metrics.ode_residual_norm)
        delta = np.diff(state.path, axis=0)
        midpoint = (state.path[:-1] + state.path[1:]) / 2
        weighted_speed = (
            cost_field(midpoint, obstacles) * np.linalg.norm(delta, axis=1) * (len(state.path) - 1)
        )
        speed_variation.append(np.std(weighted_speed) / np.mean(weighted_speed))
    assert residuals[1] < residuals[0] * 0.35
    assert residuals[2] < residuals[1] * 0.35
    assert speed_variation[2] < speed_variation[0]
