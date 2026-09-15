from dataclasses import replace
from math import erf, pi, sqrt

import numpy as np
import pytest
import sympy as sp

from path_planning_ode import (
    Obstacle,
    Scene,
    SolverOptions,
    cost_field,
    energy,
    initialize,
    jacobian,
    ode,
    presets,
    residual,
    solve,
    step,
    weighted_distance,
)


def test_symbolic_euler_lagrange():
    x, y, u, v, ax, ay = sp.symbols("x y u v ax ay")
    c = 1 + 3 * sp.exp(-((x - 2) ** 2 + (y - 3) ** 2) / 4)
    lagrangian = c**2 * (u * u + v * v)
    equations = []
    for q, velocity in [(x, u), (y, v)]:
        momentum = sp.diff(lagrangian, velocity)
        equations.append(
            sum(sp.diff(momentum, a) * b for a, b in [(x, u), (y, v), (u, ax), (v, ay)])
            - sp.diff(lagrangian, q)
        )
    solution = sp.solve(equations, [ax, ay])
    function = sp.lambdify((x, y, u, v), [solution[ax], solution[ay]], "numpy")
    np.testing.assert_allclose(
        ode(np.array([1.0, 2.0]), np.array([3.0, 4.0]), (Obstacle(2, 3, 3, 2),)),
        function(1.0, 2.0, 3.0, 4.0),
    )


def test_ode_satisfies_direct_weighted_length_equations():
    """Independently start from c|v|, without using the energy derivation."""
    x, y, u, v, ax, ay = sp.symbols("x y u v ax ay", real=True)
    c = 1 + 3 * sp.exp(-((x - 2) ** 2 + (y - 3) ** 2) / 4)
    lagrangian = c * sp.sqrt(u * u + v * v)
    equations = []
    for q, velocity in [(x, u), (y, v)]:
        momentum = sp.diff(lagrangian, velocity)
        equations.append(
            sum(sp.diff(momentum, a) * b for a, b in [(x, u), (y, v), (u, ax), (v, ay)])
            - sp.diff(lagrangian, q)
        )
    evaluate = sp.lambdify((x, y, u, v, ax, ay), equations, "numpy")
    for position, velocity in [([1, 2], [3, 4]), ([2, 4], [-2, 1]), ([0, 1], [0.2, -0.4])]:
        acceleration = ode(np.array(position), np.array(velocity), (Obstacle(2, 3, 3, 2),))
        np.testing.assert_allclose(evaluate(*position, *velocity, *acceleration), 0, atol=1e-13)


def test_weighted_distance_counts_a_thin_hill_between_vertices():
    scene = Scene(obstacles=(Obstacle(0, 0, 9, 0.1),))
    path = np.array([[-2, 0], [2, 0]])
    expected = 4 + 9 * 0.1 * sqrt(pi) * erf(20)
    assert weighted_distance(path, scene) == pytest.approx(expected, rel=1e-14)


def test_weighted_distance_is_independent_of_straight_segment_sampling():
    scene = presets()["asymmetric"]
    path = np.array([[-2, -1], [3, 4], [10, 12]], dtype=float)
    refined = np.concatenate(
        [
            a + np.array([0, 0.001, 0.15, 0.8, 1])[:, None] * (b - a)
            for a, b in zip(path[:-1], path[1:], strict=True)
        ]
    )
    expected = weighted_distance(path, scene)
    assert weighted_distance(refined, scene) == pytest.approx(expected, rel=1e-14)
    assert weighted_distance(path[::-1], scene) == pytest.approx(expected, rel=1e-14)
    assert weighted_distance(np.repeat(path, 2, axis=0), scene) == pytest.approx(expected)
    assert weighted_distance(np.array([[3, 4], [3, 4]]), scene) == 0


def test_weighted_distance_matches_independent_numerical_quadrature():
    scene = presets()["asymmetric"]
    path = np.array([[-4, 7], [2, 2], [4, 5], [12, 12]])
    # Dense trapezoidal integration, independent of the analytic erf formula.
    t = np.linspace(0, 1, 20001)
    expected = 0
    for a, b in zip(path[:-1], path[1:], strict=True):
        field = cost_field(a + t[:, None] * (b - a), scene.obstacles)
        expected += np.linalg.norm(b - a) * np.trapezoid(field, t)
    assert weighted_distance(path, scene) == pytest.approx(expected, rel=1e-8)


def test_energy_uses_squared_cost_and_reports_cost_separately():
    scene = Scene(obstacles=(Obstacle(0, 0, 2, 1),))
    path = np.array([[-1, 0], [1, 0]])
    assert energy(path, scene) == pytest.approx(3**2 * 2**2)
    state = initialize(scene)
    assert state.cost == weighted_distance(state.path, scene)
    assert state.to_dict()["cost"] == state.cost


@pytest.mark.parametrize("end", [(8, 0), (0, 8), (8, 8)])
def test_arcs_are_distinct_for_any_endpoint_direction(end):
    scene = Scene(start=(0, 0), end=end)
    straight, right, left = [initialize(scene, g).path for g in scene.guesses]
    assert not np.allclose(straight, right)
    assert not np.allclose(straight, left)
    np.testing.assert_allclose((right + left) / 2, straight, atol=1e-14)


def test_longer_detour_can_have_lower_weighted_cost():
    scene = presets()["central"]
    final = solve(scene).final
    direct, detour = final["straight"], final["bend-x"]
    assert direct.status == detour.status == "converged"
    assert detour.length > direct.length
    assert detour.cost < direct.cost * 0.6
    for state in final.values():
        assert state.cost >= state.length
        assert state.cost == weighted_distance(state.path, scene)


@pytest.mark.parametrize("n", [1, 2, 15])
def test_jacobian_matches_finite_difference(n):
    scene = replace(presets()["asymmetric"], options=SolverOptions(interior_points=n))
    path = initialize(scene, "bend-x").path
    numerical = np.zeros((2 * n, 2 * n))
    eps = 1e-5
    for i in range(2 * n):
        plus, minus = path.copy(), path.copy()
        plus[1 + i // 2, i % 2] += eps
        minus[1 + i // 2, i % 2] -= eps
        numerical[:, i] = (residual(plus, scene) - residual(minus, scene)) / (2 * eps)
    np.testing.assert_allclose(jacobian(path, scene), numerical, rtol=1e-6, atol=1e-6)


def test_empty_field_recovers_line_and_fixed_endpoints():
    scene = Scene()
    result = solve(scene)
    expected = np.linspace(scene.start, scene.end, 32)
    for history in result.histories.values():
        assert history[-1].status == "converged"
        assert history[-1].iteration <= 1
        np.testing.assert_allclose(history[-1].path, expected, atol=1e-10)
        for state in history:
            np.testing.assert_array_equal(state.path[[0, -1]], [scene.start, scene.end])


def test_symmetric_initializations_stay_symmetric():
    result = solve(presets()["central"])
    np.testing.assert_allclose(
        result.final["bend-x"].path, result.final["bend-y"].path[:, ::-1], atol=1e-7
    )


def test_mesh_refinement():
    base = Scene(
        start=(0.0, 0.0), end=(6.0, 0.0), obstacles=(Obstacle(3, 1, 0.5, 2),), guesses=("straight",)
    )
    solutions = []
    for n in (15, 31, 63):
        result = solve(replace(base, options=SolverOptions(interior_points=n))).final["straight"]
        assert result.status == "converged"
        solutions.append(result.path)
    coarse_error = np.linalg.norm(solutions[0] - solutions[1][::2])
    fine_error = np.linalg.norm(solutions[1] - solutions[2][::2]) / np.sqrt(2)
    assert fine_error < coarse_error * 0.4


@pytest.mark.parametrize("name", list(presets()))
def test_presets_are_deterministic_and_damping_reduces_residual(name):
    scene = presets()[name]
    first, second = solve(scene), solve(Scene.from_dict(scene.to_dict()))
    for guess, history in first.histories.items():
        assert history[-1].status != "running"
        assert np.isfinite(history[-1].path).all()
        norms = [s.residual_norm for s in history]
        assert np.all(np.diff(norms) <= 1e-10)
        np.testing.assert_allclose(history[-1].path, second.final[guess].path)


def test_failure_modes(monkeypatch):
    scene = Scene()
    initial = initialize(scene, "bend-x")

    def singular(*args):
        raise np.linalg.LinAlgError()

    monkeypatch.setattr(np.linalg, "solve", singular)
    assert step(scene, initial).status == "singular"
    monkeypatch.setattr(np.linalg, "solve", lambda *args: np.full(60, np.nan))
    assert step(scene, initial).status == "nonfinite"
    monkeypatch.setattr(np.linalg, "solve", lambda *args: np.zeros(60))
    assert step(scene, initial).status == "stagnated"


def test_iteration_limit_and_terminal_state():
    scene = replace(presets()["challenge"], options=SolverOptions(max_iterations=1))
    state = step(scene, initialize(scene))
    assert state.status == "iteration_limit"
    assert step(scene, state) is state


@pytest.mark.parametrize("kwargs", [{"width": 0}, {"weight": -1}, {"x": float("nan")}])
def test_invalid_obstacles(kwargs):
    with pytest.raises(ValueError):
        Obstacle(**({"x": 0, "y": 0} | kwargs))


def test_scene_validation():
    with pytest.raises(ValueError):
        Scene.from_dict({"version": 2})
    with pytest.raises(ValueError):
        SolverOptions(interior_points=1.5)
    with pytest.raises(ValueError):
        Scene(guesses=("unknown",))
