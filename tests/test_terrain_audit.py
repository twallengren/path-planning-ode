"""Independent checks for the retained terrain field and route evaluator."""

from math import exp

import numpy as np
import pytest
from scipy.integrate import quad

from path_planning_ode import TerrainField, TerrainScenario, evaluate_route


def _scenario():
    x = np.linspace(-2, 3, 8)
    y = np.linspace(-2.5, 2, 7)
    xx, yy = np.meshgrid(x, y)
    log_cost = 0.15 + 0.08 * xx - 0.06 * yy + 0.025 * xx**2 + 0.018 * xx * yy
    return TerrainScenario(
        name="independent audit",
        bounds_m=(-2, -2.5, 3, 2),
        start_m=(-1.7, -1.8),
        goal_m=(2.6, 1.7),
        field_x_m=x,
        field_y_m=y,
        elevation_m=np.zeros_like(xx),
        log_slowness=log_cost,
    )


def test_field_derivatives_match_centered_differences():
    field = TerrainField(_scenario())
    points = np.array([[-1.1, -1.4], [-0.2, 0.3], [0.8, -0.7], [2.1, 1.1]])
    epsilon = 1e-5
    gradient = np.empty_like(points)
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = epsilon
        gradient[:, axis] = (field.cost(points + offset) - field.cost(points - offset)) / (
            2 * epsilon
        )
    np.testing.assert_allclose(field.gradient(points), gradient, rtol=3e-7, atol=3e-8)


def test_route_cost_matches_adaptive_quadrature():
    scenario = _scenario()
    field = TerrainField(scenario)
    route = np.array([scenario.start_m, (-0.9, 1.2), (0.4, -0.4), (1.7, 1.3), scenario.goal_m])
    expected = sum(
        quad(lambda t: float(field.cost(a + t * (b - a))) * np.linalg.norm(b - a), 0, 1)[0]
        for a, b in zip(route[:-1], route[1:], strict=True)
    )
    assert evaluate_route(scenario, route, field=field).cost_s == pytest.approx(expected, rel=1e-5)


def test_constant_field_straight_line_and_subdivision_are_invariant():
    coordinates = tuple(float(value) for value in np.linspace(0, 10, 5))
    value = exp(0.7)
    scenario = TerrainScenario(
        name="constant",
        bounds_m=(0, 0, 10, 10),
        start_m=(1, 2),
        goal_m=(9, 8),
        field_x_m=coordinates,
        field_y_m=coordinates,
        elevation_m=np.zeros((5, 5)),
        log_slowness=np.full((5, 5), np.log(value)),
    )
    direct = evaluate_route(scenario, [scenario.start_m, scenario.goal_m])
    subdivided = evaluate_route(scenario, np.linspace(scenario.start_m, scenario.goal_m, 37))
    assert direct.cost_s == pytest.approx(value * 10)
    assert subdivided.cost_s == pytest.approx(direct.cost_s, rel=2e-14)
