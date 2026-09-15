"""Independent regression checks for domain-boundary numerical handling."""

import math

import numpy as np
import pytest

from path_planning_ode.local_planners import solve_euler_lagrange
from path_planning_ode.planners import plan
from path_planning_ode.terrain import PlannerConfig, TerrainScenario, evaluate_route
from path_planning_ode.terrain_generators import ridge_pass_terrain


def _constant_scenario() -> TerrainScenario:
    axis = np.linspace(0.0, 10.0, 5)
    values = np.full((5, 5), math.log(2.0))
    return TerrainScenario(
        name="boundary oracle",
        bounds_m=(0.0, 0.0, 10.0, 10.0),
        start_m=(0.0, 0.0),
        goal_m=(10.0, 10.0),
        field_x_m=axis,
        field_y_m=axis,
        elevation_m=np.zeros_like(values),
        log_slowness=values,
    )


def test_valid_boundary_route_matches_constant_cost_oracle_without_mutation():
    scenario = _constant_scenario()
    route = np.array(
        [
            (0.0, 0.0),
            (10.0, 0.0),
            (0.0, 10.0),
            (0.0, 0.0),
            (10.0, 10.0),
            (10.0, 0.0),
            (10.0, 10.0),
        ]
    )
    submitted = route.copy()

    result = evaluate_route(scenario, route, profile_samples=257, quadrature_order=32)

    expected_length = 40.0 + 2.0 * math.sqrt(200.0)
    assert result.feasible
    assert result.violations == ()
    assert result.length_m == pytest.approx(expected_length, rel=1e-14)
    assert result.cost_s == pytest.approx(2.0 * expected_length, rel=1e-13)
    assert result.accumulated_cost_s[-1] == pytest.approx(result.cost_s, rel=1e-15)
    assert min(result.distance_m) == 0.0
    assert max(result.distance_m) == pytest.approx(expected_length)
    np.testing.assert_array_equal(route, submitted)


def test_point_genuinely_beyond_inclusive_domain_remains_invalid():
    scenario = _constant_scenario()
    route = np.array([(0.0, 0.0), (10.0 + 1e-12, 0.0), (10.0, 10.0)])

    result = evaluate_route(scenario, route, profile_samples=257, quadrature_order=32)

    assert not result.feasible
    assert "outside_domain" in result.violations
    assert result.cost_s is None
    assert result.distance_m == ()


def test_frozen_ridge_boundary_case_returns_a_classified_candidate():
    scenario = ridge_pass_terrain(seed=0, contrast=1.0, barriers=True)
    config = PlannerConfig(
        method="slsqp",
        initialization="arc_left",
        interior_points=32,
        reference_grid_size=513,
        tolerance=1e-6,
        max_iterations=200,
        time_limit_s=60.0,
        profile_samples=65,
        options={"quadrature_order": 8},
    )

    result = plan(scenario, config)

    assert result.route_m is not None
    assert result.evaluation is not None
    assert result.evaluated_cost_s is not None
    assert result.termination_reason != "evaluation_failed"
    assert "evaluation_error" not in result.diagnostics
    route = np.asarray(result.route_m)
    xmin, ymin, xmax, ymax = scenario.bounds_m
    assert np.all((route[:, 0] >= xmin) & (route[:, 0] <= xmax))
    assert np.all((route[:, 1] >= ymin) & (route[:, 1] <= ymax))


def test_evaluator_exception_cannot_be_reported_as_feasible_or_successful(monkeypatch):
    import path_planning_ode.local_planners as local_planners

    def fail(*args, **kwargs):
        raise FloatingPointError("independent forced evaluator failure")

    monkeypatch.setattr(local_planners, "evaluate_route", fail)
    scenario = _constant_scenario()
    result = solve_euler_lagrange(
        scenario,
        PlannerConfig(
            method="euler_lagrange",
            initialization="straight",
            interior_points=4,
            max_iterations=5,
            time_limit_s=5.0,
        ),
    )

    assert result.termination_reason == "evaluation_failed"
    assert not result.feasible
    assert not result.solver_success
    assert result.evaluation is None
    assert result.evaluated_cost_s is None
    assert result.route_m is not None
    assert result.diagnostics["evaluation_error"]["exception_type"] == "FloatingPointError"
