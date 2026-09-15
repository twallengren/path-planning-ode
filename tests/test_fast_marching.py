from dataclasses import replace

import numpy as np
import pytest

from path_planning_ode.fast_marching import (
    compute_arrival_time,
    extract_route,
    solve_fast_marching,
)
from path_planning_ode.terrain import PlannerConfig, TerrainScenario, uniform_terrain_fixture


def _uniform_scenario(
    *,
    bounds=(0.0, 0.0, 100.0, 100.0),
    start=(10.0, 50.0),
    goal=(90.0, 50.0),
    barriers=(),
):
    x = np.linspace(bounds[0], bounds[2], 5)
    y = np.linspace(bounds[1], bounds[3], 5)
    values = np.zeros((len(y), len(x)))
    return TerrainScenario(
        name="fast-marching-test",
        bounds_m=bounds,
        start_m=start,
        goal_m=goal,
        field_x_m=x,
        field_y_m=y,
        elevation_m=values,
        log_slowness=values,
        barriers_geojson=barriers,
    )


def _config(grid_size):
    return PlannerConfig(
        method="fast_marching",
        initialization="none",
        reference_grid_size=grid_size,
        profile_samples=8,
    )


def test_uniform_arrival_and_extracted_route_refine_toward_analytic_cost():
    scenario = uniform_terrain_fixture()
    coarse = solve_fast_marching(scenario, _config(33))
    fine = solve_fast_marching(scenario, _config(129))
    expected = np.linalg.norm(np.asarray(scenario.goal_m) - scenario.start_m)
    assert coarse.solver_success and fine.solver_success
    assert fine.feasible
    assert abs(fine.evaluated_cost_s - expected) < abs(coarse.evaluated_cost_s - expected)
    assert fine.evaluated_cost_s == pytest.approx(expected, rel=1e-3)


def test_arrival_is_first_order_eikonal_not_graph_distance():
    scenario = _uniform_scenario(start=(0, 0), goal=(100, 100))
    solution = compute_arrival_time(scenario, 65)
    # A four-neighbour graph would report 200 seconds at the opposite corner.
    assert solution.arrival_s[0, 0] < 160
    assert solution.arrival_s[0, 0] == pytest.approx(100 * np.sqrt(2), rel=0.03)


def test_conservative_mask_routes_around_full_segment_barrier():
    barrier = {
        "type": "Polygon",
        "coordinates": [[[45, 0], [55, 0], [55, 70], [45, 70], [45, 0]]],
    }
    scenario = _uniform_scenario(barriers=(barrier,))
    solution = compute_arrival_time(scenario, 65)
    route = extract_route(scenario, solution)
    result = solve_fast_marching(scenario, _config(65))
    assert solution.masked_count > 0
    assert np.max(route[:, 1]) > 70
    assert result.solver_success and result.feasible
    assert result.evaluation.minimum_clearance_m > 0
    assert result.evaluated_cost_s > 80


def test_disconnected_conservative_grid_is_reported_as_grid_failure():
    wall = {
        "type": "Polygon",
        "coordinates": [[[45, 0], [55, 0], [55, 100], [45, 100], [45, 0]]],
    }
    result = solve_fast_marching(_uniform_scenario(barriers=(wall,)), _config(65))
    assert not result.solver_success
    assert not result.feasible
    assert result.termination_reason == "unreachable_on_grid"
    assert "continuous" not in result.diagnostics["failure_detail"].lower()


def test_uniform_reversal_and_reflection_symmetry():
    forward_scenario = _uniform_scenario(start=(10, 30), goal=(90, 70))
    reverse_scenario = replace(
        forward_scenario,
        name="reverse",
        start_m=forward_scenario.goal_m,
        goal_m=forward_scenario.start_m,
    )
    reflected_scenario = replace(
        forward_scenario,
        name="reflected",
        start_m=(10, 70),
        goal_m=(90, 30),
    )
    costs = [
        solve_fast_marching(scenario, _config(65)).evaluated_cost_s
        for scenario in (forward_scenario, reverse_scenario, reflected_scenario)
    ]
    assert costs[0] == pytest.approx(costs[1], rel=2e-3)
    assert costs[0] == pytest.approx(costs[2], rel=2e-3)


def test_non_square_physical_bounds_use_distinct_grid_spacing():
    scenario = _uniform_scenario(bounds=(0, 0, 200, 50), start=(10, 10), goal=(190, 40))
    result = solve_fast_marching(scenario, _config(65))
    expected = np.linalg.norm(np.asarray(scenario.goal_m) - scenario.start_m)
    assert result.solver_success
    assert result.evaluated_cost_s == pytest.approx(expected, rel=0.01)
    assert result.diagnostics["grid_spacing_m"] == [3.125, 0.78125]


def test_timeout_and_dense_arrival_opt_in_are_serializable():
    timeout = PlannerConfig(method="fast_marching", reference_grid_size=65, time_limit_s=1e-12)
    failed = solve_fast_marching(uniform_terrain_fixture(), timeout)
    assert failed.termination_reason == "timeout"
    assert failed.route_m is None

    included = solve_fast_marching(
        uniform_terrain_fixture(),
        PlannerConfig(
            method="fast_marching",
            reference_grid_size=17,
            profile_samples=2,
            options={"include_arrival": True},
        ),
    )
    diagnostics = included.to_dict()["diagnostics"]
    assert len(diagnostics["arrival_time_s"]) == 17
    assert len(diagnostics["conservative_mask"]) == 17
