"""Checks for the opt-in finite, smooth wall model."""

from math import hypot

import numpy as np
import pytest

from path_planning_ode import soften_walls
from path_planning_ode.planners import plan
from path_planning_ode.terrain import PlannerConfig, TerrainField, TerrainScenario, evaluate_route
from path_planning_ode.terrain_generators import obstacle_detour_fixture, uniform_terrain_fixture


def _thin_wall_scenario() -> TerrainScenario:
    axis = np.arange(9.0)
    zeros = np.zeros((len(axis), len(axis)))
    wall = {
        "type": "Polygon",
        "coordinates": [[[3.45, 1.0], [3.55, 1.0], [3.55, 7.0], [3.45, 7.0], [3.45, 1.0]]],
    }
    return TerrainScenario(
        name="sub-cell wall",
        bounds_m=(0.0, 0.0, 8.0, 8.0),
        start_m=(0.5, 4.0),
        goal_m=(7.5, 4.0),
        field_x_m=axis,
        field_y_m=axis,
        elevation_m=zeros,
        log_slowness=zeros,
        barriers_geojson=(wall,),
        provenance={"source": "soft-wall test"},
        metadata={"existing": {"preserved": True}},
    )


def test_transform_preserves_scenario_and_records_reversible_model_metadata():
    original = obstacle_detour_fixture()
    original_dict = original.to_dict()
    transformed = soften_walls(original)

    assert original.to_dict() == original_dict
    assert transformed is not original
    assert transformed.name == original.name
    assert transformed.bounds_m == original.bounds_m
    assert transformed.start_m == original.start_m
    assert transformed.goal_m == original.goal_m
    assert transformed.elevation_m == original.elevation_m
    assert transformed.provenance == original.provenance
    assert transformed.barriers_geojson == ()
    assert transformed.metadata["kind"] == original.metadata["kind"]

    expected_width = 2 * hypot(15.625, 15.625)
    assert transformed.metadata["soft_walls"] == {
        "version": 1,
        "model": "finite_smooth_high_cost",
        "base_scenario_hash": original.scenario_hash,
        "multiplier": 100.0,
        "transition_width_m": expected_width,
        "transition_profile": "outward_quintic_smootherstep",
        "geometry_geojson": original.to_dict()["barriers_geojson"],
        "source_grid_shape": [65, 65],
    }
    assert transformed.scenario_hash != original.scenario_hash
    with pytest.raises(ValueError, match="already has"):
        soften_walls(transformed)


def test_crossing_is_feasible_but_charged_and_field_has_transition_derivatives():
    original = obstacle_detour_fixture()
    transformed = soften_walls(original, multiplier=100)
    base_field = TerrainField(original)
    field = TerrainField(transformed)

    points = np.array([[500.0, 500.0], [0.0, 0.0], [430.0, 500.0], [400.0, 500.0]])
    costs = field.cost(points)
    assert costs[0] == pytest.approx(100 * base_field.cost(points[0]), rel=1e-12)
    assert costs[1] == pytest.approx(base_field.cost(points[1]), rel=1e-12)
    assert base_field.cost(points[2]) < costs[2] < costs[0]
    assert abs(field.gradient(points[2])[0]) > 1.0
    assert np.isfinite(field.gradient(points)).all()
    assert np.isfinite(field.hessian(points)).all()
    assert np.all(costs > 0)

    crossing = np.asarray([original.start_m, original.goal_m])
    hard_evaluation = evaluate_route(original, crossing, profile_samples=9)
    soft_evaluation = evaluate_route(transformed, crossing, profile_samples=9)
    assert not hard_evaluation.feasible
    assert hard_evaluation.violations == ("barrier_collision",)
    assert soft_evaluation.feasible
    assert soft_evaluation.violations == ()
    assert soft_evaluation.minimum_clearance_m is None
    assert soft_evaluation.cost_s > 10 * hard_evaluation.cost_s


def test_sub_cell_wall_is_resolved_by_fixed_physical_transition():
    original = _thin_wall_scenario()
    # No source-grid x coordinate lies inside this 0.1 m wall.
    assert not np.any(
        (np.asarray(original.field_x_m) >= 3.45) & (np.asarray(original.field_x_m) <= 3.55)
    )

    transformed = soften_walls(original)
    field = TerrainField(transformed)
    source_cost = np.exp(np.asarray(transformed.log_slowness))
    assert transformed.metadata["soft_walls"]["transition_width_m"] == pytest.approx(2 * np.sqrt(2))
    assert source_cost[4, 3] > 50
    assert source_cost[4, 4] > 50
    assert field.cost((3.5, 4.0)) > 50
    assert abs(field.gradient((3.0, 4.0))[0]) > 1

    base_crossing = evaluate_route(original, [original.start_m, original.goal_m])
    soft_crossing = evaluate_route(transformed, [original.start_m, original.goal_m])
    assert not base_crossing.feasible
    assert soft_crossing.feasible
    assert soft_crossing.cost_s > 10 * base_crossing.cost_s


@pytest.mark.parametrize("method", ["euler_lagrange", "slsqp", "fast_marching"])
def test_every_planner_uses_soft_wall_as_shared_cost_without_hidden_mask(method):
    scenario = soften_walls(obstacle_detour_fixture())
    result = plan(
        scenario,
        PlannerConfig(
            method=method,
            initialization="straight",
            interior_points=4,
            reference_grid_size=33,
            tolerance=1e-6,
            max_iterations=100,
            time_limit_s=10.0,
            profile_samples=5,
        ),
    )
    assert result.solver_success
    assert result.feasible
    assert result.scenario_hash == scenario.scenario_hash
    assert result.evaluation is not None
    assert "barrier_collision" not in result.evaluation.violations
    assert result.evaluated_cost_s == result.evaluation.cost_s
    if method == "fast_marching":
        assert result.diagnostics["masked_count"] == 0


@pytest.mark.parametrize("multiplier", [True, 1.0, 0.0, -5.0, np.nan, np.inf, 1_000_001])
def test_invalid_multiplier_is_rejected(multiplier):
    with pytest.raises(ValueError, match="multiplier"):
        soften_walls(obstacle_detour_fixture(), multiplier=multiplier)


@pytest.mark.parametrize("width", [True, 0.0, -1.0, np.nan, np.inf, 20.0])
def test_unresolved_or_invalid_transition_width_is_rejected(width):
    with pytest.raises(ValueError, match="transition_width_m"):
        soften_walls(obstacle_detour_fixture(), transition_width_m=width)


def test_empty_geometry_remains_an_unchanged_finite_field_with_model_label():
    original = uniform_terrain_fixture()
    transformed = soften_walls(original, multiplier=25.0)
    assert transformed.barriers_geojson == ()
    assert transformed.log_slowness == original.log_slowness
    assert transformed.metadata["soft_walls"]["geometry_geojson"] == []
    assert np.isfinite(TerrainField(transformed).cost((500.0, 500.0)))


def test_type_and_finite_positive_source_cost_validation():
    with pytest.raises(TypeError, match="TerrainScenario"):
        soften_walls(object())  # type: ignore[arg-type]

    original = _thin_wall_scenario()
    for log_slowness in (709.0, -1_000.0):
        invalid_cost = TerrainScenario(
            **{
                **original.to_dict(),
                "log_slowness": np.full((9, 9), log_slowness),
            }
        )
        with pytest.raises(ValueError, match="finite and positive"):
            soften_walls(invalid_cost)
