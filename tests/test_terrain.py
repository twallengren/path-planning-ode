import json
import subprocess
import sys

import numpy as np
import pytest

from path_planning_ode import (
    PlannerConfig,
    PlannerResult,
    TerrainField,
    TerrainScenario,
    evaluate_route,
    uniform_terrain_fixture,
)


def test_root_import_keeps_optional_numerical_stack_lazy():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import path_planning_ode as package; "
                "assert 'scipy' not in sys.modules; "
                "assert 'shapely' not in sys.modules; "
                "assert 'plan' in package.__all__; "
                "assert 'synthetic_terrain' in package.__all__; "
                "assert 'seed_bank' in package.__all__"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""


def test_root_lazy_exports_cover_planning_generators_and_seeds():
    from path_planning_ode import (
        SeedResult,
        plan,
        ridge_pass_terrain,
        scene_to_terrain_scenario,
        seed_bank,
        synthetic_terrain,
    )

    assert callable(plan)
    assert callable(scene_to_terrain_scenario)
    assert callable(ridge_pass_terrain)
    assert callable(synthetic_terrain)
    assert callable(seed_bank)
    assert SeedResult.__name__ == "SeedResult"


def test_uniform_fixture_round_trip_hash_and_route_evaluation():
    scenario = uniform_terrain_fixture()
    restored = TerrainScenario.from_dict(json.loads(json.dumps(scenario.to_dict())))
    assert restored == scenario
    assert restored.scenario_hash == scenario.scenario_hash

    route = np.array([scenario.start_m, scenario.goal_m])
    evaluation = evaluate_route(scenario, route)
    assert evaluation.feasible
    assert evaluation.violations == ()
    assert evaluation.minimum_clearance_m is None
    assert evaluation.length_m == pytest.approx(100.0)
    assert evaluation.cost_s == pytest.approx(100.0)
    assert evaluation.accumulated_cost_s[-1] == pytest.approx(evaluation.cost_s)
    assert evaluation.distance_m[-1] == pytest.approx(evaluation.length_m)


def test_log_spline_chain_rule_gradient_and_hessian():
    x = np.linspace(0, 4, 5)
    y = np.linspace(0, 3, 4)
    xx, yy = np.meshgrid(x, y)
    log_c = 0.02 * xx**2 + 0.03 * xx * yy - 0.01 * yy**2 + 0.2
    scenario = TerrainScenario(
        name="quadratic-log-field",
        bounds_m=(0, 0, 4, 3),
        start_m=(0, 0),
        goal_m=(4, 3),
        field_x_m=x,
        field_y_m=y,
        elevation_m=np.zeros_like(log_c),
        log_slowness=log_c,
    )
    field = TerrainField(scenario)
    points = np.array([[0.7, 1.1], [2.2, 0.8], [3.5, 2.4]])
    px, py = points.T
    expected_log = 0.02 * px**2 + 0.03 * px * py - 0.01 * py**2 + 0.2
    expected_c = np.exp(expected_log)
    grad_log = np.stack([0.04 * px + 0.03 * py, 0.03 * px - 0.02 * py], axis=-1)
    hess_log = np.array([[0.04, 0.03], [0.03, -0.02]])
    expected_hess = expected_c[:, None, None] * (
        hess_log + grad_log[:, :, None] * grad_log[:, None, :]
    )
    np.testing.assert_allclose(field.cost(points), expected_c, rtol=1e-12)
    np.testing.assert_allclose(field.gradient(points), expected_c[:, None] * grad_log, rtol=1e-11)
    np.testing.assert_allclose(field.hessian(points), expected_hess, rtol=1e-10, atol=1e-12)


def test_full_segment_barrier_contact_is_infeasible_but_domain_contact_is_allowed():
    base = uniform_terrain_fixture().to_dict()
    base["barriers_geojson"] = [
        {
            "type": "Polygon",
            "coordinates": [[[49, 49], [51, 49], [51, 51], [49, 51], [49, 49]]],
        }
    ]
    scenario = TerrainScenario.from_dict(base)
    collision = evaluate_route(scenario, [scenario.start_m, scenario.goal_m])
    assert not collision.feasible
    assert collision.violations == ("barrier_collision",)
    assert collision.minimum_clearance_m == 0

    boundary_scenario = TerrainScenario(
        **{
            key: value
            for key, value in uniform_terrain_fixture().__dict__.items()
            if key not in {"start_m", "goal_m"}
        },
        start_m=(0, 0),
        goal_m=(100, 0),
    )
    boundary = evaluate_route(boundary_scenario, [(0, 0), (100, 0)])
    assert boundary.feasible


def test_boundary_route_quadrature_and_profile_samples_stay_on_submitted_segments():
    """Roundoff in interpolation must not reject valid inclusive-boundary routes."""
    coordinates = np.linspace(0.0, 1000.0, 5)
    values = np.zeros((5, 5))
    scenario = TerrainScenario(
        name="boundary-roundoff",
        bounds_m=(0.0, 0.0, 1000.0, 1000.0),
        start_m=(1000.0, 0.0),
        goal_m=(0.0, 1000.0),
        field_x_m=coordinates,
        field_y_m=coordinates,
        elevation_m=values,
        log_slowness=values,
    )
    route = [(1000.0, 0.0), (0.0, 999.9), (0.0, 1000.0)]

    evaluation = evaluate_route(scenario, route, profile_samples=65, quadrature_order=8)

    assert evaluation.feasible
    assert evaluation.cost_s == pytest.approx(evaluation.length_m)
    assert evaluation.distance_m[0] == 0.0
    assert evaluation.distance_m[-1] == pytest.approx(evaluation.length_m)


def test_endpoint_and_domain_failures_are_reported_separately():
    scenario = uniform_terrain_fixture()
    evaluation = evaluate_route(scenario, [(-1, 20), (90, 80)])
    assert not evaluation.feasible
    assert evaluation.violations == ("start_mismatch", "outside_domain")


def test_planner_contract_round_trip_keeps_solver_and_feasibility_separate():
    scenario = uniform_terrain_fixture()
    config = PlannerConfig(method="slsqp", initialization="arc_left", options={"clearance_m": 2})
    evaluation = evaluate_route(scenario, [scenario.start_m, scenario.goal_m])
    result = PlannerResult(
        method=config.method,
        initialization=config.initialization,
        route_m=(scenario.start_m, scenario.goal_m),
        evaluated_cost_s=evaluation.cost_s,
        feasible=evaluation.feasible,
        solver_success=False,
        termination_reason="iteration_limit",
        diagnostics={"constraint_violation": 0.0},
        timing_s={"initialization": 0.01, "solve": 0.02, "evaluation": 0.01},
        scenario_hash=scenario.scenario_hash,
        config_hash=config.config_hash,
        evaluation=evaluation,
    )
    restored = PlannerResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored.to_dict() == result.to_dict()
    assert not restored.solver_success
    assert restored.feasible


@pytest.mark.parametrize(
    "change",
    [
        {"version": 1},
        {"bounds_m": (0, 0, 0, 1)},
        {"field_x_m": (0, 1, 2)},
        {"start_m": (200, 0)},
    ],
)
def test_invalid_scenario_contract(change):
    values = uniform_terrain_fixture().__dict__ | change
    with pytest.raises(ValueError):
        TerrainScenario(**values)
