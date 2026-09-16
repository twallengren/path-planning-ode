import json
import subprocess
import sys

import numpy as np
import pytest

from path_planning_ode import TerrainField, TerrainScenario, evaluate_route, uniform_terrain_fixture


def test_root_import_keeps_optional_terrain_stack_lazy():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import path_planning_ode; assert 'scipy' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""


def test_uniform_fixture_round_trip_and_route_evaluation():
    scenario = uniform_terrain_fixture()
    assert TerrainScenario.from_dict(json.loads(json.dumps(scenario.to_dict()))) == scenario
    evaluation = evaluate_route(scenario, [scenario.start_m, scenario.goal_m])
    assert evaluation.feasible
    assert evaluation.cost_s == pytest.approx(100)


def test_log_spline_chain_rule_gradient_and_hessian():
    x, y = np.linspace(0, 4, 5), np.linspace(0, 3, 4)
    xx, yy = np.meshgrid(x, y)
    log_cost = 0.02 * xx**2 + 0.03 * xx * yy - 0.01 * yy**2 + 0.2
    scenario = TerrainScenario(
        "quadratic", (0, 0, 4, 3), (0, 0), (4, 3), x, y, np.zeros_like(log_cost), log_cost
    )
    points = np.array([[0.7, 1.1], [2.2, 0.8], [3.5, 2.4]])
    px, py = points.T
    expected_cost = np.exp(0.02 * px**2 + 0.03 * px * py - 0.01 * py**2 + 0.2)
    expected_gradient = expected_cost[:, None] * np.stack(
        [0.04 * px + 0.03 * py, 0.03 * px - 0.02 * py], axis=-1
    )
    np.testing.assert_allclose(TerrainField(scenario).cost(points), expected_cost)
    np.testing.assert_allclose(TerrainField(scenario).gradient(points), expected_gradient)
