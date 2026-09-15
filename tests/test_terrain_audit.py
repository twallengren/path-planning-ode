"""Independent numerical and geometric checks for the version-2 terrain API."""

from dataclasses import replace
from math import exp

import numpy as np
import pytest
from scipy.integrate import quad

from path_planning_ode import TerrainField, TerrainScenario, evaluate_route
from path_planning_ode.fast_marching import (
    FastMarchingError,
    compute_arrival_time,
    extract_route,
    solve_fast_marching,
)
from path_planning_ode.local_planners import (
    _SLSQPFunctions,
    euler_lagrange_jacobian,
    euler_lagrange_residual,
    solve_euler_lagrange,
    solve_slsqp,
)
from path_planning_ode.terrain import PlannerConfig
from path_planning_ode.terrain_generators import (
    competing_corridors_terrain,
    layered_refraction_fixture,
    obstacle_detour_fixture,
    uniform_terrain_fixture,
)
from path_planning_ode.terrain_seeds import make_seed, resample_route


def _cubic_log_slowness(points):
    """A nonsymmetric total-degree-three polynomial, independent of the spline."""
    points = np.asarray(points, dtype=float)
    x, y = points[..., 0], points[..., 1]
    return (
        0.15
        + 0.08 * x
        - 0.06 * y
        + 0.025 * x**2
        + 0.018 * x * y
        - 0.02 * y**2
        + 0.004 * x**3
        - 0.005 * x**2 * y
        + 0.003 * x * y**2
        - 0.002 * y**3
    )


def _scenario(
    *,
    start=(0.0, 0.0),
    goal=(1.0, 1.0),
    bounds=(-2.0, -2.5, 3.0, 2.0),
    barriers=(),
    log_function=_cubic_log_slowness,
):
    x = np.linspace(bounds[0], bounds[2], 8)
    y = np.linspace(bounds[1], bounds[3], 7)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx, yy), axis=-1)
    log_slowness = log_function(points)
    elevation = 2.0 + 0.3 * xx - 0.4 * yy + 0.02 * xx * yy
    return TerrainScenario(
        name="independent-audit",
        bounds_m=bounds,
        start_m=start,
        goal_m=goal,
        field_x_m=x,
        field_y_m=y,
        elevation_m=elevation,
        log_slowness=log_slowness,
        barriers_geojson=barriers,
        provenance={"kind": "independent analytic audit"},
    )


def _constant_scenario(*, start, goal, barriers=(), value=1.0):
    return _scenario(
        start=start,
        goal=goal,
        bounds=(0.0, 0.0, 10.0, 10.0),
        barriers=barriers,
        log_function=lambda points: np.full(np.asarray(points).shape[:-1], np.log(value)),
    )


def _polygon(*coordinates):
    return {"type": "Polygon", "coordinates": [[*coordinates, coordinates[0]]]}


SQUARE = _polygon((4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0))


def test_exp_cubic_log_slowness_gradient_and_hessian_finite_differences():
    field = TerrainField(_scenario())
    points = np.array([[-1.1, -1.4], [-0.2, 0.3], [0.8, -0.7], [2.1, 1.1]])

    # A cubic spline must reproduce this cubic source surface between knots.
    np.testing.assert_allclose(field.log_slowness(points), _cubic_log_slowness(points), atol=2e-14)

    gradient_fd = np.empty_like(points)
    epsilon = 1e-5
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = epsilon
        gradient_fd[:, axis] = (field.cost(points + offset) - field.cost(points - offset)) / (
            2 * epsilon
        )
    np.testing.assert_allclose(field.gradient(points), gradient_fd, rtol=2e-8, atol=2e-9)

    hessian_fd = np.empty((len(points), 2, 2))
    epsilon = 2e-5
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = epsilon
        hessian_fd[:, :, axis] = (
            field.gradient(points + offset) - field.gradient(points - offset)
        ) / (2 * epsilon)
    np.testing.assert_allclose(field.hessian(points), hessian_fd, rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(field.hessian(points), field.hessian(points).swapaxes(-1, -2))


def test_route_cost_matches_independent_adaptive_quadrature():
    scenario = _scenario(start=(-1.7, -1.8), goal=(2.6, 1.7))
    field = TerrainField(scenario)
    route = np.array([scenario.start_m, (-0.9, 1.2), (0.4, -0.4), (1.7, 1.3), scenario.goal_m])

    expected = 0.0
    for a, b in zip(route[:-1], route[1:], strict=True):
        delta = b - a
        length = np.linalg.norm(delta)
        segment, error = quad(
            lambda t: float(field.cost(a + t * delta)) * length,
            0.0,
            1.0,
            epsabs=1e-11,
            epsrel=1e-11,
            limit=200,
        )
        assert error < 1e-9
        expected += segment

    measured = evaluate_route(scenario, route, field=field)
    assert measured.cost_s == pytest.approx(expected, rel=1e-5)
    assert measured.accumulated_cost_s[-1] == measured.cost_s


def test_constant_field_straight_line_and_subdivision_invariants():
    value = exp(0.7)
    scenario = _constant_scenario(start=(1.0, 2.0), goal=(9.0, 8.0), value=value)
    direct = np.array([scenario.start_m, scenario.goal_m])
    subdivided = np.linspace(scenario.start_m, scenario.goal_m, 37)
    length = 10.0

    direct_evaluation = evaluate_route(scenario, direct)
    subdivided_evaluation = evaluate_route(scenario, subdivided)
    assert direct_evaluation.feasible and subdivided_evaluation.feasible
    assert direct_evaluation.length_m == pytest.approx(length)
    assert direct_evaluation.cost_s == pytest.approx(value * length, rel=2e-14)
    assert subdivided_evaluation.cost_s == pytest.approx(direct_evaluation.cost_s, rel=2e-14)


def test_route_cost_is_unchanged_by_reversal_and_reflection():
    scenario = _scenario(start=(-1.5, -1.2), goal=(2.4, 1.4))
    route = np.array([scenario.start_m, (-0.7, 1.1), (0.5, -0.8), (1.7, 1.2), scenario.goal_m])
    forward = evaluate_route(scenario, route).cost_s
    reversed_scenario = replace(scenario, start_m=scenario.goal_m, goal_m=scenario.start_m)
    reversed_cost = evaluate_route(reversed_scenario, route[::-1]).cost_s
    assert reversed_cost == pytest.approx(forward, rel=2e-14)

    symmetric = _scenario(
        start=(1.0, 2.0),
        goal=(9.0, 7.0),
        bounds=(0.0, 0.0, 10.0, 10.0),
        log_function=lambda points: (
            0.01 * (np.asarray(points)[..., 0] - 5.0) ** 2
            + 0.02 * (np.asarray(points)[..., 1] - 5.0) ** 2
        ),
    )
    asymmetric_route = np.array([symmetric.start_m, (3.0, 6.5), (7.0, 4.0), symmetric.goal_m])
    reflected_route = asymmetric_route * [1.0, -1.0] + [0.0, 10.0]
    reflected_scenario = replace(
        symmetric, start_m=tuple(reflected_route[0]), goal_m=tuple(reflected_route[-1])
    )
    assert evaluate_route(reflected_scenario, reflected_route).cost_s == pytest.approx(
        evaluate_route(symmetric, asymmetric_route).cost_s, rel=2e-13
    )


def test_endpoint_tolerance_and_route_validation_are_explicit():
    scenario = _constant_scenario(start=(1.0, 1.0), goal=(9.0, 9.0))
    tolerance = 1e-4
    within = np.array([[1.0 + 0.5 * tolerance, 1.0], scenario.goal_m])
    outside = np.array([[1.0 + 2 * tolerance, 1.0], [9.0, 9.0 - 2 * tolerance]])
    assert evaluate_route(scenario, within, endpoint_tolerance_m=tolerance).feasible
    assert evaluate_route(scenario, outside, endpoint_tolerance_m=tolerance).violations == (
        "start_mismatch",
        "goal_mismatch",
    )

    invalid_routes = (
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0]]),
        np.array([[1.0, 1.0], [np.nan, 9.0]]),
    )
    for invalid in invalid_routes:
        with pytest.raises(ValueError):
            evaluate_route(scenario, invalid)


def test_scenario_rejects_invalid_endpoint_and_barrier_placement():
    with pytest.raises(ValueError, match="start_m"):
        _constant_scenario(start=(-0.01, 1.0), goal=(9.0, 9.0))
    with pytest.raises(ValueError, match="Endpoints"):
        _constant_scenario(start=(4.0, 5.0), goal=(9.0, 9.0), barriers=(SQUARE,))
    outside = _polygon((-0.1, 2.0), (1.0, 2.0), (1.0, 3.0), (-0.1, 3.0))
    with pytest.raises(ValueError, match="barrier"):
        _constant_scenario(start=(1.0, 1.0), goal=(9.0, 9.0), barriers=(outside,))


@pytest.mark.parametrize(
    ("name", "route", "barrier"),
    [
        ("corner cutting", [(1.0, 8.0), (8.0, 1.0)], SQUARE),
        (
            "thin barrier",
            [(1.0, 5.0), (9.0, 5.0)],
            _polygon((4.999, 2.0), (5.001, 2.0), (5.001, 8.0), (4.999, 8.0)),
        ),
        ("barrier touching", [(1.0, 4.0), (9.0, 4.0)], SQUARE),
    ],
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_full_segments_detect_collision(name, route, barrier):
    del name
    scenario = _constant_scenario(start=route[0], goal=route[-1], barriers=(barrier,))
    evaluation = evaluate_route(scenario, route)
    assert not evaluation.feasible
    assert evaluation.violations == ("barrier_collision",)
    assert evaluation.minimum_clearance_m == 0.0
    assert np.isfinite(evaluation.cost_s)


def test_domain_boundary_contact_is_allowed_but_domain_exit_is_not():
    scenario = _constant_scenario(start=(0.0, 0.0), goal=(10.0, 0.0))
    boundary = evaluate_route(scenario, [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)])
    assert boundary.feasible

    exits = evaluate_route(scenario, [(0.0, 0.0), (5.0, -0.01), (10.0, 0.0)])
    assert not exits.feasible
    assert exits.violations == ("outside_domain",)
    # Reporting an invalid route must not silently replace its geometry with a
    # vertex-clamped polyline. The bounded field is undefined on this route.
    assert exits.cost_s is None
    assert exits.distance_m == ()
    assert exits.elevation_m == ()
    assert exits.slowness_s_per_m == ()
    assert exits.accumulated_cost_s == ()


def test_domain_spanning_barrier_rejects_crossing_and_boundary_bypass():
    divider = _polygon((4.9, 0.0), (5.1, 0.0), (5.1, 10.0), (4.9, 10.0))
    scenario = _constant_scenario(start=(1.0, 5.0), goal=(9.0, 5.0), barriers=(divider,))

    crossing = evaluate_route(scenario, [scenario.start_m, scenario.goal_m])
    boundary_bypass = evaluate_route(
        scenario, [scenario.start_m, (4.8, 0.0), (5.2, 0.0), scenario.goal_m]
    )
    outside_bypass = evaluate_route(
        scenario, [scenario.start_m, (4.8, -0.1), (5.2, -0.1), scenario.goal_m]
    )
    assert crossing.violations == ("barrier_collision",)
    assert boundary_bypass.violations == ("barrier_collision",)
    assert "outside_domain" in outside_bypass.violations
    assert all(not result.feasible for result in (crossing, boundary_bypass, outside_bypass))


def _discrete_eikonal_residual(solution, slowness):
    """Recompute the first-order Godunov equation without using the FMM update."""
    arrival = solution.arrival_s
    dx = solution.x_m[1] - solution.x_m[0]
    dy = solution.y_m[1] - solution.y_m[0]
    seeds = set(solution.goal_seed_indices)
    residuals = []
    for row, column in np.ndindex(arrival.shape):
        if (row, column) in seeds or solution.masked[row, column]:
            continue
        if not np.isfinite(arrival[row, column]):
            continue
        horizontal = min(
            arrival[row, column - 1] if column else np.inf,
            arrival[row, column + 1] if column + 1 < arrival.shape[1] else np.inf,
        )
        vertical = min(
            arrival[row - 1, column] if row else np.inf,
            arrival[row + 1, column] if row + 1 < arrival.shape[0] else np.inf,
        )
        derivative_x = (
            max(0.0, (arrival[row, column] - horizontal) / dx) if np.isfinite(horizontal) else 0.0
        )
        derivative_y = (
            max(0.0, (arrival[row, column] - vertical) / dy) if np.isfinite(vertical) else 0.0
        )
        residuals.append(np.hypot(derivative_x, derivative_y) - slowness[row, column])
    return np.asarray(residuals)


def test_fast_marching_satisfies_first_order_isotropic_eikonal_equation():
    scenario = _scenario(start=(-1.5, -1.5), goal=(2.0, 1.5))
    solution = compute_arrival_time(scenario, grid_size=41)
    xx, yy = np.meshgrid(solution.x_m, solution.y_m)
    slowness = TerrainField(scenario).cost(np.stack((xx, yy), axis=-1))
    residual = _discrete_eikonal_residual(solution, slowness)
    assert len(residual) == solution.reachable_count - len(solution.goal_seed_indices)
    assert np.max(np.abs(residual)) < 2e-12


def test_fast_marching_mask_matches_independent_control_cell_intersections():
    from shapely.geometry import box, shape

    barrier = _polygon((4.13, 1.7), (5.07, 2.2), (6.21, 7.9), (3.81, 8.4))
    scenario = _constant_scenario(start=(1.0, 5.0), goal=(9.0, 5.0), barriers=(barrier,))
    solution = compute_arrival_time(scenario, grid_size=21)
    geometry = shape(barrier)
    dx = solution.x_m[1] - solution.x_m[0]
    dy = solution.y_m[1] - solution.y_m[0]
    expected = np.zeros_like(solution.masked)
    for row, y in enumerate(solution.y_m):
        for column, x in enumerate(solution.x_m):
            cell = box(
                max(0.0, x - dx / 2),
                max(0.0, y - dy / 2),
                min(10.0, x + dx / 2),
                min(10.0, y + dy / 2),
            )
            expected[row, column] = cell.intersects(geometry)
    np.testing.assert_array_equal(solution.masked, expected)


def _piecewise_linear_arrival(solution, point):
    """Evaluate both triangles near a point directly from their barycentric weights."""
    point = np.asarray(point)
    dx = solution.x_m[1] - solution.x_m[0]
    dy = solution.y_m[1] - solution.y_m[0]
    column = int(np.clip(np.searchsorted(solution.x_m, point[0]) - 1, 0, len(solution.x_m) - 2))
    row = int(np.clip(np.searchsorted(solution.y_m, point[1]) - 1, 0, len(solution.y_m) - 2))
    u = (point[0] - solution.x_m[column]) / dx
    v = (point[1] - solution.y_m[row]) / dy
    lower_left = solution.arrival_s[row, column]
    upper_right = solution.arrival_s[row + 1, column + 1]
    if v <= u:
        lower_right = solution.arrival_s[row, column + 1]
        return (1 - u) * lower_left + (u - v) * lower_right + v * upper_right
    upper_left = solution.arrival_s[row + 1, column]
    return (1 - v) * lower_left + (v - u) * upper_left + u * upper_right


def test_piecewise_linear_extraction_strictly_descends_and_detours():
    scenario = obstacle_detour_fixture()
    solution = compute_arrival_time(scenario, grid_size=65)
    route = extract_route(scenario, solution)
    values = np.array([_piecewise_linear_arrival(solution, point) for point in route[2:-1]])
    assert np.isfinite(values).all()
    assert np.all(np.diff(values) < 0)
    evaluation = evaluate_route(scenario, route)
    assert evaluation.feasible
    assert evaluation.minimum_clearance_m > 0
    assert np.max(np.abs(route[:, 1] - 500.0)) > 150.0


def test_piecewise_linear_extraction_has_explicit_failure_semantics():
    from dataclasses import replace as replace_dataclass

    scenario = _constant_scenario(start=(1.0, 1.0), goal=(9.0, 9.0))
    solution = compute_arrival_time(scenario, grid_size=17)
    flat = replace_dataclass(solution, arrival_s=np.zeros_like(solution.arrival_s))
    with pytest.raises(FastMarchingError) as captured:
        extract_route(scenario, flat)
    assert captured.value.reason == "extraction_failed"


def test_fast_marching_refines_uniform_cost_and_obeys_snell_invariant():
    uniform = uniform_terrain_fixture()
    expected = 0.8 * np.linalg.norm(np.asarray(uniform.goal_m) - uniform.start_m)
    costs = []
    for grid_size in (33, 129):
        result = solve_fast_marching(
            uniform,
            PlannerConfig(method="fast_marching", reference_grid_size=grid_size, profile_samples=2),
        )
        assert result.solver_success and result.feasible
        costs.append(result.evaluated_cost_s)
    assert abs(costs[-1] - expected) < abs(costs[0] - expected)
    assert costs[-1] == pytest.approx(expected, rel=1e-3)

    layered = layered_refraction_fixture()
    refracted = solve_fast_marching(
        layered,
        PlannerConfig(method="fast_marching", reference_grid_size=129, profile_samples=2),
    )
    assert refracted.solver_success and refracted.feasible
    route = np.asarray(refracted.route_m)
    invariants = []
    for lower, upper, slowness in ((220.0, 430.0, 0.75), (570.0, 780.0, 1.5)):
        region = route[(route[:, 1] > lower) & (route[:, 1] < upper)]
        slope = np.polyfit(region[:, 1], region[:, 0], 1)[0]
        invariants.append(slowness * slope / np.hypot(slope, 1.0))
    assert invariants[0] == pytest.approx(invariants[1], rel=0.01)


def _normalized_local_functions(scenario, route, *, clearance=0.0, quadrature_order=16):
    lower = np.asarray(scenario.bounds_m[:2])
    scale = np.asarray(scenario.bounds_m[2:]) - lower
    variables = ((np.asarray(route)[1:-1] - lower) / scale).ravel()
    functions = _SLSQPFunctions(
        TerrainField(scenario),
        np.asarray(scenario.start_m),
        np.asarray(scenario.goal_m),
        lower,
        scale,
        quadrature_order,
        clearance,
        float("inf"),
    )
    return functions, variables


def _finite_difference_gradient(function, variables, epsilon=1e-6):
    gradient = np.empty_like(variables)
    for index in range(len(variables)):
        offset = np.zeros_like(variables)
        offset[index] = epsilon
        gradient[index] = (function(variables + offset) - function(variables - offset)) / (
            2 * epsilon
        )
    return gradient


def test_slsqp_objective_is_weighted_length_and_gradient_is_independent_fd():
    scenario = _scenario(start=(-1.5, -1.5), goal=(2.4, 1.4))
    route = np.array([scenario.start_m, (-0.9, 0.7), (0.2, -0.3), (1.2, 0.9), scenario.goal_m])
    functions, variables = _normalized_local_functions(scenario, route)
    value, gradient = functions.objective_and_gradient(variables)

    independent = 0.0
    field = TerrainField(scenario)
    for start, end in zip(route[:-1], route[1:], strict=True):
        delta = end - start
        length = np.linalg.norm(delta)
        integral = quad(
            lambda fraction: float(field.cost(start + fraction * delta)) * length,
            0.0,
            1.0,
            epsabs=1e-11,
            epsrel=1e-11,
        )[0]
        independent += integral
    assert value == pytest.approx(independent, rel=1e-10)

    numerical = _finite_difference_gradient(
        lambda candidate: functions.objective_and_gradient(candidate)[0], variables
    )
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-6)


def test_equal_segment_jacobian_and_conservative_clearance_have_independent_oracles():
    from shapely.geometry import Point, shape

    route = np.array(
        [
            (1.0, 5.0),
            (2.0, 7.0),
            (3.0, 7.0),
            (4.0, 7.0),
            (5.0, 7.0),
            (6.0, 7.0),
            (7.0, 7.0),
            (8.0, 7.0),
            (9.0, 5.0),
        ]
    )
    scenario = _constant_scenario(start=route[0], goal=route[-1], barriers=(SQUARE,))
    clearance = 0.2
    functions, variables = _normalized_local_functions(scenario, route, clearance=clearance)
    scale = np.array([10.0, 10.0])
    characteristic = np.linalg.norm(scale)
    lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
    np.testing.assert_allclose(
        functions.equal_segments(variables), (lengths[1:] - lengths[:-1]) / characteristic
    )

    equality_fd = np.column_stack(
        [
            _finite_difference_gradient(
                lambda candidate, row=row: functions.equal_segments(candidate)[row], variables
            )
            for row in range(len(route) - 2)
        ]
    ).T
    np.testing.assert_allclose(
        functions.equal_segments_jacobian(variables), equality_fd, rtol=2e-7, atol=2e-8
    )

    barrier = shape(SQUARE)
    expected = []
    for start, end, length in zip(route[:-1], route[1:], lengths, strict=True):
        midpoint = (start + end) / 2
        distance = Point(midpoint).distance(barrier.boundary)
        signed_distance = -distance if barrier.covers(Point(midpoint)) else distance
        expected.append((signed_distance - length / 2 - clearance) / characteristic)
    np.testing.assert_allclose(functions.conservative_clearance(variables), expected, atol=1e-14)
    assert np.min(expected) > 0
    assert evaluate_route(scenario, route).minimum_clearance_m >= clearance


def test_asymmetric_euler_lagrange_jacobian_and_stationarity_are_independent():
    scenario = _scenario(start=(-1.5, -1.5), goal=(2.4, 1.4))
    field = TerrainField(scenario)
    route = np.array([scenario.start_m, (-0.9, 0.7), (0.2, -0.3), (1.2, 0.9), scenario.goal_m])
    analytic = euler_lagrange_jacobian(route, field).toarray()
    numerical = np.empty_like(analytic)
    epsilon = 1e-5
    for index in range(analytic.shape[1]):
        plus, minus = route.copy(), route.copy()
        plus[1 + index // 2, index % 2] += epsilon
        minus[1 + index // 2, index % 2] -= epsilon
        numerical[:, index] = (
            euler_lagrange_residual(plus, field) - euler_lagrange_residual(minus, field)
        ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=3e-6, atol=3e-6)

    barrier_scenario = obstacle_detour_fixture()
    result = solve_euler_lagrange(
        barrier_scenario,
        PlannerConfig(
            method="euler_lagrange",
            initialization="straight",
            interior_points=8,
            tolerance=1e-7,
            profile_samples=2,
        ),
    )
    residual_norm = np.linalg.norm(
        euler_lagrange_residual(np.asarray(result.route_m), TerrainField(barrier_scenario))
    ) / np.sqrt(16)
    assert result.solver_success
    assert residual_norm <= 1e-7
    assert not result.feasible
    assert result.evaluation.violations == ("barrier_collision",)


def test_shared_initial_seed_hash_and_resolution_limited_status_are_preserved():
    scenario = obstacle_detour_fixture()
    initial = make_seed(scenario, "barrier", 6)
    assert initial.success
    common = dict(
        initialization="barrier",
        interior_points=6,
        tolerance=1e-7,
        max_iterations=5,
        time_limit_s=10.0,
        profile_samples=2,
    )
    euler = solve_euler_lagrange(scenario, PlannerConfig(method="euler_lagrange", **common))
    constrained = solve_slsqp(scenario, PlannerConfig(method="slsqp", **common))
    expected_hash = initial.diagnostics["route_hash"]
    assert euler.diagnostics["initialization"]["initial_route_hash"] == expected_hash
    assert constrained.diagnostics["initialization"]["initial_route_hash"] == expected_hash
    assert constrained.termination_reason == "resolution_limited"
    assert not constrained.solver_success
    assert constrained.feasible
    assert constrained.diagnostics["constraint_violation"] > 1e-6
    assert constrained.diagnostics["initialization"]["representation_refinements"] > 0


def test_equal_arclength_resampling_can_create_a_collision():
    scenario = _constant_scenario(start=(1.0, 5.0), goal=(9.0, 5.0), barriers=(SQUARE,))
    safe = np.array([scenario.start_m, (3.9, 6.1), (6.1, 6.1), scenario.goal_m])
    assert evaluate_route(scenario, safe).feasible
    sampled = resample_route(safe, 3)
    assert not evaluate_route(scenario, sampled).feasible


def _independent_gauss_cost(field, route, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    fractions = (nodes + 1.0) / 2
    weights = weights / 2
    value = 0.0
    for start, end in zip(route[:-1], route[1:], strict=True):
        delta = end - start
        value += np.linalg.norm(delta) * np.dot(
            weights, field.cost(start + fractions[:, None] * delta)
        )
    return float(value)


def test_slsqp_success_has_independent_constraints_stationarity_and_feasibility():
    scenario = competing_corridors_terrain(seed=2, barriers=False)
    config = PlannerConfig(
        method="slsqp",
        initialization="straight",
        interior_points=8,
        tolerance=1e-7,
        max_iterations=200,
        time_limit_s=10.0,
        profile_samples=2,
    )
    result = solve_slsqp(scenario, config)
    assert result.solver_success and result.feasible
    route = np.asarray(result.route_m)
    field = TerrainField(scenario)
    lower = np.asarray(scenario.bounds_m[:2])
    scale = np.asarray(scenario.bounds_m[2:]) - lower
    variables = ((route[1:-1] - lower) / scale).ravel()
    assert np.all((variables > 1e-4) & (variables < 1 - 1e-4))

    def candidate_route(candidate):
        interior = lower + candidate.reshape(-1, 2) * scale
        return np.vstack((scenario.start_m, interior, scenario.goal_m))

    def objective(candidate):
        return _independent_gauss_cost(field, candidate_route(candidate), order=8)

    characteristic = np.linalg.norm(scale)

    def equality(candidate):
        lengths = np.linalg.norm(np.diff(candidate_route(candidate), axis=0), axis=1)
        return (lengths[1:] - lengths[:-1]) / characteristic

    gradient = _finite_difference_gradient(objective, variables)
    equality_jacobian = np.vstack(
        [
            _finite_difference_gradient(
                lambda candidate, row=row: equality(candidate)[row], variables
            )
            for row in range(len(route) - 2)
        ]
    )
    multipliers = np.linalg.lstsq(equality_jacobian.T, -gradient, rcond=None)[0]
    residual = gradient + equality_jacobian.T @ multipliers
    stationarity = np.linalg.norm(residual, ord=np.inf) / max(
        1.0, np.linalg.norm(gradient, ord=np.inf)
    )
    equality_violation = np.max(np.abs(equality(variables)))
    assert equality_violation == pytest.approx(
        result.diagnostics["equality_constraint_violation"], abs=2e-10
    )
    assert stationarity <= result.diagnostics["stationarity_tolerance"]
    assert stationarity == pytest.approx(result.diagnostics["stationarity_norm"], rel=0.02)
    independent_evaluation = evaluate_route(scenario, route, field=field, profile_samples=2)
    assert independent_evaluation.feasible == result.feasible
    assert independent_evaluation.cost_s == result.evaluated_cost_s
    assert objective(variables) == pytest.approx(result.evaluated_cost_s, rel=1e-5)


def test_failures_retain_routes_and_costs_and_all_timing_includes_initialization():
    scenario = obstacle_detour_fixture()
    failed = solve_slsqp(
        scenario,
        PlannerConfig(
            method="slsqp",
            initialization="straight",
            interior_points=6,
            max_iterations=1,
            time_limit_s=10.0,
            profile_samples=2,
        ),
    )
    assert not failed.solver_success
    assert failed.route_m is not None
    assert failed.evaluated_cost_s is not None
    assert failed.evaluation is not None

    for result in (
        failed,
        solve_euler_lagrange(
            uniform_terrain_fixture(),
            PlannerConfig(
                method="euler_lagrange",
                interior_points=4,
                time_limit_s=np.finfo(float).tiny,
            ),
        ),
    ):
        serialized = result.to_dict()
        assert serialized["termination_reason"] == result.termination_reason
        assert result.timing_s["initialization"] >= 0
        component_total = sum(value for key, value in result.timing_s.items() if key != "total")
        assert result.timing_s["total"] >= component_total
