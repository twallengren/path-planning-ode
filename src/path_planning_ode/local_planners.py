"""Local Euler--Lagrange and constrained SLSQP terrain planners."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .terrain import PlannerConfig, PlannerResult, TerrainField, TerrainScenario, evaluate_route
from .terrain_seeds import SeedResult, make_seed

FloatArray = NDArray[np.float64]
_NORMALIZED_BOUND_MARGIN = 8 * np.finfo(float).eps


class _DeadlineExceeded(RuntimeError):
    pass


def _check_deadline(deadline: float) -> None:
    if perf_counter() >= deadline:
        raise _DeadlineExceeded


def _ode_derivatives(
    field: TerrainField, q: FloatArray, velocity: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Acceleration and analytic position/velocity derivatives for c²|q'|²."""
    c, gradient, hessian = field.evaluate(q)
    speed2 = np.sum(velocity * velocity, axis=-1)
    gradient_velocity = np.sum(gradient * velocity, axis=-1)
    acceleration = (speed2[..., None] * gradient - 2 * velocity * gradient_velocity[..., None]) / c[
        ..., None
    ]
    hessian_velocity = np.einsum("...ij,...j->...i", hessian, velocity)
    position_derivative = (
        speed2[..., None, None] * hessian
        - 2 * velocity[..., :, None] * hessian_velocity[..., None, :]
    ) / c[..., None, None] - (
        acceleration[..., :, None] * gradient[..., None, :] / c[..., None, None]
    )
    velocity_derivative = (
        2
        * (
            gradient[..., :, None] * velocity[..., None, :]
            - velocity[..., :, None] * gradient[..., None, :]
            - gradient_velocity[..., None, None] * np.eye(2)
        )
        / c[..., None, None]
    )
    return acceleration, position_derivative, velocity_derivative


def euler_lagrange_residual(route_m: FloatArray, field: TerrainField) -> FloatArray:
    """Centered finite-difference residual of the weighted geodesic ODE."""
    route = np.asarray(route_m, dtype=float)
    step = 1.0 / (len(route) - 1)
    velocity = (route[2:] - route[:-2]) / (2 * step)
    acceleration = _ode_derivatives(field, route[1:-1], velocity)[0]
    return ((route[2:] - 2 * route[1:-1] + route[:-2]) / step**2 - acceleration).ravel()


def euler_lagrange_jacobian(route_m: FloatArray, field: TerrainField):
    """Sparse block-tridiagonal analytic Jacobian of the EL residual."""
    from scipy.sparse import lil_matrix

    route = np.asarray(route_m, dtype=float)
    interior = len(route) - 2
    step = 1.0 / (interior + 1)
    _, derivative_q, derivative_v = _ode_derivatives(
        field, route[1:-1], (route[2:] - route[:-2]) / (2 * step)
    )
    matrix = lil_matrix((2 * interior, 2 * interior), dtype=float)
    identity = np.eye(2)
    for index in range(interior):
        row = slice(2 * index, 2 * index + 2)
        matrix[row, row] = -2 * identity / step**2 - derivative_q[index]
        if index:
            matrix[row, slice(2 * index - 2, 2 * index)] = identity / step**2 + derivative_v[
                index
            ] / (2 * step)
        if index < interior - 1:
            matrix[row, slice(2 * index + 2, 2 * index + 4)] = identity / step**2 - derivative_v[
                index
            ] / (2 * step)
    return matrix.tocsr()


def _failed_initialization(
    scenario: TerrainScenario,
    config: PlannerConfig,
    seed: SeedResult,
    overall_start: float,
) -> PlannerResult:
    return PlannerResult(
        method=config.method,
        initialization=config.initialization,
        route_m=None,
        evaluated_cost_s=None,
        feasible=False,
        solver_success=False,
        termination_reason=seed.reason,
        diagnostics={"initialization": seed.diagnostics},
        timing_s={
            "initialization": seed.timing_s,
            "preprocessing": 0.0,
            "solve": 0.0,
            "evaluation": 0.0,
            "total": perf_counter() - overall_start,
        },
        scenario_hash=scenario.scenario_hash,
        config_hash=config.config_hash,
        evaluation=None,
    )


def _result(
    scenario: TerrainScenario,
    config: PlannerConfig,
    route: FloatArray,
    seed: SeedResult,
    solve_time: float,
    solver_success: bool,
    reason: str,
    diagnostics: dict,
    field: TerrainField,
    overall_start: float,
    preprocessing_time: float,
) -> PlannerResult:
    evaluation_started = perf_counter()
    try:
        evaluation = evaluate_route(
            scenario, route, field=field, profile_samples=config.profile_samples
        )
    except (FloatingPointError, ValueError) as exc:
        # Numerical solvers may terminate on a domain bound. If downstream
        # evaluation cannot classify that candidate, preserve it and return a
        # structured failure rather than escaping from the planner protocol.
        diagnostics = {
            "initialization": seed.diagnostics,
            **diagnostics,
            "evaluation_error": {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "solver_termination_reason": reason,
            },
        }
        return PlannerResult(
            method=config.method,
            initialization=config.initialization,
            route_m=tuple(map(tuple, route)),
            evaluated_cost_s=None,
            feasible=False,
            solver_success=False,
            termination_reason="evaluation_failed",
            diagnostics=diagnostics,
            timing_s={
                "initialization": seed.timing_s,
                "preprocessing": preprocessing_time,
                "solve": solve_time,
                "evaluation": perf_counter() - evaluation_started,
                "total": perf_counter() - overall_start,
            },
            scenario_hash=scenario.scenario_hash,
            config_hash=config.config_hash,
            evaluation=None,
        )
    return PlannerResult(
        method=config.method,
        initialization=config.initialization,
        route_m=tuple(map(tuple, route)),
        evaluated_cost_s=evaluation.cost_s,
        feasible=evaluation.feasible,
        solver_success=solver_success,
        termination_reason=reason,
        diagnostics={"initialization": seed.diagnostics, **diagnostics},
        timing_s={
            "initialization": seed.timing_s,
            "preprocessing": preprocessing_time,
            "solve": solve_time,
            "evaluation": evaluation.evaluation_time_s,
            "total": perf_counter() - overall_start,
        },
        scenario_hash=scenario.scenario_hash,
        config_hash=config.config_hash,
        evaluation=evaluation,
    )


def _seed_for(scenario: TerrainScenario, config: PlannerConfig, deadline: float) -> SeedResult:
    clearance = float(config.options.get("clearance_m", 0.0))
    seed = make_seed(
        scenario,
        config.initialization,
        config.interior_points,
        clearance_m=clearance,
        reference_grid_size=config.reference_grid_size,
        deadline=deadline,
    )
    seed.diagnostics["initial_route_hash"] = seed.diagnostics.get("route_hash")
    return seed


def solve_euler_lagrange(
    scenario: TerrainScenario, config: PlannerConfig, *, deadline: float | None = None
) -> PlannerResult:
    """Solve the unconstrained weighted-geodesic BVP by damped Newton steps."""
    if config.method != "euler_lagrange":
        raise ValueError("Euler--Lagrange solver requires method='euler_lagrange'.")
    overall_start = perf_counter()
    deadline = overall_start + config.time_limit_s if deadline is None else deadline
    seed = _seed_for(scenario, config, deadline)
    if not seed.success or seed.route_m is None:
        return _failed_initialization(scenario, config, seed, overall_start)
    preprocessing_started = perf_counter()
    field = TerrainField(scenario)
    preprocessing_time = perf_counter() - preprocessing_started
    route = np.asarray(seed.route_m, dtype=float).copy()
    solve_started = perf_counter()
    reason = "iteration_limit"
    success = False
    iterations = 0
    damping_used = 0.0
    residual_norm = float("inf")
    max_line_search = int(config.options.get("line_search_steps", 20))
    error: dict = {}
    try:
        import warnings

        from scipy.sparse.linalg import MatrixRankWarning, spsolve

        for iterations in range(config.max_iterations + 1):
            _check_deadline(deadline)
            residual = euler_lagrange_residual(route, field)
            residual_norm = float(np.linalg.norm(residual) / np.sqrt(residual.size))
            if not np.isfinite(residual_norm):
                reason = "nonfinite"
                break
            if residual_norm <= config.tolerance:
                success, reason = True, "stationary"
                break
            if iterations == config.max_iterations:
                break
            jacobian = euler_lagrange_jacobian(route, field)
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                try:
                    direction = spsolve(jacobian, -residual).reshape(-1, 2)
                except (MatrixRankWarning, RuntimeError, ValueError):
                    reason = "singular_jacobian"
                    break
            if not np.isfinite(direction).all():
                reason = "nonfinite"
                break
            damping = 1.0
            accepted = False
            for _ in range(max_line_search + 1):
                _check_deadline(deadline)
                candidate = route.copy()
                candidate[1:-1] += damping * direction
                try:
                    candidate_residual = euler_lagrange_residual(candidate, field)
                    candidate_norm = float(
                        np.linalg.norm(candidate_residual) / np.sqrt(candidate_residual.size)
                    )
                except ValueError:
                    candidate_norm = float("inf")
                if np.isfinite(candidate_norm) and candidate_norm < residual_norm:
                    route = candidate
                    damping_used = damping
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                reason = "stagnated"
                break
    except _DeadlineExceeded:
        reason = "time_limit"
    except Exception as exc:
        reason = "solver_error"
        error = {"exception_type": type(exc).__name__, "message": str(exc)}
    solve_time = perf_counter() - solve_started
    # Recompute independently of the iteration's bookkeeping.
    try:
        stationarity = float(
            np.linalg.norm(euler_lagrange_residual(route, field))
            / np.sqrt(2 * config.interior_points)
        )
    except ValueError:
        stationarity = None
    success = bool(success and stationarity is not None and stationarity <= config.tolerance)
    return _result(
        scenario,
        config,
        route,
        seed,
        solve_time,
        success,
        reason,
        {
            "iterations": iterations,
            "stationarity_norm": stationarity,
            "last_damping": damping_used,
            **error,
        },
        field,
        overall_start,
        preprocessing_time,
    )


@dataclass
class _SLSQPFunctions:
    field: TerrainField
    start: FloatArray
    goal: FloatArray
    lower: FloatArray
    scale: FloatArray
    quadrature_order: int
    clearance_m: float
    deadline: float

    def route(self, variables: FloatArray) -> FloatArray:
        interior = self.lower + np.asarray(variables).reshape(-1, 2) * self.scale
        return np.vstack((self.start, interior, self.goal))

    def objective_and_gradient(self, variables: FloatArray) -> tuple[float, FloatArray]:
        _check_deadline(self.deadline)
        route = self.route(variables)
        delta = np.diff(route, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature_order)
        fractions = (nodes + 1.0) / 2.0
        unit_weights = weights / 2.0
        points = route[:-1, None, :] + fractions[None, :, None] * delta[:, None, :]
        cost = self.field.cost(points)
        gradient_c = self.field.gradient(points)
        mean_cost = cost @ unit_weights
        value = float(np.sum(lengths * mean_cost))
        vertex_gradient = np.zeros_like(route)
        unit = np.divide(
            delta, lengths[:, None], out=np.zeros_like(delta), where=lengths[:, None] > 0
        )
        for index in range(len(delta)):
            field_a = np.sum(
                gradient_c[index] * (unit_weights * (1.0 - fractions))[:, None], axis=0
            )
            field_b = np.sum(gradient_c[index] * (unit_weights * fractions)[:, None], axis=0)
            vertex_gradient[index] += -unit[index] * mean_cost[index] + lengths[index] * field_a
            vertex_gradient[index + 1] += unit[index] * mean_cost[index] + lengths[index] * field_b
        normalized_gradient = vertex_gradient[1:-1] * self.scale
        return value, normalized_gradient.ravel()

    def equal_segments(self, variables: FloatArray) -> FloatArray:
        _check_deadline(self.deadline)
        route = self.route(variables)
        lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
        characteristic = max(float(np.linalg.norm(self.scale)), 1.0)
        return (lengths[1:] - lengths[:-1]) / characteristic

    def equal_segments_jacobian(self, variables: FloatArray) -> FloatArray:
        route = self.route(variables)
        delta = np.diff(route, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        unit = np.divide(
            delta, lengths[:, None], out=np.zeros_like(delta), where=lengths[:, None] > 0
        )
        vertices = len(route)
        jacobian_lengths = np.zeros((vertices - 1, 2 * (vertices - 2)))
        for segment in range(vertices - 1):
            if segment > 0:
                jacobian_lengths[segment, 2 * (segment - 1) : 2 * segment] -= (
                    unit[segment] * self.scale
                )
            if segment < vertices - 2:
                jacobian_lengths[segment, 2 * segment : 2 * segment + 2] += (
                    unit[segment] * self.scale
                )
        characteristic = max(float(np.linalg.norm(self.scale)), 1.0)
        return (jacobian_lengths[1:] - jacobian_lengths[:-1]) / characteristic

    def conservative_clearance(self, variables: FloatArray) -> FloatArray:
        _check_deadline(self.deadline)
        if self.field.barriers.is_empty:
            return np.empty(0)
        from shapely.geometry import Point

        route = self.route(variables)
        delta = np.diff(route, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        midpoints = (route[:-1] + route[1:]) / 2
        signed = []
        boundary = self.field.barriers.boundary
        for midpoint in midpoints:
            point = Point(midpoint)
            distance = float(point.distance(boundary))
            signed.append(-distance if self.field.barriers.covers(point) else distance)
        characteristic = max(float(np.linalg.norm(self.scale)), 1.0)
        return (np.asarray(signed) - lengths / 2 - self.clearance_m) / characteristic


def _finite_difference_jacobian(function: Callable[[FloatArray], FloatArray], x: FloatArray):
    base = np.asarray(function(x), dtype=float)
    jacobian = np.empty((len(base), len(x)))
    epsilon = 1e-6
    for index in range(len(x)):
        plus, minus = x.copy(), x.copy()
        plus[index] = min(1.0 - _NORMALIZED_BOUND_MARGIN, plus[index] + epsilon)
        minus[index] = max(_NORMALIZED_BOUND_MARGIN, minus[index] - epsilon)
        width = plus[index] - minus[index]
        jacobian[:, index] = (function(plus) - function(minus)) / width
    return jacobian


def _kkt_stationarity(functions: _SLSQPFunctions, variables: FloatArray) -> float:
    """Compute a normalized first-order residual with active constraints/bounds."""
    from scipy.optimize import lsq_linear

    _, gradient = functions.objective_and_gradient(variables)
    equality_jacobian = functions.equal_segments_jacobian(variables)
    inequality = functions.conservative_clearance(variables)
    active_tolerance = 1e-5
    columns = [row for row in equality_jacobian]
    unrestricted = len(columns)
    if len(inequality):
        inequality_jacobian = _finite_difference_jacobian(
            functions.conservative_clearance, variables
        )
        columns.extend(
            -row for row, value in zip(inequality_jacobian, inequality) if value <= active_tolerance
        )
    identity = np.eye(len(variables))
    columns.extend(-identity[i] for i, value in enumerate(variables) if value <= active_tolerance)
    columns.extend(
        identity[i] for i, value in enumerate(variables) if value >= 1 - active_tolerance
    )
    if not columns:
        residual = gradient
    else:
        matrix = np.column_stack(columns)
        lower = np.concatenate(
            (np.full(unrestricted, -np.inf), np.zeros(len(columns) - unrestricted))
        )
        upper = np.full(len(columns), np.inf)
        multipliers = lsq_linear(matrix, -gradient, bounds=(lower, upper)).x
        residual = gradient + matrix @ multipliers
    return float(
        np.linalg.norm(residual, ord=np.inf) / max(1.0, np.linalg.norm(gradient, ord=np.inf))
    )


def solve_slsqp(
    scenario: TerrainScenario, config: PlannerConfig, *, deadline: float | None = None
) -> PlannerResult:
    """Minimize quadrature-based weighted polyline length with hard constraints."""
    if config.method != "slsqp":
        raise ValueError("SLSQP solver requires method='slsqp'.")
    overall_start = perf_counter()
    deadline = overall_start + config.time_limit_s if deadline is None else deadline
    seed = _seed_for(scenario, config, deadline)
    if not seed.success or seed.route_m is None:
        return _failed_initialization(scenario, config, seed, overall_start)
    preprocessing_started = perf_counter()
    field = TerrainField(scenario)
    preprocessing_time = perf_counter() - preprocessing_started
    xmin, ymin, xmax, ymax = scenario.bounds_m
    lower = np.array([xmin, ymin])
    scale = np.array([xmax - xmin, ymax - ymin])
    initial = ((np.asarray(seed.route_m)[1:-1] - lower) / scale).ravel()
    clearance = float(config.options.get("clearance_m", 0.0))
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError("options.clearance_m must be finite and nonnegative.")
    quadrature_order = int(config.options.get("quadrature_order", 8))
    if quadrature_order < 2:
        raise ValueError("options.quadrature_order must be at least 2.")

    def prepare(candidate_seed: SeedResult) -> tuple[_SLSQPFunctions, FloatArray]:
        candidate_initial = ((np.asarray(candidate_seed.route_m)[1:-1] - lower) / scale).ravel()
        candidate_functions = _SLSQPFunctions(
            field,
            np.asarray(scenario.start_m),
            np.asarray(scenario.goal_m),
            lower,
            scale,
            quadrature_order,
            clearance,
            deadline,
        )
        return candidate_functions, candidate_initial

    functions, initial = prepare(seed)
    configured_route_hash = seed.diagnostics.get("route_hash")
    actual_interior_points = len(seed.route_m) - 2
    refinement_count = 0
    max_refinements = int(config.options.get("representation_refinements", 2))
    if max_refinements < 0:
        raise ValueError("options.representation_refinements must be nonnegative.")
    # The midpoint certificate can reject a sound coarse polyline. Increase the
    # representation before optimization when independent geometry says the
    # initialization itself is collision free.
    while not field.barriers.is_empty and refinement_count < max_refinements:
        conservative = functions.conservative_clearance(initial)
        if len(conservative) == 0 or np.min(conservative) >= -1e-8:
            break
        seed_evaluation = evaluate_route(
            scenario, np.asarray(seed.route_m), field=field, profile_samples=2
        )
        if not seed_evaluation.feasible:
            break
        _check_deadline(deadline)
        actual_interior_points *= 2
        refined = make_seed(
            scenario,
            config.initialization,
            actual_interior_points,
            clearance_m=clearance,
            reference_grid_size=config.reference_grid_size,
            deadline=deadline,
        )
        refinement_count += 1
        if not refined.success or refined.route_m is None:
            limited = SeedResult(
                seed.name,
                None,
                False,
                "resolution_limited",
                seed.timing_s + refined.timing_s,
                {
                    **refined.diagnostics,
                    "configured_interior_points": config.interior_points,
                    "attempted_interior_points": actual_interior_points,
                    "representation_refinements": refinement_count,
                },
            )
            return _failed_initialization(scenario, config, limited, overall_start)
        seed = SeedResult(
            refined.name,
            refined.route_m,
            True,
            refined.reason,
            seed.timing_s + refined.timing_s,
            refined.diagnostics,
        )
        functions, initial = prepare(seed)
    seed.diagnostics.update(
        {
            "initial_route_hash": configured_route_hash,
            "configured_interior_points": config.interior_points,
            "actual_interior_points": len(seed.route_m) - 2,
            "representation_refinements": refinement_count,
        }
    )
    solve_started = perf_counter()
    raw_success = False
    raw_status = -1
    raw_message = ""
    iterations = 0
    variables = initial.copy()
    reason = "solver_error"
    try:
        from scipy.optimize import minimize

        constraints: list[dict] = [
            {
                "type": "eq",
                "fun": functions.equal_segments,
                "jac": functions.equal_segments_jacobian,
            }
        ]
        if not field.barriers.is_empty:
            constraints.append({"type": "ineq", "fun": functions.conservative_clearance})

        def callback(_: FloatArray) -> None:
            _check_deadline(deadline)

        optimization = minimize(
            lambda value: functions.objective_and_gradient(value),
            initial,
            method="SLSQP",
            jac=True,
            # Keep free vertices one floating-point guard inside the inclusive
            # physical domain. This is mathematically equivalent at solver
            # precision and prevents quadrature interpolation roundoff from
            # querying the strict field evaluator just beyond a boundary.
            bounds=[(_NORMALIZED_BOUND_MARGIN, 1.0 - _NORMALIZED_BOUND_MARGIN)] * len(initial),
            constraints=constraints,
            callback=callback,
            options={
                "maxiter": config.max_iterations,
                "ftol": config.tolerance,
                "disp": False,
            },
        )
        variables = np.asarray(optimization.x)
        raw_success = bool(optimization.success)
        raw_status = int(optimization.status)
        raw_message = str(optimization.message)
        iterations = int(getattr(optimization, "nit", 0))
        reason = "converged" if raw_success else "optimization_failed"
    except _DeadlineExceeded:
        reason = "time_limit"
    except Exception as exc:
        reason = "solver_error"
        raw_message = f"{type(exc).__name__}: {exc}"
    solve_time = perf_counter() - solve_started
    route = functions.route(variables)
    try:
        equality_violation = float(np.max(np.abs(functions.equal_segments(variables))))
        inequality = functions.conservative_clearance(variables)
        inequality_violation = float(max(0.0, -np.min(inequality))) if len(inequality) else 0.0
        stationarity = _kkt_stationarity(functions, variables)
    except _DeadlineExceeded:
        equality_violation = inequality_violation = None
        stationarity = None
        reason = "time_limit"
    constraint_tolerance = max(config.tolerance * 10, 1e-6)
    stationarity_tolerance = max(np.sqrt(config.tolerance), 1e-4)
    independently_stationary = stationarity is not None and stationarity <= stationarity_tolerance
    constraints_satisfied = (
        equality_violation is not None
        and inequality_violation is not None
        and max(equality_violation, inequality_violation) <= constraint_tolerance
    )
    solver_success = bool(raw_success and constraints_satisfied and independently_stationary)
    if raw_success and not constraints_satisfied:
        reason = "constraint_violation"
    elif raw_success and not independently_stationary:
        reason = "nonstationary"

    # A collision-free seed together with a failed conservative certificate
    # indicates insufficient polyline resolution, not absence of a route.
    if inequality_violation is not None and inequality_violation > constraint_tolerance:
        seed_evaluation = evaluate_route(
            scenario, np.asarray(seed.route_m), field=field, profile_samples=2
        )
        if seed_evaluation.feasible:
            reason = "resolution_limited"

    return _result(
        scenario,
        config,
        route,
        seed,
        solve_time,
        solver_success,
        reason,
        {
            "iterations": iterations,
            "library_success": raw_success,
            "library_status": raw_status,
            "library_message": raw_message,
            "equality_constraint_violation": equality_violation,
            "barrier_constraint_violation": inequality_violation,
            "constraint_violation": (
                None
                if equality_violation is None or inequality_violation is None
                else max(equality_violation, inequality_violation)
            ),
            "stationarity_norm": stationarity,
            "stationarity_tolerance": stationarity_tolerance,
            "quadrature_order": quadrature_order,
            "clearance_m": clearance,
            "normalized_bound_margin": _NORMALIZED_BOUND_MARGIN,
        },
        field,
        overall_start,
        preprocessing_time,
    )


__all__ = [
    "euler_lagrange_jacobian",
    "euler_lagrange_residual",
    "solve_euler_lagrange",
    "solve_slsqp",
]
