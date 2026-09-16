"""Preconditioned energy descent for the shared continuous terrain field.

The solver minimizes the fixed-parameter polygonal energy
``integral c(q)^2 |q'|^2 dt``.  It uses the same positive bicubic slowness
field, shared seeds, hard barrier geometry, wall-clock budget, and independent
route evaluator as the existing terrain planners.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from time import perf_counter
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .terrain import PlannerConfig, PlannerResult, TerrainField, TerrainScenario, evaluate_route
from .terrain_seeds import SeedResult, make_seed

FloatArray = NDArray[np.float64]


class _DeadlineExceeded(RuntimeError):
    pass


def _check_deadline(deadline: float) -> None:
    if perf_counter() >= deadline:
        raise _DeadlineExceeded


@dataclass(frozen=True)
class _EnergyOptions:
    quadrature_order: int
    panel_cell_fraction: float
    step_cap_cells: float
    armijo: float
    backtrack_factor: float
    line_search_steps: int
    clearance_m: float

    @classmethod
    def from_config(cls, config: PlannerConfig) -> _EnergyOptions:
        values = cls(
            quadrature_order=int(config.options.get("quadrature_order", 8)),
            panel_cell_fraction=float(config.options.get("panel_cell_fraction", 1.0)),
            step_cap_cells=float(config.options.get("step_cap_cells", 0.5)),
            armijo=float(config.options.get("armijo", 1e-4)),
            backtrack_factor=float(config.options.get("backtrack_factor", 0.5)),
            line_search_steps=int(config.options.get("line_search_steps", 20)),
            clearance_m=float(config.options.get("clearance_m", 0.0)),
        )
        if not 2 <= values.quadrature_order <= 32:
            raise ValueError("options.quadrature_order must be from 2 through 32.")
        for name in ("panel_cell_fraction", "step_cap_cells"):
            value = getattr(values, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"options.{name} must be finite and positive.")
        if not isfinite(values.armijo) or not 0 < values.armijo < 1:
            raise ValueError("options.armijo must lie strictly between zero and one.")
        if not isfinite(values.backtrack_factor) or not 0 < values.backtrack_factor < 1:
            raise ValueError("options.backtrack_factor must lie strictly between zero and one.")
        if values.line_search_steps < 1:
            raise ValueError("options.line_search_steps must be a positive integer.")
        if not isfinite(values.clearance_m) or values.clearance_m < 0:
            raise ValueError("options.clearance_m must be finite and nonnegative.")
        return values


def _validated_route(route_m: ArrayLike) -> FloatArray:
    route = np.asarray(route_m, dtype=float)
    if route.ndim != 2 or route.shape[1:] != (2,) or len(route) < 3:
        raise ValueError("route_m must have shape (n >= 3, 2).")
    if not np.isfinite(route).all():
        raise ValueError("route_m must contain only finite coordinates.")
    return route


def _cell_size(field: TerrainField) -> float:
    scenario = field.scenario
    return float(
        min(
            np.min(np.diff(np.asarray(scenario.field_x_m))),
            np.min(np.diff(np.asarray(scenario.field_y_m))),
        )
    )


def _required_panels(
    route: FloatArray, field: TerrainField, panel_cell_fraction: float, allowance_m: float = 0.0
) -> tuple[int, ...]:
    target = panel_cell_fraction * _cell_size(field)
    lengths = np.linalg.norm(np.diff(route, axis=0), axis=1) + 2 * allowance_m
    return tuple(int(value) for value in np.maximum(1, np.ceil(lengths / target)))


def _quadrature_samples(
    route: FloatArray, panels: tuple[int, ...], order: int
) -> tuple[list[FloatArray], list[FloatArray], FloatArray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    rule_nodes = (nodes + 1.0) / 2.0
    rule_weights = weights / 2.0
    parameters_by_segment: list[FloatArray] = []
    weights_by_segment: list[FloatArray] = []
    samples = []
    for start, end, panel_count in zip(route[:-1], route[1:], panels, strict=True):
        parameters = (
            (np.arange(panel_count, dtype=float)[:, None] + rule_nodes) / panel_count
        ).ravel()
        unit_weights = np.broadcast_to(rule_weights / panel_count, (panel_count, order)).ravel()
        points = (1.0 - parameters[:, None]) * start + parameters[:, None] * end
        points = np.clip(points, np.minimum(start, end), np.maximum(start, end))
        parameters_by_segment.append(parameters)
        weights_by_segment.append(unit_weights)
        samples.append(points)
    return parameters_by_segment, weights_by_segment, np.concatenate(samples)


def _validated_panels(route: FloatArray, quadrature_panels: Sequence[int]) -> tuple[int, ...]:
    panels = tuple(quadrature_panels)
    if len(panels) != len(route) - 1 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1
        for value in panels
    ):
        raise ValueError("quadrature_panels must contain one positive integer per segment.")
    return tuple(int(value) for value in panels)


def terrain_energy_and_gradient(
    route_m: ArrayLike,
    field: TerrainField,
    *,
    quadrature_order: int = 8,
    quadrature_panels: Sequence[int] | None = None,
) -> tuple[float, FloatArray]:
    """Return physical energy in seconds squared and its exact vertex gradient.

    The analytic gradient differentiates the same composite Gauss--Legendre
    sum returned as the energy.  With no explicit panel counts, each panel is
    no longer than the smaller source-grid spacing.
    """
    route = _validated_route(route_m)
    if not isinstance(field, TerrainField):
        raise TypeError("field must be a TerrainField.")
    if (
        isinstance(quadrature_order, bool)
        or not isinstance(quadrature_order, int)
        or not 2 <= quadrature_order <= 32
    ):
        raise ValueError("quadrature_order must be from 2 through 32.")
    panels = (
        _required_panels(route, field, 1.0)
        if quadrature_panels is None
        else _validated_panels(route, quadrature_panels)
    )
    parameters_by_segment, weights_by_segment, samples = _quadrature_samples(
        route, panels, quadrature_order
    )
    cost = field.cost(samples)
    cost_gradient = field.gradient(samples)
    intervals = len(route) - 1
    value = 0.0
    gradient = np.zeros_like(route)
    offset = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for index, (start, end, parameters, weights) in enumerate(
            zip(
                route[:-1],
                route[1:],
                parameters_by_segment,
                weights_by_segment,
                strict=True,
            )
        ):
            selected = slice(offset, offset + len(parameters))
            segment_cost = cost[selected]
            segment_gradient = cost_gradient[selected]
            offset += len(parameters)
            delta = end - start
            length2 = float(delta @ delta)
            cost2_integral = float(np.dot(weights, segment_cost * segment_cost))
            spatial = (2 * weights * segment_cost)[:, None] * segment_gradient
            value += intervals * length2 * cost2_integral
            gradient[index] += intervals * (
                -2 * delta * cost2_integral
                + length2 * np.sum((1 - parameters)[:, None] * spatial, axis=0)
            )
            gradient[index + 1] += intervals * (
                2 * delta * cost2_integral + length2 * np.sum(parameters[:, None] * spatial, axis=0)
            )
    return float(value), gradient


def _terrain_energy(
    route: FloatArray,
    field: TerrainField,
    panels: tuple[int, ...],
    quadrature_order: int,
) -> float:
    _, weights_by_segment, samples = _quadrature_samples(route, panels, quadrature_order)
    cost = field.cost(samples)
    intervals = len(route) - 1
    value = 0.0
    offset = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for start, end, weights in zip(route[:-1], route[1:], weights_by_segment, strict=True):
            selected = cost[offset : offset + len(weights)]
            offset += len(weights)
            delta = end - start
            value += intervals * float(delta @ delta) * float(np.dot(weights, selected * selected))
    return float(value)


def _failed_initialization(
    scenario: TerrainScenario, config: PlannerConfig, seed: SeedResult, overall_start: float
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
    field: TerrainField,
    diagnostics: dict,
    reason: str,
    success: bool,
    preprocessing_time: float,
    solve_time: float,
    overall_start: float,
) -> PlannerResult:
    evaluation_started = perf_counter()
    try:
        evaluation = evaluate_route(
            scenario, route, field=field, profile_samples=config.profile_samples
        )
    except (FloatingPointError, ValueError) as exc:
        return PlannerResult(
            method=config.method,
            initialization=config.initialization,
            route_m=tuple(map(tuple, route)),
            evaluated_cost_s=None,
            feasible=False,
            solver_success=False,
            termination_reason="evaluation_failed",
            diagnostics={
                "initialization": seed.diagnostics,
                **diagnostics,
                "evaluation_error": {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "solver_termination_reason": reason,
                },
            },
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
        solver_success=success,
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


def _geometry_feasible(
    route: FloatArray, field: TerrainField, clearance_m: float
) -> tuple[bool, str | None]:
    from shapely.geometry import LineString, box

    line = LineString(route)
    if not box(*field.scenario.bounds_m).covers(line):
        return False, "outside_domain"
    if not field.barriers.is_empty:
        if line.intersects(field.barriers):
            return False, "barrier_collision"
        # Barrier contact is already rejected by intersects.  Positive
        # clearance is inclusive: equality satisfies the requested distance.
        if line.distance(field.barriers) < clearance_m:
            return False, "barrier_clearance"
    return True, None


def _normalization(scenario: TerrainScenario) -> tuple[FloatArray, float, float, float]:
    xmin, ymin, xmax, ymax = scenario.bounds_m
    origin = np.array([xmin, ymin], dtype=float)
    coordinate_scale = float(np.hypot(xmax - xmin, ymax - ymin))
    log_slowness = np.asarray(scenario.log_slowness, dtype=float)
    slowness_scale = exp(float(np.median(log_slowness)))
    energy_scale = (coordinate_scale * slowness_scale) ** 2
    if (
        not np.isfinite([coordinate_scale, slowness_scale, energy_scale]).all()
        or min(coordinate_scale, slowness_scale, energy_scale) <= 0
    ):
        raise FloatingPointError("Terrain normalization scale is nonfinite.")
    return origin, coordinate_scale, slowness_scale, energy_scale


def _preconditioned_direction(gradient: FloatArray) -> FloatArray:
    from scipy.linalg import solve_banded

    interior = len(gradient)
    banded = np.zeros((3, interior))
    banded[0, 1:] = -1
    banded[1] = 2
    banded[2, :-1] = -1
    return solve_banded((1, 1), banded, -gradient)


def _ode_residual_norm(route: FloatArray, field: TerrainField) -> float | None:
    from .local_planners import euler_lagrange_residual

    try:
        values = euler_lagrange_residual(route, field)
        result = float(np.linalg.norm(values) / np.sqrt(values.size))
        return result if np.isfinite(result) else None
    except (FloatingPointError, ValueError):
        return None


def solve_energy_descent(
    scenario: TerrainScenario, config: PlannerConfig, *, deadline: float | None = None
) -> PlannerResult:
    """Minimize terrain energy with preconditioned, feasible Armijo steps."""
    if config.method != "energy_descent":
        raise ValueError("Energy descent solver requires method='energy_descent'.")
    overall_start = perf_counter()
    deadline = overall_start + config.time_limit_s if deadline is None else deadline
    options = _EnergyOptions.from_config(config)
    seed = make_seed(
        scenario,
        config.initialization,
        config.interior_points,
        clearance_m=options.clearance_m,
        reference_grid_size=config.reference_grid_size,
        deadline=deadline,
    )
    seed.diagnostics["initial_route_hash"] = seed.diagnostics.get("route_hash")
    if not seed.success or seed.route_m is None:
        return _failed_initialization(scenario, config, seed, overall_start)

    preprocessing_started = perf_counter()
    field = TerrainField(scenario)
    preprocessing_time = perf_counter() - preprocessing_started
    route = np.asarray(seed.route_m, dtype=float).copy()
    initial_feasible, initial_violation = _geometry_feasible(route, field, options.clearance_m)
    solve_started = perf_counter()
    reason = "iteration_limit"
    success = False
    iterations = 0
    last_step_size = 0.0
    infeasible_rejections = 0
    error: dict = {}
    energy_history: list[float] = []
    normalized_history: list[float] = []
    accepted_baselines: list[float] = []
    accepted_energies: list[float] = []
    accepted_steps: list[float] = []
    panels: tuple[int, ...] = (1,) * (len(route) - 1)
    physical_energy: float | None = None
    normalized_energy: float | None = None
    free_gradient_norm: float | None = None
    scaled_gradient_norm: float | None = None
    initial_gradient_norm: float | None = None
    initial_energy: float | None = None
    initial_normalized_energy: float | None = None
    coordinate_scale: float | None = None
    slowness_scale: float | None = None
    energy_scale: float | None = None
    gradient_diagnostics_current = False

    try:
        _check_deadline(deadline)
        origin, coordinate_scale, slowness_scale, energy_scale = _normalization(scenario)
        step_cap_m = options.step_cap_cells * _cell_size(field)
        panels = _required_panels(route, field, options.panel_cell_fraction, allowance_m=step_cap_m)
        if not initial_feasible:
            reason = "infeasible_initialization"
        else:
            while True:
                _check_deadline(deadline)
                required = _required_panels(
                    route, field, options.panel_cell_fraction, allowance_m=step_cap_m
                )
                panels = tuple(max(old, new) for old, new in zip(panels, required, strict=True))
                physical_energy, physical_gradient = terrain_energy_and_gradient(
                    route,
                    field,
                    quadrature_order=options.quadrature_order,
                    quadrature_panels=panels,
                )
                normalized_energy = physical_energy / energy_scale
                # z=(q-origin)/L and F=E/E_scale imply dF/dz=L*dE/dq/E_scale.
                normalized_gradient = coordinate_scale * physical_gradient / energy_scale
                free_gradient_norm = float(
                    np.linalg.norm(normalized_gradient[1:-1]) / np.sqrt(2 * config.interior_points)
                )
                gradient_diagnostics_current = True
                if initial_gradient_norm is None:
                    initial_gradient_norm = max(1.0, free_gradient_norm)
                    initial_energy = physical_energy
                    initial_normalized_energy = normalized_energy
                    energy_history.append(physical_energy)
                    normalized_history.append(normalized_energy)
                scaled_gradient_norm = free_gradient_norm / initial_gradient_norm
                if not np.isfinite(
                    [
                        physical_energy,
                        normalized_energy,
                        free_gradient_norm,
                        scaled_gradient_norm,
                    ]
                ).all():
                    reason = "nonfinite"
                    break
                if scaled_gradient_norm <= config.tolerance:
                    reason, success = "stationary", True
                    break
                if iterations >= config.max_iterations:
                    break

                direction = _preconditioned_direction(normalized_gradient[1:-1])
                derivative = float(np.sum(normalized_gradient[1:-1] * direction))
                if not np.isfinite(direction).all() or not np.isfinite(derivative):
                    reason = "nonfinite"
                    break
                if derivative >= 0:
                    reason = "stagnated"
                    break
                physical_direction = coordinate_scale * direction
                largest_move = float(np.max(np.linalg.norm(physical_direction, axis=1)))
                if largest_move == 0:
                    reason = "stagnated"
                    break
                step_size = min(1.0, step_cap_m / largest_move)
                accepted = False
                rejected_for_geometry = 0
                for _ in range(options.line_search_steps):
                    _check_deadline(deadline)
                    candidate = route.copy()
                    candidate[1:-1] = origin + coordinate_scale * (
                        (route[1:-1] - origin) / coordinate_scale + step_size * direction
                    )
                    candidate[[0, -1]] = [scenario.start_m, scenario.goal_m]
                    feasible, _ = _geometry_feasible(candidate, field, options.clearance_m)
                    if not feasible:
                        infeasible_rejections += 1
                        rejected_for_geometry += 1
                    else:
                        candidate_energy = _terrain_energy(
                            candidate, field, panels, options.quadrature_order
                        )
                        candidate_normalized = candidate_energy / energy_scale
                        if np.isfinite(candidate_normalized) and candidate_normalized <= (
                            normalized_energy + options.armijo * step_size * derivative
                        ):
                            accepted_baselines.append(physical_energy)
                            accepted_energies.append(candidate_energy)
                            accepted_steps.append(step_size)
                            energy_history.append(candidate_energy)
                            normalized_history.append(candidate_normalized)
                            route = candidate
                            physical_energy = candidate_energy
                            normalized_energy = candidate_normalized
                            gradient_diagnostics_current = False
                            last_step_size = step_size
                            accepted = True
                            break
                    step_size *= options.backtrack_factor
                iterations += 1
                if not accepted:
                    reason = "constraint_blocked" if rejected_for_geometry else "stagnated"
                    break
    except _DeadlineExceeded:
        reason = "time_limit"
    except (FloatingPointError, OverflowError):
        reason = "nonfinite"
    except Exception as exc:
        reason = "solver_error"
        error = {"exception_type": type(exc).__name__, "message": str(exc)}

    solve_time = perf_counter() - solve_started
    if not gradient_diagnostics_current:
        free_gradient_norm = None
        scaled_gradient_norm = None
    ode_residual = _ode_residual_norm(route, field)
    constraint_feasible, final_violation = _geometry_feasible(route, field, options.clearance_m)
    diagnostics = {
        "iterations": iterations,
        "energy_s2": physical_energy,
        "normalized_energy": normalized_energy,
        "initial_energy_s2": initial_energy,
        "initial_normalized_energy": initial_normalized_energy,
        "energy_scale_s2": energy_scale,
        "coordinate_scale_m": coordinate_scale,
        "slowness_scale_s_per_m": slowness_scale,
        "free_gradient_norm": free_gradient_norm,
        "scaled_free_gradient_norm": scaled_gradient_norm,
        "stationarity_norm": scaled_gradient_norm,
        "stationarity_tolerance": config.tolerance,
        "gradient_diagnostics_current": gradient_diagnostics_current,
        "ode_residual_norm_m": ode_residual,
        "last_step_size": last_step_size,
        "infeasible_trial_rejections": infeasible_rejections,
        "constraint_feasible": constraint_feasible,
        "constraint_violation": final_violation or initial_violation,
        "requested_clearance_m": options.clearance_m,
        "quadrature_order": options.quadrature_order,
        "quadrature_panels": list(panels),
        "max_quadrature_panels": max(panels),
        "energy_history_s2": energy_history,
        "normalized_energy_history": normalized_history,
        "accepted_baseline_energy_s2": accepted_baselines,
        "accepted_energy_s2": accepted_energies,
        "accepted_step_sizes": accepted_steps,
        "gradient_definition": (
            "RMS over interior coordinates of d(E/energy_scale_s2)/d(q/coordinate_scale_m)"
        ),
        **error,
    }
    # Success is solely the declared unconstrained energy-gradient criterion;
    # a constraint-blocked route is retained but never called stationary.
    success = bool(
        success
        and constraint_feasible
        and scaled_gradient_norm is not None
        and scaled_gradient_norm <= config.tolerance
    )
    return _result(
        scenario,
        config,
        route,
        seed,
        field,
        diagnostics,
        reason,
        success,
        preprocessing_time,
        solve_time,
        overall_start,
    )


__all__ = ["solve_energy_descent", "terrain_energy_and_gradient"]
