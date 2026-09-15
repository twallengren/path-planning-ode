"""First-order isotropic fast marching used as a grid-based reference.

This module solves ``|grad(T)| = c`` on a conservatively masked Cartesian grid.
The extracted route descends a continuous piecewise-affine arrival-time field on
a fixed triangulation.  It is a numerical reference, not a certified continuous
optimum and not a graph shortest path.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import isfinite, sqrt
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from .terrain import (
    JSONValue,
    PlannerConfig,
    PlannerResult,
    TerrainField,
    TerrainScenario,
    _integrate_segment,
    evaluate_route,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class FastMarchingError(RuntimeError):
    """Expected numerical failure with a machine-readable reason."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class FastMarchingSolution:
    """Arrival-time grid and metadata for route extraction or warm starts."""

    x_m: FloatArray
    y_m: FloatArray
    arrival_s: FloatArray
    masked: BoolArray
    goal_seed_indices: tuple[tuple[int, int], ...]
    accepted_count: int
    preprocessing_time_s: float
    propagation_time_s: float
    scenario_hash: str

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.arrival_s.shape

    @property
    def masked_count(self) -> int:
        return int(np.count_nonzero(self.masked))

    @property
    def reachable_count(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.arrival_s)))

    def summary(self) -> dict[str, JSONValue]:
        finite = self.arrival_s[np.isfinite(self.arrival_s)]
        return {
            "grid_shape": list(self.grid_shape),
            "grid_spacing_m": [
                float(self.x_m[1] - self.x_m[0]),
                float(self.y_m[1] - self.y_m[0]),
            ],
            "accepted_count": self.accepted_count,
            "reachable_count": self.reachable_count,
            "masked_count": self.masked_count,
            "goal_seed_indices": [list(index) for index in self.goal_seed_indices],
            "arrival_range_s": (
                None if len(finite) == 0 else [float(np.min(finite)), float(np.max(finite))]
            ),
        }

    def to_dict(self, *, include_arrival: bool = True) -> dict[str, JSONValue]:
        """Serialize the solution; dense arrays are optional for benchmark records."""
        result = self.summary()
        result.update(
            {
                "x_m": self.x_m.tolist(),
                "y_m": self.y_m.tolist(),
                "preprocessing_time_s": self.preprocessing_time_s,
                "propagation_time_s": self.propagation_time_s,
                "scenario_hash": self.scenario_hash,
            }
        )
        if include_arrival:
            result["arrival_s"] = [
                [None if not isfinite(value) else float(value) for value in row]
                for row in self.arrival_s
            ]
            result["masked"] = self.masked.tolist()
        return result


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= deadline:
        raise FastMarchingError("timeout", "Fast marching exceeded its time limit.")


def _conservative_mask(
    scenario: TerrainScenario, x_m: FloatArray, y_m: FloatArray, field: TerrainField
) -> BoolArray:
    """Mask nodes whose clipped Voronoi control cell touches a barrier."""
    mask = np.zeros((len(y_m), len(x_m)), dtype=bool)
    barrier = field.barriers
    if barrier.is_empty:
        return mask

    import shapely

    dx = float(x_m[1] - x_m[0])
    dy = float(y_m[1] - y_m[0])
    xmin, ymin, xmax, ymax = scenario.bounds_m
    left = np.maximum(x_m - dx / 2, xmin)
    right = np.minimum(x_m + dx / 2, xmax)
    bottom = np.maximum(y_m - dy / 2, ymin)
    top = np.minimum(y_m + dy / 2, ymax)
    bxmin, bymin, bxmax, bymax = barrier.bounds
    x_indices = np.flatnonzero((right >= bxmin) & (left <= bxmax))
    y_indices = np.flatnonzero((top >= bymin) & (bottom <= bymax))
    if len(x_indices) == 0 or len(y_indices) == 0:
        return mask

    # Chunk geometry creation so a 1025-square study grid does not retain a
    # million temporary Python geometry objects.
    for offset in range(0, len(y_indices), 64):
        rows = y_indices[offset : offset + 64]
        cells = shapely.box(
            left[x_indices][None, :],
            bottom[rows][:, None],
            right[x_indices][None, :],
            top[rows][:, None],
        )
        mask[np.ix_(rows, x_indices)] = shapely.intersects(cells, barrier)
    return mask


def _endpoint_candidates(
    point: tuple[float, float],
    x_m: FloatArray,
    y_m: FloatArray,
    mask: BoolArray,
    field: TerrainField,
) -> list[tuple[float, int, int]]:
    from shapely.geometry import LineString

    ix = int(np.searchsorted(x_m, point[0]))
    iy = int(np.searchsorted(y_m, point[1]))
    columns = sorted({max(0, min(len(x_m) - 1, ix - 1)), max(0, min(len(x_m) - 1, ix))})
    rows = sorted({max(0, min(len(y_m) - 1, iy - 1)), max(0, min(len(y_m) - 1, iy))})
    point_array = np.asarray(point, dtype=float)
    candidates: list[tuple[float, int, int]] = []
    for row in rows:
        for column in columns:
            if mask[row, column]:
                continue
            node = np.array([x_m[column], y_m[row]])
            connector = LineString([point_array, node])
            if not field.barriers.is_empty and connector.intersects(field.barriers):
                continue
            cost = _integrate_segment(field, point_array, node, 12)
            candidates.append((cost, row, column))
    return candidates


def _upwind_update(
    row: int,
    column: int,
    arrival: FloatArray,
    accepted: BoolArray,
    slowness: FloatArray,
    dx: float,
    dy: float,
) -> float:
    ny, nx = arrival.shape
    horizontal = np.inf
    vertical = np.inf
    if column > 0 and accepted[row, column - 1]:
        horizontal = min(horizontal, arrival[row, column - 1])
    if column + 1 < nx and accepted[row, column + 1]:
        horizontal = min(horizontal, arrival[row, column + 1])
    if row > 0 and accepted[row - 1, column]:
        vertical = min(vertical, arrival[row - 1, column])
    if row + 1 < ny and accepted[row + 1, column]:
        vertical = min(vertical, arrival[row + 1, column])

    cost = slowness[row, column]
    if not isfinite(horizontal):
        return float(vertical + cost * dy) if isfinite(vertical) else np.inf
    if not isfinite(vertical):
        return float(horizontal + cost * dx)

    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2
    coefficient = inv_dx2 + inv_dy2
    linear = horizontal * inv_dx2 + vertical * inv_dy2
    constant = horizontal**2 * inv_dx2 + vertical**2 * inv_dy2 - cost**2
    discriminant = max(0.0, linear**2 - coefficient * constant)
    root = (linear + sqrt(discriminant)) / coefficient
    if root >= max(horizontal, vertical):
        return float(root)
    return float(min(horizontal + cost * dx, vertical + cost * dy))


def compute_arrival_time(
    scenario: TerrainScenario,
    grid_size: int = 129,
    *,
    field: TerrainField | None = None,
    deadline: float | None = None,
) -> FastMarchingSolution:
    """Compute a first-order arrival field from the exact scenario goal."""
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 3:
        raise ValueError("grid_size must be an integer of at least 3.")
    if deadline is not None and not isfinite(deadline):
        raise ValueError("deadline must be a finite perf_counter timestamp.")
    started = perf_counter()
    field = TerrainField(scenario) if field is None else field
    if field.scenario.scenario_hash != scenario.scenario_hash:
        raise ValueError("field and scenario do not describe the same terrain.")
    xmin, ymin, xmax, ymax = scenario.bounds_m
    x_m = np.linspace(xmin, xmax, grid_size)
    y_m = np.linspace(ymin, ymax, grid_size)
    dx = float(x_m[1] - x_m[0])
    dy = float(y_m[1] - y_m[0])
    mask = _conservative_mask(scenario, x_m, y_m, field)
    _check_deadline(deadline)
    xx, yy = np.meshgrid(x_m, y_m)
    slowness = np.asarray(field.cost(np.stack([xx, yy], axis=-1)), dtype=float)
    seeds = _endpoint_candidates(scenario.goal_m, x_m, y_m, mask, field)
    if not seeds:
        raise FastMarchingError(
            "target_resolution_limited",
            "No conservatively unmasked grid node can connect to the goal.",
        )
    preprocessing_time = perf_counter() - started

    arrival = np.full(mask.shape, np.inf)
    accepted = np.zeros(mask.shape, dtype=bool)
    trial = np.zeros(mask.shape, dtype=bool)
    heap: list[tuple[float, int, int]] = []
    for value, row, column in seeds:
        if value < arrival[row, column]:
            arrival[row, column] = value
            trial[row, column] = True
            heapq.heappush(heap, (value, row, column))

    propagation_started = perf_counter()
    accepted_count = 0
    pops = 0
    while heap:
        value, row, column = heapq.heappop(heap)
        if accepted[row, column] or value != arrival[row, column]:
            continue
        accepted[row, column] = True
        trial[row, column] = False
        accepted_count += 1
        for next_row, next_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if not (0 <= next_row < grid_size and 0 <= next_column < grid_size):
                continue
            if mask[next_row, next_column] or accepted[next_row, next_column]:
                continue
            candidate = _upwind_update(next_row, next_column, arrival, accepted, slowness, dx, dy)
            if candidate < arrival[next_row, next_column]:
                arrival[next_row, next_column] = candidate
                trial[next_row, next_column] = True
                heapq.heappush(heap, (candidate, next_row, next_column))
        pops += 1
        if pops % 2048 == 0:
            _check_deadline(deadline)
    _check_deadline(deadline)
    return FastMarchingSolution(
        x_m=x_m,
        y_m=y_m,
        arrival_s=arrival,
        masked=mask,
        goal_seed_indices=tuple((row, column) for _, row, column in seeds),
        accepted_count=accepted_count,
        preprocessing_time_s=preprocessing_time,
        propagation_time_s=perf_counter() - propagation_started,
        scenario_hash=scenario.scenario_hash,
    )


def _cell_triangles(row: int, column: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    return (
        ((row, column), (row, column + 1), (row + 1, column + 1)),
        ((row, column), (row + 1, column), (row + 1, column + 1)),
    )


def _affine_triangle(
    point: FloatArray,
    vertices: FloatArray,
    values: FloatArray,
    tolerance: float,
) -> tuple[float, FloatArray] | None:
    matrix = np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0]))
    try:
        barycentric = np.linalg.solve(matrix, point - vertices[0])
    except np.linalg.LinAlgError:
        return None
    weights = np.array([1 - barycentric.sum(), barycentric[0], barycentric[1]])
    if np.min(weights) < -tolerance or np.max(weights) > 1 + tolerance:
        return None
    coefficients = np.linalg.solve(np.column_stack((np.ones(3), vertices)), values)
    return float(coefficients[0] + coefficients[1:] @ point), coefficients[1:]


def _triangle_values(
    solution: FastMarchingSolution, point: FloatArray
) -> list[tuple[float, FloatArray]]:
    x_m, y_m = solution.x_m, solution.y_m
    if not (x_m[0] <= point[0] <= x_m[-1] and y_m[0] <= point[1] <= y_m[-1]):
        return []
    base_column = int(np.searchsorted(x_m, point[0], side="right") - 1)
    base_row = int(np.searchsorted(y_m, point[1], side="right") - 1)
    columns = {max(0, min(len(x_m) - 2, base_column))}
    rows = {max(0, min(len(y_m) - 2, base_row))}
    dx = float(x_m[1] - x_m[0])
    dy = float(y_m[1] - y_m[0])
    nearest_column = int(round((point[0] - x_m[0]) / dx))
    if 0 <= nearest_column < len(x_m) and abs(point[0] - x_m[nearest_column]) <= 1e-10 * dx:
        columns.add(max(0, min(len(x_m) - 2, base_column - 1)))
    nearest_row = int(round((point[1] - y_m[0]) / dy))
    if 0 <= nearest_row < len(y_m) and abs(point[1] - y_m[nearest_row]) <= 1e-10 * dy:
        rows.add(max(0, min(len(y_m) - 2, base_row - 1)))

    candidates: list[tuple[float, FloatArray]] = []
    for row in rows:
        for column in columns:
            for triangle in _cell_triangles(row, column):
                indices = np.asarray(triangle)
                values = solution.arrival_s[indices[:, 0], indices[:, 1]]
                if not np.isfinite(values).all():
                    continue
                vertices = np.column_stack((x_m[indices[:, 1]], y_m[indices[:, 0]]))
                candidate = _affine_triangle(point, vertices, values, 1e-9)
                if candidate is not None:
                    candidates.append(candidate)
    return candidates


def _arrival_value(solution: FastMarchingSolution, point: FloatArray) -> float | None:
    values = _triangle_values(solution, point)
    return None if not values else min(value for value, _ in values)


def _connector_clear(a: FloatArray, b: FloatArray, field: TerrainField) -> bool:
    if field.barriers.is_empty:
        return True
    from shapely.geometry import LineString

    return not LineString([a, b]).intersects(field.barriers)


def extract_route(
    scenario: TerrainScenario,
    solution: FastMarchingSolution,
    *,
    field: TerrainField | None = None,
    step_fraction: float = 0.35,
    max_steps: int | None = None,
    deadline: float | None = None,
) -> FloatArray:
    """Descend the triangulated piecewise-affine arrival-time field."""
    if solution.scenario_hash != scenario.scenario_hash:
        raise ValueError("solution and scenario do not describe the same terrain.")
    if not isfinite(step_fraction) or not (0 < step_fraction <= 1):
        raise ValueError("step_fraction must lie in (0, 1].")
    field = TerrainField(scenario) if field is None else field
    if field.scenario.scenario_hash != scenario.scenario_hash:
        raise ValueError("field and scenario do not describe the same terrain.")
    start_candidates = _endpoint_candidates(
        scenario.start_m, solution.x_m, solution.y_m, solution.masked, field
    )
    if not start_candidates:
        raise FastMarchingError(
            "start_resolution_limited",
            "No conservatively unmasked grid node can connect to the start.",
        )
    reachable = [
        (connector + solution.arrival_s[row, column], row, column)
        for connector, row, column in start_candidates
        if isfinite(solution.arrival_s[row, column])
    ]
    if not reachable:
        raise FastMarchingError(
            "unreachable_on_grid",
            "Start and goal are disconnected on this conservative grid.",
        )
    _, row, column = min(reachable)
    start = np.asarray(scenario.start_m, dtype=float)
    goal = np.asarray(scenario.goal_m, dtype=float)
    current = np.array([solution.x_m[column], solution.y_m[row]])
    route = [start]
    if np.linalg.norm(current - start) > 1e-12:
        route.append(current.copy())

    dx = float(solution.x_m[1] - solution.x_m[0])
    dy = float(solution.y_m[1] - solution.y_m[0])
    step = step_fraction * min(dx, dy)
    goal_radius = 1.5 * sqrt(dx**2 + dy**2)
    limit = max_steps if max_steps is not None else 12 * max(solution.grid_shape)
    previous_value = _arrival_value(solution, current)
    if previous_value is None:
        raise FastMarchingError("extraction_failed", "Start connector reached no valid triangle.")

    for iteration in range(limit):
        if iteration % 128 == 0:
            _check_deadline(deadline)
        if np.linalg.norm(current - goal) <= goal_radius and _connector_clear(current, goal, field):
            if np.linalg.norm(current - goal) > 1e-12:
                route.append(goal)
            return np.asarray(route)

        candidates = _triangle_values(solution, current)
        proposals: list[tuple[float, FloatArray, float]] = []
        for _, gradient in candidates:
            norm = float(np.linalg.norm(gradient))
            if not isfinite(norm) or norm <= 1e-14:
                continue
            distance = step
            for _ in range(14):
                candidate = current - distance * gradient / norm
                value = _arrival_value(solution, candidate)
                if (
                    value is not None
                    and value < previous_value - 1e-10 * max(1.0, abs(previous_value))
                    and _connector_clear(current, candidate, field)
                ):
                    proposals.append((value, candidate, distance))
                    break
                distance *= 0.5
        # At a conservative mask boundary, every adjacent affine gradient can
        # point into an unavailable triangle. Search directions on the same
        # continuous interpolant for a feasible descending tangent.
        if not proposals:
            directions = np.column_stack(
                (
                    np.cos(np.linspace(0, 2 * np.pi, 32, endpoint=False)),
                    np.sin(np.linspace(0, 2 * np.pi, 32, endpoint=False)),
                )
            )
            distance = step
            for _ in range(14):
                for direction in directions:
                    candidate = current + distance * direction
                    value = _arrival_value(solution, candidate)
                    if (
                        value is not None
                        and value < previous_value - 1e-10 * max(1.0, abs(previous_value))
                        and _connector_clear(current, candidate, field)
                    ):
                        proposals.append((value, candidate, distance))
                if proposals:
                    break
                distance *= 0.5
        if not proposals:
            raise FastMarchingError(
                "extraction_failed",
                "Arrival descent stagnated before reaching a verified goal connector.",
            )
        next_value, next_point, moved = min(proposals, key=lambda item: item[0])
        if moved < 1e-8 * min(dx, dy):
            raise FastMarchingError("extraction_failed", "Arrival descent step became too small.")
        route.append(next_point.copy())
        current = next_point
        previous_value = next_value
    raise FastMarchingError("extraction_failed", "Arrival descent exceeded its step limit.")


def solve_fast_marching(scenario: TerrainScenario, config: PlannerConfig) -> PlannerResult:
    """Run the grid reference, extract a route, and independently evaluate it."""
    if config.method != "fast_marching":
        raise ValueError("solve_fast_marching requires method='fast_marching'.")
    started = perf_counter()
    deadline = started + config.time_limit_s
    field: TerrainField | None = None
    solution: FastMarchingSolution | None = None
    timing: dict[str, float] = {}
    diagnostics: dict[str, JSONValue] = {}
    initialization = config.initialization
    try:
        initialization_started = perf_counter()
        field = TerrainField(scenario)
        timing["initialization"] = perf_counter() - initialization_started
        solution = compute_arrival_time(
            scenario,
            config.reference_grid_size,
            field=field,
            deadline=deadline,
        )
        timing["preprocessing"] = solution.preprocessing_time_s
        timing["solve"] = solution.propagation_time_s
        diagnostics.update(solution.summary())
        if bool(config.options.get("include_arrival", False)):
            dense = solution.to_dict(include_arrival=True)
            diagnostics["arrival_time_s"] = dense["arrival_s"]
            diagnostics["arrival_x_m"] = dense["x_m"]
            diagnostics["arrival_y_m"] = dense["y_m"]
            diagnostics["conservative_mask"] = dense["masked"]

        extraction_started = perf_counter()
        route = extract_route(
            scenario,
            solution,
            field=field,
            step_fraction=float(config.options.get("extraction_step_fraction", 0.35)),
            max_steps=(
                None
                if config.options.get("max_extraction_steps") is None
                else int(config.options["max_extraction_steps"])
            ),
            deadline=deadline,
        )
        timing["extraction"] = perf_counter() - extraction_started
        _check_deadline(deadline)
        evaluation_started = perf_counter()
        evaluation = evaluate_route(
            scenario,
            route,
            field=field,
            profile_samples=config.profile_samples,
            quadrature_order=int(config.options.get("quadrature_order", 16)),
        )
        timing["evaluation"] = perf_counter() - evaluation_started
        timing["total"] = perf_counter() - started
        if not evaluation.feasible:
            return PlannerResult(
                method=config.method,
                initialization=initialization,
                route_m=tuple(map(tuple, route)),
                evaluated_cost_s=evaluation.cost_s,
                feasible=False,
                solver_success=False,
                termination_reason="route_infeasible",
                diagnostics=diagnostics,
                timing_s=timing,
                scenario_hash=scenario.scenario_hash,
                config_hash=config.config_hash,
                evaluation=evaluation,
            )
        return PlannerResult(
            method=config.method,
            initialization=initialization,
            route_m=tuple(map(tuple, route)),
            evaluated_cost_s=evaluation.cost_s,
            feasible=True,
            solver_success=True,
            termination_reason="converged",
            diagnostics=diagnostics,
            timing_s=timing,
            scenario_hash=scenario.scenario_hash,
            config_hash=config.config_hash,
            evaluation=evaluation,
        )
    except FastMarchingError as exc:
        timing["total"] = perf_counter() - started
        diagnostics["failure_detail"] = str(exc)
        if solution is not None:
            diagnostics.update(solution.summary())
        return PlannerResult(
            method=config.method,
            initialization=initialization,
            route_m=None,
            evaluated_cost_s=None,
            feasible=False,
            solver_success=False,
            termination_reason=exc.reason,
            diagnostics=diagnostics,
            timing_s=timing,
            scenario_hash=scenario.scenario_hash,
            config_hash=config.config_hash,
            evaluation=None,
        )


def fast_marching_seed(
    scenario: TerrainScenario,
    grid_size: int = 129,
    *,
    time_limit_s: float = 60.0,
) -> FloatArray:
    """Return a validated grid-reference route suitable as a local-method seed."""
    started = perf_counter()
    field = TerrainField(scenario)
    solution = compute_arrival_time(
        scenario, grid_size, field=field, deadline=started + time_limit_s
    )
    route = extract_route(scenario, solution, field=field, deadline=started + time_limit_s)
    evaluation = evaluate_route(scenario, route, field=field, profile_samples=2)
    if not evaluation.feasible:
        raise FastMarchingError("route_infeasible", "Extracted warm-start route is infeasible.")
    return route


__all__ = [
    "FastMarchingError",
    "FastMarchingSolution",
    "compute_arrival_time",
    "extract_route",
    "fast_marching_seed",
    "solve_fast_marching",
]
