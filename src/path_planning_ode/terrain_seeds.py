"""Method-independent initial routes for terrain planners.

The geometric seeds in this module deliberately do not inspect terrain cost.
This keeps comparisons between local methods fair: both methods start from the
same polyline for a given scenario, initialization name, and point count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .terrain import TerrainScenario, _make_barrier_geometry

FloatArray = NDArray[np.float64]
INITIALIZATIONS = ("straight", "arc_left", "arc_right", "barrier", "fast_marching")


@dataclass(frozen=True)
class SeedResult:
    """Outcome of constructing one initialization polyline."""

    name: str
    route_m: FloatArray | None
    success: bool
    reason: str
    timing_s: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def resample_route(route_m: FloatArray, vertices: int) -> FloatArray:
    """Resample a polyline at equal arclength, retaining both endpoints."""
    route = np.asarray(route_m, dtype=float)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2:
        raise ValueError("route_m must have shape (n >= 2, 2).")
    if isinstance(vertices, bool) or not isinstance(vertices, int) or vertices < 2:
        raise ValueError("vertices must be an integer of at least 2.")
    lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > 0))
    route = route[keep]
    if len(route) < 2:
        raise ValueError("Cannot resample a zero-length route.")
    positions = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))))
    samples = np.linspace(0.0, positions[-1], vertices)
    result = np.column_stack([np.interp(samples, positions, route[:, axis]) for axis in range(2)])
    result[[0, -1]] = route[[0, -1]]
    return result


def _arc_seed(scenario: TerrainScenario, vertices: int, side: float) -> FloatArray:
    start = np.asarray(scenario.start_m)
    goal = np.asarray(scenario.goal_m)
    displacement = goal - start
    unit_normal = np.array([-displacement[1], displacement[0]]) / np.linalg.norm(displacement)
    xmin, ymin, xmax, ymax = scenario.bounds_m
    # Pick the largest modest bend that remains in the rectangular domain.
    amplitude = 0.2 * np.linalg.norm(displacement)
    t = np.linspace(0.0, 1.0, vertices)
    shape = np.sin(np.pi * t)
    normal = side * unit_normal
    for axis, low, high in ((0, xmin, xmax), (1, ymin, ymax)):
        coefficient = shape * normal[axis]
        base = start[axis] + t * displacement[axis]
        positive = coefficient > 0
        negative = coefficient < 0
        if np.any(positive):
            amplitude = min(
                amplitude, float(np.min((high - base[positive]) / coefficient[positive]))
            )
        if np.any(negative):
            amplitude = min(
                amplitude, float(np.min((low - base[negative]) / coefficient[negative]))
            )
    amplitude = max(0.0, 0.95 * amplitude)
    route = start + t[:, None] * displacement + amplitude * shape[:, None] * normal
    route[[0, -1]] = [start, goal]
    return route


def _polygon_vertices(geometry: Any) -> Iterable[tuple[float, float]]:
    if geometry.geom_type == "Polygon":
        yield from list(geometry.exterior.coords)[:-1]
        for ring in geometry.interiors:
            yield from list(ring.coords)[:-1]
    elif geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            yield from _polygon_vertices(polygon)


def _barrier_seed(
    scenario: TerrainScenario, vertices: int, clearance_m: float, deadline: float | None
) -> tuple[FloatArray | None, str, dict[str, Any]]:
    """Shortest visibility path around barrier geometry, using Euclidean length."""
    from heapq import heappop, heappush

    from shapely.geometry import LineString, Point, box

    barriers = _make_barrier_geometry(scenario.barriers_geojson)
    if barriers.is_empty:
        return (
            resample_route(np.asarray([scenario.start_m, scenario.goal_m]), vertices),
            "ok",
            {"visibility_vertices": 2},
        )

    xmin, ymin, xmax, ymax = scenario.bounds_m
    scale = max(xmax - xmin, ymax - ymin)
    # Padding by two nominal output segments keeps equal-arclength resampling
    # from shaving an obstacle corner between retained vertices.
    offset = max(float(clearance_m), 2 * scale / (vertices - 1), scale * 1e-8)
    expanded = barriers.buffer(offset, join_style=2)
    domain = box(xmin, ymin, xmax, ymax)
    candidate_points = [scenario.start_m, scenario.goal_m]
    for point in _polygon_vertices(expanded):
        if domain.covers(Point(point)):
            candidate_points.append((float(point[0]), float(point[1])))
    # Stable de-duplication matters for deterministic benchmark hashes/results.
    points = np.asarray(list(dict.fromkeys(candidate_points)), dtype=float)
    count = len(points)
    graph: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            if deadline is not None and perf_counter() >= deadline:
                return None, "time_limit", {"visibility_vertices": count}
            segment = LineString([points[i], points[j]])
            if domain.covers(segment) and not segment.intersects(barriers):
                length = float(np.linalg.norm(points[j] - points[i]))
                graph[i].append((j, length))
                graph[j].append((i, length))

    distances = [float("inf")] * count
    previous = [-1] * count
    distances[0] = 0.0
    queue: list[tuple[float, int]] = [(0.0, 0)]
    while queue:
        distance, node = heappop(queue)
        if distance != distances[node]:
            continue
        if node == 1:
            break
        for neighbor, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heappush(queue, (candidate, neighbor))
    if not isfinite(distances[1]):
        return None, "no_geometric_route", {"visibility_vertices": count}
    indices = [1]
    while indices[-1] != 0:
        indices.append(previous[indices[-1]])
    route = points[np.asarray(indices[::-1])]
    sampled = resample_route(route, vertices)
    if LineString(sampled).intersects(barriers):
        return (
            None,
            "initialization_resolution_limited",
            {
                "visibility_vertices": count,
                "visibility_path_vertices": len(route),
                "requested_vertices": vertices,
            },
        )
    return (
        sampled,
        "ok",
        {
            "visibility_vertices": count,
            "visibility_path_vertices": len(route),
            "clearance_m": offset,
        },
    )


def make_seed(
    scenario: TerrainScenario,
    initialization: str,
    interior_points: int,
    *,
    clearance_m: float = 0.0,
    reference_grid_size: int = 129,
    deadline: float | None = None,
    fast_marching_route: FloatArray | None = None,
) -> SeedResult:
    """Construct one named seed with exactly ``interior_points + 2`` vertices."""
    started = perf_counter()
    vertices = interior_points + 2
    try:
        if deadline is not None and started >= deadline:
            raise TimeoutError
        if initialization == "straight":
            route = np.linspace(scenario.start_m, scenario.goal_m, vertices)
            reason, diagnostics = "ok", {}
        elif initialization == "arc_left":
            route = _arc_seed(scenario, vertices, 1.0)
            reason, diagnostics = "ok", {}
        elif initialization == "arc_right":
            route = _arc_seed(scenario, vertices, -1.0)
            reason, diagnostics = "ok", {}
        elif initialization == "barrier":
            route, reason, diagnostics = _barrier_seed(scenario, vertices, clearance_m, deadline)
        elif initialization == "fast_marching":
            if fast_marching_route is None:
                from .fast_marching import compute_arrival_time, extract_route

                solution = compute_arrival_time(
                    scenario, grid_size=reference_grid_size, deadline=deadline
                )
                fast_marching_route = extract_route(scenario, solution, deadline=deadline)
            route = resample_route(np.asarray(fast_marching_route), vertices)
            reason, diagnostics = "ok", {"reference_grid_size": reference_grid_size}
        else:
            route, reason, diagnostics = None, "unknown_initialization", {}
        if route is not None and initialization in {"barrier", "fast_marching"}:
            from shapely.geometry import LineString, box

            line = LineString(route)
            domain = box(*scenario.bounds_m)
            barriers = _make_barrier_geometry(scenario.barriers_geojson)
            if not domain.covers(line) or line.intersects(barriers):
                route = None
                reason = "initialization_resolution_limited"
                diagnostics = {
                    **diagnostics,
                    "resampling_collision": line.intersects(barriers),
                    "resampling_outside_domain": not domain.covers(line),
                }
    except TimeoutError:
        route, reason, diagnostics = None, "time_limit", {}
    except Exception as exc:  # Initialization failure is a result, not a planner crash.
        failure_reason = getattr(exc, "reason", "initialization_failed")
        if failure_reason in {"timeout", "time_limit"}:
            failure_reason = "time_limit"
        route, reason, diagnostics = (
            None,
            failure_reason,
            {
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
    diagnostics = {
        **diagnostics,
        "requested_vertices": vertices,
        "actual_vertices": None if route is None else len(route),
        "route_hash": (
            None
            if route is None
            else sha256(np.asarray(route, dtype="<f8").tobytes(order="C")).hexdigest()
        ),
    }
    elapsed = perf_counter() - started
    return SeedResult(initialization, route, route is not None, reason, elapsed, diagnostics)


def seed_bank(
    scenario: TerrainScenario,
    interior_points: int,
    *,
    names: Iterable[str] = INITIALIZATIONS,
    clearance_m: float = 0.0,
    reference_grid_size: int = 129,
    deadline: float | None = None,
) -> dict[str, SeedResult]:
    """Build a deterministic bank; local planners call this same function."""
    return {
        name: make_seed(
            scenario,
            name,
            interior_points,
            clearance_m=clearance_m,
            reference_grid_size=reference_grid_size,
            deadline=deadline,
        )
        for name in names
    }


__all__ = ["INITIALIZATIONS", "SeedResult", "make_seed", "resample_route", "seed_bank"]
