"""Shared terrain problem, planner result, and route-evaluation contracts.

Coordinates are horizontal metres in ``(x, y)`` order.  Elevations are metres,
the cost field is slowness in seconds per horizontal metre, and route costs are
seconds.  The source field grid is part of a scenario and is intentionally
independent of every planner's discretization.

SciPy and Shapely are imported only when this module's v2 API is used.  The
package's version-1 Gaussian API can therefore remain lightweight in Pyodide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from math import isfinite
from time import perf_counter
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
PlannerMethod = Literal["euler_lagrange", "slsqp", "fast_marching"]


def _json_value(value: Any, *, path: str = "value") -> JSONValue:
    """Copy supported data into a deterministic, JSON-native representation."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), path=path)
    if isinstance(value, np.generic):
        return _json_value(value.item(), path=path)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings.")
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, path=f"{path}[]") for item in value]
    raise ValueError(f"{path} contains non-JSON value {type(value).__name__}.")


def _canonical_hash(data: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_value(data),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _point(value: Sequence[float], name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain two finite coordinates.")
    return float(array[0]), float(array[1])


def _float_tuple(value: Sequence[float], name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional sequence.")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class TerrainScenario:
    """Version-2, JSON-serializable definition of one fixed terrain problem.

    ``bounds_m`` is ``(xmin, ymin, xmax, ymax)``.  Grid-valued arrays use the
    conventional image shape ``(len(field_y_m), len(field_x_m))``.
    ``log_slowness`` is the natural logarithm of seconds per horizontal metre.
    Barriers are GeoJSON Polygon or MultiPolygon geometry objects and are kept
    separate from the finite cost field.
    """

    name: str
    bounds_m: tuple[float, float, float, float]
    start_m: tuple[float, float]
    goal_m: tuple[float, float]
    field_x_m: tuple[float, ...]
    field_y_m: tuple[float, ...]
    elevation_m: tuple[tuple[float, ...], ...]
    log_slowness: tuple[tuple[float, ...], ...]
    barriers_geojson: tuple[dict[str, JSONValue], ...] = ()
    provenance: dict[str, JSONValue] = field(default_factory=dict)
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    version: int = 2

    def __post_init__(self) -> None:
        if self.version != 2:
            raise ValueError("Unsupported terrain scenario version; expected 2.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string.")

        bounds = np.asarray(self.bounds_m, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError("bounds_m must contain finite (xmin, ymin, xmax, ymax).")
        xmin, ymin, xmax, ymax = (float(item) for item in bounds)
        if not (xmin < xmax and ymin < ymax):
            raise ValueError("bounds_m must have positive width and height.")
        object.__setattr__(self, "bounds_m", (xmin, ymin, xmax, ymax))

        start = _point(self.start_m, "start_m")
        goal = _point(self.goal_m, "goal_m")
        if start == goal:
            raise ValueError("start_m and goal_m must be distinct.")
        for label, point in (("start_m", start), ("goal_m", goal)):
            if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax):
                raise ValueError(f"{label} must lie inside the inclusive bounds.")
        object.__setattr__(self, "start_m", start)
        object.__setattr__(self, "goal_m", goal)

        x = _float_tuple(self.field_x_m, "field_x_m")
        y = _float_tuple(self.field_y_m, "field_y_m")
        if len(x) < 4 or len(y) < 4:
            raise ValueError("Cubic terrain fields require at least four coordinates per axis.")
        if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
            raise ValueError("Terrain field coordinates must be strictly increasing.")
        scale = max(xmax - xmin, ymax - ymin, 1.0)
        if not np.allclose(
            (x[0], y[0], x[-1], y[-1]), (xmin, ymin, xmax, ymax), atol=1e-12 * scale, rtol=0
        ):
            raise ValueError("Terrain field coordinates must cover bounds_m exactly.")
        object.__setattr__(self, "field_x_m", x)
        object.__setattr__(self, "field_y_m", y)

        expected_shape = (len(y), len(x))
        for label in ("elevation_m", "log_slowness"):
            array = np.asarray(getattr(self, label), dtype=float)
            if array.shape != expected_shape or not np.isfinite(array).all():
                raise ValueError(f"{label} must be finite with shape {expected_shape}.")
            object.__setattr__(self, label, tuple(tuple(float(v) for v in row) for row in array))

        barriers = _json_value(self.barriers_geojson, path="barriers_geojson")
        provenance = _json_value(self.provenance, path="provenance")
        metadata = _json_value(self.metadata, path="metadata")
        if not isinstance(barriers, list) or not all(isinstance(item, dict) for item in barriers):
            raise ValueError("barriers_geojson must be a sequence of GeoJSON geometry objects.")
        if not isinstance(provenance, dict) or not isinstance(metadata, dict):
            raise ValueError("provenance and metadata must be JSON objects.")
        object.__setattr__(self, "barriers_geojson", tuple(barriers))
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "metadata", metadata)

        # Validate topology at construction so malformed or misplaced barriers
        # never reach a planner.
        barriers_geometry = _make_barrier_geometry(self.barriers_geojson)
        if not barriers_geometry.is_empty:
            from shapely.geometry import Point, box

            domain = box(xmin, ymin, xmax, ymax)
            if not domain.covers(barriers_geometry):
                raise ValueError("Every barrier must lie inside bounds_m.")
            if barriers_geometry.intersects(Point(start)) or barriers_geometry.intersects(
                Point(goal)
            ):
                raise ValueError("Endpoints must not touch or lie inside a barrier.")

    @property
    def scenario_hash(self) -> str:
        """SHA-256 of the canonical serialized problem definition."""
        return _canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "version": self.version,
            "name": self.name,
            "bounds_m": list(self.bounds_m),
            "start_m": list(self.start_m),
            "goal_m": list(self.goal_m),
            "field_x_m": list(self.field_x_m),
            "field_y_m": list(self.field_y_m),
            "elevation_m": [list(row) for row in self.elevation_m],
            "log_slowness": [list(row) for row in self.log_slowness],
            "barriers_geojson": _json_value(self.barriers_geojson),
            "provenance": _json_value(self.provenance),
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TerrainScenario":
        if not isinstance(data, Mapping) or data.get("version") != 2:
            raise ValueError("Expected a version 2 terrain scenario object.")
        allowed = {
            "version",
            "name",
            "bounds_m",
            "start_m",
            "goal_m",
            "field_x_m",
            "field_y_m",
            "elevation_m",
            "log_slowness",
            "barriers_geojson",
            "provenance",
            "metadata",
        }
        if set(data) - allowed:
            raise ValueError("Terrain scenario contains unknown fields.")
        try:
            return cls(
                version=data["version"],
                name=data["name"],
                bounds_m=data["bounds_m"],
                start_m=data["start_m"],
                goal_m=data["goal_m"],
                field_x_m=data["field_x_m"],
                field_y_m=data["field_y_m"],
                elevation_m=data["elevation_m"],
                log_slowness=data["log_slowness"],
                barriers_geojson=tuple(data.get("barriers_geojson", ())),
                provenance=data.get("provenance", {}),
                metadata=data.get("metadata", {}),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed terrain scenario.") from exc


def _make_barrier_geometry(barriers_geojson: Sequence[Mapping[str, Any]]):
    from shapely.geometry import GeometryCollection, shape
    from shapely.ops import unary_union

    if not barriers_geojson:
        return GeometryCollection()
    geometries = []
    for item in barriers_geojson:
        try:
            geometry = shape(item)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid barrier GeoJSON geometry.") from exc
        if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Barriers must be GeoJSON Polygon or MultiPolygon geometries.")
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("Barrier geometries must be non-empty and valid.")
        geometries.append(geometry)
    return unary_union(geometries)


class TerrainField:
    """Bicubic elevation and positive log-slowness interpolants for a scenario."""

    def __init__(self, scenario: TerrainScenario):
        from scipy.interpolate import RectBivariateSpline

        self.scenario = scenario
        x = np.asarray(scenario.field_x_m)
        y = np.asarray(scenario.field_y_m)
        # RectBivariateSpline expects z.shape == (len(x), len(y)).
        self._log_spline = RectBivariateSpline(
            x, y, np.asarray(scenario.log_slowness).T, kx=3, ky=3, s=0
        )
        self._elevation_spline = RectBivariateSpline(
            x, y, np.asarray(scenario.elevation_m).T, kx=3, ky=3, s=0
        )
        self.barriers = _make_barrier_geometry(scenario.barriers_geojson)

    def _coordinates(self, points_m: ArrayLike) -> tuple[FloatArray, tuple[int, ...]]:
        points = np.asarray(points_m, dtype=float)
        if points.shape == (2,):
            shape: tuple[int, ...] = ()
            flat = points.reshape(1, 2)
        elif points.ndim >= 1 and points.shape[-1] == 2:
            shape = points.shape[:-1]
            flat = points.reshape(-1, 2)
        else:
            raise ValueError("points_m must have shape (..., 2).")
        if not np.isfinite(flat).all():
            raise ValueError("points_m must be finite.")
        xmin, ymin, xmax, ymax = self.scenario.bounds_m
        if np.any(
            (flat[:, 0] < xmin) | (flat[:, 0] > xmax) | (flat[:, 1] < ymin) | (flat[:, 1] > ymax)
        ):
            raise ValueError("Terrain field queries must lie inside bounds_m.")
        return flat, shape

    def _spline_value(
        self, spline: Any, points_m: ArrayLike, dx: int = 0, dy: int = 0
    ) -> FloatArray:
        points, shape = self._coordinates(points_m)
        return np.asarray(spline.ev(points[:, 0], points[:, 1], dx=dx, dy=dy)).reshape(shape)

    def log_slowness(self, points_m: ArrayLike) -> FloatArray:
        """Natural log of slowness, with units log(seconds / metre)."""
        return self._spline_value(self._log_spline, points_m)

    def cost(self, points_m: ArrayLike) -> FloatArray:
        """Positive slowness in seconds per horizontal metre."""
        return np.exp(self.log_slowness(points_m))

    def gradient(self, points_m: ArrayLike) -> FloatArray:
        """Slowness gradient in seconds per square metre."""
        points, shape = self._coordinates(points_m)
        log_value = self._log_spline.ev(points[:, 0], points[:, 1])
        log_gradient = np.stack(
            [
                self._log_spline.ev(points[:, 0], points[:, 1], dx=1, dy=0),
                self._log_spline.ev(points[:, 0], points[:, 1], dx=0, dy=1),
            ],
            axis=-1,
        )
        return (np.exp(log_value)[:, None] * log_gradient).reshape(shape + (2,))

    def hessian(self, points_m: ArrayLike) -> FloatArray:
        """Slowness Hessian in seconds per cubic metre."""
        points, shape = self._coordinates(points_m)
        x, y = points[:, 0], points[:, 1]
        log_value = self._log_spline.ev(x, y)
        gradient = np.stack(
            [self._log_spline.ev(x, y, dx=1), self._log_spline.ev(x, y, dy=1)], axis=-1
        )
        dxx = self._log_spline.ev(x, y, dx=2)
        dxy = self._log_spline.ev(x, y, dx=1, dy=1)
        dyy = self._log_spline.ev(x, y, dy=2)
        log_hessian = np.empty((len(points), 2, 2))
        log_hessian[:, 0, 0] = dxx
        log_hessian[:, 0, 1] = log_hessian[:, 1, 0] = dxy
        log_hessian[:, 1, 1] = dyy
        result = np.exp(log_value)[:, None, None] * (
            log_hessian + gradient[:, :, None] * gradient[:, None, :]
        )
        return result.reshape(shape + (2, 2))

    def elevation(self, points_m: ArrayLike) -> FloatArray:
        """Bicubic terrain elevation in metres."""
        return self._spline_value(self._elevation_spline, points_m)

    def evaluate(self, points_m: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return slowness, its gradient, and its Hessian."""
        return self.cost(points_m), self.gradient(points_m), self.hessian(points_m)


@dataclass(frozen=True)
class PlannerConfig:
    """Version-2 planner controls shared by native and browser implementations."""

    method: PlannerMethod
    initialization: str = "straight"
    interior_points: int = 32
    reference_grid_size: int = 129
    tolerance: float = 1e-7
    max_iterations: int = 1000
    time_limit_s: float = 60.0
    profile_samples: int = 129
    options: dict[str, JSONValue] = field(default_factory=dict)
    version: int = 2

    def __post_init__(self) -> None:
        if self.version != 2:
            raise ValueError("Unsupported planner configuration version; expected 2.")
        if self.method not in ("euler_lagrange", "slsqp", "fast_marching"):
            raise ValueError("Unknown planner method.")
        if not isinstance(self.initialization, str) or not self.initialization:
            raise ValueError("initialization must be a non-empty string.")
        for name in ("interior_points", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            isinstance(self.reference_grid_size, bool)
            or not isinstance(self.reference_grid_size, int)
            or self.reference_grid_size < 3
            or self.reference_grid_size % 2 == 0
        ):
            raise ValueError("reference_grid_size must be an odd integer of at least 3.")
        if (
            isinstance(self.profile_samples, bool)
            or not isinstance(self.profile_samples, int)
            or self.profile_samples < 2
        ):
            raise ValueError("profile_samples must be an integer of at least 2.")
        for name in ("tolerance", "time_limit_s"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        options = _json_value(self.options, path="options")
        if not isinstance(options, dict):
            raise ValueError("options must be a JSON object.")
        object.__setattr__(self, "options", options)

    @property
    def config_hash(self) -> str:
        return _canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "version": self.version,
            "method": self.method,
            "initialization": self.initialization,
            "interior_points": self.interior_points,
            "reference_grid_size": self.reference_grid_size,
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "time_limit_s": self.time_limit_s,
            "profile_samples": self.profile_samples,
            "options": _json_value(self.options),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlannerConfig":
        if not isinstance(data, Mapping) or data.get("version") != 2:
            raise ValueError("Expected a version 2 planner configuration object.")
        allowed = {
            "version",
            "method",
            "initialization",
            "interior_points",
            "reference_grid_size",
            "tolerance",
            "max_iterations",
            "time_limit_s",
            "profile_samples",
            "options",
        }
        if set(data) - allowed:
            raise ValueError("Planner configuration contains unknown fields.")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError("Malformed planner configuration.") from exc


@dataclass(frozen=True)
class RouteEvaluation:
    """Independent measurements and feasibility decision for a complete route."""

    cost_s: float | None
    length_m: float
    feasible: bool
    minimum_clearance_m: float | None
    violations: tuple[str, ...]
    distance_m: tuple[float, ...]
    elevation_m: tuple[float, ...]
    slowness_s_per_m: tuple[float, ...]
    accumulated_cost_s: tuple[float, ...]
    evaluation_time_s: float

    def __post_init__(self) -> None:
        for name in ("length_m", "evaluation_time_s"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.cost_s is not None and (not isfinite(self.cost_s) or self.cost_s < 0):
            raise ValueError("cost_s must be None or finite and nonnegative.")
        if self.minimum_clearance_m is not None and (
            not isfinite(self.minimum_clearance_m) or self.minimum_clearance_m < 0
        ):
            raise ValueError("minimum_clearance_m must be None or finite and nonnegative.")
        sizes = {
            len(self.distance_m),
            len(self.elevation_m),
            len(self.slowness_s_per_m),
            len(self.accumulated_cost_s),
        }
        size = next(iter(sizes)) if len(sizes) == 1 else -1
        if (
            size < 0
            or (self.cost_s is None and size != 0)
            or (self.cost_s is not None and size < 2)
        ):
            raise ValueError(
                "Profiles must be empty when cost is unavailable, otherwise length >= 2."
            )
        if self.feasible != (len(self.violations) == 0):
            raise ValueError("feasible must agree with violations.")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "cost_s": self.cost_s,
            "length_m": self.length_m,
            "feasible": self.feasible,
            "minimum_clearance_m": self.minimum_clearance_m,
            "violations": list(self.violations),
            "profile": {
                "distance_m": list(self.distance_m),
                "elevation_m": list(self.elevation_m),
                "slowness_s_per_m": list(self.slowness_s_per_m),
                "accumulated_cost_s": list(self.accumulated_cost_s),
            },
            "evaluation_time_s": self.evaluation_time_s,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouteEvaluation":
        try:
            profile = data["profile"]
            return cls(
                cost_s=None if data.get("cost_s") is None else float(data["cost_s"]),
                length_m=float(data["length_m"]),
                feasible=bool(data["feasible"]),
                minimum_clearance_m=(
                    None
                    if data.get("minimum_clearance_m") is None
                    else float(data["minimum_clearance_m"])
                ),
                violations=tuple(data["violations"]),
                distance_m=tuple(float(v) for v in profile["distance_m"]),
                elevation_m=tuple(float(v) for v in profile["elevation_m"]),
                slowness_s_per_m=tuple(float(v) for v in profile["slowness_s_per_m"]),
                accumulated_cost_s=tuple(float(v) for v in profile["accumulated_cost_s"]),
                evaluation_time_s=float(data["evaluation_time_s"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed route evaluation.") from exc


@dataclass(frozen=True)
class PlannerResult:
    """One planner outcome with solver and independent evaluation kept separate."""

    method: PlannerMethod
    initialization: str
    route_m: tuple[tuple[float, float], ...] | None
    evaluated_cost_s: float | None
    feasible: bool
    solver_success: bool
    termination_reason: str
    diagnostics: dict[str, JSONValue]
    timing_s: dict[str, float]
    scenario_hash: str
    config_hash: str
    evaluation: RouteEvaluation | None = None
    version: int = 2

    def __post_init__(self) -> None:
        if self.version != 2:
            raise ValueError("Unsupported planner result version; expected 2.")
        if self.method not in ("euler_lagrange", "slsqp", "fast_marching"):
            raise ValueError("Unknown planner method.")
        if not isinstance(self.initialization, str) or not self.initialization:
            raise ValueError("initialization must be a non-empty string.")
        if not isinstance(self.termination_reason, str) or not self.termination_reason:
            raise ValueError("termination_reason must be a non-empty string.")
        if self.route_m is not None:
            route = np.asarray(self.route_m, dtype=float)
            if (
                route.ndim != 2
                or route.shape[1] != 2
                or len(route) < 2
                or not np.isfinite(route).all()
            ):
                raise ValueError("route_m must be None or a finite array with shape (n >= 2, 2).")
            object.__setattr__(
                self, "route_m", tuple(tuple(float(v) for v in row) for row in route)
            )
        if self.evaluated_cost_s is not None and (
            not isfinite(self.evaluated_cost_s) or self.evaluated_cost_s < 0
        ):
            raise ValueError("evaluated_cost_s must be None or finite and nonnegative.")
        diagnostics = _json_value(self.diagnostics, path="diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError("diagnostics must be a JSON object.")
        timing = _json_value(self.timing_s, path="timing_s")
        if not isinstance(timing, dict) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            for value in timing.values()
        ):
            raise ValueError("timing_s values must be finite nonnegative numbers.")
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "timing_s", {key: float(value) for key, value in timing.items()})
        for name in ("scenario_hash", "config_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hexadecimal digest.")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"{name} must be a SHA-256 hexadecimal digest.") from exc
        if self.evaluation is None:
            if self.evaluated_cost_s is not None or self.feasible:
                raise ValueError(
                    "A result without an evaluation cannot have a cost or be feasible."
                )
        elif (
            self.feasible != self.evaluation.feasible
            or ((self.evaluated_cost_s is None) != (self.evaluation.cost_s is None))
            or (
                self.evaluated_cost_s is not None
                and not np.isclose(
                    self.evaluated_cost_s, self.evaluation.cost_s, rtol=0, atol=1e-12
                )
            )
        ):
            raise ValueError("Top-level cost and feasibility must agree with evaluation.")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "version": self.version,
            "method": self.method,
            "initialization": self.initialization,
            "route_m": None if self.route_m is None else [list(point) for point in self.route_m],
            "evaluated_cost_s": self.evaluated_cost_s,
            "feasible": self.feasible,
            "solver_success": self.solver_success,
            "termination_reason": self.termination_reason,
            "diagnostics": _json_value(self.diagnostics),
            "timing_s": _json_value(self.timing_s),
            "scenario_hash": self.scenario_hash,
            "config_hash": self.config_hash,
            "evaluation": None if self.evaluation is None else self.evaluation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlannerResult":
        if not isinstance(data, Mapping) or data.get("version") != 2:
            raise ValueError("Expected a version 2 planner result object.")
        allowed = {
            "version",
            "method",
            "initialization",
            "route_m",
            "evaluated_cost_s",
            "feasible",
            "solver_success",
            "termination_reason",
            "diagnostics",
            "timing_s",
            "scenario_hash",
            "config_hash",
            "evaluation",
        }
        if set(data) - allowed:
            raise ValueError("Planner result contains unknown fields.")
        try:
            evaluation = data.get("evaluation")
            return cls(
                version=data["version"],
                method=data["method"],
                initialization=data["initialization"],
                route_m=None if data.get("route_m") is None else tuple(map(tuple, data["route_m"])),
                evaluated_cost_s=data.get("evaluated_cost_s"),
                feasible=data["feasible"],
                solver_success=data["solver_success"],
                termination_reason=data["termination_reason"],
                diagnostics=data.get("diagnostics", {}),
                timing_s=data.get("timing_s", {}),
                scenario_hash=data["scenario_hash"],
                config_hash=data["config_hash"],
                evaluation=None if evaluation is None else RouteEvaluation.from_dict(evaluation),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed planner result.") from exc


def _route_profile_points(route: FloatArray, count: int) -> tuple[FloatArray, FloatArray]:
    segment_lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
    vertices = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    length = vertices[-1]
    if length == 0:
        return route[[0, -1]], np.array([0.0, 0.0])
    distances = np.unique(np.concatenate((vertices, np.linspace(0.0, length, count))))
    segment_indices = np.searchsorted(vertices[1:], distances, side="left")
    segment_indices = np.minimum(segment_indices, len(segment_lengths) - 1)
    local = distances - vertices[segment_indices]
    fractions = np.divide(
        local,
        segment_lengths[segment_indices],
        out=np.zeros_like(local),
        where=segment_lengths[segment_indices] > 0,
    )
    # Cumulative-distance subtraction can leave a fraction a few ulps outside
    # [0, 1].  Use a convex interpolation and contain its floating-point result
    # within the submitted segment.  The input polyline is never changed, and
    # genuinely out-of-domain routes have already been rejected by the caller.
    fractions = np.clip(fractions, 0.0, 1.0)
    starts = route[segment_indices]
    ends = route[segment_indices + 1]
    points = (1.0 - fractions[:, None]) * starts + fractions[:, None] * ends
    points = np.clip(points, np.minimum(starts, ends), np.maximum(starts, ends))
    points[0] = route[0]
    points[-1] = route[-1]
    return points, distances


def _segment_breaks(a: FloatArray, b: FloatArray, scenario: TerrainScenario) -> FloatArray:
    delta = b - a
    values = [0.0, 1.0]
    if delta[0] != 0:
        values.extend((np.asarray(scenario.field_x_m[1:-1]) - a[0]) / delta[0])
    if delta[1] != 0:
        values.extend((np.asarray(scenario.field_y_m[1:-1]) - a[1]) / delta[1])
    result = np.asarray(values, dtype=float)
    return np.unique(result[(result >= 0) & (result <= 1)])


def _integrate_segment(
    field: TerrainField, a: FloatArray, b: FloatArray, quadrature_order: int
) -> float:
    length = float(np.linalg.norm(b - a))
    if length == 0:
        return 0.0
    nodes, weights = _quadrature_rule(quadrature_order)
    breaks = _segment_breaks(a, b, field.scenario)
    lower = breaks[:-1]
    upper = breaks[1:]
    t = ((upper - lower)[:, None] * nodes + (upper + lower)[:, None]) / 2
    points = (1.0 - t[..., None]) * a + t[..., None] * b
    # Legendre nodes lie strictly inside the segment, but the affine expression
    # can round a coordinate a few ulps beyond an endpoint on the inclusive
    # domain boundary. Clamp only to the submitted segment's coordinate range;
    # this changes no route geometry and prevents spurious field extrapolation.
    points = np.clip(points, np.minimum(a, b), np.maximum(a, b))
    values = field.cost(points.reshape(-1, 2)).reshape(t.shape)
    integral = np.sum((upper - lower) * np.sum(values * weights, axis=1) / 2)
    return float(length * integral)


@lru_cache(maxsize=None)
def _quadrature_rule(order: int) -> tuple[FloatArray, FloatArray]:
    return np.polynomial.legendre.leggauss(order)


def evaluate_route(
    scenario: TerrainScenario,
    route_m: ArrayLike,
    *,
    field: TerrainField | None = None,
    profile_samples: int = 129,
    quadrature_order: int = 16,
    endpoint_tolerance_m: float = 1e-7,
) -> RouteEvaluation:
    """Measure cost and validate every segment of a complete polyline route.

    Domain-boundary contact is permitted.  Any contact with an impassable
    barrier is a collision.  Cost is evaluated independently even for a route
    that collides.  An out-of-domain route has no cost or profile because the
    scenario deliberately leaves the field undefined beyond its bounds.
    """
    started = perf_counter()
    route = np.asarray(route_m, dtype=float)
    if route.ndim != 2 or route.shape[1:] != (2,) or len(route) < 2:
        raise ValueError("route_m must have shape (n >= 2, 2).")
    if not np.isfinite(route).all():
        raise ValueError("route_m must contain only finite coordinates.")
    if (
        isinstance(profile_samples, bool)
        or not isinstance(profile_samples, int)
        or profile_samples < 2
    ):
        raise ValueError("profile_samples must be an integer of at least 2.")
    if (
        isinstance(quadrature_order, bool)
        or not isinstance(quadrature_order, int)
        or quadrature_order < 2
    ):
        raise ValueError("quadrature_order must be an integer of at least 2.")
    if not isfinite(endpoint_tolerance_m) or endpoint_tolerance_m < 0:
        raise ValueError("endpoint_tolerance_m must be finite and nonnegative.")
    if field is None:
        field = TerrainField(scenario)
    elif field.scenario.scenario_hash != scenario.scenario_hash:
        raise ValueError("field and scenario do not describe the same terrain.")

    from shapely.geometry import LineString, box

    violations: list[str] = []
    if np.linalg.norm(route[0] - scenario.start_m) > endpoint_tolerance_m:
        violations.append("start_mismatch")
    if np.linalg.norm(route[-1] - scenario.goal_m) > endpoint_tolerance_m:
        violations.append("goal_mismatch")

    line = LineString(route)
    domain = box(*scenario.bounds_m)
    if not domain.covers(line):
        violations.append("outside_domain")
    if not field.barriers.is_empty and line.intersects(field.barriers):
        violations.append("barrier_collision")
    clearance = None if field.barriers.is_empty else float(line.distance(field.barriers))

    segment_lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    if "outside_domain" in violations:
        return RouteEvaluation(
            cost_s=None,
            length_m=total_length,
            feasible=False,
            minimum_clearance_m=clearance,
            violations=tuple(violations),
            distance_m=(),
            elevation_m=(),
            slowness_s_per_m=(),
            accumulated_cost_s=(),
            evaluation_time_s=perf_counter() - started,
        )

    segment_costs = np.array(
        [
            _integrate_segment(field, a, b, quadrature_order)
            for a, b in zip(route[:-1], route[1:], strict=True)
        ]
    )
    total_cost = float(np.sum(segment_costs))

    profile_points, profile_distance = _route_profile_points(route, profile_samples)
    profile_step_costs = np.array(
        [
            _integrate_segment(field, a, b, quadrature_order)
            for a, b in zip(profile_points[:-1], profile_points[1:], strict=True)
        ]
    )
    accumulated = np.concatenate(([0.0], np.cumsum(profile_step_costs)))
    accumulated[-1] = total_cost
    elevation = field.elevation(profile_points)
    slowness = field.cost(profile_points)

    return RouteEvaluation(
        cost_s=total_cost,
        length_m=total_length,
        feasible=not violations,
        minimum_clearance_m=clearance,
        violations=tuple(violations),
        distance_m=tuple(float(v) for v in profile_distance),
        elevation_m=tuple(float(v) for v in elevation),
        slowness_s_per_m=tuple(float(v) for v in slowness),
        accumulated_cost_s=tuple(float(v) for v in accumulated),
        evaluation_time_s=perf_counter() - started,
    )


def uniform_terrain_fixture() -> TerrainScenario:
    """Small deterministic end-to-end fixture with one second/metre slowness."""
    coordinates = tuple(float(value) for value in np.linspace(0.0, 100.0, 5))
    zeros = tuple(tuple(0.0 for _ in coordinates) for _ in coordinates)
    return TerrainScenario(
        name="uniform-100m",
        bounds_m=(0.0, 0.0, 100.0, 100.0),
        start_m=(10.0, 20.0),
        goal_m=(90.0, 80.0),
        field_x_m=coordinates,
        field_y_m=coordinates,
        elevation_m=zeros,
        log_slowness=zeros,
        provenance={"kind": "analytic", "description": "Uniform validation fixture"},
    )


__all__ = [
    "PlannerConfig",
    "PlannerMethod",
    "PlannerResult",
    "RouteEvaluation",
    "TerrainField",
    "TerrainScenario",
    "evaluate_route",
    "uniform_terrain_fixture",
]
