"""Shared analytic fields for the unified path playground.

The module deliberately imports only NumPy and the small Gaussian core at
module import time.  Terrain, SciPy, Shapely, generators, and packaged data are
loaded only when a terrain field or preset is requested.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import Obstacle

Array = NDArray[np.float64]


def _obstacles(values: Sequence[Obstacle | Mapping]) -> tuple[Obstacle, ...]:
    try:
        return tuple(
            item if isinstance(item, Obstacle) else Obstacle(**dict(item)) for item in values
        )
    except TypeError as exc:
        raise ValueError("Malformed Gaussian obstacle.") from exc


def _bounds(value: ArrayLike | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError("bounds must contain finite (xmin, ymin, xmax, ymax).")
    xmin, ymin, xmax, ymax = (float(item) for item in result)
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("bounds must have positive width and height.")
    return xmin, ymin, xmax, ymax


def _canonical_hash(value: Mapping) -> str:
    def normalize_numbers(item):
        if isinstance(item, bool) or item is None or isinstance(item, str):
            return item
        if isinstance(item, (int, np.integer)):
            return int(item)
        if isinstance(item, (float, np.floating)):
            number = float(item)
            if not np.isfinite(number):
                raise ValueError("Field specification contains a non-finite number.")
            if number == 0:
                return 0
            if number.is_integer():
                return int(number)
            return number
        if isinstance(item, Mapping):
            return {key: normalize_numbers(nested) for key, nested in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize_numbers(nested) for nested in item]
        raise ValueError(f"Field specification contains non-JSON value {type(item).__name__}.")

    payload = json.dumps(
        normalize_numbers(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256(payload.encode()).hexdigest()


def _point_array(points: ArrayLike) -> tuple[Array, tuple[int, ...]]:
    result = np.asarray(points, dtype=float)
    if result.shape == (2,):
        shape: tuple[int, ...] = ()
        flat = result.reshape(1, 2)
    elif result.ndim >= 1 and result.shape[-1] == 2:
        shape = result.shape[:-1]
        flat = result.reshape(-1, 2)
    else:
        raise ValueError("points must have shape (..., 2).")
    if not np.isfinite(flat).all():
        raise ValueError("points must be finite.")
    return flat, shape


def _gaussian_evaluate(
    points: Array, obstacles: tuple[Obstacle, ...]
) -> tuple[Array, Array, Array]:
    """Return ``1+b``, its gradient, and Hessian for Gaussian bumps."""
    factor = np.ones(len(points))
    gradient = np.zeros((len(points), 2))
    hessian = np.zeros((len(points), 2, 2))
    identity = np.eye(2)
    for obstacle in obstacles:
        delta = points - (obstacle.x, obstacle.y)
        inverse_width2 = 1.0 / obstacle.width**2
        bump = obstacle.weight * np.exp(-inverse_width2 * np.sum(delta * delta, axis=1))
        factor += bump
        gradient += (-2 * inverse_width2 * bump)[:, None] * delta
        hessian += bump[:, None, None] * (
            4 * inverse_width2**2 * delta[:, :, None] * delta[:, None, :]
            - 2 * inverse_width2 * identity
        )
    return factor, gradient, hessian


def _gaussian_cost(points: Array, obstacles: tuple[Obstacle, ...]) -> Array:
    factor = np.ones(len(points))
    for obstacle in obstacles:
        delta = points - (obstacle.x, obstacle.y)
        factor += obstacle.weight * np.exp(-np.sum(delta * delta, axis=1) / obstacle.width**2)
    return factor


def _gaussian_cost_gradient(points: Array, obstacles: tuple[Obstacle, ...]) -> tuple[Array, Array]:
    factor = np.ones(len(points))
    gradient = np.zeros((len(points), 2))
    for obstacle in obstacles:
        delta = points - (obstacle.x, obstacle.y)
        inverse_width2 = 1.0 / obstacle.width**2
        bump = obstacle.weight * np.exp(-inverse_width2 * np.sum(delta * delta, axis=1))
        factor += bump
        gradient += (-2 * inverse_width2 * bump)[:, None] * delta
    return factor, gradient


class PlaygroundField:
    """Positive analytic base field multiplied by optional Gaussian bumps."""

    def __init__(
        self,
        base_field: Mapping[str, Any] | None = None,
        gaussians: Sequence[Obstacle | Mapping] = (),
        *,
        bounds: ArrayLike | None = None,
    ) -> None:
        base = dict(base_field or {"kind": "uniform", "cost": 1.0, "elevation": 0.0})
        kind = base.get("kind")
        if kind not in ("uniform", "terrain"):
            raise ValueError("base_field.kind must be 'uniform' or 'terrain'.")
        self.gaussians = _obstacles(gaussians)
        self._terrain = None
        if kind == "uniform":
            allowed = {"kind", "cost", "elevation"}
            if set(base) - allowed:
                raise ValueError("Uniform base_field contains unknown fields.")
            cost = float(base.get("cost", 1.0))
            elevation = float(base.get("elevation", 0.0))
            if not np.isfinite([cost, elevation]).all() or cost <= 0:
                raise ValueError("Uniform cost must be positive and elevation finite.")
            self.base_spec = {"kind": "uniform", "cost": cost, "elevation": elevation}
            self.bounds_m = _bounds(bounds)
            self.cost_scale = cost
        else:
            if set(base) != {"kind", "scenario", "scenario_hash"}:
                raise ValueError("Terrain base_field requires scenario and scenario_hash only.")
            from .terrain import TerrainField, TerrainScenario

            scenario_value = base["scenario"]
            scenario = (
                scenario_value
                if isinstance(scenario_value, TerrainScenario)
                else TerrainScenario.from_dict(scenario_value)
            )
            if scenario.barriers_geojson:
                raise ValueError(
                    "Terrain playground fields cannot contain hard barriers; "
                    "explicitly soften them."
                )
            if base["scenario_hash"] != scenario.scenario_hash:
                raise ValueError("Terrain base_field scenario_hash does not match the scenario.")
            self._terrain = TerrainField(scenario)
            self.base_spec = {
                "kind": "terrain",
                "scenario": scenario.to_dict(),
                "scenario_hash": scenario.scenario_hash,
            }
            self.bounds_m = scenario.bounds_m
            if bounds is not None and _bounds(bounds) != self.bounds_m:
                raise ValueError("Explicit bounds must match terrain scenario bounds_m.")
            self.cost_scale = float(np.exp(np.median(np.asarray(scenario.log_slowness))))

        self.field_hash = _canonical_hash(self.to_spec())

    @classmethod
    def from_spec(
        cls,
        base_field: Mapping[str, Any] | None,
        gaussians: Sequence[Obstacle | Mapping] = (),
        *,
        bounds: ArrayLike | None = None,
    ) -> PlaygroundField:
        """Construct from a base descriptor or a complete :meth:`to_spec` value."""
        if base_field is not None and "base_field" in base_field:
            complete = dict(base_field)
            if complete.get("version") != 1 or set(complete) != {
                "version",
                "base_field",
                "gaussians",
                "bounds_m",
            }:
                raise ValueError("Malformed playground field specification.")
            if gaussians:
                raise ValueError("Do not provide gaussians twice.")
            return cls(
                complete["base_field"],
                complete["gaussians"],
                bounds=complete["bounds_m"],
            )
        return cls(base_field, gaussians, bounds=bounds)

    def to_spec(self) -> dict:
        return {
            "version": 1,
            "base_field": self.base_spec,
            "gaussians": [asdict(item) for item in self.gaussians],
            "bounds_m": list(self.bounds_m) if self.bounds_m is not None else None,
        }

    @property
    def minimum_scale(self) -> float:
        candidates = [item.width for item in self.gaussians]
        if self._terrain is not None:
            scenario = self._terrain.scenario
            candidates.extend(np.diff(scenario.field_x_m))
            candidates.extend(np.diff(scenario.field_y_m))
        if candidates:
            return float(min(candidates))
        if self.bounds_m is not None:
            xmin, ymin, xmax, ymax = self.bounds_m
            return float(np.hypot(xmax - xmin, ymax - ymin) / 32)
        return float("inf")

    def coordinate_scale(self, path: ArrayLike) -> float:
        if self.bounds_m is not None:
            xmin, ymin, xmax, ymax = self.bounds_m
            return float(np.hypot(xmax - xmin, ymax - ymin))
        points, _ = _point_array(path)
        extent = np.ptp(points, axis=0)
        return max(float(np.hypot(*extent)), 1.0)

    def in_domain(self, points: ArrayLike) -> bool:
        values, _ = _point_array(points)
        if self.bounds_m is None:
            return True
        xmin, ymin, xmax, ymax = self.bounds_m
        return bool(
            np.all(
                (values[:, 0] >= xmin)
                & (values[:, 0] <= xmax)
                & (values[:, 1] >= ymin)
                & (values[:, 1] <= ymax)
            )
        )

    def evaluate(self, points: ArrayLike) -> tuple[Array, Array, Array]:
        values, shape = _point_array(points)
        if not self.in_domain(values):
            raise ValueError("Field queries must lie inside bounds_m.")
        factor, factor_gradient, factor_hessian = _gaussian_evaluate(values, self.gaussians)
        if self._terrain is None:
            base_cost = np.full(len(values), self.base_spec["cost"])
            base_gradient = np.zeros((len(values), 2))
            base_hessian = np.zeros((len(values), 2, 2))
        else:
            base_cost, base_gradient, base_hessian = self._terrain.evaluate(values)
        cost = base_cost * factor
        gradient = base_gradient * factor[:, None] + base_cost[:, None] * factor_gradient
        hessian = (
            base_hessian * factor[:, None, None]
            + base_cost[:, None, None] * factor_hessian
            + base_gradient[:, :, None] * factor_gradient[:, None, :]
            + factor_gradient[:, :, None] * base_gradient[:, None, :]
        )
        return (
            cost.reshape(shape),
            gradient.reshape(shape + (2,)),
            hessian.reshape(shape + (2, 2)),
        )

    def cost(self, points: ArrayLike) -> Array:
        values, shape = _point_array(points)
        if not self.in_domain(values):
            raise ValueError("Field queries must lie inside bounds_m.")
        factor = _gaussian_cost(values, self.gaussians)
        if self._terrain is None:
            base_cost = np.full(len(values), self.base_spec["cost"])
        else:
            base_cost = self._terrain.cost(values)
        return (base_cost * factor).reshape(shape)

    def gradient(self, points: ArrayLike) -> Array:
        values, shape = _point_array(points)
        if not self.in_domain(values):
            raise ValueError("Field queries must lie inside bounds_m.")
        factor, factor_gradient = _gaussian_cost_gradient(values, self.gaussians)
        if self._terrain is None:
            base_cost = np.full(len(values), self.base_spec["cost"])
            base_gradient = np.zeros((len(values), 2))
        else:
            base_cost = self._terrain.cost(values)
            base_gradient = self._terrain.gradient(values)
        return (base_gradient * factor[:, None] + base_cost[:, None] * factor_gradient).reshape(
            shape + (2,)
        )

    def hessian(self, points: ArrayLike) -> Array:
        return self.evaluate(points)[2]

    def elevation(self, points: ArrayLike) -> Array:
        values, shape = _point_array(points)
        if not self.in_domain(values):
            raise ValueError("Field queries must lie inside bounds_m.")
        if self._terrain is None:
            result = np.full(len(values), self.base_spec["elevation"])
        else:
            result = self._terrain.elevation(values)
        return np.asarray(result).reshape(shape)


def terrain_to_playground_field(
    scenario,
    gaussians: Sequence[Obstacle | Mapping] = (),
    *,
    soften_barriers: bool = False,
    wall_multiplier: float = 100.0,
    transition_width_m: float | None = None,
) -> PlaygroundField:
    """Adapt an exact terrain scenario, explicitly converting hard barriers if requested."""
    from .terrain import TerrainScenario

    if not isinstance(scenario, TerrainScenario):
        scenario = TerrainScenario.from_dict(scenario)
    if scenario.barriers_geojson:
        if not soften_barriers:
            raise ValueError("Terrain contains hard barriers; set soften_barriers=True explicitly.")
        from .soft_walls import soften_walls

        scenario = soften_walls(
            scenario,
            multiplier=wall_multiplier,
            transition_width_m=transition_width_m,
        )
    base = {
        "kind": "terrain",
        "scenario": scenario.to_dict(),
        "scenario_hash": scenario.scenario_hash,
    }
    return PlaygroundField(base, gaussians)


def _gaussian_preset(name: str, seed: int) -> tuple[list[Obstacle], dict]:
    rng = np.random.default_rng(seed)
    if name == "blank":
        obstacles: list[Obstacle] = []
    elif name == "random_hills":
        obstacles = [
            Obstacle(
                float(rng.uniform(-3.6, 3.6)),
                float(rng.uniform(-2.7, 2.7)),
                float(rng.uniform(2.5, 10.0)),
                float(rng.uniform(0.45, 1.0)),
            )
            for _ in range(7)
        ]
    elif name == "corridor":
        obstacles = [
            Obstacle(x, y, 12.0, 0.62) for x in (-2.4, -0.8, 0.8, 2.4) for y in (-1.35, 1.35)
        ]
    elif name == "slalom":
        obstacles = [
            Obstacle(x, y, 13.0, 0.72)
            for x, y in ((-2.6, 0.8), (-1.3, -0.8), (0.0, 0.8), (1.3, -0.8), (2.6, 0.8))
        ]
    else:
        raise ValueError("Unknown Gaussian playground preset.")
    return obstacles, {"kind": "gaussian", "preset": name, "seed": seed}


def build_playground_preset(
    name: str,
    seed: int = 0,
    *,
    wall_multiplier: float = 100.0,
    transition_width_m: float | None = None,
) -> dict:
    """Build a deterministic JSON-ready Gaussian or terrain playground scene."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    seed = int(seed)
    gaussian_names = {"blank", "random_hills", "corridor", "slalom"}
    if name in gaussian_names:
        gaussians, metadata = _gaussian_preset(name, seed)
        return {
            "name": name,
            "bounds": [-6.0, 6.0, -4.0, 4.0],
            "start": [-4.8, 0.0],
            "end": [4.8, 0.0],
            "base_field": {"kind": "uniform", "cost": 1.0, "elevation": 0.0},
            "gaussians": [asdict(item) for item in gaussians],
            "strokes": [],
            "metadata": metadata,
        }

    from .terrain_generators import FAMILY_NAMES, mount_tamalpais_terrain, synthetic_terrain

    if name == "mount_tamalpais":
        scenario = mount_tamalpais_terrain(barriers=False)
        recorded_seed: int | None = None
    elif name in FAMILY_NAMES:
        scenario = synthetic_terrain(name, seed=seed, barriers=True)
        recorded_seed = seed
    else:
        expected = sorted(gaussian_names | set(FAMILY_NAMES) | {"mount_tamalpais"})
        raise ValueError(f"Unknown playground preset; expected one of {expected}.")
    adapted = terrain_to_playground_field(
        scenario,
        soften_barriers=bool(scenario.barriers_geojson),
        wall_multiplier=wall_multiplier,
        transition_width_m=transition_width_m,
    )
    final_scenario = adapted.base_spec["scenario"]
    xmin, ymin, xmax, ymax = final_scenario["bounds_m"]
    return {
        "name": name,
        "bounds": [xmin, xmax, ymin, ymax],
        "start": list(final_scenario["start_m"]),
        "end": list(final_scenario["goal_m"]),
        "base_field": adapted.base_spec,
        "gaussians": [],
        "strokes": [],
        "metadata": {
            "kind": "terrain",
            "preset": name,
            "seed": recorded_seed,
            "scenario_hash": adapted.base_spec["scenario_hash"],
            "soft_walls": final_scenario["metadata"].get("soft_walls"),
        },
    }


__all__ = ["PlaygroundField", "build_playground_preset", "terrain_to_playground_field"]
