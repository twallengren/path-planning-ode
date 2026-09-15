"""Convert impassable terrain barriers into finite, smooth cost walls.

This module provides an opt-in model transform.  It does not change the
meaning of ``TerrainScenario.barriers_geojson``: barriers stored there remain
impassable.  Instead, :func:`soften_walls` samples a smooth log-slowness
penalty on the scenario's fixed source grid and removes the hard geometry from
the returned scenario.
"""

from __future__ import annotations

from math import hypot, isfinite, log

import numpy as np

from .terrain import TerrainScenario

_METADATA_KEY = "soft_walls"
_PROFILE = "outward_quintic_smootherstep"
_MAX_MULTIPLIER = 1_000_000.0


def _source_cell_diagonal_m(scenario: TerrainScenario) -> float:
    """Return the largest source-grid cell diagonal in physical metres."""
    x_step = float(np.max(np.diff(np.asarray(scenario.field_x_m, dtype=float))))
    y_step = float(np.max(np.diff(np.asarray(scenario.field_y_m, dtype=float))))
    return hypot(x_step, y_step)


def _positive_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _wall_weight(scenario: TerrainScenario, transition_width_m: float) -> np.ndarray:
    """Sample a compact C2 wall profile on the fixed source field grid."""
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union

    geometries = [shape(item) for item in scenario.barriers_geojson]
    if not geometries:
        return np.zeros((len(scenario.field_y_m), len(scenario.field_x_m)), dtype=float)
    walls = unary_union(geometries)

    distances = np.empty((len(scenario.field_y_m), len(scenario.field_x_m)), dtype=float)
    for row, y in enumerate(scenario.field_y_m):
        for column, x in enumerate(scenario.field_x_m):
            # Geometry distance is zero on and inside a polygon.  This gives
            # even sub-cell-width walls an outward transition sampled by the
            # fixed source grid rather than by a planner discretization.
            distances[row, column] = walls.distance(Point(x, y))

    fraction = np.clip(distances / transition_width_m, 0.0, 1.0)
    smootherstep = fraction**3 * (fraction * (fraction * 6.0 - 15.0) + 10.0)
    return 1.0 - smootherstep


def soften_walls(
    scenario: TerrainScenario,
    multiplier: float = 100.0,
    transition_width_m: float | None = None,
) -> TerrainScenario:
    """Return ``scenario`` with hard barriers represented by finite cost.

    The transform adds ``log(multiplier) * weight`` to log-slowness at each
    source-grid node.  ``weight`` is one on the barrier geometry and follows a
    compact quintic smootherstep to zero over ``transition_width_m`` outside
    it.  The scalar profile's first two derivatives join smoothly at both ends;
    the resulting bicubic log-field supplies the spatial derivatives needed by
    the Euler--Lagrange method without tying smoothing to solver resolution.

    ``multiplier`` is the nominal factor at source nodes on or inside a wall.
    Bicubic interpolation can overshoot this nominal value between nodes and
    can ring below the base field near a transition.  It is bounded
    to at most one million to keep the finite model numerically useful.

    When omitted, the transition width is twice the largest source-grid cell
    diagonal.  Explicit widths must span at least one such diagonal so that a
    thin wall cannot silently disappear between source samples.

    The returned scenario has no impassable barriers.  Original geometry and
    all transform parameters are retained under ``metadata["soft_walls"]``.
    Applying the transform twice is rejected.
    """
    if not isinstance(scenario, TerrainScenario):
        raise TypeError("scenario must be a TerrainScenario.")
    if _METADATA_KEY in scenario.metadata:
        raise ValueError("scenario already has the soft-wall transform applied.")

    multiplier_value = _positive_float(multiplier, "multiplier")
    if multiplier_value <= 1.0:
        raise ValueError("multiplier must be greater than 1.")
    if multiplier_value > _MAX_MULTIPLIER:
        raise ValueError(f"multiplier must be at most {_MAX_MULTIPLIER:g}.")

    cell_diagonal_m = _source_cell_diagonal_m(scenario)
    if transition_width_m is None:
        width_m = 2.0 * cell_diagonal_m
    else:
        width_m = _positive_float(transition_width_m, "transition_width_m")
        if width_m < cell_diagonal_m:
            raise ValueError(
                "transition_width_m must be at least the largest source-grid "
                f"cell diagonal ({cell_diagonal_m:.12g} m)."
            )

    weight = _wall_weight(scenario, width_m)
    transformed_log_slowness = (
        np.asarray(scenario.log_slowness, dtype=float) + log(multiplier_value) * weight
    )
    with np.errstate(over="ignore", invalid="ignore"):
        source_cost = np.exp(transformed_log_slowness)
    if not np.isfinite(source_cost).all() or np.any(source_cost <= 0):
        raise ValueError("soft-wall source costs must remain finite and positive.")

    serialized = scenario.to_dict()
    geometry = serialized["barriers_geojson"]
    metadata = dict(scenario.metadata)
    metadata[_METADATA_KEY] = {
        "version": 1,
        "model": "finite_smooth_high_cost",
        "base_scenario_hash": scenario.scenario_hash,
        "multiplier": multiplier_value,
        "transition_width_m": width_m,
        "transition_profile": _PROFILE,
        "geometry_geojson": geometry,
        "source_grid_shape": [len(scenario.field_y_m), len(scenario.field_x_m)],
    }

    return TerrainScenario(
        name=scenario.name,
        bounds_m=scenario.bounds_m,
        start_m=scenario.start_m,
        goal_m=scenario.goal_m,
        field_x_m=scenario.field_x_m,
        field_y_m=scenario.field_y_m,
        elevation_m=scenario.elevation_m,
        log_slowness=transformed_log_slowness,
        barriers_geojson=(),
        provenance=scenario.provenance,
        metadata=metadata,
        version=scenario.version,
    )


__all__ = ["soften_walls"]
