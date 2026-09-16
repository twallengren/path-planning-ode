"""Common entry point for version-2 terrain planners."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from .core import Scene, cost_field
from .local_planners import solve_euler_lagrange, solve_slsqp
from .terrain import PlannerConfig, PlannerResult, TerrainScenario


def scene_to_terrain_scenario(scene: Scene, *, samples: int = 65) -> TerrainScenario:
    """Adapt a v1 Gaussian scene through an explicit sampled-spline approximation."""
    if not isinstance(scene, Scene):
        raise TypeError("scene must be a version-1 Scene.")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 4:
        raise ValueError("samples must be an integer of at least 4.")
    endpoint_values = np.asarray([scene.start, scene.end], dtype=float)
    if np.array_equal(endpoint_values[0], endpoint_values[1]):
        raise ValueError("A v1 Scene must have distinct endpoints for terrain planning.")
    if scene.obstacles:
        obstacle_values = np.asarray(
            [
                [obstacle.x - 5 * obstacle.width, obstacle.y - 5 * obstacle.width]
                for obstacle in scene.obstacles
            ]
            + [
                [obstacle.x + 5 * obstacle.width, obstacle.y + 5 * obstacle.width]
                for obstacle in scene.obstacles
            ]
        )
        values = np.vstack((endpoint_values, obstacle_values))
    else:
        values = endpoint_values
    low, high = np.min(values, axis=0), np.max(values, axis=0)
    span = np.maximum(high - low, np.linalg.norm(np.asarray(scene.end) - scene.start) * 0.1)
    low -= 0.1 * span
    high += 0.1 * span
    x = np.linspace(low[0], high[0], samples)
    y = np.linspace(low[1], high[1], samples)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx, yy), axis=-1)
    slowness = cost_field(points, scene.obstacles)
    return TerrainScenario(
        name="v1 sampled Gaussian scene",
        bounds_m=(float(x[0]), float(y[0]), float(x[-1]), float(y[-1])),
        start_m=scene.start,
        goal_m=scene.end,
        field_x_m=tuple(x),
        field_y_m=tuple(y),
        elevation_m=np.zeros_like(slowness),
        log_slowness=np.log(slowness),
        provenance={
            "kind": "v1_scene_adapter",
            "approximation": (
                "Gaussian slowness sampled on a regular grid then bicubic in log-space"
            ),
            "samples": [samples, samples],
            "source_version": 1,
        },
        metadata={"v1_scene": scene.to_dict()},
    )


def plan(scenario: TerrainScenario | Scene, config: PlannerConfig) -> PlannerResult:
    """Run the configured planner under one end-to-end wall-clock budget."""
    started = perf_counter()
    if not isinstance(config, PlannerConfig):
        raise TypeError("config must be a PlannerConfig.")
    adapter_time = 0.0
    if isinstance(scenario, Scene):
        adapter_started = perf_counter()
        scenario = scene_to_terrain_scenario(
            scenario, samples=int(config.options.get("adapter_samples", 65))
        )
        adapter_time = perf_counter() - adapter_started
    if not isinstance(scenario, TerrainScenario):
        raise TypeError("scenario must be a TerrainScenario or version-1 Scene.")
    deadline = started + config.time_limit_s
    if config.method == "euler_lagrange":
        result = solve_euler_lagrange(scenario, config, deadline=deadline)
    elif config.method == "slsqp":
        result = solve_slsqp(scenario, config, deadline=deadline)
    elif config.method == "energy_descent":
        from .terrain_energy import solve_energy_descent

        result = solve_energy_descent(scenario, config, deadline=deadline)
    else:
        from .fast_marching import solve_fast_marching

        result = solve_fast_marching(scenario, config)

    timing = dict(result.timing_s)
    timing["preprocessing"] = timing.get("preprocessing", 0.0) + adapter_time
    timing["total"] = perf_counter() - started
    return replace(result, timing_s=timing)


__all__ = ["plan", "scene_to_terrain_scenario"]
