"""Euler–Lagrange path planning, shared by Python and the browser."""

from importlib import import_module

from .core import (
    IterationState,
    Obstacle,
    Scene,
    SolveResult,
    SolverOptions,
    cost_field,
    energy,
    initialize,
    jacobian,
    ode,
    residual,
    solve,
    step,
    weighted_distance,
)
from .presets import presets

__all__ = [
    "IterationState",
    "Obstacle",
    "Scene",
    "SolveResult",
    "SolverOptions",
    "cost_field",
    "energy",
    "initialize",
    "jacobian",
    "ode",
    "presets",
    "residual",
    "solve",
    "step",
    "weighted_distance",
    "PlannerConfig",
    "PlannerResult",
    "RouteEvaluation",
    "TerrainField",
    "TerrainScenario",
    "evaluate_route",
    "uniform_terrain_fixture",
    "FastMarchingError",
    "FastMarchingSolution",
    "compute_arrival_time",
    "extract_route",
    "fast_marching_seed",
    "solve_fast_marching",
    "plan",
    "scene_to_terrain_scenario",
    "BENCHMARK_SEEDS",
    "DIFFICULTY_LEVELS",
    "FAMILY_NAMES",
    "SOURCE_RESOLUTION",
    "VALIDATION_NAMES",
    "competing_corridors_terrain",
    "correlated_roughness_terrain",
    "dead_ends_terrain",
    "disconnected_fixture",
    "layered_refraction_fixture",
    "mount_tamalpais_terrain",
    "obstacle_detour_fixture",
    "ridge_pass_terrain",
    "symmetry_fixture",
    "synthetic_terrain",
    "validation_terrain",
    "INITIALIZATIONS",
    "SeedResult",
    "make_seed",
    "resample_route",
    "seed_bank",
    "soften_walls",
]

_TERRAIN_EXPORTS = {
    "PlannerConfig",
    "PlannerResult",
    "RouteEvaluation",
    "TerrainField",
    "TerrainScenario",
    "evaluate_route",
    "uniform_terrain_fixture",
}

_FAST_MARCHING_EXPORTS = {
    "FastMarchingError",
    "FastMarchingSolution",
    "compute_arrival_time",
    "extract_route",
    "fast_marching_seed",
    "solve_fast_marching",
}

_PLANNER_EXPORTS = {"plan", "scene_to_terrain_scenario"}

_TERRAIN_GENERATOR_EXPORTS = {
    "BENCHMARK_SEEDS",
    "DIFFICULTY_LEVELS",
    "FAMILY_NAMES",
    "SOURCE_RESOLUTION",
    "VALIDATION_NAMES",
    "competing_corridors_terrain",
    "correlated_roughness_terrain",
    "dead_ends_terrain",
    "disconnected_fixture",
    "layered_refraction_fixture",
    "mount_tamalpais_terrain",
    "obstacle_detour_fixture",
    "ridge_pass_terrain",
    "symmetry_fixture",
    "synthetic_terrain",
    "validation_terrain",
}

_TERRAIN_SEED_EXPORTS = {
    "INITIALIZATIONS",
    "SeedResult",
    "make_seed",
    "resample_route",
    "seed_bank",
}

_SOFT_WALL_EXPORTS = {"soften_walls"}


def __getattr__(name: str):
    """Load the SciPy/Shapely terrain API only when a caller requests it."""
    if name in _TERRAIN_EXPORTS:
        value = getattr(import_module(".terrain", __name__), name)
        globals()[name] = value
        return value
    if name in _FAST_MARCHING_EXPORTS:
        value = getattr(import_module(".fast_marching", __name__), name)
        globals()[name] = value
        return value
    if name in _PLANNER_EXPORTS:
        value = getattr(import_module(".planners", __name__), name)
        globals()[name] = value
        return value
    if name in _TERRAIN_GENERATOR_EXPORTS:
        value = getattr(import_module(".terrain_generators", __name__), name)
        globals()[name] = value
        return value
    if name in _TERRAIN_SEED_EXPORTS:
        value = getattr(import_module(".terrain_seeds", __name__), name)
        globals()[name] = value
        return value
    if name in _SOFT_WALL_EXPORTS:
        value = getattr(import_module(".soft_walls", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
