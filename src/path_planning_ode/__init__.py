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
    "RouteEvaluation",
    "TerrainField",
    "TerrainScenario",
    "evaluate_route",
    "uniform_terrain_fixture",
    "mount_tamalpais_terrain",
    "soften_walls",
    "PlaygroundField",
    "build_playground_preset",
    "terrain_to_playground_field",
    "BrushStroke",
    "MAX_GAUSSIANS",
    "PlaygroundMetrics",
    "PlaygroundOptions",
    "PlaygroundState",
    "advance_playground",
    "build_playground_field",
    "evaluate_playground",
    "initialize_playground",
    "playground_energy_gradient",
    "set_playground_pin",
    "strokes_to_obstacles",
]

_TERRAIN_EXPORTS = {
    "RouteEvaluation",
    "TerrainField",
    "TerrainScenario",
    "evaluate_route",
    "uniform_terrain_fixture",
}
_GENERATOR_EXPORTS = {"mount_tamalpais_terrain"}
_FIELD_EXPORTS = {"PlaygroundField", "build_playground_preset", "terrain_to_playground_field"}
_SOFT_WALL_EXPORTS = {"soften_walls"}
_PLAYGROUND_EXPORTS = {
    "BrushStroke",
    "MAX_GAUSSIANS",
    "PlaygroundMetrics",
    "PlaygroundOptions",
    "PlaygroundState",
    "advance_playground",
    "build_playground_field",
    "evaluate_playground",
    "initialize_playground",
    "playground_energy_gradient",
    "set_playground_pin",
    "strokes_to_obstacles",
}


def __getattr__(name: str):
    """Load optional terrain and playground modules only when requested."""
    if name in _TERRAIN_EXPORTS:
        module = ".terrain"
    elif name in _GENERATOR_EXPORTS:
        module = ".terrain_generators"
    elif name in _FIELD_EXPORTS:
        module = ".field_adapter"
    elif name in _SOFT_WALL_EXPORTS:
        module = ".soft_walls"
    elif name in _PLAYGROUND_EXPORTS:
        module = ".playground"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
