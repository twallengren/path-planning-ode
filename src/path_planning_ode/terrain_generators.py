"""Deterministic terrain scenarios for benchmarks and validation.

The arrays in this module are source fields.  Their 65 x 65 resolution is fixed
and deliberately unrelated to the number of points used by a path solver.
"""

from __future__ import annotations

import json
import math
from importlib.resources import files
from typing import Callable

import numpy as np

from .terrain import TerrainScenario

SOURCE_RESOLUTION = 65
BENCHMARK_SEEDS = tuple(range(20))
DIFFICULTY_LEVELS = ("easy", "medium", "hard")
FAMILY_NAMES = (
    "ridge_pass",
    "competing_corridors",
    "dead_ends",
    "correlated_roughness",
)
VALIDATION_NAMES = ("uniform", "layered_refraction", "symmetry", "obstacle_detour", "disconnected")


def _grid(size_m: float = 1_000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, size_m, SOURCE_RESOLUTION)
    x, y = np.meshgrid(axis, axis)
    return axis, axis.copy(), x, y


def _rectangle(x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


def _circle(cx: float, cy: float, radius: float, vertices: int = 20) -> dict:
    ring = [
        [
            cx + radius * math.cos(2 * math.pi * i / vertices),
            cy + radius * math.sin(2 * math.pi * i / vertices),
        ]
        for i in range(vertices)
    ]
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def _scenario(
    name: str,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    elevation: np.ndarray,
    log_slowness: np.ndarray,
    *,
    start: tuple[float, float] = (60.0, 500.0),
    goal: tuple[float, float] = (940.0, 500.0),
    barriers: tuple[dict, ...] = (),
    provenance: dict | None = None,
    metadata: dict | None = None,
) -> TerrainScenario:
    return TerrainScenario(
        name=name,
        bounds_m=(float(x_axis[0]), float(y_axis[0]), float(x_axis[-1]), float(y_axis[-1])),
        start_m=start,
        goal_m=goal,
        field_x_m=x_axis.tolist(),
        field_y_m=y_axis.tolist(),
        elevation_m=np.asarray(elevation, dtype=float).tolist(),
        log_slowness=np.asarray(log_slowness, dtype=float).tolist(),
        barriers_geojson=barriers,
        provenance=provenance or {},
        metadata=metadata or {},
    )


def _common_metadata(family: str, seed: int, contrast: float, barriers: bool) -> dict:
    return {
        "kind": "synthetic_benchmark",
        "family": family,
        "seed": seed,
        "benchmark_seed": seed in BENCHMARK_SEEDS,
        "difficulty": DIFFICULTY_LEVELS[seed % len(DIFFICULTY_LEVELS)],
        "difficulty_rule": "difficulty = ('easy', 'medium', 'hard')[seed % 3]",
        "contrast": contrast,
        "barriers_enabled": barriers,
        "source_resolution": [SOURCE_RESOLUTION, SOURCE_RESOLUTION],
        "cost_units": "seconds per horizontal metre",
        "cost_model": "static, isotropic, strictly positive; exp(bicubic(log_slowness))",
    }


def _validate_controls(seed: int, contrast: float, barriers: bool) -> tuple[int, float, bool]:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if not np.isfinite(contrast) or contrast < 0:
        raise ValueError("contrast must be finite and nonnegative")
    if not isinstance(barriers, bool):
        raise ValueError("barriers must be a bool")
    return int(seed), float(contrast), barriers


def ridge_pass_terrain(
    *, seed: int = 0, contrast: float = 1.0, barriers: bool = True
) -> TerrainScenario:
    seed, contrast, barriers = _validate_controls(seed, contrast, barriers)
    rng = np.random.default_rng(seed)
    xa, ya, x, y = _grid()
    level = seed % 3
    ridge_x = 500.0 + rng.uniform(-35.0, 35.0)
    pass_y = 500.0 + rng.uniform(-150.0, 150.0)
    ridge = np.exp(-0.5 * ((x - ridge_x) / (58.0 - 7.0 * level)) ** 2)
    opening = np.exp(-0.5 * ((y - pass_y) / (105.0 - 15.0 * level)) ** 2)
    elevation = 35.0 + 330.0 * ridge * (1.0 - 0.78 * opening)
    log_s = math.log(0.8) + contrast * (1.25 + 0.2 * level) * ridge * (1.0 - 0.9 * opening)
    hard = ()
    if barriers:
        gap = 90.0 - 10.0 * level
        hard = (
            _rectangle(ridge_x - 18, 0, ridge_x + 18, pass_y - gap),
            _rectangle(ridge_x - 18, pass_y + gap, ridge_x + 18, 1_000),
        )
    return _scenario(
        "Synthetic ridge and pass",
        xa,
        ya,
        elevation,
        log_s,
        barriers=hard,
        metadata=_common_metadata("ridge_pass", seed, contrast, barriers),
    )


def competing_corridors_terrain(
    *, seed: int = 0, contrast: float = 1.0, barriers: bool = True
) -> TerrainScenario:
    seed, contrast, barriers = _validate_controls(seed, contrast, barriers)
    rng = np.random.default_rng(seed)
    xa, ya, x, y = _grid()
    level = seed % 3
    upper = 690.0 + rng.uniform(-35.0, 35.0)
    lower = 315.0 + rng.uniform(-35.0, 35.0)
    upper_corridor = np.exp(-0.5 * ((y - upper - 45 * np.sin(x / 175)) / (72 - 7 * level)) ** 2)
    lower_corridor = np.exp(-0.5 * ((y - lower + 25 * np.sin(x / 145)) / (105 - 8 * level)) ** 2)
    basin = np.maximum(upper_corridor, 0.82 * lower_corridor)
    elevation = 80.0 + 160.0 * (1.0 - basin) + 20.0 * np.sin(x / 140) * np.sin(y / 125)
    log_s = math.log(0.8) + contrast * (1.15 + 0.15 * level) * (1.0 - basin)
    hard = ()
    if barriers:
        hard = (
            _rectangle(350, 405, 650, 570),
            _rectangle(440, 0, 500, 205 + 15 * level),
        )
    return _scenario(
        "Synthetic competing corridors",
        xa,
        ya,
        elevation,
        log_s,
        barriers=hard,
        metadata=_common_metadata("competing_corridors", seed, contrast, barriers),
    )


def dead_ends_terrain(
    *, seed: int = 0, contrast: float = 1.0, barriers: bool = True
) -> TerrainScenario:
    seed, contrast, barriers = _validate_controls(seed, contrast, barriers)
    rng = np.random.default_rng(seed)
    xa, ya, x, y = _grid()
    level = seed % 3
    lure_y = 500.0 + rng.uniform(-65.0, 65.0)
    wall = np.exp(-0.5 * ((x - 700) / 42) ** 2) * np.exp(-0.5 * ((y - lure_y) / 235) ** 8)
    arms = np.exp(-0.5 * ((y - (lure_y + 225)) / 36) ** 2) + np.exp(
        -0.5 * ((y - (lure_y - 225)) / 36) ** 2
    )
    arms *= np.exp(-0.5 * ((x - 520) / 190) ** 8)
    elevation = 30.0 + 240.0 * np.minimum(1.0, wall + arms)
    log_s = math.log(0.8) + contrast * (1.35 + 0.15 * level) * np.minimum(1.0, wall + arms)
    hard = ()
    if barriers:
        thickness = 24.0 + 5.0 * level
        hard = (
            _rectangle(680, lure_y - 225, 680 + thickness, lure_y + 225),
            _rectangle(395, lure_y - 225, 704, lure_y - 225 + thickness),
            _rectangle(395, lure_y + 225 - thickness, 704, lure_y + 225),
        )
    return _scenario(
        "Synthetic dead ends",
        xa,
        ya,
        elevation,
        log_s,
        barriers=hard,
        metadata=_common_metadata("dead_ends", seed, contrast, barriers),
    )


def correlated_roughness_terrain(
    *, seed: int = 0, contrast: float = 1.0, barriers: bool = True
) -> TerrainScenario:
    seed, contrast, barriers = _validate_controls(seed, contrast, barriers)
    rng = np.random.default_rng(seed)
    xa, ya, x, y = _grid()
    level = seed % 3
    white = rng.normal(size=(SOURCE_RESOLUTION, SOURCE_RESOLUTION))
    ky = np.fft.fftfreq(SOURCE_RESOLUTION)[:, None]
    kx = np.fft.rfftfreq(SOURCE_RESOLUTION)[None, :]
    scale = 0.045 + 0.012 * level
    low_pass = np.exp(-(kx * kx + ky * ky) / (2 * scale * scale))
    rough = np.fft.irfft2(np.fft.rfft2(white) * low_pass, s=white.shape)
    rough = (rough - rough.mean()) / rough.std()
    elevation = 120.0 + 55.0 * rough
    log_s = math.log(0.8) + contrast * (0.28 + 0.06 * level) * rough
    hard: tuple[dict, ...] = ()
    if barriers:
        circles = []
        for _ in range(2 + level):
            circles.append(
                _circle(
                    float(rng.uniform(300, 700)),
                    float(rng.uniform(170, 830)),
                    float(rng.uniform(35, 60)),
                )
            )
        hard = tuple(circles)
    return _scenario(
        "Synthetic correlated roughness",
        xa,
        ya,
        elevation,
        log_s,
        barriers=hard,
        metadata=_common_metadata("correlated_roughness", seed, contrast, barriers),
    )


_FAMILIES: dict[str, Callable[..., TerrainScenario]] = {
    "ridge_pass": ridge_pass_terrain,
    "competing_corridors": competing_corridors_terrain,
    "dead_ends": dead_ends_terrain,
    "correlated_roughness": correlated_roughness_terrain,
}


def synthetic_terrain(
    family: str, *, seed: int = 0, contrast: float = 1.0, barriers: bool = True
) -> TerrainScenario:
    """Build one of four seeded benchmark families on the fixed source grid."""
    try:
        generator = _FAMILIES[family]
    except KeyError as exc:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILY_NAMES}") from exc
    return generator(seed=seed, contrast=contrast, barriers=barriers)


def _fixture_metadata(name: str, **extra: object) -> dict:
    return {
        "kind": "analytic_validation",
        "fixture": name,
        "source_resolution": [SOURCE_RESOLUTION, SOURCE_RESOLUTION],
        "cost_units": "seconds per horizontal metre",
        **extra,
    }


def uniform_terrain_fixture() -> TerrainScenario:
    xa, ya, x, _ = _grid()
    return _scenario(
        "Uniform analytic fixture",
        xa,
        ya,
        np.zeros_like(x),
        np.full_like(x, math.log(0.8)),
        start=(100, 200),
        goal=(900, 800),
        metadata=_fixture_metadata("uniform", expected_path="straight line"),
    )


def layered_refraction_fixture() -> TerrainScenario:
    xa, ya, x, y = _grid()
    # The interface is a grid row, making the two sampled homogeneous media explicit.
    slow = np.where(y < 500.0, 0.75, 1.5)
    return _scenario(
        "Layered refraction analytic fixture",
        xa,
        ya,
        0.04 * y,
        np.log(slow),
        start=(100, 150),
        goal=(900, 850),
        metadata=_fixture_metadata(
            "layered_refraction",
            interface_y_m=500.0,
            lower_slowness_s_per_m=0.75,
            upper_slowness_s_per_m=1.5,
            invariant="sin(theta)/speed is continuous across the ideal layer interface",
        ),
    )


def symmetry_fixture() -> TerrainScenario:
    xa, ya, x, y = _grid()
    radial = np.exp(-((x - 500) ** 2 + (y - 500) ** 2) / (2 * 125**2))
    return _scenario(
        "Reflection symmetry fixture",
        xa,
        ya,
        180 * radial,
        math.log(0.8) + 1.2 * radial,
        metadata=_fixture_metadata("symmetry", symmetry_axis_y_m=500.0),
    )


def obstacle_detour_fixture() -> TerrainScenario:
    xa, ya, x, _ = _grid()
    return _scenario(
        "Obstacle detour fixture",
        xa,
        ya,
        np.zeros_like(x),
        np.full_like(x, math.log(0.8)),
        barriers=(_rectangle(440, 350, 560, 650),),
        metadata=_fixture_metadata(
            "obstacle_detour", expected="route must pass above or below rectangle"
        ),
    )


def disconnected_fixture() -> TerrainScenario:
    xa, ya, x, _ = _grid()
    return _scenario(
        "Disconnected fixture",
        xa,
        ya,
        np.zeros_like(x),
        np.full_like(x, math.log(0.8)),
        barriers=(_rectangle(480, 0, 520, 1_000),),
        metadata=_fixture_metadata("disconnected", expected="no collision-free route"),
    )


_VALIDATION: dict[str, Callable[[], TerrainScenario]] = {
    "uniform": uniform_terrain_fixture,
    "layered_refraction": layered_refraction_fixture,
    "symmetry": symmetry_fixture,
    "obstacle_detour": obstacle_detour_fixture,
    "disconnected": disconnected_fixture,
}


def validation_terrain(name: str) -> TerrainScenario:
    try:
        return _VALIDATION[name]()
    except KeyError as exc:
        raise ValueError(
            f"unknown validation fixture {name!r}; expected one of {VALIDATION_NAMES}"
        ) from exc


def mount_tamalpais_terrain(*, contrast: float = 1.0, barriers: bool = False) -> TerrainScenario:
    """Load the offline, provenance-pinned Mount Tamalpais elevation crop.

    Elevation is observed data.  The cost transformation and optional closure
    polygons are explicitly labelled modelling assumptions in the scenario.
    """
    _, contrast, barriers = _validate_controls(0, contrast, barriers)
    path = files("path_planning_ode.data").joinpath("mount_tamalpais.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    base_log = np.asarray(payload.pop("model_base_log_slowness"), dtype=float)
    base = math.log(0.8)
    payload["log_slowness"] = (base + contrast * (base_log - base)).tolist()
    payload["metadata"]["contrast"] = contrast
    payload["metadata"]["barriers_enabled"] = barriers
    illustrative_barriers = tuple(payload.pop("illustrative_barriers_geojson"))
    payload["barriers_geojson"] = illustrative_barriers if barriers else ()
    return TerrainScenario.from_dict(payload)
