"""Incremental path optimization for the freehand path playground.

The displayed path is a polygonal curve with a uniform parameter interval per
segment.  Its discrete energy is evaluated by composite Gauss--Legendre
quadrature.  Panel counts are part of :class:`PlaygroundState`: a step first
reserves enough panels for the largest geometrically permitted trial, then uses
those same nodes for its baseline, analytic gradient, and every line-search
trial.  This avoids differentiating an objective whose integration partition
changes during the step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from time import perf_counter
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import Obstacle, Scene, SolverOptions, jacobian, residual, weighted_distance

Array = NDArray[np.float64]
Method = Literal["descent", "newton"]
PlaygroundStatus = Literal[
    "running",
    "converged",
    "iteration_limit",
    "backtracking_failed",
    "singular",
    "nonfinite",
]

MAX_GAUSSIANS = 256


@dataclass(frozen=True)
class BrushStroke:
    """A freehand gesture whose brush settings remain attached to the gesture."""

    id: str
    points: tuple[tuple[float, float], ...]
    width: float
    strength: float

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("A stroke id must be a nonempty string.")
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 1:
            raise ValueError("Stroke points must have shape (n, 2) with n >= 1.")
        if not np.isfinite(points).all():
            raise ValueError("Stroke points must be finite.")
        if not np.isfinite([self.width, self.strength]).all():
            raise ValueError("Stroke width and strength must be finite.")
        if self.width <= 0 or self.strength < 1:
            raise ValueError("Stroke width must be positive and strength at least one.")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "points": [list(point) for point in self.points],
            "width": self.width,
            "strength": self.strength,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> BrushStroke:
        try:
            return cls(
                id=value["id"],
                points=tuple(tuple(point) for point in value["points"]),
                width=value["width"],
                strength=value["strength"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed brush stroke.") from exc


@dataclass(frozen=True)
class PlaygroundOptions:
    """Numerical controls for incremental playground updates.

    ``tolerance`` applies to the scaled free-gradient RMS in ``descent`` mode
    and to the scaled free-equation ODE-residual RMS in ``newton`` mode.
    """

    interior_points: int = 32
    method: Method = "descent"
    max_iterations: int = 2000
    tolerance: float = 1e-6
    armijo: float = 1e-4
    backtrack_factor: float = 0.5
    max_backtracks: int = 20
    step_cap_widths: float = 0.5
    quadrature_order: int = 8
    panel_widths: float = 1.0

    def __post_init__(self):
        for name in ("interior_points", "max_iterations", "max_backtracks", "quadrature_order"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_iterations > 2000:
            raise ValueError("max_iterations must be at most 2000.")
        if self.method not in ("descent", "newton"):
            raise ValueError("method must be 'descent' or 'newton'.")
        finite_positive = ("tolerance", "step_cap_widths", "panel_widths")
        if any(
            not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0
            for name in finite_positive
        ):
            raise ValueError("Tolerance and geometric scale options must be finite and positive.")
        if not np.isfinite(self.armijo) or not 0 < self.armijo < 1:
            raise ValueError("armijo must lie strictly between zero and one.")
        if not np.isfinite(self.backtrack_factor) or not 0 < self.backtrack_factor < 1:
            raise ValueError("backtrack_factor must lie strictly between zero and one.")
        if not 2 <= self.quadrature_order <= 32:
            raise ValueError("quadrature_order must be from 2 through 32.")

    @classmethod
    def from_dict(cls, value: Mapping | None) -> PlaygroundOptions:
        try:
            return cls(**dict(value or {}))
        except TypeError as exc:
            raise ValueError("Malformed playground options.") from exc


@dataclass(frozen=True)
class PlaygroundMetrics:
    """Energy, route cost, and two distinct stationarity diagnostics.

    Raw norms are root-mean-square values over the unpinned interior vertex
    coordinates (or matching ODE equations).  Each scaled norm divides its raw
    value by ``max(1, its initialization value)``; this is the value tested
    against :attr:`PlaygroundOptions.tolerance` for the corresponding method.
    """

    energy: float
    route_cost: float
    free_gradient_norm: float
    scaled_free_gradient_norm: float
    ode_residual_norm: float
    scaled_ode_residual_norm: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlaygroundState:
    """A complete, serializable incremental-solver snapshot."""

    path: Array
    obstacles: tuple[Obstacle, ...]
    options: PlaygroundOptions
    quadrature_panels: tuple[int, ...]
    metrics: PlaygroundMetrics
    initial_gradient_norm: float
    initial_residual_norm: float
    iteration: int = 0
    status: PlaygroundStatus = "running"
    pin_index: int | None = None
    pin_position: tuple[float, float] | None = None
    step_size: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "path": self.path.tolist(),
            "obstacles": [asdict(item) for item in self.obstacles],
            "options": asdict(self.options),
            "quadrature_panels": [int(item) for item in self.quadrature_panels],
            "metrics": self.metrics.to_dict(),
            "initial_gradient_norm": self.initial_gradient_norm,
            "initial_residual_norm": self.initial_residual_norm,
            "iteration": self.iteration,
            "status": self.status,
            "pin_index": self.pin_index,
            "pin_position": list(self.pin_position) if self.pin_position is not None else None,
            "step_size": self.step_size,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _coerce_strokes(strokes: Sequence[BrushStroke | Mapping]) -> tuple[BrushStroke, ...]:
    result = tuple(
        item if isinstance(item, BrushStroke) else BrushStroke.from_dict(item) for item in strokes
    )
    ids = [item.id for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Stroke ids must be unique.")
    return result


def _coerce_obstacles(obstacles: Sequence[Obstacle | Mapping]) -> tuple[Obstacle, ...]:
    try:
        return tuple(
            item if isinstance(item, Obstacle) else Obstacle(**dict(item)) for item in obstacles
        )
    except TypeError as exc:
        raise ValueError("Malformed Gaussian obstacle.") from exc


def _polyline_at(points: Array, cumulative: Array, distances: Array) -> Array:
    """Interpolate points at arc distances, including repeated-input safety."""
    segment = np.searchsorted(cumulative[1:], distances, side="right")
    segment = np.minimum(segment, len(points) - 2)
    lengths = np.diff(cumulative)
    fraction = (distances - cumulative[segment]) / lengths[segment]
    return points[segment] + fraction[:, None] * (points[segment + 1] - points[segment])


def strokes_to_obstacles(
    strokes: Sequence[BrushStroke | Mapping],
    *,
    max_gaussians: int = MAX_GAUSSIANS,
    spacing_widths: float = 0.5,
) -> tuple[Obstacle, ...]:
    """Convert freehand strokes to sampling-rate-invariant Gaussian line brushes.

    Nonzero strokes are split into equal arc-length cells no wider than
    ``spacing_widths * stroke.width``.  A Gaussian is placed at each cell
    midpoint and gets weight
    ``(strength - 1) * cell_length / (sqrt(pi) * width)``.  Thus pointer event
    density does not alter the represented field, and a long stroke has an
    interior cost approaching the nominal multiplicative ``strength``.  A
    one-point stroke (or a gesture containing only duplicate points) produces a
    Gaussian of weight ``strength - 1``.  The cap is strict: callers can reject
    the newest gesture without silently changing older strokes.
    """
    if isinstance(max_gaussians, bool) or not isinstance(max_gaussians, int) or max_gaussians < 1:
        raise ValueError("max_gaussians must be a positive integer.")
    if not np.isfinite(spacing_widths) or spacing_widths <= 0:
        raise ValueError("spacing_widths must be finite and positive.")

    result: list[Obstacle] = []
    for stroke in _coerce_strokes(strokes):
        points = np.asarray(stroke.points, dtype=float)
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0]
        points = points[keep]
        if len(points) == 1:
            count = 1
        else:
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            cumulative = np.r_[0.0, np.cumsum(lengths)]
            total = float(cumulative[-1])
            count = max(1, int(np.ceil(total / (spacing_widths * stroke.width))))
        if len(result) + count > max_gaussians:
            required = len(result) + count
            raise ValueError(
                f"Stroke field requires {required} Gaussian bumps; maximum is {max_gaussians}."
            )
        if len(points) == 1:
            generated = [Obstacle(*points[0], weight=stroke.strength - 1, width=stroke.width)]
        else:
            cell_length = total / count
            centers = _polyline_at(points, cumulative, (np.arange(count) + 0.5) * cell_length)
            generated = [
                Obstacle(
                    *center,
                    weight=(stroke.strength - 1) * cell_length / (np.sqrt(np.pi) * stroke.width),
                    width=stroke.width,
                )
                for center in centers
            ]
        result.extend(generated)
    return tuple(result)


def build_playground_field(
    strokes: Sequence[BrushStroke | Mapping],
    *,
    max_gaussians: int = MAX_GAUSSIANS,
    spacing_widths: float = 0.5,
) -> dict:
    """Return a JSON-ready validated stroke field and its generated Gaussians."""
    validated = _coerce_strokes(strokes)
    obstacles = strokes_to_obstacles(
        validated, max_gaussians=max_gaussians, spacing_widths=spacing_widths
    )
    return {
        "strokes": [item.to_dict() for item in validated],
        "obstacles": [asdict(item) for item in obstacles],
        "gaussian_count": len(obstacles),
    }


def _validated_path(path: ArrayLike, interior_points: int) -> Array:
    result = np.asarray(path, dtype=float)
    if result.shape != (interior_points + 2, 2):
        raise ValueError(f"path must have shape ({interior_points + 2}, 2).")
    if not np.isfinite(result).all():
        raise ValueError("path coordinates must be finite.")
    if np.array_equal(result[0], result[-1]):
        raise ValueError("Path endpoints must be distinct.")
    return result.copy()


def _validated_pin(
    path: Array, pin_index: int | None, pin_position: ArrayLike | None
) -> tuple[int | None, tuple[float, float] | None]:
    if pin_index is None:
        if pin_position is not None:
            raise ValueError("pin_position requires pin_index.")
        return None, None
    if (
        isinstance(pin_index, bool)
        or not isinstance(pin_index, int)
        or not 1 <= pin_index < len(path) - 1
    ):
        raise ValueError("pin_index must identify an interior full-path vertex.")
    position = path[pin_index] if pin_position is None else np.asarray(pin_position, dtype=float)
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("pin_position must contain two finite coordinates.")
    return pin_index, (float(position[0]), float(position[1]))


def _minimum_width(obstacles: tuple[Obstacle, ...]) -> float:
    return min((item.width for item in obstacles), default=float("inf"))


def _step_cap(path: Array, obstacles: tuple[Obstacle, ...], options: PlaygroundOptions) -> float:
    width = _minimum_width(obstacles)
    if np.isfinite(width):
        return options.step_cap_widths * width
    return max(float(np.linalg.norm(path[-1] - path[0])) / (len(path) - 1), 1.0)


def _required_panels(
    path: Array,
    obstacles: tuple[Obstacle, ...],
    options: PlaygroundOptions,
    *,
    movement_allowance: float = 0.0,
) -> tuple[int, ...]:
    width = _minimum_width(obstacles)
    if not np.isfinite(width):
        return (1,) * (len(path) - 1)
    # Either endpoint of a segment can move by the allowance, so its length can
    # grow by at most twice that amount during the upcoming line search.
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1) + 2 * movement_allowance
    target = options.panel_widths * width
    return tuple(int(item) for item in np.maximum(1, np.ceil(lengths / target).astype(int)))


@lru_cache(maxsize=31)
def _gauss_rule(order: int) -> tuple[Array, Array]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1) / 2, weights / 2


def _gaussian_cost_gradient(
    points: Array, obstacles: tuple[Obstacle, ...], *, gradient: bool
) -> tuple[Array, Array | None]:
    """Evaluate Gaussian cost and optionally gradient without unused Hessians."""
    cost = np.ones(points.shape[:-1])
    cost_gradient = np.zeros_like(points) if gradient else None
    if not obstacles:
        return cost, cost_gradient
    centers = np.array([[item.x, item.y] for item in obstacles])
    weights = np.array([item.weight for item in obstacles])
    inverse_width2 = 1 / np.square([item.width for item in obstacles])
    # Bound temporary (sample, obstacle, coordinate) arrays for unusually fine
    # quadrature while vectorizing the normal interactive workload.
    flat_points = points.reshape(-1, 2)
    flat_cost = cost.reshape(-1)
    flat_gradient = None if cost_gradient is None else cost_gradient.reshape(-1, 2)
    for first in range(0, len(flat_points), 4096):
        selected = slice(first, first + 4096)
        delta = flat_points[selected, None, :] - centers[None, :, :]
        bump = weights[None, :] * np.exp(-np.sum(delta * delta, axis=-1) * inverse_width2[None, :])
        flat_cost[selected] += np.sum(bump, axis=1)
        if flat_gradient is not None:
            flat_gradient[selected] += np.sum(
                (-2 * inverse_width2[None, :] * bump)[..., None] * delta, axis=1
            )
    return cost, cost_gradient


def _quadrature_samples(
    path: Array, panels: tuple[int, ...], order: int
) -> tuple[list[Array], list[Array], Array]:
    rule_nodes, rule_weights = _gauss_rule(order)
    parameters_by_segment = []
    weights_by_segment = []
    samples = []
    for start, end, panel_count in zip(path[:-1], path[1:], panels, strict=True):
        parameters = (
            (np.arange(panel_count, dtype=float)[:, None] + rule_nodes) / panel_count
        ).ravel()
        weights = np.broadcast_to(rule_weights / panel_count, (panel_count, order)).ravel()
        parameters_by_segment.append(parameters)
        weights_by_segment.append(weights)
        samples.append(start + parameters[:, None] * (end - start))
    return parameters_by_segment, weights_by_segment, np.concatenate(samples)


def playground_energy_gradient(
    path: ArrayLike,
    obstacles: Sequence[Obstacle | Mapping] = (),
    *,
    quadrature_panels: Sequence[int] | None = None,
    quadrature_order: int = 8,
    panel_widths: float = 1.0,
) -> tuple[float, Array]:
    """Evaluate the discrete energy and its exact analytic vertex gradient."""
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
        raise ValueError("path must have shape (n, 2) with n >= 3.")
    if not np.isfinite(points).all():
        raise ValueError("path coordinates must be finite.")
    field = _coerce_obstacles(obstacles)
    options = PlaygroundOptions(
        interior_points=len(points) - 2,
        quadrature_order=quadrature_order,
        panel_widths=panel_widths,
    )
    if quadrature_panels is None:
        panels = _required_panels(points, field, options)
    else:
        panels = tuple(quadrature_panels)
        if len(panels) != len(points) - 1 or any(
            isinstance(item, bool) or not isinstance(item, (int, np.integer)) or item < 1
            for item in panels
        ):
            raise ValueError("quadrature_panels must contain one positive integer per segment.")

    gradient = np.zeros_like(points)
    energy_value = 0.0
    interval_count = len(points) - 1
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        parameters_by_segment, weights_by_segment, samples = _quadrature_samples(
            points, panels, quadrature_order
        )
        costs, cost_gradients = _gaussian_cost_gradient(samples, field, gradient=True)
        assert cost_gradients is not None
        offset = 0
        for index, (start, end, parameters, weights) in enumerate(
            zip(
                points[:-1],
                points[1:],
                parameters_by_segment,
                weights_by_segment,
                strict=True,
            )
        ):
            delta = end - start
            selected = slice(offset, offset + len(parameters))
            cost = costs[selected]
            cost_gradient = cost_gradients[selected]
            offset += len(parameters)
            squared_cost_integral = float(np.dot(weights, cost * cost))
            spatial = (2 * weights * cost)[:, None] * cost_gradient
            length2 = float(np.dot(delta, delta))
            energy_value += interval_count * length2 * squared_cost_integral
            gradient[index] += interval_count * (
                -2 * delta * squared_cost_integral
                + length2 * np.sum((1 - parameters)[:, None] * spatial, axis=0)
            )
            gradient[index + 1] += interval_count * (
                2 * delta * squared_cost_integral
                + length2 * np.sum(parameters[:, None] * spatial, axis=0)
            )
    return float(energy_value), gradient


def _playground_energy(
    path: Array,
    obstacles: tuple[Obstacle, ...],
    panels: tuple[int, ...],
    quadrature_order: int,
) -> float:
    """Energy-only line-search evaluation on an already frozen partition."""
    interval_count = len(path) - 1
    value = 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        _, weights_by_segment, samples = _quadrature_samples(path, panels, quadrature_order)
        costs = _gaussian_cost_gradient(samples, obstacles, gradient=False)[0]
        offset = 0
        for start, end, weights in zip(path[:-1], path[1:], weights_by_segment, strict=True):
            delta = end - start
            cost = costs[offset : offset + len(weights)]
            offset += len(weights)
            value += interval_count * float(delta @ delta) * float(np.dot(weights, cost * cost))
    return float(value)


def _free_vertices(length: int, pin_index: int | None) -> Array:
    result = np.arange(1, length - 1)
    return result if pin_index is None else result[result != pin_index]


def _core_scene(path: Array, obstacles: tuple[Obstacle, ...]) -> Scene:
    return Scene(
        start=tuple(path[0]),
        end=tuple(path[-1]),
        obstacles=obstacles,
        options=SolverOptions(interior_points=len(path) - 2),
        guesses=("straight",),
    )


def _rms(values: Array) -> float:
    return 0.0 if values.size == 0 else float(np.linalg.norm(values) / np.sqrt(values.size))


def _evaluate(
    path: Array,
    obstacles: tuple[Obstacle, ...],
    panels: tuple[int, ...],
    options: PlaygroundOptions,
    pin_index: int | None,
    gradient_scale: float,
    residual_scale: float,
) -> tuple[PlaygroundMetrics, Array]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        energy_value, gradient = playground_energy_gradient(
            path,
            obstacles,
            quadrature_panels=panels,
            quadrature_order=options.quadrature_order,
            panel_widths=options.panel_widths,
        )
        free = _free_vertices(len(path), pin_index)
        gradient_norm = _rms(gradient[free])
        scene = _core_scene(path, obstacles)
        ode_values = residual(path, scene).reshape(-1, 2)
        residual_norm = _rms(ode_values[free - 1])
        route_cost = weighted_distance(path, scene)
    metrics = PlaygroundMetrics(
        energy=energy_value,
        route_cost=route_cost,
        free_gradient_norm=gradient_norm,
        scaled_free_gradient_norm=gradient_norm / max(1.0, gradient_scale),
        ode_residual_norm=residual_norm,
        scaled_ode_residual_norm=residual_norm / max(1.0, residual_scale),
    )
    return metrics, gradient


def _status_for_metrics(
    metrics: PlaygroundMetrics, options: PlaygroundOptions, iteration: int
) -> PlaygroundStatus:
    if not np.isfinite(list(metrics.to_dict().values())).all():
        return "nonfinite"
    convergence = (
        metrics.scaled_free_gradient_norm
        if options.method == "descent"
        else metrics.scaled_ode_residual_norm
    )
    if convergence <= options.tolerance:
        return "converged"
    if iteration >= options.max_iterations:
        return "iteration_limit"
    return "running"


def initialize_playground(
    start: ArrayLike,
    end: ArrayLike,
    obstacles: Sequence[Obstacle | Mapping] = (),
    *,
    path: ArrayLike | None = None,
    options: PlaygroundOptions | Mapping | None = None,
    pin_index: int | None = None,
    pin_position: ArrayLike | None = None,
) -> PlaygroundState:
    """Validate and preserve an arbitrary path, or create a straight one."""
    started = perf_counter()
    start_point, end_point = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if (
        start_point.shape != (2,)
        or end_point.shape != (2,)
        or not np.isfinite([start_point, end_point]).all()
    ):
        raise ValueError("start and end must contain two finite coordinates.")
    if np.array_equal(start_point, end_point):
        raise ValueError("Path endpoints must be distinct.")

    if options is None and path is not None:
        candidate = np.asarray(path)
        if candidate.ndim != 2 or candidate.shape[1:] != (2,) or len(candidate) < 3:
            raise ValueError("path must have shape (n, 2) with n >= 3.")
        settings = PlaygroundOptions(interior_points=len(candidate) - 2)
    else:
        settings = (
            options
            if isinstance(options, PlaygroundOptions)
            else PlaygroundOptions.from_dict(options)
        )
    if path is None:
        points = np.linspace(start_point, end_point, settings.interior_points + 2)
    else:
        points = _validated_path(path, settings.interior_points)
        if not np.array_equal(points[0], start_point) or not np.array_equal(points[-1], end_point):
            raise ValueError("Custom path endpoints must exactly match start and end.")

    index, position = _validated_pin(points, pin_index, pin_position)
    if index is not None:
        points[index] = position
    field = _coerce_obstacles(obstacles)
    cap = _step_cap(points, field, settings)
    panels = _required_panels(points, field, settings, movement_allowance=cap)
    provisional, _ = _evaluate(points, field, panels, settings, index, 1.0, 1.0)
    gradient_scale = max(1.0, provisional.free_gradient_norm)
    residual_scale = max(1.0, provisional.ode_residual_norm)
    metrics, _ = _evaluate(points, field, panels, settings, index, gradient_scale, residual_scale)
    return PlaygroundState(
        path=points,
        obstacles=field,
        options=settings,
        quadrature_panels=panels,
        metrics=metrics,
        initial_gradient_norm=gradient_scale,
        initial_residual_norm=residual_scale,
        status=_status_for_metrics(metrics, settings, 0),
        pin_index=index,
        pin_position=position,
        elapsed_seconds=perf_counter() - started,
    )


def evaluate_playground(
    path: ArrayLike,
    obstacles: Sequence[Obstacle | Mapping] = (),
    *,
    options: PlaygroundOptions | Mapping | None = None,
    pin_index: int | None = None,
    quadrature_panels: Sequence[int] | None = None,
) -> PlaygroundMetrics:
    """Evaluate all displayed diagnostics without taking a solver step."""
    candidate = np.asarray(path, dtype=float)
    if candidate.ndim != 2 or candidate.shape[1:] != (2,) or len(candidate) < 3:
        raise ValueError("path must have shape (n, 2) with n >= 3.")
    settings = (
        PlaygroundOptions(interior_points=len(candidate) - 2)
        if options is None
        else options
        if isinstance(options, PlaygroundOptions)
        else PlaygroundOptions.from_dict(options)
    )
    points = _validated_path(candidate, settings.interior_points)
    index, _ = _validated_pin(points, pin_index, None)
    field = _coerce_obstacles(obstacles)
    panels = (
        _required_panels(points, field, settings)
        if quadrature_panels is None
        else tuple(quadrature_panels)
    )
    metrics, _ = _evaluate(points, field, panels, settings, index, 1.0, 1.0)
    return metrics


def _expanded_panels(state: PlaygroundState) -> tuple[int, ...]:
    required = _required_panels(
        state.path,
        state.obstacles,
        state.options,
        movement_allowance=_step_cap(state.path, state.obstacles, state.options),
    )
    return tuple(max(old, new) for old, new in zip(state.quadrature_panels, required, strict=True))


def _descent_direction(state: PlaygroundState, gradient: Array) -> Array:
    free = _free_vertices(len(state.path), state.pin_index)
    direction = np.zeros_like(state.path)
    if free.size == 0:
        return direction
    interior_count = len(state.path) - 2
    stiffness = 2 * np.eye(interior_count)
    stiffness += np.diag(-np.ones(interior_count - 1), 1)
    stiffness += np.diag(-np.ones(interior_count - 1), -1)
    selected = free - 1
    restricted = stiffness[np.ix_(selected, selected)]
    direction[free] = np.linalg.solve(restricted, -gradient[free])
    return direction


def _newton_direction(state: PlaygroundState) -> tuple[Array, Array]:
    scene = _core_scene(state.path, state.obstacles)
    all_residual = residual(state.path, scene)
    matrix = jacobian(state.path, scene)
    free = _free_vertices(len(state.path), state.pin_index) - 1
    scalar = np.ravel(np.column_stack((2 * free, 2 * free + 1)))
    direction = np.zeros_like(state.path)
    if scalar.size:
        solved = np.linalg.solve(matrix[np.ix_(scalar, scalar)], -all_residual[scalar])
        direction[free + 1] = solved.reshape(-1, 2)
    return direction, all_residual[scalar]


def _free_residual_norm(path: Array, state: PlaygroundState) -> float:
    scene = _core_scene(path, state.obstacles)
    values = residual(path, scene).reshape(-1, 2)
    free = _free_vertices(len(path), state.pin_index)
    return _rms(values[free - 1])


def _single_step(state: PlaygroundState) -> PlaygroundState:
    if state.status != "running":
        return state
    started = perf_counter()
    panels = _expanded_panels(state)
    metrics, gradient = _evaluate(
        state.path,
        state.obstacles,
        panels,
        state.options,
        state.pin_index,
        state.initial_gradient_norm,
        state.initial_residual_norm,
    )
    baseline_status = _status_for_metrics(metrics, state.options, state.iteration)
    if baseline_status != "running":
        return replace(
            state,
            quadrature_panels=panels,
            metrics=metrics,
            status=baseline_status,
            elapsed_seconds=perf_counter() - started,
        )

    try:
        if state.options.method == "descent":
            direction = _descent_direction(state, gradient)
            derivative = float(np.sum(gradient * direction))
            if not np.isfinite(derivative) or derivative >= 0:
                failure: PlaygroundStatus = (
                    "nonfinite" if not np.isfinite(derivative) else "singular"
                )
                return replace(
                    state,
                    quadrature_panels=panels,
                    metrics=metrics,
                    iteration=state.iteration + 1,
                    status=failure,
                    elapsed_seconds=perf_counter() - started,
                )
            baseline_merit = metrics.energy
        else:
            direction, free_residual = _newton_direction(state)
            baseline_merit = float(np.dot(free_residual, free_residual))
            derivative = -baseline_merit
    except np.linalg.LinAlgError:
        return replace(
            state,
            quadrature_panels=panels,
            metrics=metrics,
            iteration=state.iteration + 1,
            status="singular",
            elapsed_seconds=perf_counter() - started,
        )

    if not np.isfinite(direction).all():
        return replace(
            state,
            quadrature_panels=panels,
            metrics=metrics,
            iteration=state.iteration + 1,
            status="nonfinite",
            elapsed_seconds=perf_counter() - started,
        )
    largest_move = float(np.max(np.linalg.norm(direction, axis=1)))
    if largest_move == 0:
        return replace(
            state,
            quadrature_panels=panels,
            metrics=metrics,
            iteration=state.iteration + 1,
            status="backtracking_failed",
            elapsed_seconds=perf_counter() - started,
        )
    step_size = min(1.0, _step_cap(state.path, state.obstacles, state.options) / largest_move)
    accepted_path = None
    accepted_metrics = None
    for _ in range(state.options.max_backtracks):
        candidate = state.path + step_size * direction
        candidate[[0, -1]] = state.path[[0, -1]]
        if state.pin_index is not None:
            candidate[state.pin_index] = state.pin_position
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if state.options.method == "descent":
                candidate_merit = _playground_energy(
                    candidate,
                    state.obstacles,
                    panels,
                    state.options.quadrature_order,
                )
                accepted = np.isfinite(candidate_merit) and candidate_merit <= (
                    baseline_merit + state.options.armijo * step_size * derivative
                )
            else:
                candidate_residual = _free_residual_norm(candidate, state)
                candidate_merit = candidate_residual**2
                accepted = np.isfinite(candidate_merit) and candidate_merit <= (
                    1 - state.options.armijo * step_size
                ) * (metrics.ode_residual_norm**2)
        if accepted:
            accepted_path = candidate
            accepted_metrics, _ = _evaluate(
                candidate,
                state.obstacles,
                panels,
                state.options,
                state.pin_index,
                state.initial_gradient_norm,
                state.initial_residual_norm,
            )
            break
        step_size *= state.options.backtrack_factor

    iteration = state.iteration + 1
    elapsed = perf_counter() - started
    if accepted_path is None or accepted_metrics is None:
        return replace(
            state,
            quadrature_panels=panels,
            metrics=metrics,
            iteration=iteration,
            status="backtracking_failed",
            step_size=0.0,
            elapsed_seconds=elapsed,
        )
    return replace(
        state,
        path=accepted_path,
        quadrature_panels=panels,
        metrics=accepted_metrics,
        iteration=iteration,
        status=_status_for_metrics(accepted_metrics, state.options, iteration),
        step_size=step_size,
        elapsed_seconds=elapsed,
    )


def advance_playground(
    state: PlaygroundState,
    obstacles: Sequence[Obstacle | Mapping] | None = None,
    *,
    iterations: int = 1,
) -> PlaygroundState:
    """Advance up to ``iterations`` accepted-or-terminal solver attempts.

    Supplying a changed field reinitializes diagnostics and convergence scales
    around the exact current geometry before advancing.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    current = state
    if obstacles is not None:
        field = _coerce_obstacles(obstacles)
        if field != state.obstacles:
            current = initialize_playground(
                state.path[0],
                state.path[-1],
                field,
                path=state.path,
                options=state.options,
                pin_index=state.pin_index,
                pin_position=state.pin_position,
            )
    for _ in range(iterations):
        next_state = _single_step(current)
        if next_state is current or next_state.status != "running":
            return next_state
        current = next_state
    return current


def set_playground_pin(
    state: PlaygroundState, pin_index: int | None, pin_position: ArrayLike | None = None
) -> PlaygroundState:
    """Set, move, or release the interior pin while retaining all other vertices."""
    return initialize_playground(
        state.path[0],
        state.path[-1],
        state.obstacles,
        path=state.path,
        options=state.options,
        pin_index=pin_index,
        pin_position=pin_position,
    )
