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
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import Obstacle, Scene, SolverOptions, weighted_distance
from .field_adapter import PlaygroundField

Array = NDArray[np.float64]
Method = Literal["auto", "descent", "newton"]
Phase = Literal["descent", "newton"]
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
    method: Method = "auto"
    max_iterations: int = 2000
    tolerance: float = 1e-6
    armijo: float = 1e-4
    backtrack_factor: float = 0.5
    max_backtracks: int = 20
    step_cap_widths: float = 0.5
    quadrature_order: int = 8
    panel_widths: float = 1.0
    auto_descent_updates: int = 8
    auto_stall_window: int = 5
    auto_relative_improvement: float = 1e-4
    auto_gradient_trigger: float = 1e-3
    auto_cooldown: int = 8
    auto_residual_tolerance: float = 1e-5
    newton_cost_guard: float = 1e-3
    newton_energy_guard: float = 1e-2

    def __post_init__(self):
        for name in (
            "interior_points",
            "max_iterations",
            "max_backtracks",
            "quadrature_order",
            "auto_descent_updates",
            "auto_stall_window",
            "auto_cooldown",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_iterations > 2000:
            raise ValueError("max_iterations must be at most 2000.")
        if self.method not in ("auto", "descent", "newton"):
            raise ValueError("method must be 'auto', 'descent', or 'newton'.")
        finite_positive = (
            "tolerance",
            "step_cap_widths",
            "panel_widths",
            "auto_relative_improvement",
            "auto_gradient_trigger",
            "auto_residual_tolerance",
            "newton_cost_guard",
            "newton_energy_guard",
        )
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
    normalized_energy: float
    route_cost: float
    free_gradient_norm: float
    normalized_free_gradient_norm: float
    scaled_free_gradient_norm: float
    ode_residual_norm: float
    normalized_ode_residual_norm: float
    scaled_ode_residual_norm: float
    coordinate_scale: float
    cost_scale: float
    energy_scale: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlaygroundState:
    """A complete, serializable incremental-solver snapshot."""

    path: Array
    obstacles: tuple[Obstacle, ...]
    field: PlaygroundField
    coordinate_scale: float
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
    phase: Phase = "descent"
    accepted_descent_updates: int = 0
    relative_energy_improvements: tuple[float, ...] = ()
    newton_cooldown: int = 0
    phase_reference_energy: float | None = None
    phase_reference_cost: float | None = None
    phase_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "path": self.path.tolist(),
            "obstacles": [asdict(item) for item in self.obstacles],
            "field_spec": self.field.to_spec(),
            "field_hash": self.field.field_hash,
            "coordinate_scale": self.coordinate_scale,
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
            "phase": self.phase,
            "accepted_descent_updates": self.accepted_descent_updates,
            "relative_energy_improvements": list(self.relative_energy_improvements),
            "newton_cooldown": self.newton_cooldown,
            "phase_reference_energy": self.phase_reference_energy,
            "phase_reference_cost": self.phase_reference_cost,
            "phase_reason": self.phase_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PlaygroundState:
        """Restore a strict JSON snapshot, including its immutable field."""
        try:
            field = PlaygroundField.from_spec(value["field_spec"])
            if value.get("field_hash") != field.field_hash:
                raise ValueError("Playground state field_hash does not match field_spec.")
            options = PlaygroundOptions.from_dict(value["options"])
            path = _validated_path(value["path"], options.interior_points)
            metrics = PlaygroundMetrics(**dict(value["metrics"]))
            return cls(
                path=path,
                obstacles=field.gaussians,
                field=field,
                coordinate_scale=float(value["coordinate_scale"]),
                options=options,
                quadrature_panels=tuple(int(item) for item in value["quadrature_panels"]),
                metrics=metrics,
                initial_gradient_norm=float(value["initial_gradient_norm"]),
                initial_residual_norm=float(value["initial_residual_norm"]),
                iteration=int(value.get("iteration", 0)),
                status=value.get("status", "running"),
                pin_index=value.get("pin_index"),
                pin_position=(
                    tuple(value["pin_position"]) if value.get("pin_position") is not None else None
                ),
                step_size=float(value.get("step_size", 0.0)),
                elapsed_seconds=float(value.get("elapsed_seconds", 0.0)),
                phase=value.get("phase", "descent"),
                accepted_descent_updates=int(value.get("accepted_descent_updates", 0)),
                relative_energy_improvements=tuple(
                    float(item) for item in value.get("relative_energy_improvements", ())
                ),
                newton_cooldown=int(value.get("newton_cooldown", 0)),
                phase_reference_energy=value.get("phase_reference_energy"),
                phase_reference_cost=value.get("phase_reference_cost"),
                phase_reason=value.get("phase_reason"),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed playground state.") from exc


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


def _coerce_field(
    field: PlaygroundField | Mapping | None,
    obstacles: Sequence[Obstacle | Mapping],
    *,
    bounds: ArrayLike | None = None,
) -> PlaygroundField:
    gaussian_values = _coerce_obstacles(obstacles)
    if isinstance(field, PlaygroundField):
        if gaussian_values and gaussian_values != field.gaussians:
            return PlaygroundField(field.base_spec, gaussian_values, bounds=field.bounds_m)
        return field
    return PlaygroundField.from_spec(field, gaussian_values, bounds=bounds)


def _step_cap(path: Array, field: PlaygroundField, options: PlaygroundOptions) -> float:
    scale = field.minimum_scale
    if np.isfinite(scale):
        return options.step_cap_widths * scale
    return max(field.coordinate_scale(path) / max(len(path) - 1, 1), 1.0)


def _required_panels(
    path: Array,
    field: PlaygroundField,
    options: PlaygroundOptions,
    *,
    movement_allowance: float = 0.0,
) -> tuple[int, ...]:
    scale = field.minimum_scale
    if not np.isfinite(scale):
        return (1,) * (len(path) - 1)
    # Either endpoint of a segment can move by the allowance, so its length can
    # grow by at most twice that amount during the upcoming line search.
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1) + 2 * movement_allowance
    target = options.panel_widths * scale
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
    field: PlaygroundField | Mapping | None = None,
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
    analytic_field = _coerce_field(field, obstacles)
    options = PlaygroundOptions(
        interior_points=len(points) - 2,
        quadrature_order=quadrature_order,
        panel_widths=panel_widths,
    )
    if quadrature_panels is None:
        panels = _required_panels(points, analytic_field, options)
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
        costs = analytic_field.cost(samples)
        cost_gradients = analytic_field.gradient(samples)
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
    field: PlaygroundField,
    panels: tuple[int, ...],
    quadrature_order: int,
) -> float:
    """Energy-only line-search evaluation on an already frozen partition."""
    interval_count = len(path) - 1
    value = 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        _, weights_by_segment, samples = _quadrature_samples(path, panels, quadrature_order)
        costs = field.cost(samples)
        offset = 0
        for start, end, weights in zip(path[:-1], path[1:], weights_by_segment, strict=True):
            delta = end - start
            cost = costs[offset : offset + len(weights)]
            offset += len(weights)
            value += interval_count * float(delta @ delta) * float(np.dot(weights, cost * cost))
    return float(value)


def _playground_route_cost(
    path: Array,
    field: PlaygroundField,
    panels: tuple[int, ...],
    quadrature_order: int,
) -> float:
    """Independently integrate ``c ds`` along the displayed polyline."""
    if field.base_spec["kind"] == "uniform":
        scene = _core_scene(path, field.gaussians)
        return float(field.base_spec["cost"] * weighted_distance(path, scene))
    _, weights_by_segment, samples = _quadrature_samples(path, panels, quadrature_order)
    costs = field.cost(samples)
    value = 0.0
    offset = 0
    for start, end, weights in zip(path[:-1], path[1:], weights_by_segment, strict=True):
        selected = costs[offset : offset + len(weights)]
        offset += len(weights)
        value += float(np.linalg.norm(end - start) * np.dot(weights, selected))
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


def _ode_derivatives(
    field: PlaygroundField, q: Array, velocity: Array
) -> tuple[Array, Array, Array]:
    cost, gradient, hessian = field.evaluate(q)
    speed2 = np.sum(velocity * velocity, axis=-1)
    gradient_velocity = np.sum(gradient * velocity, axis=-1)
    acceleration = (
        speed2[..., None] * gradient - 2 * velocity * gradient_velocity[..., None]
    ) / cost[..., None]
    hessian_velocity = np.einsum("...ij,...j->...i", hessian, velocity)
    derivative_q = (
        speed2[..., None, None] * hessian
        - 2 * velocity[..., :, None] * hessian_velocity[..., None, :]
    ) / cost[..., None, None] - (
        acceleration[..., :, None] * gradient[..., None, :] / cost[..., None, None]
    )
    derivative_v = (
        2
        * (
            gradient[..., :, None] * velocity[..., None, :]
            - velocity[..., :, None] * gradient[..., None, :]
            - gradient_velocity[..., None, None] * np.eye(2)
        )
        / cost[..., None, None]
    )
    return acceleration, derivative_q, derivative_v


def _field_residual(path: Array, field: PlaygroundField) -> Array:
    step = 1.0 / (len(path) - 1)
    velocity = (path[2:] - path[:-2]) / (2 * step)
    acceleration = _ode_derivatives(field, path[1:-1], velocity)[0]
    return ((path[2:] - 2 * path[1:-1] + path[:-2]) / step**2 - acceleration).reshape(-1, 2)


def _field_jacobian(path: Array, field: PlaygroundField) -> Array:
    interior = len(path) - 2
    step = 1.0 / (interior + 1)
    _, derivative_q, derivative_v = _ode_derivatives(
        field, path[1:-1], (path[2:] - path[:-2]) / (2 * step)
    )
    matrix = np.zeros((2 * interior, 2 * interior))
    identity = np.eye(2)
    for index in range(interior):
        row = slice(2 * index, 2 * index + 2)
        matrix[row, row] = -2 * identity / step**2 - derivative_q[index]
        if index:
            matrix[row, 2 * index - 2 : 2 * index] = identity / step**2 + derivative_v[index] / (
                2 * step
            )
        if index < interior - 1:
            matrix[row, 2 * index + 2 : 2 * index + 4] = identity / step**2 - derivative_v[
                index
            ] / (2 * step)
    return matrix


def _evaluate(
    path: Array,
    field: PlaygroundField,
    panels: tuple[int, ...],
    options: PlaygroundOptions,
    pin_index: int | None,
    gradient_scale: float,
    residual_scale: float,
    coordinate_scale: float | None = None,
) -> tuple[PlaygroundMetrics, Array]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        energy_value, gradient = playground_energy_gradient(
            path,
            field=field,
            quadrature_panels=panels,
            quadrature_order=options.quadrature_order,
            panel_widths=options.panel_widths,
        )
        free = _free_vertices(len(path), pin_index)
        coordinate_scale = (
            field.coordinate_scale(path) if coordinate_scale is None else coordinate_scale
        )
        energy_scale = (coordinate_scale * field.cost_scale) ** 2
        normalized_gradient = coordinate_scale * gradient / energy_scale
        gradient_norm = _rms(gradient[free])
        normalized_gradient_norm = _rms(normalized_gradient[free])
        ode_values = _field_residual(path, field)
        residual_norm = _rms(ode_values[free - 1])
        normalized_residual_norm = residual_norm / coordinate_scale
        route_cost = _playground_route_cost(path, field, panels, options.quadrature_order)
    metrics = PlaygroundMetrics(
        energy=energy_value,
        normalized_energy=energy_value / energy_scale,
        route_cost=route_cost,
        free_gradient_norm=gradient_norm,
        normalized_free_gradient_norm=normalized_gradient_norm,
        scaled_free_gradient_norm=normalized_gradient_norm / max(1.0, gradient_scale),
        ode_residual_norm=residual_norm,
        normalized_ode_residual_norm=normalized_residual_norm,
        scaled_ode_residual_norm=normalized_residual_norm / max(1.0, residual_scale),
        coordinate_scale=coordinate_scale,
        cost_scale=field.cost_scale,
        energy_scale=energy_scale,
    )
    return metrics, gradient


def _status_for_metrics(
    metrics: PlaygroundMetrics, options: PlaygroundOptions, iteration: int
) -> PlaygroundStatus:
    if not np.isfinite(list(metrics.to_dict().values())).all():
        return "nonfinite"
    convergence = {
        "auto": metrics.scaled_ode_residual_norm,
        "descent": metrics.scaled_free_gradient_norm,
        "newton": metrics.scaled_ode_residual_norm,
    }[options.method]
    tolerance = options.auto_residual_tolerance if options.method == "auto" else options.tolerance
    if convergence <= tolerance:
        return "converged"
    if iteration >= options.max_iterations:
        return "iteration_limit"
    return "running"


def initialize_playground(
    start: ArrayLike,
    end: ArrayLike,
    obstacles: Sequence[Obstacle | Mapping] = (),
    *,
    field: PlaygroundField | Mapping | None = None,
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
    analytic_field = _coerce_field(field, obstacles)
    if not analytic_field.in_domain(points):
        raise ValueError("The complete path must lie inside field bounds.")
    cap = _step_cap(points, analytic_field, settings)
    panels = _required_panels(points, analytic_field, settings, movement_allowance=cap)
    coordinate_scale = analytic_field.coordinate_scale(points)
    provisional, _ = _evaluate(
        points,
        analytic_field,
        panels,
        settings,
        index,
        1.0,
        1.0,
        coordinate_scale,
    )
    gradient_scale = max(1.0, provisional.normalized_free_gradient_norm)
    residual_scale = max(1.0, provisional.normalized_ode_residual_norm)
    metrics, _ = _evaluate(
        points,
        analytic_field,
        panels,
        settings,
        index,
        gradient_scale,
        residual_scale,
        coordinate_scale,
    )
    return PlaygroundState(
        path=points,
        obstacles=analytic_field.gaussians,
        field=analytic_field,
        coordinate_scale=coordinate_scale,
        options=settings,
        quadrature_panels=panels,
        metrics=metrics,
        initial_gradient_norm=gradient_scale,
        initial_residual_norm=residual_scale,
        status=_status_for_metrics(metrics, settings, 0),
        pin_index=index,
        pin_position=position,
        elapsed_seconds=perf_counter() - started,
        phase="newton" if settings.method == "newton" else "descent",
    )


def evaluate_playground(
    path: ArrayLike,
    obstacles: Sequence[Obstacle | Mapping] = (),
    *,
    field: PlaygroundField | Mapping | None = None,
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
    analytic_field = _coerce_field(field, obstacles)
    if not analytic_field.in_domain(points):
        raise ValueError("The complete path must lie inside field bounds.")
    panels = (
        _required_panels(points, analytic_field, settings)
        if quadrature_panels is None
        else tuple(quadrature_panels)
    )
    metrics, _ = _evaluate(points, analytic_field, panels, settings, index, 1.0, 1.0)
    return metrics


def _expanded_panels(state: PlaygroundState) -> tuple[int, ...]:
    required = _required_panels(
        state.path,
        state.field,
        state.options,
        movement_allowance=_step_cap(state.path, state.field, state.options),
    )
    return tuple(max(old, new) for old, new in zip(state.quadrature_panels, required, strict=True))


def _descent_direction(state: PlaygroundState, gradient: Array) -> Array:
    free = _free_vertices(len(state.path), state.pin_index)
    direction = np.zeros_like(state.path)
    if free.size == 0:
        return direction
    # Removing one pinned row/column splits the Dirichlet stiffness matrix into
    # at most two tridiagonal blocks.  Thomas elimination keeps Gaussian-only
    # updates NumPy-only and linear in the vertex count.
    split = np.flatnonzero(np.diff(free) != 1) + 1
    for block in np.split(free, split):
        diagonal = np.full(len(block), 2.0)
        right = -gradient[block].copy()
        for index in range(1, len(block)):
            multiplier = -1.0 / diagonal[index - 1]
            diagonal[index] += multiplier
            right[index] -= multiplier * right[index - 1]
        solved = np.empty_like(right)
        solved[-1] = right[-1] / diagonal[-1]
        for index in range(len(block) - 2, -1, -1):
            solved[index] = (right[index] + solved[index + 1]) / diagonal[index]
        direction[block] = solved
    return direction


def _newton_direction(state: PlaygroundState) -> tuple[Array, Array]:
    all_residual = _field_residual(state.path, state.field).ravel()
    matrix = _field_jacobian(state.path, state.field)
    free = _free_vertices(len(state.path), state.pin_index) - 1
    scalar = np.ravel(np.column_stack((2 * free, 2 * free + 1)))
    direction = np.zeros_like(state.path)
    if scalar.size:
        solved = np.linalg.solve(matrix[np.ix_(scalar, scalar)], -all_residual[scalar])
        direction[free + 1] = solved.reshape(-1, 2)
    return direction, all_residual[scalar]


def _free_residual_norm(path: Array, state: PlaygroundState) -> float:
    values = _field_residual(path, state.field)
    free = _free_vertices(len(path), state.pin_index)
    return _rms(values[free - 1])


def _single_step(state: PlaygroundState) -> PlaygroundState:
    if state.status != "running":
        return state
    started = perf_counter()
    panels = _expanded_panels(state)
    metrics, gradient = _evaluate(
        state.path,
        state.field,
        panels,
        state.options,
        state.pin_index,
        state.initial_gradient_norm,
        state.initial_residual_norm,
        state.coordinate_scale,
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

    automatic = state.options.method == "auto"
    working = replace(state, quadrature_panels=panels, metrics=metrics)
    phase = working.phase if automatic else working.options.method

    if automatic and phase == "descent" and working.newton_cooldown == 0:
        stationary_gradient = metrics.scaled_free_gradient_norm <= working.options.tolerance
        stalled = len(
            working.relative_energy_improvements
        ) >= working.options.auto_stall_window and all(
            value < working.options.auto_relative_improvement
            for value in working.relative_energy_improvements[-working.options.auto_stall_window :]
        )
        ready = working.accepted_descent_updates >= working.options.auto_descent_updates
        if stationary_gradient or (
            ready
            and (
                stalled or metrics.scaled_free_gradient_norm < working.options.auto_gradient_trigger
            )
        ):
            phase = "newton"
            working = replace(
                working,
                phase="newton",
                phase_reference_energy=metrics.energy,
                phase_reference_cost=metrics.route_cost,
                phase_reason=(
                    "descent_stationary"
                    if stationary_gradient
                    else "gradient"
                    if not stalled
                    else "small_energy_improvements"
                ),
            )

    # A failed/stationary descent may request one immediate guarded Newton
    # attempt.  The loop has at most the descent attempt and that Newton trial.
    for _phase_attempt in range(2):
        try:
            if phase == "descent":
                normalized_gradient = metrics.coordinate_scale * gradient / metrics.energy_scale
                normalized_direction = _descent_direction(working, normalized_gradient)
                derivative = float(np.sum(normalized_gradient * normalized_direction))
                direction = metrics.coordinate_scale * normalized_direction
                baseline_merit = metrics.normalized_energy
                invalid_direction = not np.isfinite(derivative) or derivative >= 0
            else:
                direction, _free_residual = _newton_direction(working)
                baseline_merit = metrics.ode_residual_norm**2
                derivative = -baseline_merit
                invalid_direction = False
        except np.linalg.LinAlgError:
            direction = np.full_like(working.path, np.nan)
            invalid_direction = True

        largest_move = (
            float(np.max(np.linalg.norm(direction, axis=1)))
            if np.isfinite(direction).all()
            else float("nan")
        )
        failed_before_search = (
            invalid_direction or not np.isfinite(largest_move) or largest_move == 0
        )
        accepted_path = None
        accepted_metrics = None
        step_size = 0.0
        if not failed_before_search:
            step_size = min(
                1.0, _step_cap(working.path, working.field, working.options) / largest_move
            )
            for _ in range(working.options.max_backtracks):
                candidate = working.path + step_size * direction
                candidate[[0, -1]] = working.path[[0, -1]]
                if working.pin_index is not None:
                    candidate[working.pin_index] = working.pin_position
                if not np.isfinite(candidate).all() or not working.field.in_domain(candidate):
                    accepted = False
                else:
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        if phase == "descent":
                            candidate_energy = _playground_energy(
                                candidate,
                                working.field,
                                panels,
                                working.options.quadrature_order,
                            )
                            candidate_merit = candidate_energy / metrics.energy_scale
                            accepted = np.isfinite(candidate_merit) and candidate_merit <= (
                                baseline_merit + working.options.armijo * step_size * derivative
                            )
                        else:
                            candidate_residual = _free_residual_norm(candidate, working)
                            candidate_merit = candidate_residual**2
                            accepted = (
                                np.isfinite(candidate_merit)
                                and candidate_merit
                                <= (1 - working.options.armijo * step_size) * baseline_merit
                            )
                            if accepted and automatic:
                                candidate_energy = _playground_energy(
                                    candidate,
                                    working.field,
                                    panels,
                                    working.options.quadrature_order,
                                )
                                candidate_cost = _playground_route_cost(
                                    candidate,
                                    working.field,
                                    panels,
                                    working.options.quadrature_order,
                                )
                                reference_energy = (
                                    working.phase_reference_energy
                                    if working.phase_reference_energy is not None
                                    else metrics.energy
                                )
                                reference_cost = (
                                    working.phase_reference_cost
                                    if working.phase_reference_cost is not None
                                    else metrics.route_cost
                                )
                                accepted = bool(
                                    np.isfinite([candidate_energy, candidate_cost]).all()
                                    and candidate_energy
                                    <= reference_energy * (1 + working.options.newton_energy_guard)
                                    and candidate_cost
                                    <= reference_cost * (1 + working.options.newton_cost_guard)
                                )
                if accepted:
                    accepted_path = candidate
                    accepted_metrics, _ = _evaluate(
                        candidate,
                        working.field,
                        panels,
                        working.options,
                        working.pin_index,
                        working.initial_gradient_norm,
                        working.initial_residual_norm,
                        working.coordinate_scale,
                    )
                    break
                step_size *= working.options.backtrack_factor

        iteration = working.iteration + 1
        elapsed = perf_counter() - started
        if accepted_path is not None and accepted_metrics is not None:
            if phase == "descent" and automatic:
                improvement = max(
                    0.0,
                    (metrics.energy - accepted_metrics.energy)
                    / max(abs(metrics.energy), np.finfo(float).tiny),
                )
                history = (working.relative_energy_improvements + (improvement,))[
                    -working.options.auto_stall_window :
                ]
                cooldown = max(0, working.newton_cooldown - 1)
                return replace(
                    working,
                    path=accepted_path,
                    metrics=accepted_metrics,
                    iteration=iteration,
                    status=_status_for_metrics(accepted_metrics, working.options, iteration),
                    step_size=step_size,
                    elapsed_seconds=elapsed,
                    phase="descent",
                    accepted_descent_updates=working.accepted_descent_updates + 1,
                    relative_energy_improvements=history,
                    newton_cooldown=cooldown,
                    phase_reason="cooldown" if cooldown else "descent",
                )
            return replace(
                working,
                path=accepted_path,
                metrics=accepted_metrics,
                iteration=iteration,
                status=_status_for_metrics(accepted_metrics, working.options, iteration),
                step_size=step_size,
                elapsed_seconds=elapsed,
                phase=phase,
                phase_reason="newton" if automatic else working.phase_reason,
            )

        if automatic and phase == "descent":
            if working.newton_cooldown > 0:
                return replace(
                    working,
                    iteration=iteration,
                    status=(
                        "iteration_limit"
                        if iteration >= working.options.max_iterations
                        else "backtracking_failed"
                    ),
                    step_size=0.0,
                    elapsed_seconds=elapsed,
                    phase_reason="descent_failed_during_cooldown",
                )
            phase = "newton"
            working = replace(
                working,
                phase="newton",
                phase_reference_energy=metrics.energy,
                phase_reference_cost=metrics.route_cost,
                phase_reason="descent_failed",
            )
            continue
        if automatic:
            return replace(
                working,
                iteration=iteration,
                status="iteration_limit"
                if iteration >= working.options.max_iterations
                else "running",
                step_size=0.0,
                elapsed_seconds=elapsed,
                phase="descent",
                newton_cooldown=working.options.auto_cooldown,
                relative_energy_improvements=(),
                phase_reference_energy=None,
                phase_reference_cost=None,
                phase_reason="newton_failed",
            )
        failure: PlaygroundStatus = (
            "nonfinite"
            if not np.isfinite(direction).all()
            else "singular"
            if invalid_direction
            else "backtracking_failed"
        )
        return replace(
            working,
            iteration=iteration,
            status=failure,
            step_size=0.0,
            elapsed_seconds=elapsed,
        )
    raise AssertionError("automatic phase loop exceeded its fixed bound")


def advance_playground(
    state: PlaygroundState,
    obstacles: Sequence[Obstacle | Mapping] | None = None,
    *,
    field: PlaygroundField | Mapping | None = None,
    iterations: int = 1,
) -> PlaygroundState:
    """Advance up to ``iterations`` accepted-or-terminal solver attempts.

    Supplying a changed field reinitializes diagnostics and convergence scales
    around the exact current geometry before advancing.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    current = state
    if obstacles is not None or field is not None:
        replacement_field = _coerce_field(
            state.field if field is None else field,
            state.obstacles if obstacles is None else obstacles,
        )
        if replacement_field.field_hash != state.field.field_hash:
            current = initialize_playground(
                state.path[0],
                state.path[-1],
                replacement_field.gaussians,
                field=replacement_field,
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
        field=state.field,
        path=state.path,
        options=state.options,
        pin_index=pin_index,
        pin_position=pin_position,
    )
