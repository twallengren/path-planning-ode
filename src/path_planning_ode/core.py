"""Finite-difference boundary-value solver for E = integral c(q) |q'|² dt.

Points are ordered (x, y), including both fixed endpoints. There are N interior
points and N+1 intervals. Newton solves the Euler–Lagrange residual, which is
not a minimization algorithm and does not enforce collision constraints.
"""

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Status = Literal["running", "converged", "stagnated", "singular", "nonfinite", "iteration_limit"]


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    weight: float = 1.0
    width: float = 1.0

    def __post_init__(self):
        if not np.isfinite([self.x, self.y, self.weight, self.width]).all():
            raise ValueError("Obstacle values must be finite.")
        if self.weight < 0 or self.width <= 0:
            raise ValueError("Obstacle weight must be nonnegative and width positive.")


@dataclass(frozen=True)
class SolverOptions:
    interior_points: int = 30
    max_iterations: int = 100
    tolerance: float = 1e-7
    mode: Literal["damped", "undamped"] = "damped"

    def __post_init__(self):
        for name in ("interior_points", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Tolerance must be finite and positive.")
        if self.mode not in ("damped", "undamped"):
            raise ValueError("Mode must be damped or undamped.")


@dataclass(frozen=True)
class Scene:
    start: tuple[float, float] = (-2.0, -2.0)
    end: tuple[float, float] = (12.0, 12.0)
    obstacles: tuple[Obstacle, ...] = ()
    options: SolverOptions = field(default_factory=SolverOptions)
    guesses: tuple[str, ...] = ("straight", "bend-x", "bend-y")
    version: int = 1

    def __post_init__(self):
        if self.version != 1:
            raise ValueError("Unsupported scene version; expected 1.")
        if np.shape(self.start) != (2,) or np.shape(self.end) != (2,):
            raise ValueError("Endpoints must contain two coordinates.")
        if not np.isfinite([self.start, self.end]).all():
            raise ValueError("Endpoints must be finite.")
        if not self.guesses or len(set(self.guesses)) != len(self.guesses):
            raise ValueError("Choose at least one initial guess, without duplicates.")
        if any(g not in ("straight", "bend-x", "bend-y") for g in self.guesses):
            raise ValueError("Unknown initial guess.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Expected a version 1 scene object.")
        allowed = {"version", "start", "end", "obstacles", "options", "guesses"}
        if set(data) - allowed:
            raise ValueError("Scene contains unknown fields.")
        try:
            return cls(
                start=tuple(data["start"]),
                end=tuple(data["end"]),
                obstacles=tuple(Obstacle(**o) for o in data.get("obstacles", [])),
                options=SolverOptions(**data.get("options", {})),
                guesses=tuple(data.get("guesses", ("straight", "bend-x", "bend-y"))),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed scene.") from exc


@dataclass
class IterationState:
    path: Array
    iteration: int
    residual_norm: float
    energy: float
    length: float
    status: Status = "running"
    damping: float = 0.0

    def to_dict(self) -> dict:
        return {
            "path": self.path.tolist(),
            "iteration": self.iteration,
            "residual_norm": self.residual_norm,
            "energy": self.energy,
            "length": self.length,
            "status": self.status,
            "damping": self.damping,
        }


@dataclass
class SolveResult:
    scene: Scene
    histories: dict[str, list[IterationState]]

    @property
    def final(self) -> dict[str, IterationState]:
        return {name: history[-1] for name, history in self.histories.items()}

    def to_dict(self) -> dict:
        return {
            "scene": self.scene.to_dict(),
            "histories": {
                name: [s.to_dict() for s in states] for name, states in self.histories.items()
            },
        }


def _field(points: Array, obstacles: tuple[Obstacle, ...]) -> tuple[Array, Array, Array]:
    p = np.asarray(points, dtype=float)
    c = np.ones(p.shape[:-1])
    grad = np.zeros_like(p)
    hess = np.zeros(p.shape[:-1] + (2, 2))
    for obstacle in obstacles:
        d = p - [obstacle.x, obstacle.y]
        a = 1 / obstacle.width**2
        bump = obstacle.weight * np.exp(-a * np.sum(d * d, axis=-1))
        c += bump
        grad += (-2 * a * bump)[..., None] * d
        hess += bump[..., None, None] * (
            4 * a * a * d[..., :, None] * d[..., None, :] - 2 * a * np.eye(2)
        )
    return c, grad, hess


def cost_field(points: Array, obstacles: tuple[Obstacle, ...] = ()) -> Array:
    """Evaluate c(x,y); points have shape (..., 2)."""
    return _field(points, obstacles)[0]


def _ode_derivatives(q: Array, v: Array, obstacles: tuple[Obstacle, ...]):
    c, grad, hess = _field(q, obstacles)
    speed2 = np.sum(v * v, axis=-1)
    gv = np.sum(grad * v, axis=-1)
    f = (speed2[..., None] * grad - 2 * v * gv[..., None]) / (2 * c[..., None])
    hv = np.einsum("...ij,...j->...i", hess, v)
    fq = (speed2[..., None, None] * hess - 2 * v[..., :, None] * hv[..., None, :]) / (
        2 * c[..., None, None]
    ) - f[..., :, None] * grad[..., None, :] / c[..., None, None]
    fv = (
        grad[..., :, None] * v[..., None, :]
        - v[..., :, None] * grad[..., None, :]
        - gv[..., None, None] * np.eye(2)
    ) / c[..., None, None]
    return f, fq, fv


def ode(q: Array, velocity: Array, obstacles: tuple[Obstacle, ...] = ()) -> Array:
    """Acceleration in the Euler–Lagrange boundary-value ODE."""
    return _ode_derivatives(np.asarray(q), np.asarray(velocity), obstacles)[0]


def residual(path: Array, scene: Scene) -> Array:
    h = 1 / (len(path) - 1)
    v = (path[2:] - path[:-2]) / (2 * h)
    return (
        (path[2:] - 2 * path[1:-1] + path[:-2]) / h**2 - ode(path[1:-1], v, scene.obstacles)
    ).ravel()


def jacobian(path: Array, scene: Scene) -> Array:
    n = len(path) - 2
    h = 1 / (n + 1)
    _, fq, fv = _ode_derivatives(path[1:-1], (path[2:] - path[:-2]) / (2 * h), scene.obstacles)
    matrix = np.zeros((2 * n, 2 * n))
    for i in range(n):
        row = slice(2 * i, 2 * i + 2)
        matrix[row, row] = -2 * np.eye(2) / h**2 - fq[i]
        if i > 0:
            matrix[row, 2 * i - 2 : 2 * i] = np.eye(2) / h**2 + fv[i] / (2 * h)
        if i < n - 1:
            matrix[row, 2 * i + 2 : 2 * i + 4] = np.eye(2) / h**2 - fv[i] / (2 * h)
    return matrix


def energy(path: Array, scene: Scene) -> float:
    """Midpoint quadrature of the continuous energy, for diagnostics."""
    d = np.diff(path, axis=0)
    c = cost_field((path[1:] + path[:-1]) / 2, scene.obstacles)
    return float(np.sum(c * np.sum(d * d, axis=1)) * (len(path) - 1))


def _state(path: Array, scene: Scene, iteration: int, damping: float = 0) -> IterationState:
    norm = float(np.linalg.norm(residual(path, scene)) / np.sqrt(2 * (len(path) - 2)))
    value = energy(path, scene)
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    status: Status = "running"
    if not np.isfinite([norm, value, length]).all():
        status = "nonfinite"
    elif norm <= scene.options.tolerance:
        status = "converged"
    elif iteration >= scene.options.max_iterations:
        status = "iteration_limit"
    return IterationState(path.copy(), iteration, norm, value, length, status, damping)


def initialize(scene: Scene, guess: str = "straight") -> IterationState:
    exponents = {"straight": (1, 1), "bend-x": (5, 1), "bend-y": (1, 5)}
    if guess not in exponents:
        raise ValueError("Unknown initial guess.")
    t = np.linspace(0, 1, scene.options.interior_points + 2)[:, None]
    path = np.asarray(scene.start) + (np.asarray(scene.end) - scene.start) * t ** exponents[guess]
    return _state(path, scene, 0)


def step(scene: Scene, state: IterationState) -> IterationState:
    """One Newton iteration. Terminal states are returned unchanged."""
    if state.status != "running":
        return state

    def stopped(status: Status) -> IterationState:
        return IterationState(
            state.path.copy(),
            state.iteration + 1,
            state.residual_norm,
            state.energy,
            state.length,
            status,
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        r = residual(state.path, scene)
        j = jacobian(state.path, scene)
        if not np.isfinite(r).all() or not np.isfinite(j).all():
            return stopped("nonfinite")
        try:
            direction = np.linalg.solve(j, -r).reshape(-1, 2)
        except np.linalg.LinAlgError:
            return stopped("singular")
        if not np.isfinite(direction).all():
            return stopped("nonfinite")
        damping = 1.0
        for _ in range(21 if scene.options.mode == "damped" else 1):
            candidate = state.path.copy()
            candidate[1:-1] += damping * direction
            next_state = _state(candidate, scene, state.iteration + 1, damping)
            if next_state.status != "nonfinite" and (
                scene.options.mode == "undamped"
                or next_state.residual_norm**2 <= (1 - 1e-4 * damping) * state.residual_norm**2
            ):
                return next_state
            damping *= 0.5
        return stopped("nonfinite" if next_state.status == "nonfinite" else "stagnated")


def solve(scene: Scene) -> SolveResult:
    histories = {}
    for guess in scene.guesses:
        history = [initialize(scene, guess)]
        while history[-1].status == "running":
            history.append(step(scene, history[-1]))
        histories[guess] = history
    return SolveResult(scene, histories)
