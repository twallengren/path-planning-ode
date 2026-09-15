"""Deterministic experiments used by the essay, notebook, and tests."""

from .core import Obstacle, Scene


def presets() -> dict[str, Scene]:
    return {
        "empty": Scene(),
        "central": Scene(obstacles=(Obstacle(5, 5, 6, 2),)),
        "asymmetric": Scene(
            obstacles=(
                Obstacle(3, 4, 5, 1.6),
                Obstacle(7, 7, 8, 1.8),
                Obstacle(8, 2, 3, 1.3),
            )
        ),
        "passage": Scene(
            start=(-2, 5),
            end=(12, 5),
            obstacles=tuple(
                Obstacle(x, y, 12, 1.2) for x, y in [(4, 2.8), (4, 7.2), (7, 2.8), (7, 7.2)]
            ),
        ),
        "challenge": Scene(
            obstacles=(
                Obstacle(2, 3, 35, 1.1),
                Obstacle(5, 5, 50, 1.4),
                Obstacle(8, 7, 35, 1.1),
            )
        ),
    }
