"""Run with uv run --extra plot python examples/explore.py [scene.json]."""

import argparse
import json
from pathlib import Path

from path_planning_ode import Scene, presets, solve

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("scene", nargs="?", type=Path, help="Exported version 1 scene JSON")
parser.add_argument("--no-plot", action="store_true", help="Print results without opening a plot")
args = parser.parse_args()
scene = (
    Scene.from_dict(json.loads(args.scene.read_text())) if args.scene else presets()["asymmetric"]
)
result = solve(scene)
for name, state in result.final.items():
    print(
        f"{name:10} {state.status:16} residual={state.residual_norm:.3g} "
        f"energy={state.energy:.2f} length={state.length:.2f}"
    )

if not args.no_plot:
    import matplotlib.pyplot as plt

    for name, state in result.final.items():
        plt.plot(*state.path.T, label=f"{name}: {state.status}")
    for obstacle in scene.obstacles:
        plt.scatter(obstacle.x, obstacle.y, marker="x", color="black")
    plt.axis("equal")
    plt.legend()
    plt.title("Euler–Lagrange stationary paths · soft obstacle costs")
    plt.show()
