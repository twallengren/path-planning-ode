"""Build website data from the installed Python package; never duplicate the solver."""

import json
from pathlib import Path
from shutil import copytree

import numpy as np

import path_planning_ode
from path_planning_ode import cost_field, presets, solve, weighted_distance

root = Path(__file__).resolve().parents[1]
public = root / "web" / "public"
public.mkdir(parents=True, exist_ok=True)
copytree(
    Path(path_planning_ode.__file__).parent,
    public / "python" / "path_planning_ode",
    dirs_exist_ok=True,
    ignore=lambda directory, names: [n for n in names if n == "__pycache__"],
)
(public / "presets.json").write_text(json.dumps({k: v.to_dict() for k, v in presets().items()}))
scene = presets()["asymmetric"]
preview = solve(scene).to_dict()
preview["straight_cost"] = weighted_distance(np.array([scene.start, scene.end]), scene)
xx, yy = np.meshgrid(np.linspace(-4, 14, 90), np.linspace(-4, 14, 90))
preview["field"] = cost_field(np.stack([xx, yy], axis=-1), scene.obstacles).ravel().tolist()
(public / "preview.json").write_text(json.dumps(preview))
