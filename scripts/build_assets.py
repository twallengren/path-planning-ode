"""Copy the browser package and generate the playground's initial raster."""

import json
from pathlib import Path
from shutil import copytree, ignore_patterns, rmtree

import numpy as np

import path_planning_ode
from path_planning_ode.field_adapter import PlaygroundField, build_playground_preset

root = Path(__file__).resolve().parents[1]
public = root / "web" / "public"
package_root = Path(path_planning_ode.__file__).parent
python_destination = public / "python" / "path_planning_ode"

# A clean copy is essential: dirs_exist_ok would retain modules removed from
# the source package and let an old local build republish them.
if python_destination.exists():
    rmtree(python_destination)
copytree(package_root, python_destination, ignore=ignore_patterns("__pycache__"))

for obsolete in (
    public / "preview.json",
    public / "presets.json",
    public / "terrain",
    public / "study",
):
    if obsolete.exists():
        if obsolete.is_dir():
            rmtree(obsolete)
        else:
            obsolete.unlink()

manifest = sorted(
    str(path.relative_to(package_root))
    for path in package_root.rglob("*")
    if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json"}
)
(public / "python-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

# The static first paint and the live worker use the same preset definition.
scene = build_playground_preset("random_hills", seed=0)
field = PlaygroundField(scene["base_field"], scene["gaussians"])
xmin, xmax, ymin, ymax = scene["bounds"]
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 120), np.linspace(ymax, ymin, 80))
points = np.stack([xx, yy], axis=-1)
(public / "playground-preview.json").write_text(
    json.dumps(
        {
            "width": 120,
            "height": 80,
            "field": np.asarray(field.cost(points)).ravel().tolist(),
            "scene": scene,
        }
    ),
    encoding="utf-8",
)
