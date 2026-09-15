"""Build website data from the installed Python package; never duplicate the solver."""

import gzip
import json
import subprocess
import sys
from pathlib import Path
from shutil import copy2, copytree, rmtree

import numpy as np

import path_planning_ode
from path_planning_ode import cost_field, presets, solve, weighted_distance
from path_planning_ode.planners import plan
from path_planning_ode.playground import build_playground_field
from path_planning_ode.soft_walls import soften_walls
from path_planning_ode.terrain import PlannerConfig, TerrainScenario
from path_planning_ode.terrain_generators import (
    FAMILY_NAMES,
    mount_tamalpais_terrain,
    synthetic_terrain,
)

root = Path(__file__).resolve().parents[1]
public = root / "web" / "public"


def clone_or_copy(source: str | Path, destination: str | Path) -> str:
    """Use an APFS clone when available, with a portable byte-copy fallback."""
    source_path = Path(source)
    destination_path = Path(destination)
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["cp", "-c", "-f", str(source_path), str(destination_path)],
            check=False,
            capture_output=True,
        )
        if completed.returncode == 0:
            return str(destination_path)
    return copy2(source_path, destination_path)


public.mkdir(parents=True, exist_ok=True)
copytree(
    Path(path_planning_ode.__file__).parent,
    public / "python" / "path_planning_ode",
    dirs_exist_ok=True,
    ignore=lambda directory, names: [n for n in names if n == "__pycache__"],
)
package_root = Path(path_planning_ode.__file__).parent
manifest = sorted(
    str(path.relative_to(package_root))
    for path in package_root.rglob("*")
    if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json"}
)
(public / "python-manifest.json").write_text(json.dumps(manifest))
(public / "presets.json").write_text(json.dumps({k: v.to_dict() for k, v in presets().items()}))
scene = presets()["asymmetric"]
preview = solve(scene).to_dict()
preview["straight_cost"] = weighted_distance(np.array([scene.start, scene.end]), scene)
xx, yy = np.meshgrid(np.linspace(-4, 14, 90), np.linspace(-4, 14, 90))
preview["field"] = cost_field(np.stack([xx, yy], axis=-1), scene.obstacles).ravel().tolist()
(public / "preview.json").write_text(json.dumps(preview))

# The homepage can draw its seeded field before Pyodide finishes loading. This
# preview uses the same Python stroke builder and cost field as the live worker.
playground_strokes = [
    {"id": "seed-1", "points": [[0, 0.15]], "width": 0.85, "strength": 18},
    {
        "id": "seed-2",
        "points": [[2.25, -2.25], [2.55, -1.8], [2.8, -1.25]],
        "width": 0.5,
        "strength": 8,
    },
]
playground_field = build_playground_field(playground_strokes)
playground_obstacles = tuple(
    path_planning_ode.Obstacle(**item) for item in playground_field["obstacles"]
)
playground_xx, playground_yy = np.meshgrid(
    np.linspace(-6, 6, 120), np.linspace(4, -4, 80)
)
(public / "playground-preview.json").write_text(
    json.dumps(
        {
            "width": 120,
            "height": 80,
            "field": cost_field(
                np.stack([playground_xx, playground_yy], axis=-1), playground_obstacles
            )
            .ravel()
            .tolist(),
        }
    )
)

terrain_dir = public / "terrain"
scenario_dir = terrain_dir / "scenarios"
scenario_dir.mkdir(parents=True, exist_ok=True)
terrain_scenarios = {
    family: soften_walls(synthetic_terrain(family, seed=0, contrast=1.0, barriers=True))
    for family in FAMILY_NAMES
}
for family, terrain_scenario in terrain_scenarios.items():
    (scenario_dir / f"{family}-0.json").write_text(json.dumps(terrain_scenario.to_dict()))
(scenario_dir / "mount_tamalpais.json").write_text(
    json.dumps(soften_walls(mount_tamalpais_terrain(barriers=True)).to_dict())
)

# The initial screen is a real planner result produced by the same public
# dispatcher as native and browser runs. Missing planner code is a build error.
preview_scenario = terrain_scenarios["ridge_pass"]
preview_config = PlannerConfig(
    method="fast_marching", initialization="fast_marching", reference_grid_size=129
)
preview_results = [plan(preview_scenario, preview_config)]
(terrain_dir / "preview.json").write_text(
    json.dumps(
        {
            "version": 2,
            "scenario": preview_scenario.to_dict(),
            "configs": [preview_config.to_dict()],
            "results": [result.to_dict() for result in preview_results],
        }
    )
)

# Copy the checked-in published study, including the exact frozen scenario
# assets used by its records, so every row can load back into the live explorer.
published_dir = root / "experiments" / "published"
published_index = published_dir / "index.json"
if published_index.is_file():
    study_dir = public / "study"
    if study_dir.exists():
        rmtree(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    for source in sorted(published_dir.iterdir()):
        if source.is_file():
            clone_or_copy(source, study_dir / source.name)
            assets.append(source.name)
        elif source.is_dir():
            copytree(source, study_dir / source.name, copy_function=clone_or_copy)
    report = root / "docs" / "study-report.md"
    if report.is_file() and not (study_dir / report.name).is_file():
        clone_or_copy(report, study_dir / report.name)
        if report.name not in assets:
            assets.append(report.name)

    study = json.loads(published_index.read_text(encoding="utf-8"))
    study_scenarios = study_dir / "scenarios"
    study_scenarios.mkdir(parents=True, exist_ok=True)
    scenario_hashes = set()
    for run in study.get("runs", []):
        result = run.get("result") or {}
        scenario_hash = run.get("scenario_hash") or result.get("scenario_hash")
        reference = run.get("scenario_ref")
        if not scenario_hash or not reference or scenario_hash in scenario_hashes:
            continue
        scenario_asset = study_scenarios / f"{scenario_hash}.json.gz"
        if not scenario_asset.is_file():
            raise RuntimeError(f"Missing frozen scenario asset for {reference}.")
        try:
            recorded_scenario = TerrainScenario.from_dict(
                json.loads(gzip.decompress(scenario_asset.read_bytes()))
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid frozen scenario asset for {reference}.") from error
        if recorded_scenario.scenario_hash != scenario_hash:
            raise RuntimeError(f"Frozen scenario hash mismatch for {reference}.")
        scenario_hashes.add(scenario_hash)
    (study_dir / "assets.json").write_text(json.dumps(sorted(assets)))
