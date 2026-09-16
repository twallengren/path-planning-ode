#!/usr/bin/env python3
"""Run the small, non-publication terrain energy-descent comparison matrix."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy
import shapely

from path_planning_ode import PlannerConfig, plan
from path_planning_ode.soft_walls import soften_walls
from path_planning_ode.terrain_generators import synthetic_terrain

FAMILIES = (
    ("ridge_pass", 0),
    ("competing_corridors", 1),
    ("dead_ends", 2),
    ("correlated_roughness", 3),
)
INTERIOR_POINTS = (32, 64)
INITIALIZATIONS = ("straight", "arc_left")
METHODS = ("energy_descent", "euler_lagrange")


def _environment() -> dict:
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ("git", "status", "--porcelain"), text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "shapely": shapely.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "git_commit": commit,
        "git_worktree_dirty": dirty,
    }


def _config(method: str, initialization: str, interior_points: int) -> PlannerConfig:
    return PlannerConfig(
        method=method,
        initialization=initialization,
        interior_points=interior_points,
        tolerance=1e-6,
        max_iterations=1000,
        time_limit_s=8.0,
        profile_samples=65,
        options={"quadrature_order": 8},
    )


def run_comparison(output_dir: Path) -> list[dict]:
    """Execute the cold matrix and separate FMM-warm check, then write results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for family, seed in FAMILIES:
        scenario = soften_walls(
            synthetic_terrain(family, seed=seed, contrast=1.0, barriers=True),
            multiplier=100.0,
        )
        for interior_points in INTERIOR_POINTS:
            for initialization in INITIALIZATIONS:
                for method in METHODS:
                    config = _config(method, initialization, interior_points)
                    result = plan(scenario, config)
                    records.append(
                        {
                            "case_id": f"{family}-s{seed:02d}-soft100",
                            "family": family,
                            "seed": seed,
                            "soft_wall_multiplier": 100.0,
                            "scenario_hash": scenario.scenario_hash,
                            "method": method,
                            "initialization": initialization,
                            "interior_points": interior_points,
                            "config_hash": config.config_hash,
                            "config": config.to_dict(),
                            "initial_route_hash": result.diagnostics.get("initialization", {}).get(
                                "initial_route_hash"
                            ),
                            "termination_reason": result.termination_reason,
                            "solver_success": result.solver_success,
                            "feasible": result.feasible,
                            "evaluated_cost_s": result.evaluated_cost_s,
                            "length_m": (
                                None if result.evaluation is None else result.evaluation.length_m
                            ),
                            "iterations": result.diagnostics.get("iterations"),
                            "stationarity_norm": result.diagnostics.get("stationarity_norm"),
                            "ode_residual_norm_m": result.diagnostics.get("ode_residual_norm_m"),
                            "total_time_s": result.timing_s.get("total"),
                            "result": result.to_dict(),
                        }
                    )

    warm_checks = run_warm_start_check(output_dir, write_files=False)

    payload = {
        "kind": "terrain-energy-focused-comparison",
        "version": 1,
        "scope": (
            "Focused reproducible comparison only; not part of the frozen terrain study "
            "and not evidence of general method superiority."
        ),
        "matrix": {
            "families_and_seeds": [list(item) for item in FAMILIES],
            "soft_wall_multiplier": 100.0,
            "interior_points": list(INTERIOR_POINTS),
            "initializations": list(INITIALIZATIONS),
            "methods": list(METHODS),
            "tolerance": 1e-6,
            "max_iterations": 1000,
            "time_limit_s": 8.0,
            "profile_samples": 65,
            "quadrature_order": 8,
        },
        "environment": _environment(),
        "runs": records,
        "warm_start_check": {
            "scope": (
                "Separate four-case N=32 FMM-warm validation using the exact hybrid "
                "default controls; not part of the 32-run cold matrix or frozen study."
            ),
            "runs": warm_checks,
        },
    }
    (output_dir / "runs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    columns = [key for key in records[0] if key not in {"config", "result"}]
    with (output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: record[key] for key in columns} for record in records)
    return records


def run_warm_start_check(output_dir: Path, *, write_files: bool = True) -> list[dict]:
    """Run the exact four-family hybrid default and a separate FMM reference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    warm_checks = []
    for family, seed in FAMILIES:
        scenario = soften_walls(
            synthetic_terrain(family, seed=seed, contrast=1.0, barriers=True),
            multiplier=100.0,
        )
        reference_config = PlannerConfig(
            method="fast_marching",
            initialization="fast_marching",
            interior_points=32,
            reference_grid_size=129,
            tolerance=1e-5,
            max_iterations=400,
            time_limit_s=60.0,
            profile_samples=65,
        )
        warm_config = PlannerConfig(
            method="energy_descent",
            initialization="fast_marching",
            interior_points=32,
            reference_grid_size=129,
            tolerance=1e-5,
            max_iterations=400,
            time_limit_s=60.0,
            profile_samples=65,
            options={"quadrature_order": 8},
        )
        reference = plan(scenario, reference_config)
        energy = plan(scenario, warm_config)
        warm_checks.append(
            {
                "case_id": f"{family}-s{seed:02d}-soft100",
                "family": family,
                "seed": seed,
                "scenario_hash": scenario.scenario_hash,
                "interior_points": 32,
                "reference_grid_size": 129,
                "initialization": "fast_marching",
                "tolerance": 1e-5,
                "max_iterations": 400,
                "time_limit_s": 60.0,
                "energy_config_hash": warm_config.config_hash,
                "energy_config": warm_config.to_dict(),
                "reference_config_hash": reference_config.config_hash,
                "reference_config": reference_config.to_dict(),
                "energy_termination_reason": energy.termination_reason,
                "energy_solver_success": energy.solver_success,
                "energy_feasible": energy.feasible,
                "energy_evaluated_cost_s": energy.evaluated_cost_s,
                "energy_stationarity_norm": energy.diagnostics.get("stationarity_norm"),
                "energy_ode_residual_norm_m": energy.diagnostics.get("ode_residual_norm_m"),
                "energy_initialization_time_s": energy.timing_s.get("initialization"),
                "energy_solve_time_s": energy.timing_s.get("solve"),
                "energy_total_time_s": energy.timing_s.get("total"),
                "energy_initial_route_hash": energy.diagnostics.get("initialization", {}).get(
                    "initial_route_hash"
                ),
                "reference_termination_reason": reference.termination_reason,
                "reference_solver_success": reference.solver_success,
                "reference_feasible": reference.feasible,
                "reference_evaluated_cost_s": reference.evaluated_cost_s,
                "reference_total_time_s": reference.timing_s.get("total"),
                "energy_result": energy.to_dict(),
                "reference_result": reference.to_dict(),
                "signed_cost_difference_s": (
                    None
                    if energy.evaluated_cost_s is None or reference.evaluated_cost_s is None
                    else energy.evaluated_cost_s - reference.evaluated_cost_s
                ),
            }
        )

    payload = {
        "kind": "terrain-energy-warm-start-check",
        "version": 1,
        "scope": (
            "Separate four-case N=32 FMM-warm validation using the exact hybrid "
            "default controls; not part of the cold matrix or frozen study."
        ),
        "environment": _environment(),
        "runs": warm_checks,
    }
    if write_files:
        (output_dir / "warm-start.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    columns = [
        key
        for key in warm_checks[0]
        if key not in {"energy_config", "reference_config", "energy_result", "reference_result"}
    ]
    with (output_dir / "warm-start.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: record[key] for key in columns} for record in warm_checks)
    return warm_checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "terrain-energy-comparison",
        help="Output directory (default: system temporary directory).",
    )
    parser.add_argument(
        "--warm-only",
        action="store_true",
        help="Run only the four-family exact hybrid-default check.",
    )
    args = parser.parse_args()
    if args.warm_only:
        records = run_warm_start_check(args.output_dir)
        successes = sum(record["energy_solver_success"] for record in records)
        feasible = sum(record["energy_feasible"] for record in records)
        print(
            f"wrote {len(records)} warm-start runs to {args.output_dir} · "
            f"solver_success={successes} feasible={feasible}"
        )
        return
    records = run_comparison(args.output_dir)
    returned = sum(record["evaluated_cost_s"] is not None for record in records)
    successes = sum(record["solver_success"] for record in records)
    feasible = sum(record["feasible"] for record in records)
    print(
        f"wrote {len(records)} runs to {args.output_dir} · "
        f"evaluated={returned} solver_success={successes} feasible={feasible}"
    )


if __name__ == "__main__":
    main()
