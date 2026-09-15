"""Reproducible benchmark execution and analysis for terrain routing."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import platform
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from .terrain import PlannerConfig
from .terrain_generators import synthetic_terrain

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
_REPOSITORY_CHECKOUT = (REPOSITORY_ROOT / "pyproject.toml").exists()
DEFAULT_PROTOCOL_PATH = (
    REPOSITORY_ROOT / "experiments" / "protocol.json" if _REPOSITORY_CHECKOUT else None
)
DEFAULT_PUBLISHED_DIR = (
    REPOSITORY_ROOT / "experiments" / "published"
    if _REPOSITORY_CHECKOUT
    else Path.cwd() / "terrain-study-results"
)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    family: str
    seed: int
    contrast: float
    barriers: bool
    difficulty: str
    scenario_ref: str
    scenario_hash: str
    experiment: str = "baseline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "seed": self.seed,
            "contrast": self.contrast,
            "barriers": self.barriers,
            "difficulty": self.difficulty,
            "scenario_ref": self.scenario_ref,
            "scenario_hash": self.scenario_hash,
            "experiment": self.experiment,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseSpec":
        return cls(**data)


@dataclass(frozen=True)
class RunSpec:
    case: CaseSpec
    config: PlannerConfig
    profile: str
    protocol_sha256: str = ""

    @property
    def run_id(self) -> str:
        material = {
            "case_id": self.case.case_id,
            "scenario_hash": self.case.scenario_hash,
            "config_hash": self.config.config_hash,
            "profile": self.profile,
            "protocol_sha256": self.protocol_sha256,
        }
        return hashlib.sha256(_canonical_json(material).encode()).hexdigest()[:24]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
    elif DEFAULT_PROTOCOL_PATH is not None:
        text = DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8")
    else:
        text = files("path_planning_ode.data").joinpath("protocol.json").read_text(encoding="utf-8")
    data = json.loads(text)
    if data.get("version") != 1:
        raise ValueError("Expected experiment protocol version 1.")
    digest_data = dict(data)
    data["protocol_sha256"] = hashlib.sha256(_canonical_json(digest_data).encode()).hexdigest()
    return data


def _case_spec(
    family: str,
    seed: int,
    contrast: float,
    barriers: bool,
    *,
    experiment: str,
) -> CaseSpec:
    scenario = synthetic_terrain(family, seed=seed, contrast=contrast, barriers=barriers)
    contrast_label = str(contrast).replace(".", "p")
    if experiment == "baseline":
        case_id = f"{family}-s{seed:02d}"
        scenario_ref = f"synthetic/{family}/{seed}"
    else:
        case_id = f"contrast-{family}-s{seed:02d}-c{contrast_label}"
        scenario_ref = (
            f"synthetic/{family}/{seed}?contrast={contrast:g}&barriers={str(barriers).lower()}"
        )
    return CaseSpec(
        case_id=case_id,
        family=family,
        seed=seed,
        contrast=contrast,
        barriers=barriers,
        difficulty=str(scenario.metadata["difficulty"]),
        scenario_ref=scenario_ref,
        scenario_hash=scenario.scenario_hash,
        experiment=experiment,
    )


def frozen_cases(protocol: Mapping[str, Any], *, include_sweeps: bool = True) -> list[CaseSpec]:
    cases = [
        _case_spec(
            family,
            int(seed),
            float(protocol["baseline"]["contrast"]),
            bool(protocol["baseline"]["barriers"]),
            experiment="baseline",
        )
        for family in protocol["families"]
        for seed in protocol["benchmark_seeds"]
    ]
    if len(cases) != int(protocol["baseline"]["case_count"]):
        raise ValueError("Protocol baseline case_count disagrees with its frozen matrix.")
    if include_sweeps:
        sweep = protocol["contrast_sweep"]
        cases.extend(
            _case_spec(
                family,
                int(sweep["seed"]),
                float(contrast),
                bool(sweep["barriers"]),
                experiment="contrast_sweep",
            )
            for family in protocol["families"]
            for contrast in sweep["values"]
        )
    return cases


def _planner_config(
    method: str,
    initialization: str,
    *,
    interior_points: int,
    reference_grid_size: int,
    time_limit_s: float,
    max_iterations: int = 200,
) -> PlannerConfig:
    options: dict[str, Any] = {"quadrature_order": 8}
    if initialization == "fast_marching":
        options["warm_start_cost_included"] = True
    return PlannerConfig(
        method=method,
        initialization=initialization,
        interior_points=interior_points,
        reference_grid_size=reference_grid_size,
        tolerance=1e-6,
        max_iterations=max_iterations,
        time_limit_s=time_limit_s,
        profile_samples=65,
        options=options,
    )


def build_run_specs(
    protocol: Mapping[str, Any], profile: str, *, time_limit_s: float | None = None
) -> list[RunSpec]:
    """Expand the declared full protocol or a clearly labelled reduced profile."""
    if profile not in {"smoke", "representative", "full"}:
        raise ValueError("profile must be smoke, representative, or full.")
    declared_limit = float(protocol["planner_time_limit_s"])
    limit = declared_limit if time_limit_s is None else min(declared_limit, time_limit_s)
    warm_grid = int(protocol["warm_start_reference_grid_size"])

    if profile == "full":
        cases = frozen_cases(protocol)
        baseline_cases = [case for case in cases if case.experiment == "baseline"]
        sweep_cases = [case for case in cases if case.experiment == "contrast_sweep"]
        specs: list[RunSpec] = []
        for case in baseline_cases:
            for grid in protocol["reference_grid_sizes"]:
                specs.append(
                    RunSpec(
                        case,
                        _planner_config(
                            "fast_marching",
                            "fast_marching",
                            interior_points=32,
                            reference_grid_size=int(grid),
                            time_limit_s=limit,
                        ),
                        profile,
                    )
                )
            for interior in protocol["local_interior_points"]:
                for initialization in protocol["initializations"]:
                    for method in protocol["local_methods"]:
                        specs.append(
                            RunSpec(
                                case,
                                _planner_config(
                                    method,
                                    initialization,
                                    interior_points=int(interior),
                                    reference_grid_size=warm_grid,
                                    time_limit_s=limit,
                                ),
                                profile,
                            )
                        )
        # The contrast sweep fixes one middle resolution and all predeclared
        # starts, plus the warm-start reference grid.
        for case in sweep_cases:
            specs.append(
                RunSpec(
                    case,
                    _planner_config(
                        "fast_marching",
                        "fast_marching",
                        interior_points=64,
                        reference_grid_size=warm_grid,
                        time_limit_s=limit,
                    ),
                    profile,
                )
            )
            for initialization in protocol["initializations"]:
                for method in protocol["local_methods"]:
                    specs.append(
                        RunSpec(
                            case,
                            _planner_config(
                                method,
                                initialization,
                                interior_points=64,
                                reference_grid_size=warm_grid,
                                time_limit_s=limit,
                            ),
                            profile,
                        )
                    )
        return [
            RunSpec(spec.case, spec.config, spec.profile, str(protocol["protocol_sha256"]))
            for spec in specs
        ]

    seeds = (0, 1, 2, 3) if profile == "representative" else (0,) * len(protocol["families"])
    cases = [
        _case_spec(family, seed, 1.0, True, experiment="baseline")
        for family, seed in zip(protocol["families"], seeds, strict=True)
    ]
    specs = []
    if profile == "smoke":
        local_points = 8
        grids = (33,)
        initializations = ("straight", "barrier")
        iterations = 40
    else:
        local_points = 32
        grids = (129, 257)
        initializations = ("straight", "barrier", "fast_marching")
        iterations = 120
    for case in cases:
        for grid in grids:
            specs.append(
                RunSpec(
                    case,
                    _planner_config(
                        "fast_marching",
                        "fast_marching",
                        interior_points=local_points,
                        reference_grid_size=grid,
                        time_limit_s=limit,
                        max_iterations=iterations,
                    ),
                    profile,
                )
            )
        for initialization in initializations:
            for method in protocol["local_methods"]:
                specs.append(
                    RunSpec(
                        case,
                        _planner_config(
                            method,
                            initialization,
                            interior_points=local_points,
                            reference_grid_size=(129 if profile == "smoke" else 257),
                            time_limit_s=limit,
                            max_iterations=iterations,
                        ),
                        profile,
                    )
                )
    return [
        RunSpec(spec.case, spec.config, spec.profile, str(protocol["protocol_sha256"]))
        for spec in specs
    ]


def environment_metadata() -> dict[str, Any]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for package in ("scipy", "shapely"):
        try:
            module = __import__(package)
            versions[package] = module.__version__
        except ImportError:
            versions[package] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, "unavailable"
    execution_hash = _execution_source_hash()
    analysis_hash = _analysis_source_hash()
    return {
        "runtime_label": "native",
        "versions": versions,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "thread_limits": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "commit": commit,
        "working_tree_dirty": bool(status),
        "numerical_execution_source_sha256": execution_hash,
        "analysis_source_sha256": analysis_hash,
        "working_tree_source_sha256": execution_hash,
    }


def _repository_paths() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        paths = sorted(set(completed.stdout.splitlines()))
    except (OSError, subprocess.CalledProcessError):
        paths = []
    return paths


def _hash_paths(paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = REPOSITORY_ROOT / relative
        if path.is_file():
            payload = path.read_bytes()
        elif not _REPOSITORY_CHECKOUT and relative.startswith("src/path_planning_ode/"):
            installed_path = PACKAGE_ROOT / relative.removeprefix("src/path_planning_ode/")
            payload = installed_path.read_bytes() if installed_path.is_file() else None
        elif not _REPOSITORY_CHECKOUT and relative == "experiments/protocol.json":
            installed_path = PACKAGE_ROOT / "data" / "protocol.json"
            payload = installed_path.read_bytes() if installed_path.is_file() else None
        else:
            payload = None
        if payload is None:
            if _REPOSITORY_CHECKOUT:
                continue
            # Wheels do not contain the repository lock/build files or CLI shim.
            # Include installed distribution metadata under each absent path so
            # the omission is explicit and deterministic rather than hashing an
            # empty input.
            try:
                from importlib.metadata import distribution

                installed = distribution("path-planning-ode")
                marker = {
                    "name": installed.metadata.get("Name"),
                    "version": installed.version,
                    "requires": sorted(installed.requires or ()),
                    "files": sorted(str(item) for item in (installed.files or ())),
                }
            except Exception:
                marker = {"name": "path-planning-ode", "version": "unavailable"}
            payload = ("installed-distribution:" + _canonical_json(marker)).encode()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _execution_source_hash() -> str:
    numerical_files = {
        "src/path_planning_ode/terrain.py",
        "src/path_planning_ode/terrain_generators.py",
        "src/path_planning_ode/terrain_seeds.py",
        "src/path_planning_ode/fast_marching.py",
        "src/path_planning_ode/local_planners.py",
        "src/path_planning_ode/planners.py",
        "experiments/protocol.json",
        "pyproject.toml",
        "uv.lock",
    }
    digest = hashlib.sha256(_hash_paths(numerical_files).encode())
    for function in (_planner_config, build_run_specs, _planner_process, execute_run):
        digest.update(inspect.getsource(function).encode())
    return digest.hexdigest()


def _analysis_source_hash() -> str:
    analysis_files = {
        "src/path_planning_ode/experiments.py",
        "src/path_planning_ode/study_reporting.py",
        "scripts/run_benchmarks.py",
        "experiments/protocol.json",
    }
    return _hash_paths(analysis_files)


def _planner_process(case_data: dict[str, Any], config_data: dict[str, Any], output: Any) -> None:
    """Child entry point: scenario construction is inside the hard wall budget."""
    started = perf_counter()
    try:
        from .planners import plan

        case = CaseSpec.from_dict(case_data)
        scenario = synthetic_terrain(
            case.family, seed=case.seed, contrast=case.contrast, barriers=case.barriers
        )
        generation_time = perf_counter() - started
        if scenario.scenario_hash != case.scenario_hash:
            raise RuntimeError("Generated scenario hash differs from the frozen case hash.")
        config = PlannerConfig.from_dict(config_data)
        result = plan(scenario, config)
        output.put(
            {
                "record_status": "completed",
                "scenario_generation_time_s": generation_time,
                "child_wall_time_s": perf_counter() - started,
                "result": result.to_dict(),
            }
        )
    except BaseException as exc:
        output.put(
            {
                "record_status": "worker_error",
                "scenario_generation_time_s": None,
                "child_wall_time_s": perf_counter() - started,
                "result": None,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )


def execute_run(spec: RunSpec, environment: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one planner in a disposable subprocess with a hard wall timeout."""
    context = mp.get_context("spawn")
    output = context.Queue(maxsize=1)
    process = context.Process(
        target=_planner_process,
        args=(spec.case.to_dict(), spec.config.to_dict(), output),
        daemon=False,
    )
    started = perf_counter()
    process.start()
    payload: dict[str, Any] | None = None
    deadline = started + spec.config.time_limit_s
    while perf_counter() < deadline:
        try:
            payload = output.get(timeout=min(0.1, max(0.0, deadline - perf_counter())))
            break
        except queue.Empty:
            if not process.is_alive():
                break
    if payload is None and process.is_alive():
        process.terminate()
        process.join(timeout=2)
        payload = {
            "record_status": "timeout",
            "scenario_generation_time_s": None,
            "child_wall_time_s": perf_counter() - started,
            "result": None,
            "error": {
                "type": "HardTimeout",
                "message": "Subprocess exceeded the end-to-end planner budget.",
            },
        }
    else:
        process.join(timeout=2)
    if payload is None:
        payload = {
            "record_status": "worker_error",
            "scenario_generation_time_s": None,
            "child_wall_time_s": perf_counter() - started,
            "result": None,
            "error": {
                "type": "WorkerExit",
                "message": f"Planner subprocess exited with code {process.exitcode}.",
            },
        }
    try:
        output.close()
        output.join_thread()
    except (OSError, ValueError):
        pass
    record = {
        "schema_version": 1,
        "run_id": spec.run_id,
        "profile": spec.profile,
        "case": spec.case.to_dict(),
        "config": spec.config.to_dict(),
        "config_hash": spec.config.config_hash,
        "protocol_sha256": spec.protocol_sha256,
        "protocol_runtime_label": "native",
        "environment": dict(environment),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    return record


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    if source.suffix == ".gz":
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            lines = handle.readlines()
    else:
        lines = source.read_text(encoding="utf-8").splitlines()
    records = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {source}:{line_number}.") from exc
    return records


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _canonical_json(record) + "\n"
    if path.suffix == ".gz":
        with gzip.open(path, "at", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.buffer.fileobj.fileno())
    else:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def run_experiments(
    specs: Sequence[RunSpec],
    output_path: str | Path,
    *,
    resume: bool = False,
    workers: int = 1,
    environment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run and durably append records; existing run IDs are skipped on resume."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1 or workers > 8:
        raise ValueError("workers must be an integer from 1 through 8.")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    path = Path(output_path)
    existing = read_jsonl(path) if resume else []
    if path.exists() and not resume:
        path.unlink()
    metadata = environment_metadata() if environment is None else dict(environment)
    metadata["benchmark_workers"] = workers
    metadata["benchmark_concurrency"] = "thread_supervisor_with_spawned_process_per_run"
    expected_protocols = {spec.protocol_sha256 for spec in specs}
    expected_profile = specs[0].profile if specs else None
    expected_by_id = {spec.run_id: spec for spec in specs}
    if len(expected_protocols) > 1:
        raise ValueError("Run specs must use one protocol digest.")
    for record in existing:
        if expected_profile is None or record.get("profile") != expected_profile:
            raise ValueError("Cannot resume records from a different execution profile.")
        if record.get("protocol_sha256") not in expected_protocols:
            raise ValueError("Cannot resume records from a different protocol digest.")
        requested = expected_by_id.get(record.get("run_id"))
        if requested is None:
            raise ValueError("Cannot resume a run ID outside the requested specification set.")
        if record.get("config_hash") != requested.config.config_hash:
            raise ValueError("Cannot resume a run whose configuration hash changed.")
        previous_source = record.get("environment", {}).get(
            "numerical_execution_source_sha256",
            record.get("environment", {}).get("working_tree_source_sha256"),
        )
        current_source = metadata.get(
            "numerical_execution_source_sha256", metadata.get("working_tree_source_sha256")
        )
        if previous_source != current_source:
            raise ValueError("Cannot resume after the numerical source hash changed.")
    completed_ids = {record["run_id"] for record in existing}
    pending = [spec for spec in specs if spec.run_id not in completed_ids]
    produced: list[dict[str, Any]] = []
    if workers == 1:
        for spec in pending:
            record = execute_run(spec, metadata)
            _append_jsonl(path, record)
            produced.append(record)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute_run, spec, metadata): spec for spec in pending}
            for future in as_completed(futures):
                record = future.result()
                _append_jsonl(path, record)
                produced.append(record)
    return sorted(existing + produced, key=lambda record: record["run_id"])


def _successful_cost(record: Mapping[str, Any]) -> float | None:
    result = record.get("result")
    if (
        record.get("record_status") != "completed"
        or not result
        or not result.get("feasible")
        or result.get("evaluated_cost_s") is None
    ):
        return None
    return float(result["evaluated_cost_s"])


def annotate_reference_differences(
    records: Sequence[dict[str, Any]],
    threshold_percent: float = 0.5,
    expected_grids: Mapping[str, Sequence[int]] | None = None,
) -> list[dict[str, Any]]:
    """Add signed finest-grid differences and empirical refinement flags."""
    references: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        if record["config"]["method"] != "fast_marching":
            continue
        cost = _successful_cost(record)
        if cost is not None:
            references.setdefault(record["case"]["case_id"], []).append(
                (int(record["config"]["reference_grid_size"]), cost)
            )
    annotated = []
    for source in records:
        record = dict(source)
        values = sorted(references.get(record["case"]["case_id"], []))
        expected = (
            sorted(set(expected_grids.get(record["case"]["case_id"], ())))
            if expected_grids is not None
            else sorted({grid for grid, _ in values})
        )
        successful_by_grid = dict(values)
        declared_finest = expected[-1] if expected else None
        if declared_finest in successful_by_grid:
            finest = (declared_finest, successful_by_grid[declared_finest])
            basis = "declared_finest"
        elif values:
            finest = values[-1]
            basis = "fallback_finest_available"
        else:
            finest = None
            basis = "unavailable"
        unresolved = True
        refinement_change = None
        unresolved_reason = "reference_unavailable"
        if len(expected) >= 2 and all(grid in successful_by_grid for grid in expected[-2:]):
            coarse = successful_by_grid[expected[-2]]
            fine = successful_by_grid[expected[-1]]
            refinement_change = 100 * (fine - coarse) / fine
            unresolved = abs(refinement_change) > threshold_percent
            unresolved_reason = "refinement_over_threshold" if unresolved else None
        elif declared_finest not in successful_by_grid:
            unresolved_reason = "missing_declared_finest"
        elif len(expected) < 2:
            unresolved_reason = "insufficient_declared_refinement_levels"
        else:
            unresolved_reason = "missing_finest_pair"
        cost = _successful_cost(record)
        difference = None
        if cost is not None and finest is not None:
            difference = 100 * (cost - finest[1]) / finest[1]
        record["reference_difference_percent"] = difference
        record["reference_unresolved"] = unresolved
        record["reference_refinement_change_percent"] = refinement_change
        record["reference_grid_size"] = None if finest is None else finest[0]
        record["reference_basis"] = basis
        record["comparison_basis"] = basis
        record["reference_unresolved_reason"] = unresolved_reason
        record["declared_finest_reference_grid_size"] = declared_finest
        annotated.append(record)
    return annotated


def _bootstrap_mean_interval(
    values: Sequence[tuple[str, float]], *, samples: int, seed: int
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    clusters: dict[str, list[float]] = {}
    for cluster, value in values:
        clusters.setdefault(cluster, []).append(value)
    names = sorted(clusters)
    if len(names) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for index in range(samples):
        selected = rng.choice(names, size=len(names), replace=True)
        estimates[index] = np.mean([value for name in selected for value in clusters[name]])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize_records(
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260915,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        config = record["config"]
        key = (
            record["case"]["experiment"],
            record["case"]["family"],
            (
                record["case"]["contrast"]
                if record["case"]["experiment"] == "contrast_sweep"
                else None
            ),
            config["method"],
            config["initialization"],
            config["interior_points"] if config["method"] != "fast_marching" else None,
            config["reference_grid_size"] if config["method"] == "fast_marching" else None,
        )
        groups.setdefault(key, []).append(record)
    rows = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        costs = [
            record["reference_difference_percent"]
            for record in group
            if not record["reference_unresolved"]
        ]
        costs = [value for value in costs if value is not None]
        runtimes = [
            float(record["child_wall_time_s"])
            for record in group
            if record.get("child_wall_time_s") is not None
        ]
        feasible = [record for record in group if _successful_cost(record) is not None]
        evaluated_costs = [_successful_cost(record) for record in group]
        evaluated_costs = [value for value in evaluated_costs if value is not None]
        clearances = [
            record["result"]["evaluation"]["minimum_clearance_m"]
            for record in feasible
            if record["result"]["evaluation"]["minimum_clearance_m"] is not None
        ]
        stationarity = [
            record["result"]["diagnostics"].get("stationarity_norm")
            for record in group
            if record.get("result")
            and record["result"]["diagnostics"].get("stationarity_norm") is not None
        ]
        terminated = [
            record
            for record in group
            if record.get("result") and record["result"].get("solver_success")
        ]
        failures: dict[str, int] = {}
        termination_modes: dict[str, int] = {}
        for record in group:
            if record["record_status"] != "completed":
                reason = record["record_status"]
            elif record.get("result"):
                reason = record["result"]["termination_reason"]
            else:
                reason = "missing_result"
            termination_modes[reason] = termination_modes.get(reason, 0) + 1
            if _successful_cost(record) is None or not (
                record.get("result") and record["result"].get("solver_success")
            ):
                failures[reason] = failures.get(reason, 0) + 1
        clustered = [
            (f"{record['case']['family']}:{record['case']['seed']}", value)
            for record in group
            if not record["reference_unresolved"]
            and (value := record["reference_difference_percent"]) is not None
        ]
        low, high = _bootstrap_mean_interval(
            clustered, samples=bootstrap_samples, seed=bootstrap_seed
        )
        rows.append(
            {
                "experiment": key[0],
                "family": key[1],
                "contrast": key[2],
                "method": key[3],
                "initialization": key[4],
                "interior_points": key[5],
                "reference_grid_size": key[6],
                "runs": len(group),
                "feasible_runs": len(feasible),
                "feasibility_rate": len(feasible) / len(group),
                "solver_success_rate": len(terminated) / len(group),
                "median_evaluated_cost_s": (
                    None if not evaluated_costs else float(np.median(evaluated_costs))
                ),
                "median_clearance_m": None if not clearances else float(np.median(clearances)),
                "median_stationarity_norm": (
                    None if not stationarity else float(np.median(stationarity))
                ),
                "median_actual_interior_points": (
                    None
                    if key[3] == "fast_marching"
                    else float(
                        np.median(
                            [
                                record["result"]["diagnostics"].get(
                                    "actual_interior_points", record["config"]["interior_points"]
                                )
                                for record in group
                                if record.get("result")
                            ]
                            or [key[5]]
                        )
                    )
                ),
                "mean_reference_difference_percent": None if not costs else float(np.mean(costs)),
                "median_reference_difference_percent": None
                if not costs
                else float(np.median(costs)),
                "bootstrap_95_low_percent": low,
                "bootstrap_95_high_percent": high,
                "median_runtime_s": None if not runtimes else float(np.median(runtimes)),
                "reference_unresolved_runs": sum(bool(r["reference_unresolved"]) for r in group),
                "failure_modes": "; ".join(
                    f"{name}:{count}" for name, count in sorted(failures.items())
                ),
                "termination_modes": "; ".join(
                    f"{name}:{count}" for name, count in sorted(termination_modes.items())
                ),
            }
        )
    return rows


def initialization_sensitivity_summary(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize within-case cost ranges across identical geometric seed banks."""
    groups: dict[tuple[str, str, float | None, str, int], list[float]] = {}
    for record in records:
        method = record["config"]["method"]
        if method not in {"euler_lagrange", "slsqp"}:
            continue
        cost = _successful_cost(record)
        if cost is None:
            continue
        experiment = record["case"]["experiment"]
        contrast = record["case"]["contrast"] if experiment == "contrast_sweep" else None
        key = (
            record["case"]["case_id"],
            experiment,
            record["case"]["family"],
            contrast,
            method,
            int(record["config"]["interior_points"]),
        )
        groups.setdefault(key, []).append(cost)
    spreads: dict[tuple[str, str, float | None, str, int], list[float]] = {}
    for (_, experiment, family, contrast, method, interior), costs in groups.items():
        if len(costs) >= 2:
            spreads.setdefault((experiment, family, contrast, method, interior), []).append(
                100 * (max(costs) - min(costs)) / min(costs)
            )
    return [
        {
            "experiment": experiment,
            "family": family,
            "contrast": contrast,
            "method": method,
            "interior_points": interior,
            "cases_with_multiple_feasible_initializations": len(values),
            "median_within_case_cost_range_percent": float(np.median(values)),
            "maximum_within_case_cost_range_percent": float(np.max(values)),
        }
        for (experiment, family, contrast, method, interior), values in sorted(spreads.items())
    ]


def local_refinement_summary(
    records: Sequence[dict[str, Any]], declared_levels: Sequence[int] = (32, 64, 128)
) -> list[dict[str, Any]]:
    """Compare only the two finest declared local discretizations."""
    levels = sorted(set(int(level) for level in declared_levels))
    if len(levels) < 2:
        raise ValueError("At least two declared local refinement levels are required.")
    required = tuple(levels[-2:])
    by_case: dict[tuple[Any, ...], dict[int, float | None]] = {}
    for record in records:
        config = record["config"]
        if config["method"] not in {"euler_lagrange", "slsqp"}:
            continue
        key = (
            record["case"]["case_id"],
            record["case"]["experiment"],
            record["case"]["family"],
            record["case"]["contrast"],
            config["method"],
            config["initialization"],
        )
        by_case.setdefault(key, {})[int(config["interior_points"])] = _successful_cost(record)
    changes: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, values in by_case.items():
        summary = changes.setdefault(key[1:], {"cases": 0, "resolved": []})
        summary["cases"] += 1
        if all(values.get(level) is not None for level in required):
            coarse_cost = values[required[0]]
            fine_cost = values[required[1]]
            summary["resolved"].append(100 * (fine_cost - coarse_cost) / fine_cost)
    rows = []
    for key, summary in sorted(changes.items()):
        values = summary["resolved"]
        rows.append(
            {
                "experiment": key[0],
                "family": key[1],
                "contrast": key[2] if key[0] == "contrast_sweep" else None,
                "method": key[3],
                "initialization": key[4],
                "declared_coarse_interior_points": required[0],
                "declared_fine_interior_points": required[1],
                "cases": summary["cases"],
                "cases_with_two_feasible_finest_levels": len(values),
                "unresolved_cases": summary["cases"] - len(values),
                "median_signed_change_percent": (None if not values else float(np.median(values))),
                "median_absolute_change_percent": (
                    None if not values else float(np.median(np.abs(values)))
                ),
                "over_0p5_percent": int(np.count_nonzero(np.abs(values) > 0.5)),
            }
        )
    return rows


def residual_cost_examples(
    records: Sequence[dict[str, Any]], limit: int = 8
) -> list[dict[str, Any]]:
    """Select feasible EL candidates with the largest signed reference differences."""
    examples = []
    for record in records:
        result = record.get("result") or {}
        if record["config"]["method"] != "euler_lagrange" or not result.get("feasible"):
            continue
        residual = result.get("diagnostics", {}).get("stationarity_norm")
        difference = record.get("reference_difference_percent")
        if residual is None or difference is None or record.get("reference_unresolved", True):
            continue
        examples.append(
            {
                "case_id": record["case"]["case_id"],
                "initialization": record["config"]["initialization"],
                "interior_points": record["config"]["interior_points"],
                "stationarity_norm": residual,
                "reference_difference_percent": difference,
                "solver_success": result["solver_success"],
            }
        )
    return sorted(examples, key=lambda row: abs(row["reference_difference_percent"]), reverse=True)[
        :limit
    ]


def paired_method_summary(
    records: Sequence[dict[str, Any]], *, bootstrap_samples: int, bootstrap_seed: int
) -> list[dict[str, Any]]:
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for record in records:
        config = record["config"]
        if config["method"] not in {"euler_lagrange", "slsqp"}:
            continue
        key = (
            record["case"]["case_id"],
            config["initialization"],
            config["interior_points"],
        )
        pairs.setdefault(key, {})[config["method"]] = record
    grouped: dict[tuple[str, str, float | None, str, int], dict[str, Any]] = {}
    for key, methods in pairs.items():
        if set(methods) != {"euler_lagrange", "slsqp"}:
            continue
        case = methods["slsqp"]["case"]
        experiment = case["experiment"]
        contrast = case["contrast"] if experiment == "contrast_sweep" else None
        group_key = (experiment, case["family"], contrast, key[1], key[2])
        group = grouped.setdefault(group_key, {"pairs": 0, "matched": 0, "values": []})
        group["pairs"] += 1
        euler_diagnostics = (methods["euler_lagrange"].get("result") or {}).get("diagnostics", {})
        slsqp_diagnostics = (methods["slsqp"].get("result") or {}).get("diagnostics", {})
        euler_hash = euler_diagnostics.get("initialization", {}).get("initial_route_hash")
        slsqp_hash = slsqp_diagnostics.get("initialization", {}).get("initial_route_hash")
        if euler_hash is None or euler_hash != slsqp_hash:
            continue
        group["matched"] += 1
        euler_cost = _successful_cost(methods["euler_lagrange"])
        slsqp_cost = _successful_cost(methods["slsqp"])
        if euler_cost is None or slsqp_cost is None:
            continue
        cluster = f"{case['family']}:{case['seed']}"
        difference = 100 * (slsqp_cost - euler_cost) / euler_cost
        group["values"].append((cluster, difference))
    rows = []
    for group_key, group in sorted(grouped.items()):
        experiment, family, contrast, initialization, interior = group_key
        values = group["values"]
        low, high = _bootstrap_mean_interval(values, samples=bootstrap_samples, seed=bootstrap_seed)
        rows.append(
            {
                "experiment": experiment,
                "family": family,
                "contrast": contrast,
                "initialization": initialization,
                "interior_points": interior,
                "paired_cases": group["pairs"],
                "matched_seed_pairs": group["matched"],
                "paired_feasible_cases": len(values),
                "paired_feasible_rate": (
                    len(values) / group["matched"] if group["matched"] else 0.0
                ),
                "mean_slsqp_minus_euler_percent": (
                    None if not values else float(np.mean([value for _, value in values]))
                ),
                "bootstrap_95_low_percent": low,
                "bootstrap_95_high_percent": high,
            }
        )
    return rows


def paired_seed_outcomes(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose every local-method pair, including survivorship and hash mismatches."""
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for record in records:
        config = record["config"]
        if config["method"] in {"euler_lagrange", "slsqp"}:
            key = (
                record["case"]["case_id"],
                config["initialization"],
                config["interior_points"],
            )
            pairs.setdefault(key, {})[config["method"]] = record
    outcomes = []
    for (case_id, initialization, interior), methods in sorted(pairs.items()):
        if set(methods) != {"euler_lagrange", "slsqp"}:
            continue
        hashes, costs, feasible = {}, {}, {}
        for method, record in methods.items():
            result = record.get("result") or {}
            hashes[method] = (
                result.get("diagnostics", {}).get("initialization", {}).get("initial_route_hash")
            )
            costs[method] = _successful_cost(record)
            feasible[method] = costs[method] is not None
        matched = hashes["euler_lagrange"] is not None and len(set(hashes.values())) == 1
        difference = None
        if matched and all(feasible.values()):
            difference = 100 * (costs["slsqp"] - costs["euler_lagrange"]) / costs["euler_lagrange"]
        outcomes.append(
            {
                "case_id": case_id,
                "initialization": initialization,
                "interior_points": interior,
                "seed_hash_match": matched,
                "euler_feasible": feasible["euler_lagrange"],
                "slsqp_feasible": feasible["slsqp"],
                "slsqp_minus_euler_percent": difference,
            }
        )
    return outcomes


def write_summary_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dashboard_result(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    evaluation = result.get("evaluation")
    evaluation_summary = None
    if evaluation is not None:
        evaluation_summary = {
            key: evaluation.get(key)
            for key in (
                "cost_s",
                "length_m",
                "feasible",
                "minimum_clearance_m",
                "violations",
                "evaluation_time_s",
            )
        }
    return {
        key: result.get(key)
        for key in (
            "version",
            "method",
            "initialization",
            "evaluated_cost_s",
            "feasible",
            "solver_success",
            "termination_reason",
            "diagnostics",
            "timing_s",
            "scenario_hash",
            "config_hash",
        )
    } | {
        "route_point_count": len(result.get("route_m") or ()),
        "evaluation": evaluation_summary,
    }


def build_published_index(
    records: Sequence[dict[str, Any]],
    protocol: Mapping[str, Any],
    *,
    profile: str,
    bootstrap_samples: int | None = None,
    expected_specs: Sequence[RunSpec] | None = None,
) -> dict[str, Any]:
    threshold = float(protocol["reference_refinement_unresolved_percent"])
    declared_specs = build_run_specs(protocol, profile)
    execution_specs = list(declared_specs if expected_specs is None else expected_specs)
    execution_ids = {spec.run_id for spec in execution_specs}
    declared_ids = {spec.run_id for spec in declared_specs}
    expected_ids = declared_ids if profile == "full" else execution_ids
    recorded_ids = {record["run_id"] for record in records}
    expected_grids: dict[str, set[int]] = {}
    for spec in declared_specs:
        if spec.config.method == "fast_marching":
            expected_grids.setdefault(spec.case.case_id, set()).add(spec.config.reference_grid_size)
    annotated = annotate_reference_differences(records, threshold, expected_grids)
    bootstrap = protocol["bootstrap"]
    samples = int(bootstrap["samples"] if bootstrap_samples is None else bootstrap_samples)
    aggregate = summarize_records(
        annotated,
        bootstrap_samples=samples,
        bootstrap_seed=int(bootstrap["seed"]),
    )
    paired = paired_method_summary(
        annotated,
        bootstrap_samples=samples,
        bootstrap_seed=int(bootstrap["seed"]),
    )
    sensitivity = initialization_sensitivity_summary(annotated)
    refinement = local_refinement_summary(annotated, protocol["local_interior_points"])
    paired_outcomes = paired_seed_outcomes(annotated)
    residual_examples = residual_cost_examples(annotated)
    runs = [
        {
            "run_id": record["run_id"],
            "record_status": record["record_status"],
            "case_id": record["case"]["case_id"],
            "family": record["case"]["family"],
            "seed": record["case"]["seed"],
            "difficulty": record["case"]["difficulty"],
            "scenario_ref": record["case"]["scenario_ref"],
            "scenario_hash": record["case"]["scenario_hash"],
            "scenario_generation_time_s": record.get("scenario_generation_time_s"),
            "child_wall_time_s": record.get("child_wall_time_s"),
            "config": record["config"],
            "result": _dashboard_result(record.get("result")),
            "result_ref": (
                None if record.get("result") is None else f"runs/{record['run_id']}.json.gz"
            ),
            "error": record.get("error"),
            "reference_difference_percent": record["reference_difference_percent"],
            "reference_unresolved": record["reference_unresolved"],
            "reference_basis": record["reference_basis"],
            "comparison_basis": record["comparison_basis"],
            "reference_unresolved_reason": record["reference_unresolved_reason"],
            "reference_grid_size": record["reference_grid_size"],
            "declared_finest_reference_grid_size": record["declared_finest_reference_grid_size"],
            "reference_refinement_change_percent": record["reference_refinement_change_percent"],
        }
        for record in annotated
    ]
    environments = {
        record["environment"]["working_tree_source_sha256"]: record["environment"]
        for record in records
    }
    missing = expected_ids - recorded_ids
    complete = not missing and len(recorded_ids) == len(expected_ids)
    protocol_configuration_match = execution_ids == declared_ids
    full_protocol = profile == "full" and complete and protocol_configuration_match
    return {
        "title": protocol["title"],
        "protocol": dict(protocol),
        "runtime_label": "native",
        "scope": {
            "profile": profile,
            "run_count": len(records),
            "expected_run_count": len(expected_ids),
            "missing_run_count": len(missing),
            "coverage_fraction": len(recorded_ids & expected_ids) / len(expected_ids),
            "completed_count": sum(r["record_status"] == "completed" for r in records),
            "profile_complete": complete,
            "protocol_configuration_match": protocol_configuration_match,
            "full_protocol": full_protocol,
            "statement": (
                "Complete frozen study matrix."
                if full_protocol
                else (
                    f"Incomplete full profile ({len(recorded_ids & declared_ids)}/"
                    f"{len(declared_ids)} declared runs); it does not support full-study claims."
                )
                if profile == "full"
                else (
                    f"Incomplete {profile} profile ({len(recorded_ids & expected_ids)}/"
                    f"{len(expected_ids)} declared runs); it does not support full-study claims."
                    if not complete
                    else (
                        f"Complete reduced {profile} profile; "
                        "it does not support full-study claims."
                    )
                )
            ),
        },
        "aggregate": aggregate,
        "paired": paired,
        "initialization_sensitivity": sensitivity,
        "local_refinement": refinement,
        "paired_outcomes": paired_outcomes,
        "residual_cost_examples": residual_examples,
        "runs": runs,
        "environments": list(environments.values()),
    }


def generate_figures(index: Mapping[str, Any], output_dir: str | Path) -> list[str]:
    """Generate compact descriptive figures when Matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = index["aggregate"]
    labels = [
        f"{row['method']}\n{row['initialization']}"
        for row in rows
        if row["experiment"] == "baseline"
    ]
    feasibility = [row["feasibility_rate"] for row in rows if row["experiment"] == "baseline"]
    if not labels:
        return []
    width = max(7, min(16, 0.55 * len(labels)))
    figure, axis = plt.subplots(figsize=(width, 4.2))
    axis.bar(np.arange(len(labels)), feasibility, color="#2f6f73")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Feasible-route fraction")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axis.set_title("Independent route feasibility by configuration")
    figure.tight_layout()
    path = destination / "feasibility.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return [path.name]


def generate_report(index: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scope = index["scope"]
    runs = index["runs"]
    completed = sum(run["record_status"] == "completed" for run in runs)
    feasible = sum(run.get("result") is not None and run["result"].get("feasible") for run in runs)
    failures: dict[str, int] = {}
    for run in runs:
        if run["record_status"] != "completed":
            reason = run["record_status"]
        elif run.get("result"):
            reason = run["result"]["termination_reason"]
        else:
            reason = "missing_result"
        if not (
            run.get("result")
            and run["result"].get("feasible")
            and run["result"].get("solver_success")
        ):
            failures[reason] = failures.get(reason, 0) + 1
    failure_text = (
        ", ".join(f"{name}: {count}" for name, count in sorted(failures.items())) or "none"
    )
    lines = [
        f"# {index['title']}",
        "",
        "## Published scope",
        "",
        scope["statement"],
        (
            f"This artifact contains {len(runs)} attempted runs, {completed} completed "
            f"subprocess records, and {feasible} independently feasible routes."
        ),
        "",
        "## Interpretation boundary",
        "",
        index["protocol"]["model_boundary"],
        (
            "Fast marching is reported as a grid-based reference. Signed differences from its "
            "finest available grid may be negative and are not called optimality gaps. A reference "
            "is unresolved when its two finest declared costs differ by more than 0.5%, or when "
            "either of those levels is unavailable."
        ),
        "",
        "## Unsuccessful runs",
        "",
        f"Observed failure or infeasibility reasons: {failure_text}.",
        "No infeasible route enters cost rankings or paired cost summaries.",
        "",
        "## Evidence for the hypotheses",
        "",
    ]
    if scope["full_protocol"]:
        lines.extend(
            [
                (
                    "The published aggregate and paired tables quantify initialization "
                    "sensitivity, feasibility, termination, runtime, and signed reference "
                    "differences over the frozen 80-case matrix. Bootstrap intervals resample "
                    "family-and-seed clusters, so resolution and initialization variants are "
                    "not treated as independent terrain cases."
                ),
                (
                    "Claims should use configurations whose reference refinement is resolved "
                    "and retain the unsuccessful runs listed above."
                ),
            ]
        )
    else:
        lines.extend(
            [
                (
                    "This reduced profile checks the complete data path and exposes individual "
                    "successes and failures. Its case and resolution count is too small to accept "
                    "or reject the study hypotheses."
                ),
                (
                    "Run `run-terrain-benchmarks --profile full --resume` before drawing "
                    "study-level conclusions."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run run-terrain-benchmarks --profile full --resume --workers 4",
            "```",
            "",
            f"Protocol SHA-256: `{index['protocol']['protocol_sha256']}`.",
            (
                "Each JSONL run stores scenario and configuration hashes, the seed route hash "
                "when available, dependency and hardware metadata, commit identity, numerical "
                "source hash, timings, diagnostics, and independent route evaluation."
            ),
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish_results(
    records: Sequence[dict[str, Any]],
    protocol: Mapping[str, Any],
    output_dir: str | Path,
    *,
    profile: str,
    bootstrap_samples: int | None = None,
    figures: bool = True,
    report_path: str | Path | None = None,
    expected_specs: Sequence[RunSpec] | None = None,
) -> dict[str, Any]:
    from .study_reporting import (
        capture_numerical_build_metadata,
        generate_study_figures,
        write_study_report,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    details = destination / "runs"
    details.mkdir(parents=True, exist_ok=True)
    for record in records:
        if record.get("result") is None:
            continue
        detail_path = details / f"{record['run_id']}.json.gz"
        payload = json.dumps(record["result"], separators=(",", ":"), allow_nan=False).encode()
        compressed = gzip.compress(payload, compresslevel=6, mtime=0)
        if not detail_path.is_file() or detail_path.read_bytes() != compressed:
            detail_path.write_bytes(compressed)
    index = build_published_index(
        records,
        protocol,
        profile=profile,
        bootstrap_samples=bootstrap_samples,
        expected_specs=expected_specs,
    )
    raw_archive = destination / "runs.jsonl.gz"
    index["publication"] = {
        "analysis_source_sha256": _analysis_source_hash(),
        "generated_from_latest_record_utc": max(
            (record.get("recorded_at_utc", "") for record in records), default=""
        ),
        "raw_archive_sha256": (
            hashlib.sha256(raw_archive.read_bytes()).hexdigest() if raw_archive.is_file() else None
        ),
        "raw_archive_size_bytes": raw_archive.stat().st_size if raw_archive.is_file() else None,
        "numerical_build": capture_numerical_build_metadata(),
    }
    index_path = destination / "index.json"
    temporary_index = destination / ".index.json.tmp"
    temporary_index.write_text(
        json.dumps(index, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_index.replace(index_path)
    write_summary_csv(index["aggregate"], destination / "summary.csv")
    write_summary_csv(index["paired"], destination / "paired.csv")
    write_summary_csv(
        index["initialization_sensitivity"], destination / "initialization-sensitivity.csv"
    )
    write_summary_csv(index["local_refinement"], destination / "local-refinement.csv")
    write_summary_csv(index["paired_outcomes"], destination / "paired-outcomes.csv")
    if figures:
        generate_study_figures(index, destination)
    output_report = destination / "study-report.md"
    documentation_url = "https://github.com/twallengren/path-planning-ode/blob/master/docs"
    write_study_report(
        index,
        output_report,
        artifact_prefix="",
        documentation_prefix=documentation_url,
    )
    extra_report = Path(report_path) if report_path is not None else None
    if (
        extra_report is None
        and _REPOSITORY_CHECKOUT
        and profile == "full"
        and destination.resolve() == DEFAULT_PUBLISHED_DIR.resolve()
    ):
        extra_report = REPOSITORY_ROOT / "docs" / "study-report.md"
    if extra_report is not None and extra_report.resolve() != output_report.resolve():
        relative_artifacts = os.path.relpath(destination.resolve(), extra_report.parent.resolve())
        write_study_report(
            index,
            extra_report,
            artifact_prefix="" if relative_artifacts == "." else relative_artifacts,
            documentation_prefix=(
                ""
                if extra_report.parent.resolve() == (REPOSITORY_ROOT / "docs").resolve()
                else documentation_url
            ),
        )
    return index


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen terrain-routing benchmark protocol."
    )
    parser.add_argument("--profile", choices=("smoke", "representative", "full"), default="smoke")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="result directory (default: ./terrain-study-results/<profile>)",
    )
    parser.add_argument("--report-path", type=Path, help="optional additional report path")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--time-limit", type=float, help="cap the declared 60-second per-run budget"
    )
    parser.add_argument("--limit", type=int, help="run only the first N deterministic run specs")
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--records-only",
        action="store_true",
        help="write resumable JSONL records without generating summaries",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be from 1 through 8")
    output_dir = args.output_dir or Path.cwd() / "terrain-study-results" / args.profile
    protocol = load_protocol(args.protocol)
    specs = build_run_specs(protocol, args.profile, time_limit_s=args.time_limit)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        specs = specs[: args.limit]
    records = run_experiments(
        specs,
        output_dir / "runs.jsonl.gz",
        resume=args.resume,
        workers=args.workers,
    )
    if args.records_only:
        print(f"Recorded {len(records)} {args.profile} runs in {output_dir.resolve()}")
        return 0
    index = publish_results(
        records,
        protocol,
        output_dir,
        profile=args.profile,
        bootstrap_samples=args.bootstrap_samples,
        figures=not args.no_figures,
        report_path=args.report_path,
        expected_specs=specs,
    )
    print(f"Published {index['scope']['run_count']} {args.profile} runs to {output_dir.resolve()}")
    return 0


__all__ = [
    "CaseSpec",
    "RunSpec",
    "annotate_reference_differences",
    "build_published_index",
    "build_run_specs",
    "environment_metadata",
    "execute_run",
    "frozen_cases",
    "generate_report",
    "initialization_sensitivity_summary",
    "local_refinement_summary",
    "load_protocol",
    "paired_method_summary",
    "paired_seed_outcomes",
    "publish_results",
    "read_jsonl",
    "residual_cost_examples",
    "run_experiments",
    "summarize_records",
    "write_summary_csv",
    "cli_main",
]
