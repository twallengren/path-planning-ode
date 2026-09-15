import hashlib
import json
from importlib.resources import files

import numpy as np
import pytest

from path_planning_ode.terrain import TerrainField, TerrainScenario
from path_planning_ode.terrain_generators import (
    BENCHMARK_SEEDS,
    FAMILY_NAMES,
    SOURCE_RESOLUTION,
    VALIDATION_NAMES,
    mount_tamalpais_terrain,
    synthetic_terrain,
    validation_terrain,
)


@pytest.mark.parametrize("family", FAMILY_NAMES)
@pytest.mark.parametrize("seed", BENCHMARK_SEEDS)
def test_benchmark_matrix_is_deterministic_and_has_fixed_source_grid(family, seed):
    first = synthetic_terrain(family, seed=seed)
    second = synthetic_terrain(family, seed=seed)
    assert first.scenario_hash == second.scenario_hash
    assert len(first.field_x_m) == len(first.field_y_m) == SOURCE_RESOLUTION
    assert np.shape(first.log_slowness) == (SOURCE_RESOLUTION, SOURCE_RESOLUTION)
    assert first.metadata["difficulty"] == ("easy", "medium", "hard")[seed % 3]
    assert first.metadata["cost_units"] == "seconds per horizontal metre"
    assert np.isfinite(first.log_slowness).all()


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_family_controls_are_explicit(family):
    without_contrast = synthetic_terrain(family, seed=4, contrast=0, barriers=False)
    with_contrast = synthetic_terrain(family, seed=4, contrast=1.7, barriers=True)
    np.testing.assert_allclose(without_contrast.log_slowness, np.log(0.8))
    assert not np.allclose(without_contrast.log_slowness, with_contrast.log_slowness)
    assert not without_contrast.barriers_geojson
    assert with_contrast.barriers_geojson


def test_different_seeds_change_each_family():
    for family in FAMILY_NAMES:
        first = synthetic_terrain(family, seed=0, barriers=False)
        second = synthetic_terrain(family, seed=1, barriers=False)
        assert first.scenario_hash != second.scenario_hash


@pytest.mark.parametrize("name", VALIDATION_NAMES)
def test_validation_fixtures_round_trip(name):
    fixture = validation_terrain(name)
    restored = TerrainScenario.from_dict(fixture.to_dict())
    assert restored.scenario_hash == fixture.scenario_hash
    assert fixture.metadata["kind"] == "analytic_validation"


def test_uniform_fixture_field_is_exactly_constant():
    field = TerrainField(validation_terrain("uniform"))
    points = np.array([[0, 0], [123, 456], [500, 500], [1_000, 1_000]], dtype=float)
    np.testing.assert_allclose(field.cost(points), 0.8, atol=1e-14)
    np.testing.assert_allclose(field.gradient(points), 0, atol=1e-14)
    np.testing.assert_allclose(field.hessian(points), 0, atol=1e-14)


def test_hard_barrier_fixtures_have_expected_connectivity_intent():
    detour = validation_terrain("obstacle_detour")
    disconnected = validation_terrain("disconnected")
    assert detour.barriers_geojson[0]["coordinates"][0][0] == [440, 350]
    ring = disconnected.barriers_geojson[0]["coordinates"][0]
    assert min(point[1] for point in ring) == disconnected.bounds_m[1]
    assert max(point[1] for point in ring) == disconnected.bounds_m[3]


def test_mount_tamalpais_is_offline_hash_pinned_observed_elevation(monkeypatch):
    def network_forbidden(*args, **kwargs):
        raise AssertionError("runtime terrain loading attempted network access")

    monkeypatch.setattr("urllib.request.urlopen", network_forbidden)
    scenario = mount_tamalpais_terrain()
    data_path = files("path_planning_ode.data").joinpath("mount_tamalpais.json")
    manifest_path = files("path_planning_ode.data").joinpath("mount_tamalpais.provenance.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == manifest["processed_sha256"]
    assert [source["sha256"] for source in scenario.provenance["raw_sources"]] == [
        "f4c9f7170e8f247f096febb316ca405c959183538e4dc3582fd145c87b9dc4db",
        "414a9017e59684c4dd686c529d7b1574293a40957d85f014a534a19cfe9c8d9f",
    ]
    assert scenario.provenance["source_crs"] == "EPSG:3857 (Web Mercator metres)"
    assert "illustrative" in scenario.metadata["model_assumptions"].lower()
    assert np.ptp(np.asarray(scenario.elevation_m)) > 400
    assert not scenario.barriers_geojson


def test_mount_tamalpais_model_controls_do_not_change_observed_elevation():
    plain = mount_tamalpais_terrain(contrast=0, barriers=False)
    modelled = mount_tamalpais_terrain(contrast=2, barriers=True)
    assert plain.elevation_m == modelled.elevation_m
    np.testing.assert_allclose(plain.log_slowness, np.log(0.8))
    assert modelled.barriers_geojson
    assert not np.allclose(plain.log_slowness, modelled.log_slowness)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": 1.5},
        {"contrast": -1},
        {"contrast": float("nan")},
        {"barriers": 1},
    ],
)
def test_invalid_synthetic_controls(kwargs):
    with pytest.raises(ValueError):
        synthetic_terrain("ridge_pass", **kwargs)


def test_unknown_names_fail_clearly():
    with pytest.raises(ValueError, match="unknown family"):
        synthetic_terrain("volcano")
    with pytest.raises(ValueError, match="unknown validation fixture"):
        validation_terrain("triangle")
