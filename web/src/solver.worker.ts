/// <reference lib="webworker" />
import type { PyodideAPI } from 'pyodide';

const context = self as unknown as DedicatedWorkerGlobalScope;
let runtime: PyodideAPI;
let baseUrl = '';
let playgroundReady = false;
let terrainReady = false;
let residentRevision = -1;

const GAUSSIAN_FILES = [
  '__init__.py',
  'core.py',
  'presets.py',
  'field_adapter.py',
  'playground.py',
];
const TERRAIN_FILES = [
  'terrain.py',
  'soft_walls.py',
  'terrain_generators.py',
  'data/__init__.py',
  'data/mount_tamalpais.json',
  'data/mount_tamalpais.provenance.json',
];

async function installFiles(files: string[]) {
  runtime.FS.mkdirTree('/home/pyodide/path_planning_ode');
  for (const name of files) {
    const response = await fetch(new URL(`python/path_planning_ode/${name}`, baseUrl));
    if (!response.ok) throw new Error(`Could not load solver module ${name}.`);
    const parts = name.split('/');
    if (parts.length > 1)
      runtime.FS.mkdirTree(`/home/pyodide/path_planning_ode/${parts.slice(0, -1).join('/')}`);
    runtime.FS.writeFile(
      `/home/pyodide/path_planning_ode/${name}`,
      new Uint8Array(await response.arrayBuffer()),
    );
  }
}

async function ensureTerrain() {
  if (terrainReady) return;
  context.postMessage({ progress: 'Loading terrain tools…' });
  try {
    await runtime.loadPackage(['scipy', 'shapely']);
    await installFiles(TERRAIN_FILES);
    runtime.runPython(`
from path_planning_ode.terrain import TerrainScenario
from path_planning_ode.field_adapter import terrain_to_playground_field
`);
    terrainReady = true;
    context.postMessage({ progress: 'Ready · runs on this device' });
  } catch (error) {
    context.postMessage({
      progress: 'Ready for Gaussian scenes · terrain tools unavailable',
    });
    throw error;
  }
}

function setJson(name: string, value: unknown) {
  runtime.globals.set(name, JSON.stringify(value));
}

function pythonJson(source: string) {
  return JSON.parse(runtime.runPython(source));
}

context.onmessage = async ({ data }: MessageEvent<Record<string, any>>) => {
  const id = data.id as number,
    action = String(data.action);
  try {
    if (action === 'playgroundBoot') {
      baseUrl = String(data.base);
      context.postMessage({ progress: 'Loading solver…' });
      const originalFetch = context.fetch.bind(context);
      context.fetch = async (...args: Parameters<typeof fetch>) => {
        try {
          const response = await originalFetch(...args);
          if (!response.ok) throw new Error(`Download failed (HTTP ${response.status}).`);
          return response;
        } catch (error) {
          context.postMessage({ id, error: `Runtime download failed. ${String(error)}` });
          throw error;
        }
      };
      const { loadPyodide } = await import(
        /* @vite-ignore */ new URL('runtime/pyodide.mjs', baseUrl).href
      );
      runtime = await loadPyodide({ indexURL: new URL('runtime/', baseUrl).href });
      context.postMessage({ progress: 'Loading NumPy…' });
      await runtime.loadPackage('numpy');
      await installFiles(GAUSSIAN_FILES);
      runtime.runPython(`
import json, numpy as np
from dataclasses import asdict
from path_planning_ode.core import Obstacle
from path_planning_ode.field_adapter import PlaygroundField, build_playground_preset
from path_planning_ode.playground import (
    MAX_GAUSSIANS, PlaygroundOptions, advance_playground, build_playground_field,
    evaluate_playground, initialize_playground,
)

def playground_field_from_scene(scene):
    explicit = tuple(Obstacle(**item) for item in scene.get('gaussians', ()))
    painted = build_playground_field(scene.get('strokes', ()))['obstacles']
    combined = explicit + tuple(Obstacle(**item) for item in painted)
    if len(combined) > MAX_GAUSSIANS:
        raise ValueError(f'Field requires {len(combined)} Gaussian bumps; maximum is {MAX_GAUSSIANS}.')
    bounds = scene['bounds']
    terrain_bounds = [bounds[0], bounds[2], bounds[1], bounds[3]]
    return PlaygroundField.from_spec(scene['base_field'], combined, bounds=terrain_bounds), explicit, combined

def playground_options(settings, interior_points=None):
    return PlaygroundOptions(
        interior_points=settings.get('interior_points', interior_points),
        method='auto',
        tolerance=settings.get('tolerance', 1e-5),
        max_iterations=settings.get('max_iterations', 2000),
    )

def playground_payload(state, revision):
    item = state.to_dict()
    metrics = dict(item['metrics'])
    metrics.update(
        iteration=item['iteration'], status=item['status'], elapsed_seconds=item['elapsed_seconds'],
        phase=item['phase'], phase_reason=item['phase_reason'],
    )
    return {
        'revision': revision, 'path': item['path'], 'metrics': metrics,
        'state': {
            'resident': True, 'revision': revision, 'pin_index': item['pin_index'],
            'phase': item['phase'], 'field_hash': item['field_hash'],
        },
    }
`);
      context.fetch = originalFetch;
      playgroundReady = true;
      context.postMessage({ id, result: {} });
      return;
    }
    if (!playgroundReady) throw new Error('Playground runtime is not ready.');

    if (action === 'playgroundPreset') {
      const name = String(data.name);
      if (!['blank', 'random_hills', 'corridor', 'slalom'].includes(name)) await ensureTerrain();
      runtime.globals.set('playground_preset_name', name);
      runtime.globals.set('playground_seed_value', data.seed);
      runtime.globals.set('playground_revision_value', data.revision ?? 0);
      const result = pythonJson(`
playground_preset_value = build_playground_preset(playground_preset_name, playground_seed_value)
json.dumps({'revision': playground_revision_value, 'preset': playground_preset_value}, allow_nan=False)
`);
      context.postMessage({ id, result });
      return;
    }

    if (action === 'playgroundAdaptTerrain') {
      await ensureTerrain();
      setJson('playground_terrain_json', data.scenario);
      runtime.globals.set('playground_revision_value', data.revision ?? 0);
      const result = pythonJson(`
playground_source_terrain = TerrainScenario.from_dict(json.loads(playground_terrain_json))
playground_had_barriers = bool(playground_source_terrain.barriers_geojson)
playground_adapted_field = terrain_to_playground_field(playground_source_terrain, soften_barriers=True)
playground_adapted_scenario = playground_adapted_field.base_spec['scenario']
playground_terrain_bounds = playground_adapted_scenario['bounds_m']
playground_terrain_preset = {
    'name': 'imported',
    'bounds': [playground_terrain_bounds[0], playground_terrain_bounds[2], playground_terrain_bounds[1], playground_terrain_bounds[3]],
    'start': playground_adapted_scenario['start_m'],
    'end': playground_adapted_scenario['goal_m'],
    'base_field': playground_adapted_field.base_spec,
    'gaussians': [], 'strokes': [],
    'metadata': {
        'kind': 'terrain', 'preset': 'imported', 'seed': None,
        'scenario_hash': playground_adapted_field.base_spec['scenario_hash'],
        'soft_walls': playground_adapted_scenario.get('metadata', {}).get('soft_walls'),
        'imported_from': 'terrain-v2',
    },
}
json.dumps({
    'revision': playground_revision_value,
    'preset': playground_terrain_preset,
    'message': (
        'Terrain v2 imported. Hard barriers were converted to finite soft walls.'
        if playground_had_barriers else 'Terrain v2 imported.'
    ),
}, allow_nan=False)
`);
      context.postMessage({ id, result });
      return;
    }

    if (action === 'playgroundField') {
      if (data.scene?.base_field?.kind === 'terrain') await ensureTerrain();
      setJson('playground_scene_json', data.scene);
      setJson('playground_paths_json', data.paths ?? []);
      runtime.globals.set('playground_grid_width', data.width ?? 144);
      runtime.globals.set('playground_grid_height', data.height ?? 96);
      runtime.globals.set('playground_revision_value', data.revision ?? 0);
      const result = pythonJson(`
playground_scene_data = json.loads(playground_scene_json)
playground_analytic_field, playground_explicit, playground_combined = playground_field_from_scene(playground_scene_data)
playground_bounds = playground_scene_data['bounds']
playground_xx, playground_yy = np.meshgrid(
    np.linspace(playground_bounds[0], playground_bounds[1], playground_grid_width),
    np.linspace(playground_bounds[3], playground_bounds[2], playground_grid_height),
)
playground_grid_points = np.stack([playground_xx, playground_yy], axis=-1)
playground_field_values = playground_analytic_field.cost(playground_grid_points).ravel().tolist()
playground_elevation_values = playground_analytic_field.elevation(playground_grid_points).ravel()
playground_costs = []
for playground_path_item in json.loads(playground_paths_json):
    playground_options_item = PlaygroundOptions(interior_points=len(playground_path_item) - 2, method='auto')
    playground_costs.append(evaluate_playground(playground_path_item, field=playground_analytic_field, options=playground_options_item).route_cost)
json.dumps({
    'revision': playground_revision_value, 'bounds': playground_bounds,
    'width': playground_grid_width, 'height': playground_grid_height,
    'field': playground_field_values,
    'field_min': float(np.min(playground_field_values)),
    'field_max': float(np.max(playground_field_values)),
    'elevation': playground_elevation_values.tolist(),
    'elevation_min': float(np.min(playground_elevation_values)),
    'elevation_max': float(np.max(playground_elevation_values)),
    'gaussians': [asdict(item) for item in playground_explicit],
    'obstacles': [asdict(item) for item in playground_combined],
    'ghost_costs': playground_costs,
    'field_hash': playground_analytic_field.field_hash,
}, allow_nan=False)
`);
      context.postMessage({ id, result });
      return;
    }

    if (action === 'playgroundInitialize') {
      if (data.scene?.base_field?.kind === 'terrain') await ensureTerrain();
      setJson('playground_scene_json', data.scene);
      setJson('playground_path_json', data.path);
      setJson('playground_settings_json', data.settings);
      setJson('playground_pin_json', data.pin ?? null);
      runtime.globals.set('playground_revision_value', data.revision ?? 0);
      const result = pythonJson(`
playground_scene_data = json.loads(playground_scene_json)
playground_settings_data = json.loads(playground_settings_json)
playground_pin_data = json.loads(playground_pin_json)
playground_analytic_field, _, _ = playground_field_from_scene(playground_scene_data)
playground_options_value = playground_options(playground_settings_data)
playground_state = initialize_playground(
    playground_scene_data['start'], playground_scene_data['end'],
    field=playground_analytic_field, path=json.loads(playground_path_json), options=playground_options_value,
    pin_index=playground_pin_data['index'] if playground_pin_data else None,
    pin_position=playground_pin_data['position'] if playground_pin_data else None,
)
json.dumps(playground_payload(playground_state, playground_revision_value), allow_nan=False)
`);
      residentRevision = data.revision ?? 0;
      context.postMessage({ id, result });
      return;
    }

    if (action === 'playgroundStep') {
      const revision = data.revision ?? 0;
      const iterations = data.iterations ?? 1;
      if (!Number.isInteger(iterations) || iterations < 1 || iterations > 20)
        throw new Error('Step iterations must be an integer from 1 through 20.');
      runtime.globals.set('playground_revision_value', revision);
      runtime.globals.set('playground_iterations_value', iterations);
      if (revision !== residentRevision) {
        if (data.scene?.base_field?.kind === 'terrain') await ensureTerrain();
        setJson('playground_scene_json', data.scene);
        setJson('playground_path_json', data.path);
        setJson('playground_settings_json', data.settings);
        setJson('playground_pin_json', data.pin ?? null);
        runtime.runPython(`
playground_scene_data = json.loads(playground_scene_json)
playground_settings_data = json.loads(playground_settings_json)
playground_pin_data = json.loads(playground_pin_json)
playground_analytic_field, _, _ = playground_field_from_scene(playground_scene_data)
playground_state = initialize_playground(
    playground_scene_data['start'], playground_scene_data['end'],
    field=playground_analytic_field, path=json.loads(playground_path_json),
    options=playground_options(playground_settings_data),
    pin_index=playground_pin_data['index'] if playground_pin_data else None,
    pin_position=playground_pin_data['position'] if playground_pin_data else None,
)
`);
        residentRevision = revision;
      }
      const result = pythonJson(`
playground_state = advance_playground(playground_state, iterations=playground_iterations_value)
json.dumps(playground_payload(playground_state, playground_revision_value), allow_nan=False)
`);
      context.postMessage({ id, result });
      return;
    }

    if (action === 'playgroundEvaluate') {
      if (data.scene?.base_field?.kind === 'terrain') await ensureTerrain();
      setJson('playground_scene_json', data.scene);
      setJson('playground_paths_json', data.paths ?? []);
      runtime.globals.set('playground_revision_value', data.revision ?? 0);
      const result = pythonJson(`
playground_scene_data = json.loads(playground_scene_json)
playground_analytic_field, _, _ = playground_field_from_scene(playground_scene_data)
playground_costs = []
for playground_path_item in json.loads(playground_paths_json):
    playground_options_item = PlaygroundOptions(interior_points=len(playground_path_item) - 2, method='auto')
    playground_costs.append(evaluate_playground(playground_path_item, field=playground_analytic_field, options=playground_options_item).route_cost)
json.dumps({'revision': playground_revision_value, 'ghost_costs': playground_costs}, allow_nan=False)
`);
      context.postMessage({ id, result });
      return;
    }

    throw new Error(`Unknown worker action: ${action}`);
  } catch (error) {
    context.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
};
