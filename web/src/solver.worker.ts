/// <reference lib="webworker" />
import type { PyodideAPI } from 'pyodide';
import type { Scene, Bounds, TerrainScenario, TerrainConfig } from './types';

let runtime: PyodideAPI;
let terrainReady = false;
const context = self as unknown as DedicatedWorkerGlobalScope;
context.onmessage = async ({
  data,
}: MessageEvent<{
  id: number;
  action: string;
  scene?: Scene | TerrainScenario;
  bounds?: Bounds;
  base?: string;
  family?: string;
  seed?: number;
  contrast?: number;
  barriers?: boolean;
  wallMode?: 'soft' | 'hard';
  wallMultiplier?: number;
  configs?: TerrainConfig[];
}>) => {
  const { id, action, scene, bounds } = data;
  try {
    if (action === 'boot' || action === 'terrainBoot') {
      context.postMessage({ progress: 'Loading Python runtime…' });
      // Emscripten can leave its initialization promise pending after a fetch fails.
      // Forward bootstrap network failures directly so the UI can terminate/retry.
      const originalFetch = context.fetch.bind(context);
      context.fetch = async (...args: Parameters<typeof fetch>) => {
        try {
          const response = await originalFetch(...args);
          if (!response.ok) throw new Error(`Runtime download failed (HTTP ${response.status}).`);
          return response;
        } catch (error) {
          context.postMessage({ id, error: `Runtime download failed. ${String(error)}` });
          throw error;
        }
      };
      const { loadPyodide } = await import(
        /* @vite-ignore */ new URL('runtime/pyodide.mjs', data.base).href
      );
      runtime = await loadPyodide({ indexURL: new URL('runtime/', data.base).href });
      context.postMessage({ progress: 'Loading NumPy…' });
      await runtime.loadPackage('numpy');
      runtime.FS.mkdirTree('/home/pyodide/path_planning_ode');
      const files =
        action === 'terrainBoot'
          ? ((await (await fetch(new URL('python-manifest.json', data.base))).json()) as string[])
          : ['__init__.py', 'core.py', 'presets.py'];
      for (const name of files) {
        const response = await fetch(new URL(`python/path_planning_ode/${name}`, data.base));
        if (!response.ok) throw new Error('Could not load the solver package.');
        const parts = name.split('/');
        if (parts.length > 1)
          runtime.FS.mkdirTree(`/home/pyodide/path_planning_ode/${parts.slice(0, -1).join('/')}`);
        runtime.FS.writeFile(`/home/pyodide/path_planning_ode/${name}`, await response.text());
      }
      if (action === 'terrainBoot') {
        context.postMessage({ progress: 'Loading SciPy and geometry tools…' });
        await runtime.loadPackage(['scipy', 'shapely']);
        runtime.runPython(
          'import json, numpy as np\nfrom path_planning_ode.terrain import TerrainScenario, PlannerConfig\nfrom path_planning_ode.terrain_generators import synthetic_terrain, mount_tamalpais_terrain\nfrom path_planning_ode.soft_walls import soften_walls\nfrom path_planning_ode.planners import plan, scene_to_terrain_scenario',
        );
        terrainReady = true;
      } else {
        runtime.runPython(
          'import json, numpy as np\nfrom path_planning_ode import Scene, initialize, step, cost_field, weighted_distance',
        );
      }
      context.fetch = originalFetch;
      context.postMessage({ id, result: {} });
    } else if (action === 'initialize') {
      runtime.globals.set('scene_json', JSON.stringify(scene));
      runtime.globals.set('bounds_json', JSON.stringify(bounds));
      const result = runtime.runPython(`
scene = Scene.from_dict(json.loads(scene_json))
states = {guess: initialize(scene, guess) for guess in scene.guesses}
bounds = json.loads(bounds_json)
xx, yy = np.meshgrid(np.linspace(bounds[0], bounds[1], 90), np.linspace(bounds[2], bounds[3], 90))
field = cost_field(np.stack([xx, yy], axis=-1), scene.obstacles).ravel().tolist()
json.dumps({'states': {k: v.to_dict() for k, v in states.items()}, 'field': field, 'bounds': bounds, 'straight_cost': weighted_distance(np.array([scene.start, scene.end]), scene)}, allow_nan=False)
`);
      context.postMessage({ id, result: JSON.parse(result) });
    } else if (action === 'step') {
      const result = runtime.runPython(`
states = {k: step(scene, v) for k, v in states.items()}
json.dumps({'states': {k: v.to_dict() for k, v in states.items()}}, allow_nan=False)
`);
      context.postMessage({ id, result: JSON.parse(result) });
    } else if (action === 'terrainGenerate') {
      if (!terrainReady) throw new Error('Terrain runtime is not ready.');
      runtime.globals.set('family_name', data.family);
      runtime.globals.set('terrain_seed', data.seed);
      runtime.globals.set('terrain_contrast', data.contrast);
      runtime.globals.set('terrain_barriers', data.barriers);
      runtime.globals.set('terrain_wall_mode', data.wallMode);
      runtime.globals.set('terrain_wall_multiplier', data.wallMultiplier);
      const result = runtime.runPython(`
generated_terrain = mount_tamalpais_terrain(contrast=terrain_contrast, barriers=terrain_barriers) if family_name == 'mount_tamalpais' else synthetic_terrain(family_name, seed=terrain_seed, contrast=terrain_contrast, barriers=terrain_barriers)
if terrain_wall_mode == 'soft':
    generated_terrain = soften_walls(generated_terrain, multiplier=terrain_wall_multiplier)
json.dumps(generated_terrain.to_dict(), allow_nan=False)
`);
      context.postMessage({ id, result: JSON.parse(result) });
    } else if (action === 'terrainSolve') {
      if (!terrainReady) throw new Error('Terrain runtime is not ready.');
      runtime.globals.set('terrain_scene_json', JSON.stringify(data.scene));
      runtime.globals.set('terrain_configs_json', JSON.stringify(data.configs));
      const result = runtime.runPython(`
terrain_scene = TerrainScenario.from_dict(json.loads(terrain_scene_json))
terrain_configs = [PlannerConfig.from_dict(item) for item in json.loads(terrain_configs_json)]
json.dumps([plan(terrain_scene, item).to_dict() for item in terrain_configs], allow_nan=False)
`);
      context.postMessage({ id, result: JSON.parse(result) });
    } else if (action === 'terrainAdaptV1') {
      if (!terrainReady) throw new Error('Terrain runtime is not ready.');
      runtime.globals.set('legacy_scene_json', JSON.stringify(data.scene));
      const result = runtime.runPython(`
from path_planning_ode import Scene as LegacyScene
legacy = LegacyScene.from_dict(json.loads(legacy_scene_json))
adapted = scene_to_terrain_scenario(legacy)
json.dumps(adapted.to_dict(), allow_nan=False)
`);
      context.postMessage({ id, result: JSON.parse(result) });
    }
  } catch (error) {
    context.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
};
