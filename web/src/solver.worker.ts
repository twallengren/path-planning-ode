/// <reference lib="webworker" />
import type { PyodideAPI } from 'pyodide';
import type { Scene, Bounds } from './types';

let runtime: PyodideAPI;
const context = self as unknown as DedicatedWorkerGlobalScope;
context.onmessage = async ({
  data,
}: MessageEvent<{ id: number; action: string; scene?: Scene; bounds?: Bounds; base?: string }>) => {
  const { id, action, scene, bounds } = data;
  try {
    if (action === 'boot') {
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
      for (const name of ['__init__.py', 'core.py', 'presets.py']) {
        const response = await fetch(new URL(`python/path_planning_ode/${name}`, data.base));
        if (!response.ok) throw new Error('Could not load the solver package.');
        runtime.FS.writeFile(`/home/pyodide/path_planning_ode/${name}`, await response.text());
      }
      runtime.runPython(
        'import json, numpy as np\nfrom path_planning_ode import Scene, initialize, step, cost_field',
      );
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
json.dumps({'states': {k: v.to_dict() for k, v in states.items()}, 'field': field, 'bounds': bounds}, allow_nan=False)
`);
      context.postMessage({ id, result: JSON.parse(result) });
    } else if (action === 'step') {
      const result = runtime.runPython(`
states = {k: step(scene, v) for k, v in states.items()}
json.dumps({'states': {k: v.to_dict() for k, v in states.items()}}, allow_nan=False)
`);
      context.postMessage({ id, result: JSON.parse(result) });
    }
  } catch (error) {
    context.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
};
