import 'katex/dist/katex.min.css';
import './style.css';
import { content } from './content';
import { Plot, chart, colors, names, sceneBounds } from './plot';
import { validateScene, type Scene, type Frame, type State, type Guess } from './types';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
$('app').innerHTML = content;
const plot = new Plot($<HTMLCanvasElement>('landscape'));
let scenes: Record<string, Scene>, scene: Scene;
let histories: Record<string, State[]> = {};
let straightCost: number | undefined;
let cursor = 0,
  selected = 0,
  generation = 0;
let ready = false,
  playing = false,
  pending = false,
  preview = false;
let worker: Worker,
  serial = 0,
  queue = Promise.resolve();
const waiting = new Map<number, { resolve: (frame: Frame) => void; reject: (e: Error) => void }>();
const status = (text: string) => {
  $('runtime').textContent = text;
};
const message = (text: string) => {
  $('message').textContent = text;
};
const errorMessage = (error: unknown) => (error instanceof Error ? error.message : String(error));
const base = new URL('.', location.href);
const maxCursor = () => Math.max(0, ...Object.values(histories).map((h) => h.length - 1));

function request(action: string, extra = {}): Promise<Frame> {
  const id = ++serial;
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject });
    worker.postMessage({ id, action, ...extra });
  });
}
function enqueue<T>(action: () => Promise<T>): Promise<T> {
  const result = queue.then(action);
  queue = result.then(
    () => undefined,
    () => undefined,
  );
  return result;
}
function toggleControls() {
  const terminal = Object.values(histories).every((h) => h.at(-1)?.status !== 'running');
  $<HTMLButtonElement>('play').disabled =
    !ready || (pending && !playing) || (terminal && cursor >= maxCursor());
  $<HTMLButtonElement>('step').disabled = !ready || pending || (terminal && cursor >= maxCursor());
  $<HTMLButtonElement>('reset').disabled = !ready;
  $('play').textContent = playing ? 'Ⅱ Pause' : cursor < maxCursor() ? '▶ Replay' : '▶ Run solver';
}
function render() {
  plot.scene = scene;
  plot.selected = selected;
  plot.states = Object.fromEntries(
    Object.entries(histories).map(([name, h]) => [name, h[Math.min(cursor, h.length - 1)]]),
  );
  plot.draw();
  $('iteration-label').textContent = `ITERATION ${String(cursor).padStart(2, '0')}`;
  const timeline = $<HTMLInputElement>('timeline');
  timeline.max = String(maxCursor());
  timeline.value = String(cursor);
  $('timeline-value').textContent = `${cursor} / ${maxCursor()}`;
  const best = Object.entries(plot.states)
    .filter(([, state]) => Number.isFinite(state.cost))
    .sort((a, b) => a[1].cost - b[1].cost)[0];
  $('cost-comparison').textContent =
    straightCost === undefined
      ? 'Computing route costs…'
      : !best
        ? `Direct route cost: ${straightCost.toFixed(2)}.`
        : `Direct route cost: ${straightCost.toFixed(2)}. Lowest cost shown: ${names[best[0]]}, ${best[1].cost.toFixed(2)}.` +
          (straightCost > 0 && best[1].cost < straightCost * (1 - 1e-6)
            ? ` That’s ${(100 * (1 - best[1].cost / straightCost)).toFixed(1)}% less.`
            : ' No cheaper detour shown.');
  $('metrics').innerHTML = Object.entries(plot.states)
    .map(([name, state]) => {
      const index = ['straight', 'bend-x', 'bend-y'].indexOf(name);
      return `<div class="metric"><h4 style="color:${colors[index]}">${names[name]} <span class="status ${state.status}">${state.status.replace('_', ' ')}</span></h4><p>${state.cost.toFixed(2)} <span>total cost</span> &nbsp; ${state.length.toFixed(1)} <span>distance</span></p><details><summary>Numerical details</summary><small>RMS ${state.residual_norm.toExponential(1)} · step fraction ${state.damping.toPrecision(2)}</small></details></div>`;
    })
    .join('');
  renderCharts();
  const outOfView = Object.values(plot.states).some((state) =>
    state.path.some(
      ([x, y]) =>
        x < plot.bounds[0] || x > plot.bounds[1] || y < plot.bounds[2] || y > plot.bounds[3],
    ),
  );
  $('playback-label').textContent = outOfView
    ? 'Path extends beyond the view'
    : 'Drag points to reshape the scene';
  toggleControls();
}
function renderCharts() {
  chart($('residual-chart'), histories, 'residual_norm', cursor);
  chart($('cost-chart'), histories, 'cost', cursor);
}
const chartResize = new ResizeObserver(renderCharts);
for (const id of ['cost-chart', 'residual-chart']) chartResize.observe($(id));
function populateControls() {
  for (const endpoint of ['start', 'end'] as const)
    for (const [index, axis] of ['x', 'y'].entries()) {
      $<HTMLInputElement>(`${endpoint}-${axis}`).value = String(scene[endpoint][index]);
    }
  $<HTMLSelectElement>('mode').value = scene.options.mode;
  $<HTMLInputElement>('resolution').value = String(scene.options.interior_points);
  $('resolution-value').textContent = String(scene.options.interior_points);
  document.querySelectorAll<HTMLInputElement>('[data-guess]').forEach((input) => {
    input.checked = scene.guesses.includes(input.dataset.guess as Guess);
  });
  $('rover-path').innerHTML = scene.guesses
    .map((g) => `<option value="${g}">${names[g]}</option>`)
    .join('');
  selected = Math.max(0, Math.min(selected, scene.obstacles.length - 1));
  $('obstacle-count').textContent = String(scene.obstacles.length).padStart(2, '0');
  $('obstacle-select').innerHTML = scene.obstacles.length
    ? scene.obstacles.map((_, i) => `<option value="${i}">Obstacle ${i + 1}</option>`).join('')
    : '<option>No obstacles</option>';
  $<HTMLSelectElement>('obstacle-select').value = String(selected);
  $('obstacle-editor').hidden = !scene.obstacles.length;
  const obstacle = scene.obstacles[selected];
  if (obstacle)
    for (const key of ['x', 'y', 'weight', 'width'] as const) {
      $<HTMLInputElement>(key === 'x' || key === 'y' ? `obstacle-${key}` : key).value = String(
        obstacle[key],
      );
      if (key === 'weight' || key === 'width')
        $(`${key}-value`).textContent = String(obstacle[key]);
    }
}
function initialPaths() {
  // Only an initial-guess display; the Python worker supplies all computed values.
  histories = Object.fromEntries(
    scene.guesses.map((guess) => {
      const side = guess === 'bend-x' ? -1 : guess === 'bend-y' ? 1 : 0;
      const delta = scene.end.map((v, axis) => v - scene.start[axis]);
      const normal = [-delta[1], delta[0]];
      const path = Array.from({ length: scene.options.interior_points + 2 }, (_, i) => {
        const t = i / (scene.options.interior_points + 1);
        return scene.start.map(
          (v, axis) => v + delta[axis] * t + side * 0.3 * Math.sin(Math.PI * t) * normal[axis],
        ) as [number, number];
      });
      return [
        guess,
        [
          {
            path,
            iteration: 0,
            residual_norm: 0,
            energy: 0,
            cost: 0,
            length: 0,
            status: 'running',
            damping: 0,
          } as State,
        ],
      ];
    }),
  );
}
async function resetScene() {
  const token = ++generation;
  playing = false;
  stopRover();
  cursor = 0;
  preview = false;
  straightCost = undefined;
  plot.bounds = sceneBounds(scene);
  plot.field = [];
  initialPaths();
  render();
  $('metrics').textContent = 'Computing initial values…';
  $('view-label').textContent = ready ? 'Computing initial state…' : 'Waiting for Python';
  if (!ready) return;
  pending = true;
  toggleControls();
  const snapshot = structuredClone(scene),
    bounds = [...plot.bounds];
  try {
    const frame = await enqueue(() => request('initialize', { scene: snapshot, bounds }));
    if (token !== generation) return;
    histories = Object.fromEntries(
      Object.entries(frame.states).map(([name, state]) => [name, [state]]),
    );
    straightCost = frame.straight_cost;
    plot.setField(frame.field!);
    pending = false;
    $('view-label').textContent = 'Live Python solver';
    render();
  } catch (error) {
    if (token === generation) {
      pending = false;
      message(errorMessage(error));
      toggleControls();
    }
  }
}
async function advance() {
  if (pending || !ready) return;
  if (cursor < maxCursor()) {
    cursor++;
    render();
    return;
  }
  if (Object.values(histories).every((h) => h.at(-1)!.status !== 'running')) {
    playing = false;
    render();
    return;
  }
  pending = true;
  const token = generation;
  toggleControls();
  try {
    const frame = await enqueue(() => request('step'));
    if (token !== generation) return;
    for (const [name, state] of Object.entries(frame.states)) {
      if (state.iteration !== histories[name].at(-1)!.iteration) histories[name].push(state);
    }
    cursor = maxCursor();
    pending = false;
    render();
  } catch (error) {
    if (token === generation) {
      pending = false;
      playing = false;
      message(errorMessage(error));
      toggleControls();
    }
  }
}
let runTimer: ReturnType<typeof setTimeout>;
async function runLoop() {
  clearTimeout(runTimer);
  if (!playing) return;
  await advance();
  if (playing) runTimer = setTimeout(runLoop, 100);
}
function edit(change: (draft: Scene) => void) {
  try {
    const draft = structuredClone(scene);
    change(draft);
    scene = validateScene(draft);
    markCustom();
    message('');
    populateControls();
    void resetScene();
  } catch (error) {
    message(errorMessage(error));
    populateControls();
  }
}
function markCustom() {
  document.querySelectorAll('[data-preset]').forEach((el) => el.classList.remove('active'));
  $('experiment-note').textContent =
    'Your landscape. Change a hill, run the routes, and compare their total costs.';
}
const notes: Record<string, string> = {
  empty:
    'With uniform cost, each initial guess should recover the straight line in one Newton step.',
  central:
    'Is the detour worth it? Run the routes, then change the hill’s strength and width. Compare total cost, not just distance.',
  asymmetric:
    'Try routes on either side of the hills. Which one is cheapest, and is it also the shortest?',
  passage:
    'The gap offers cheaper ground between hills. Compare going through with going around; a solver may also get stuck.',
  challenge:
    'Crossing these strong hills is expensive. Do the arc-shaped starting routes find worthwhile detours? Check the costs and solver status.',
};
function selectPreset(key: string) {
  if (!scenes) return;
  scene = structuredClone(scenes[key]);
  selected = 0;
  document
    .querySelectorAll('[data-preset]')
    .forEach((el) => el.classList.toggle('active', (el as HTMLElement).dataset.preset === key));
  $('experiment-note').textContent = notes[key];
  message('');
  populateControls();
  void resetScene();
}
document.querySelectorAll<HTMLButtonElement>('[data-preset]').forEach(
  (button) =>
    (button.onclick = () => {
      selectPreset(button.dataset.preset!);
      if (!button.closest('.experiment-tabs'))
        $('playground').scrollIntoView({
          behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
        });
    }),
);
$('play').onclick = () => {
  playing = !playing;
  toggleControls();
  if (playing) void runLoop();
};
$('step').onclick = () => {
  playing = false;
  void advance();
};
$('reset').onclick = () => {
  void resetScene();
};
$('empty').onclick = () => selectPreset('empty');
$('timeline').oninput = () => {
  playing = false;
  stopRover();
  cursor = Number($<HTMLInputElement>('timeline').value);
  render();
};
for (const endpoint of ['start', 'end'] as const)
  for (const [index, axis] of ['x', 'y'].entries()) {
    $(`${endpoint}-${axis}`).onchange = () =>
      edit((draft) => {
        draft[endpoint][index] = $<HTMLInputElement>(`${endpoint}-${axis}`).valueAsNumber;
      });
  }
$('mode').onchange = () =>
  edit((draft) => {
    draft.options.mode = $<HTMLSelectElement>('mode').value as 'damped' | 'undamped';
  });
$('resolution').oninput = () => {
  $('resolution-value').textContent = $<HTMLInputElement>('resolution').value;
};
$('resolution').onchange = () =>
  edit((draft) => {
    draft.options.interior_points = $<HTMLInputElement>('resolution').valueAsNumber;
  });
document.querySelectorAll<HTMLInputElement>('[data-guess]').forEach(
  (input) =>
    (input.onchange = () =>
      edit((draft) => {
        draft.guesses = Array.from(
          document.querySelectorAll<HTMLInputElement>('[data-guess]:checked'),
        ).map((el) => el.dataset.guess as Guess);
      })),
);
$('add').onclick = () =>
  edit((draft) => {
    selected = draft.obstacles.length;
    draft.obstacles.push({
      x: (draft.start[0] + draft.end[0]) / 2,
      y: (draft.start[1] + draft.end[1]) / 2,
      weight: 5,
      width: 1.5,
    });
  });
$('remove').onclick = () =>
  edit((draft) => {
    draft.obstacles.splice(selected, 1);
  });
$('obstacle-select').onchange = () => {
  selected = Number($<HTMLSelectElement>('obstacle-select').value);
  populateControls();
  render();
};
for (const key of ['x', 'y', 'weight', 'width'] as const) {
  const id = key === 'x' || key === 'y' ? `obstacle-${key}` : key;
  $(id).onchange = () =>
    edit((draft) => {
      draft.obstacles[selected][key] = $<HTMLInputElement>(id).valueAsNumber;
    });
  if (key === 'weight' || key === 'width')
    $(id).oninput = () => {
      $(`${key}-value`).textContent = $<HTMLInputElement>(id).value;
    };
}
for (const key of ['heatmap', 'contours', 'samples'] as const)
  $(key).onchange = () => {
    plot[key] = $<HTMLInputElement>(key).checked;
    plot.draw();
  };
let drag: 'start' | 'end' | number | null = null;
plot.canvas.onpointerdown = (event) => {
  const p = plot.fromPixel(event.clientX, event.clientY),
    radius = (plot.bounds[1] - plot.bounds[0]) * 0.035;
  const distance = (q: number[]) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  if (distance(scene.start) < radius) drag = 'start';
  else if (distance(scene.end) < radius) drag = 'end';
  else {
    const i = scene.obstacles.findIndex((o) => distance([o.x, o.y]) < radius);
    drag = i < 0 ? null : i;
  }
  if (drag !== null) {
    playing = false;
    ++generation;
    stopRover();
    plot.canvas.setPointerCapture(event.pointerId);
    if (typeof drag === 'number') selected = drag;
  }
};
plot.canvas.onpointermove = (event) => {
  if (drag === null) return;
  const p = plot
    .fromPixel(event.clientX, event.clientY)
    .map((v) => Math.max(-100, Math.min(100, Math.round(v * 100) / 100))) as [number, number];
  if (typeof drag === 'number') Object.assign(scene.obstacles[drag], { x: p[0], y: p[1] });
  else scene[drag] = p;
  markCustom();
  plot.field = [];
  cursor = 0;
  initialPaths();
  populateControls();
  render();
  $('metrics').textContent = 'Release the point to compute the new scene.';
};
plot.canvas.onpointerup = plot.canvas.onpointercancel = () => {
  if (drag !== null) {
    drag = null;
    void resetScene();
  }
};

let roverFrame = 0;
function stopRover() {
  cancelAnimationFrame(roverFrame);
  roverFrame = 0;
  plot.rover = null;
  $('rover').textContent = 'Play rover';
}
$('rover').onclick = () => {
  if (roverFrame) {
    stopRover();
    render();
    return;
  }
  playing = false;
  toggleControls();
  const started = performance.now();
  $('rover').textContent = 'Stop rover';
  const animate = (time: number) => {
    plot.rover = Math.min((time - started) / 5000, 1);
    const all = plot.states,
      chosen = $<HTMLSelectElement>('rover-path').value;
    plot.states = all[chosen] ? { [chosen]: all[chosen] } : all;
    plot.draw();
    plot.states = all;
    if (plot.rover < 1) roverFrame = requestAnimationFrame(animate);
    else {
      stopRover();
      render();
    }
  };
  roverFrame = requestAnimationFrame(animate);
};
function download(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob),
    anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('export').onclick = () =>
  download(
    'path-scene.json',
    new Blob([JSON.stringify(scene, null, 2)], { type: 'application/json' }),
  );
$('png').onclick = () =>
  plot.canvas.toBlob((blob) => {
    if (blob) download('path-planning.png', blob);
  });
$('share').onclick = async () => {
  const url = new URL(location.href);
  url.hash = `scene=${encodeURIComponent(JSON.stringify(scene))}`;
  try {
    await navigator.clipboard.writeText(url.href);
    message('Scene link copied. It includes the setup; the solver starts from iteration zero.');
  } catch {
    history.replaceState(null, '', url);
    message('The scene link is in your address bar. Copy it to share.');
  }
};
$('import').onclick = () => $<HTMLInputElement>('file').click();
$('file').onchange = async () => {
  const input = $<HTMLInputElement>('file'),
    file = input.files?.[0];
  if (!file) return;
  try {
    if (file.size > 100_000) throw new Error('Scene file exceeds 100 KB.');
    scene = validateScene(JSON.parse(await file.text()));
    markCustom();
    populateControls();
    await resetScene();
    message('Scene imported.');
  } catch (error) {
    message(errorMessage(error));
  } finally {
    input.value = '';
  }
};
async function boot() {
  ready = false;
  toggleControls();
  $('retry').hidden = true;
  if (worker) worker.terminate();
  for (const promise of waiting.values()) promise.reject(new Error('Worker restarted.'));
  waiting.clear();
  queue = Promise.resolve();
  worker = new Worker(new URL('./solver.worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => {
    if (data.progress) {
      status(data.progress);
      return;
    }
    const promise = waiting.get(data.id);
    if (!promise) return;
    waiting.delete(data.id);
    if (data.error) promise.reject(new Error(data.error));
    else promise.resolve(data.result);
  };
  worker.onerror = (event) => {
    for (const promise of waiting.values())
      promise.reject(new Error(event.message || 'Browser worker failed.'));
    waiting.clear();
    ready = false;
    playing = false;
    pending = false;
    status('Python worker unavailable. Retry to restart.');
    $('retry').hidden = false;
    toggleControls();
  };
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      request('boot', { base: base.href }),
      new Promise((_, reject) => {
        timeout = setTimeout(() => reject(new Error('Loading timed out.')), 90_000);
      }),
    ]);
    clearTimeout(timeout);
    ready = true;
    status('Python + NumPy ready · all computation stays in your browser');
    await resetScene();
  } catch (error) {
    clearTimeout(timeout);
    worker.terminate();
    status(`Live solver unavailable. ${errorMessage(error)}`);
    $('retry').hidden = false;
    $('view-label').textContent = preview
      ? 'Precomputed example · not live'
      : 'Live computation unavailable';
    toggleControls();
  }
}
$('retry').onclick = () => {
  void boot();
};
async function start() {
  const response = await fetch(new URL('presets.json', base));
  if (!response.ok) throw new Error('Could not load example scenes. Reload the page to try again.');
  scenes = await response.json();
  scene = structuredClone(scenes.asymmetric);
  let shared = false;
  if (location.hash.startsWith('#scene=')) {
    try {
      if (location.hash.length > 100_000) throw new Error('Scene link is too large.');
      scene = validateScene(JSON.parse(decodeURIComponent(location.hash.slice(7))));
      shared = true;
      markCustom();
    } catch (error) {
      message(`Invalid scene link: ${errorMessage(error)}`);
    }
  }
  plot.bounds = sceneBounds(scene);
  populateControls();
  initialPaths();
  render();
  $('metrics').textContent = 'Waiting for Python…';
  if (!shared) {
    try {
      const response = await fetch(new URL('preview.json', base));
      if (response.ok) {
        const example = await response.json();
        histories = example.histories;
        straightCost = example.straight_cost;
        plot.setField(example.field);
        preview = true;
        cursor = maxCursor();
        render();
        $('view-label').textContent = 'Precomputed example · Python is loading';
      }
    } catch {
      /* Live runtime can still load. */
    }
  }
  await boot();
}
void start().catch((error) => {
  status(errorMessage(error));
  message('The essay remains available. Check the local setup instructions in the repository.');
});
