import './playground.css';
import { PlaygroundCanvas } from './canvas';
import {
  clampToBounds,
  movePathNeighborhood,
  resamplePath,
  straightPath,
  strokeHit,
} from './geometry';
import { GestureHistory } from './history';
import {
  DEFAULT_BOUNDS,
  DEFAULT_SETTINGS,
  cloneSnapshot,
  validatePlaygroundFile,
  type FieldResult,
  type GhostRoute,
  type Metrics,
  type Pin,
  type PlaygroundFile,
  type PlaygroundScene,
  type PlaygroundSettings,
  type PlaygroundSnapshot,
  type Point,
  type StepResult,
  type Stroke,
  type Tool,
  type WorkerResult,
} from './types';
import { validateScene, type Scene } from '../types';
import { LatestTaskQueue, PlaygroundWorkerClient } from './worker-client';

const $ = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T | null;
const all = <T extends Element = HTMLElement>(selector: string) => [
  ...document.querySelectorAll<T>(selector),
];
const initialViewWidth = DEFAULT_BOUNDS[1] - DEFAULT_BOUNDS[0];

const seedStrokes = (): Stroke[] => [
  { id: 'seed-1', points: [[0, 0.15]], width: 0.85, strength: 18 },
  {
    id: 'seed-2',
    points: [
      [2.25, -2.25],
      [2.55, -1.8],
      [2.8, -1.25],
    ],
    width: 0.5,
    strength: 8,
  },
];

function defaultSnapshot(): PlaygroundSnapshot {
  const settings = structuredClone(DEFAULT_SETTINGS),
    start: Point = [-4.8, -1.8],
    end: Point = [4.8, 1.8];
  return {
    scene: {
      bounds: [...DEFAULT_BOUNDS],
      start,
      end,
      strokes: seedStrokes(),
      gaussians: [],
    },
    path: straightPath(start, end, settings.interior_points),
    settings,
    ghosts: [],
    metrics: null,
  };
}

const presets: Record<string, () => PlaygroundSnapshot> = {
  seeded: defaultSnapshot,
  blank: () => {
    const result = defaultSnapshot();
    result.scene.strokes = [];
    return result;
  },
  corridor: () => {
    const result = defaultSnapshot();
    result.scene.strokes = [
      {
        id: 'corridor-top',
        points: [
          [-1.2, 1.25],
          [0, 1.1],
          [1.2, 1.25],
        ],
        width: 0.6,
        strength: 22,
      },
      {
        id: 'corridor-bottom',
        points: [
          [-1.2, -1.25],
          [0, -1.1],
          [1.2, -1.25],
        ],
        width: 0.6,
        strength: 22,
      },
    ];
    return result;
  },
  slalom: () => {
    const result = defaultSnapshot();
    result.scene.strokes = [
      { id: 'slalom-1', points: [[-2.2, 0.9]], width: 0.72, strength: 24 },
      { id: 'slalom-2', points: [[0, -0.9]], width: 0.72, strength: 24 },
      { id: 'slalom-3', points: [[2.2, 0.9]], width: 0.72, strength: 24 },
    ];
    return result;
  },
};

function formatMetric(value: number | undefined, digits = 3) {
  if (value === undefined || !Number.isFinite(value)) return '—';
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function download(filename: string, contents: string) {
  const url = URL.createObjectURL(new Blob([contents], { type: 'application/json' })),
    anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

class PlaygroundApp {
  private snapshot = defaultSnapshot();
  private readonly history = new GestureHistory(50);
  private readonly canvas: PlaygroundCanvas;
  private readonly client: PlaygroundWorkerClient;
  private readonly queue: LatestTaskQueue<WorkerResult>;
  private revision = 0;
  private fieldRevision = -1;
  private ready = false;
  private paused = false;
  private tool: Tool = 'grab';
  private currentPreset = 'seeded';
  private pin: Pin = null;
  private solverState: unknown = null;
  private previousPath: Point[] | null = null;
  private runTimer = 0;
  private readonly fieldRollbacks = new Map<number, PlaygroundSnapshot>();
  private pendingImportRevision: number | null = null;
  private gesture:
    | { kind: 'grab'; pointerId: number; index: number; changed: boolean }
    | {
        kind: 'paint';
        pointerId: number;
        stroke: Stroke;
        changed: boolean;
        before: PlaygroundSnapshot;
      }
    | { kind: 'erase'; pointerId: number; changed: boolean }
    | null = null;
  private strokeSerial = 2;
  private keyboardIndex: number;
  private keyboardTarget: string;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = new PlaygroundCanvas(canvas, this.snapshot.scene, this.snapshot.path);
    this.keyboardIndex = Math.floor(this.snapshot.path.length / 2);
    this.keyboardTarget = `path:${this.keyboardIndex}`;
    this.client = new PlaygroundWorkerClient(
      () => new Worker(new URL('../solver.worker.ts', import.meta.url), { type: 'module' }),
      (progress) => this.setRuntime(progress),
      (error) => this.failWorker(error),
    );
    this.queue = new LatestTaskQueue(
      (result, revision) => this.accept(result, revision),
      (error, revision) => this.reject(error, revision),
    );
    this.bind();
    this.sync();
    this.render();
    void this.loadPreview();
    void this.boot();
  }

  private async loadPreview() {
    const initialRevision = this.revision;
    try {
      const response = await fetch(new URL('playground-preview.json', new URL('.', location.href)));
      if (!response.ok) return;
      const preview = (await response.json()) as { field: number[]; width: number; height: number };
      if (this.revision === initialRevision && this.fieldRevision < 0) {
        this.canvas.setField(preview.field, preview.width, preview.height);
        this.canvas.draw();
      }
    } catch {
      // The grid and painted-stroke overlay remain usable if the static preview is unavailable.
    }
  }

  private bind() {
    for (const button of all<HTMLElement>('[data-playground-tool]'))
      button.onclick = () => this.setTool(button.dataset.playgroundTool as Tool);
    for (const button of all<HTMLElement>('[data-playground-preset]'))
      button.onclick = () => this.usePreset(button.dataset.playgroundPreset || 'seeded');
    $('playground-retry')?.addEventListener('click', () => void this.boot());
    $('playground-undo')?.addEventListener('click', () => this.undo());
    $('playground-redo')?.addEventListener('click', () => this.redo());
    $('playground-toggle')?.addEventListener('click', () => this.togglePause());
    $('playground-keep')?.addEventListener('click', () => this.keepPath());
    $('playground-reset')?.addEventListener('click', () => this.usePreset(this.currentPreset));
    $('playground-fit')?.addEventListener('click', () => this.fitView());
    $('playground-focus-canvas')?.addEventListener('click', () => this.canvas.canvas.focus());
    $('playground-save')?.addEventListener('click', () => this.save());
    $('playground-load')?.addEventListener('click', () => $('playground-file')?.click());
    $('playground-file')?.addEventListener('change', () => void this.loadFile());
    for (const [id, key] of [
      ['playground-show-points', 'showPoints'],
      ['playground-show-previous', 'showPrevious'],
    ] as const) {
      const input = $<HTMLInputElement>(id);
      if (input)
        input.onchange = () => {
          this.canvas[key] = input.checked;
          this.canvas.draw();
        };
    }

    const resolution = $<HTMLSelectElement>('playground-resolution');
    if (resolution)
      resolution.onchange = () => {
        const count = Number(resolution.value) as 32 | 64 | 128;
        this.commitControlChange((draft) => {
          draft.settings.interior_points = count;
          draft.path = resamplePath(draft.path, count);
        }, true);
      };
    const method = $<HTMLSelectElement>('playground-method');
    if (method)
      method.onchange = () =>
        this.commitControlChange((draft) => {
          draft.settings.method = method.value === 'newton' ? 'newton' : 'descent';
        });
    const keyboardTarget = $<HTMLSelectElement>('playground-keyboard-target');
    if (keyboardTarget)
      keyboardTarget.onchange = () => {
        this.keyboardTarget = keyboardTarget.value;
        if (this.keyboardTarget.startsWith('path:'))
          this.keyboardIndex = Number(this.keyboardTarget.slice(5));
      };
    for (const [id, key] of [
      ['playground-brush-radius', 'brush_radius'],
      ['playground-brush-strength', 'brush_strength'],
    ] as const) {
      const input = $<HTMLInputElement>(id);
      if (!input) continue;
      input.oninput = () => {
        this.snapshot.settings[key] =
          key === 'brush_radius'
            ? (input.valueAsNumber / 100) * initialViewWidth
            : input.valueAsNumber;
        this.sync();
      };
    }

    this.canvas.canvas.addEventListener('pointerdown', (event) => this.pointerDown(event));
    this.canvas.canvas.addEventListener('pointermove', (event) => this.pointerMove(event));
    this.canvas.canvas.addEventListener('pointerup', (event) => this.pointerUp(event));
    this.canvas.canvas.addEventListener('pointercancel', (event) => this.pointerCancel(event));
    this.canvas.canvas.addEventListener('keydown', (event) => this.keyDown(event));
    window.addEventListener('keydown', (event) => {
      const editable =
        event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement;
      if (editable) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        event.shiftKey ? this.redo() : this.undo();
      }
    });
  }

  private setTool(tool: Tool) {
    this.tool = tool;
    all<HTMLElement>('[data-playground-tool]').forEach((button) => {
      const active = button.dataset.playgroundTool === tool;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    const brush = $('playground-brush-section');
    if (brush) brush.hidden = tool !== 'paint';
    this.setStatus(
      tool === 'grab'
        ? 'Grab a path point or endpoint and drag it.'
        : tool === 'paint'
          ? 'Paint a smooth region of higher travel cost.'
          : 'Touch a painted stroke to remove the whole stroke.',
    );
  }

  private snapshotForHistory() {
    return cloneSnapshot(this.snapshot);
  }

  private restore(snapshot: PlaygroundSnapshot, fieldChanged = true) {
    this.snapshot = cloneSnapshot(snapshot);
    this.pin = null;
    this.solverState = null;
    this.previousPath = null;
    this.revision++;
    if (fieldChanged) this.fieldRevision = -1;
    this.keyboardIndex = Math.min(this.keyboardIndex, this.snapshot.path.length - 2);
    this.sync();
    this.render();
    this.schedule(fieldChanged ? 'field' : 'initialize');
  }

  private commitControlChange(change: (draft: PlaygroundSnapshot) => void, resample = false) {
    const before = this.snapshotForHistory();
    this.history.begin(before);
    const oldPath = this.snapshot.path;
    change(this.snapshot);
    if (resample && this.snapshot.path === oldPath)
      this.snapshot.path = resamplePath(oldPath, this.snapshot.settings.interior_points);
    this.history.commit(true);
    this.solverState = null;
    this.revision++;
    this.sync();
    this.render();
    this.schedule('initialize');
  }

  private addGhost(path: Point[], cost = this.snapshot.metrics?.route_cost ?? null) {
    if (path.length < 2) return;
    const ghost: GhostRoute = {
      id: `ghost-${Date.now()}-${this.revision}`,
      path: path.map(([x, y]) => [x, y]),
      route_cost: cost,
    };
    this.snapshot.ghosts = [...this.snapshot.ghosts, ghost].slice(-3);
  }

  private pointerDown(event: PointerEvent) {
    if (event.button !== 0) return;
    this.previousPath = null;
    const point = this.canvas.fromClient(event.clientX, event.clientY),
      hitRadius = this.canvas.modelRadiusFromPixels(event.pointerType === 'touch' ? 26 : 18);
    if (this.tool === 'grab') {
      const nearest = this.canvas.nearestVisiblePoint(point);
      if (nearest.index < 0 || nearest.distance > hitRadius) {
        this.setStatus('Grab closer to the path or either endpoint.');
        return;
      }
      this.history.begin(this.snapshotForHistory());
      this.keyboardIndex = nearest.index;
      this.keyboardTarget = `path:${nearest.index}`;
      this.gesture = {
        kind: 'grab',
        pointerId: event.pointerId,
        index: nearest.index,
        changed: false,
      };
      this.pin =
        nearest.index > 0 && nearest.index < this.snapshot.path.length - 1
          ? { index: nearest.index, position: point }
          : null;
      this.canvas.heldIndex = nearest.index;
      this.solverState = null;
      this.revision++;
      this.schedule('initialize');
    } else if (this.tool === 'paint') {
      const before = this.snapshotForHistory();
      this.history.begin(before);
      const stroke: Stroke = {
        id: this.nextStrokeId(),
        points: [clampToBounds(point, this.snapshot.scene.bounds)],
        width: this.snapshot.settings.brush_radius,
        strength: this.snapshot.settings.brush_strength,
      };
      this.gesture = { kind: 'paint', pointerId: event.pointerId, stroke, changed: true, before };
      this.canvas.activeStroke = stroke;
    } else {
      this.history.begin(this.snapshotForHistory());
      let index = -1;
      for (let candidate = this.snapshot.scene.strokes.length - 1; candidate >= 0; candidate--)
        if (strokeHit(this.snapshot.scene.strokes[candidate], point, hitRadius)) {
          index = candidate;
          break;
        }
      const changed = index >= 0;
      if (changed) {
        this.snapshot.scene.strokes.splice(index, 1);
        this.fieldRevision = -1;
        this.revision++;
        this.render();
      }
      this.gesture = { kind: 'erase', pointerId: event.pointerId, changed };
    }
    this.canvas.canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
    this.render();
  }

  private pointerMove(event: PointerEvent) {
    if (!this.gesture || this.gesture.pointerId !== event.pointerId) return;
    const point = this.canvas.fromClient(event.clientX, event.clientY);
    if (this.gesture.kind === 'grab') {
      const index = this.gesture.index,
        next = clampToBounds(point, this.snapshot.scene.bounds);
      this.snapshot.path = movePathNeighborhood(this.snapshot.path, index, next);
      if (index === 0) this.snapshot.scene.start = next;
      if (index === this.snapshot.path.length - 1) this.snapshot.scene.end = next;
      this.pin =
        index > 0 && index < this.snapshot.path.length - 1 ? { index, position: next } : null;
      this.gesture.changed = true;
      this.solverState = null;
      this.revision++;
      this.schedule('initialize');
    } else if (this.gesture.kind === 'paint') {
      const next = clampToBounds(point, this.snapshot.scene.bounds),
        points = this.gesture.stroke.points,
        minimum = this.gesture.stroke.width * 0.12;
      if (Math.hypot(next[0] - points.at(-1)![0], next[1] - points.at(-1)![1]) >= minimum)
        points.push(next);
      this.canvas.activeStroke = this.gesture.stroke;
      this.fieldRevision = -1;
      this.revision++;
      this.fieldRollbacks.clear();
      this.fieldRollbacks.set(this.revision, this.gesture.before);
      this.schedule('field');
    }
    this.render();
    event.preventDefault();
  }

  private pointerUp(event: PointerEvent) {
    if (!this.gesture || this.gesture.pointerId !== event.pointerId) return;
    const gesture = this.gesture;
    if (gesture.kind === 'paint') {
      this.snapshot.scene.strokes.push(structuredClone(gesture.stroke));
      this.canvas.activeStroke = null;
      this.fieldRevision = -1;
      this.solverState = null;
      this.revision++;
      this.fieldRollbacks.clear();
      this.fieldRollbacks.set(this.revision, gesture.before);
    } else if (gesture.kind === 'grab') {
      this.pin = null;
      this.canvas.heldIndex = null;
      this.revision++;
    }
    this.history.commit(gesture.changed);
    this.gesture = null;
    this.sync();
    this.render();
    this.schedule(gesture.kind === 'paint' || gesture.kind === 'erase' ? 'field' : 'initialize');
  }

  private pointerCancel(event: PointerEvent) {
    if (!this.gesture || this.gesture.pointerId !== event.pointerId) return;
    const before = this.history.cancel();
    this.gesture = null;
    this.pin = null;
    this.canvas.activeStroke = null;
    this.canvas.heldIndex = null;
    if (before) this.restore(before);
  }

  private keyDown(event: KeyboardEvent) {
    if (event.key.toLowerCase() === 'g') return this.setTool('grab');
    if (event.key.toLowerCase() === 'p') return this.setTool('paint');
    if (event.key.toLowerCase() === 'e') return this.setTool('erase');
    if (event.key === ' ') {
      event.preventDefault();
      return this.togglePause();
    }
    const direction: Record<string, Point> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, 1],
      ArrowDown: [0, -1],
    };
    if (direction[event.key] || event.key === 'Delete' || event.key === 'Backspace')
      this.previousPath = null;
    if (
      (event.key === 'Delete' || event.key === 'Backspace') &&
      this.keyboardTarget.startsWith('stroke:')
    ) {
      event.preventDefault();
      const id = this.keyboardTarget.slice(7),
        index = this.snapshot.scene.strokes.findIndex((stroke) => stroke.id === id);
      if (index < 0) return;
      this.history.begin(this.snapshotForHistory());
      this.snapshot.scene.strokes.splice(index, 1);
      this.history.commit(true);
      this.keyboardTarget = `path:${this.keyboardIndex}`;
      this.fieldRevision = -1;
      this.revision++;
      this.render();
      this.schedule('field');
      this.setStatus('Deleted the selected painted stroke.');
      return;
    }
    if (!direction[event.key]) return;
    event.preventDefault();
    const step = initialViewWidth * (event.shiftKey ? 0.01 : 0.0025),
      delta = direction[event.key];
    if (this.keyboardTarget.startsWith('stroke:')) {
      const id = this.keyboardTarget.slice(7),
        stroke = this.snapshot.scene.strokes.find((candidate) => candidate.id === id);
      if (!stroke) return;
      this.history.begin(this.snapshotForHistory());
      stroke.points = stroke.points.map((point) =>
        clampToBounds(
          [point[0] + delta[0] * step, point[1] + delta[1] * step],
          this.snapshot.scene.bounds,
        ),
      );
      this.history.commit(true);
      this.fieldRevision = -1;
      this.revision++;
      this.render();
      this.schedule('field');
      this.setStatus('Moved the selected painted stroke.');
      return;
    }
    const selectedIndex = this.keyboardTarget.startsWith('path:')
        ? Number(this.keyboardTarget.slice(5))
        : this.keyboardIndex,
      index = Math.max(0, Math.min(this.snapshot.path.length - 1, selectedIndex)),
      previous = this.snapshot.path[index],
      next = clampToBounds(
        [previous[0] + delta[0] * step, previous[1] + delta[1] * step],
        this.snapshot.scene.bounds,
      );
    this.history.begin(this.snapshotForHistory());
    this.snapshot.path = movePathNeighborhood(this.snapshot.path, index, next);
    if (index === 0) this.snapshot.scene.start = next;
    if (index === this.snapshot.path.length - 1) this.snapshot.scene.end = next;
    this.history.commit(true);
    this.revision++;
    this.solverState = null;
    this.render();
    this.schedule('initialize');
    this.setStatus(`Moved path point ${index + 1} of ${this.snapshot.path.length}.`);
  }

  private undo() {
    const previous = this.history.undo(this.snapshotForHistory());
    if (previous) this.restore(previous);
  }

  private redo() {
    const next = this.history.redo(this.snapshotForHistory());
    if (next) this.restore(next);
  }

  private usePreset(name: string) {
    const make = presets[name];
    if (!make) return;
    this.currentPreset = name;
    this.history.begin(this.snapshotForHistory());
    const next = make();
    next.ghosts = this.snapshot.ghosts.slice(-3);
    this.snapshot = next;
    this.previousPath = null;
    this.history.commit(true);
    this.revision++;
    this.fieldRevision = -1;
    this.solverState = null;
    this.sync();
    this.render();
    this.schedule('field');
  }

  private togglePause() {
    this.paused = !this.paused;
    window.clearTimeout(this.runTimer);
    this.queue.clear();
    this.solverState = null;
    this.revision++;
    this.sync();
    this.setStatus(this.paused ? 'Solver paused. Editing stays available.' : 'Solver resumed.');
    if (!this.paused) this.schedule('initialize');
  }

  private keepPath() {
    this.addGhost(this.snapshot.path);
    this.revision++;
    this.render();
    this.schedule('evaluate');
    this.setStatus(`Kept this path for comparison (${this.snapshot.ghosts.length} of 3).`);
  }

  private fitView() {
    const points = [
      ...this.snapshot.path,
      ...this.snapshot.scene.strokes.flatMap((stroke) => stroke.points),
    ];
    const xs = points.map((point) => point[0]),
      ys = points.map((point) => point[1]),
      minX = Math.min(...xs),
      maxX = Math.max(...xs),
      minY = Math.min(...ys),
      maxY = Math.max(...ys),
      padding = Math.max(0.5, 0.12 * Math.max(maxX - minX, maxY - minY));
    let width = maxX - minX + 2 * padding,
      height = maxY - minY + 2 * padding;
    const aspect = Math.max(0.2, this.canvas.canvas.clientWidth / this.canvas.canvas.clientHeight);
    if (width / height < aspect) width = height * aspect;
    else height = width / aspect;
    const centerX = (minX + maxX) / 2,
      centerY = (minY + maxY) / 2;
    this.snapshot.scene.bounds = [
      centerX - width / 2,
      centerX + width / 2,
      centerY - height / 2,
      centerY + height / 2,
    ];
    this.fieldRevision = -1;
    this.revision++;
    this.render();
    this.schedule('field');
  }

  private save() {
    const file: PlaygroundFile = {
      kind: 'path-playground',
      version: 1,
      strokes: this.snapshot.scene.strokes,
      endpoints: { start: this.snapshot.scene.start, end: this.snapshot.scene.end },
      bounds: this.snapshot.scene.bounds,
      path: this.snapshot.path,
      settings: this.snapshot.settings,
      ghosts: this.snapshot.ghosts,
    };
    download('path-playground-v1.json', JSON.stringify(file, null, 2));
    this.setMessage('Playground saved.');
  }

  private nextStrokeId() {
    const ids = new Set(this.snapshot.scene.strokes.map((stroke) => stroke.id));
    let id: string;
    do id = `stroke-${++this.strokeSerial}`;
    while (ids.has(id));
    return id;
  }

  private async loadFile() {
    const input = $<HTMLInputElement>('playground-file'),
      file = input?.files?.[0];
    if (!file) return;
    try {
      if (file.size > 2_000_000) throw new Error('The file exceeds the 2 MB browser limit.');
      const raw = JSON.parse(await file.text());
      if (raw?.version === 2 && (raw?.scenario || raw?.field_x_m))
        throw new Error(
          'Terrain v2 files belong in the Terrain lab. Open terrain.html to load this file.',
        );
      const imported =
        raw?.kind === 'path-playground' ? validatePlaygroundFile(raw) : this.adaptGaussianV1(raw);
      const beforeImport = this.snapshotForHistory();
      this.history.begin(beforeImport);
      this.snapshot = {
        scene: {
          bounds: imported.bounds,
          start: imported.endpoints.start,
          end: imported.endpoints.end,
          strokes: imported.strokes,
          gaussians: [],
        },
        path: imported.path,
        settings: imported.settings,
        ghosts: imported.ghosts,
        metrics: null,
      };
      this.previousPath = null;
      this.revision++;
      this.fieldRevision = -1;
      this.solverState = null;
      this.fieldRollbacks.clear();
      this.fieldRollbacks.set(this.revision, beforeImport);
      this.pendingImportRevision = this.revision;
      this.sync();
      this.render();
      this.schedule('field');
      this.setMessage(
        raw?.kind === 'path-playground' ? 'Playground loaded.' : 'Gaussian explorer scene adapted.',
      );
    } catch (error) {
      this.setMessage(error instanceof Error ? error.message : String(error), true);
    } finally {
      if (input) input.value = '';
    }
  }

  private adaptGaussianV1(value: unknown): PlaygroundFile {
    if (!value || typeof value !== 'object')
      throw new Error('Expected a playground or Gaussian explorer JSON file.');
    const legacy = validateScene(value) as Scene;
    const settings: PlaygroundSettings = { ...DEFAULT_SETTINGS };
    const start = legacy.start as Point,
      end = legacy.end as Point,
      strokes: Stroke[] = legacy.obstacles.map((obstacle: any, index: number) => ({
        id: `legacy-${index}`,
        points: [[Number(obstacle.x), Number(obstacle.y)]],
        width: Number(obstacle.width),
        strength: obstacle.weight + 1,
      }));
    const points = [
      start,
      end,
      ...legacy.obstacles.map((obstacle) => [obstacle.x, obstacle.y] as Point),
    ];
    const xs = points.map((point) => point[0]),
      ys = points.map((point) => point[1]);
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1);
    const padding = span * 0.15;
    const bounds: [number, number, number, number] = [
      Math.min(...xs) - padding,
      Math.max(...xs) + padding,
      Math.min(...ys) - padding,
      Math.max(...ys) + padding,
    ];
    return validatePlaygroundFile({
      kind: 'path-playground',
      version: 1,
      strokes,
      endpoints: { start, end },
      bounds,
      path: straightPath(start, end, settings.interior_points),
      settings,
      ghosts: [],
    });
  }

  private async boot() {
    window.clearTimeout(this.runTimer);
    this.ready = false;
    this.queue.clear();
    this.client.start();
    this.setRuntime('Starting Python + NumPy…');
    const retry = $('playground-retry');
    if (retry) retry.hidden = true;
    this.sync();
    try {
      await Promise.race([
        this.client.request('playgroundBoot', { base: new URL('.', location.href).href }),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Loading timed out.')), 90_000),
        ),
      ]);
      this.ready = true;
      this.setRuntime('Python + NumPy ready · solving on this device');
      this.schedule('field');
    } catch (error) {
      this.failWorker(error instanceof Error ? error : new Error(String(error)));
    }
  }

  private failWorker(error: Error) {
    this.ready = false;
    this.client.stop(error);
    this.setRuntime(`Live solver unavailable. ${error.message}`);
    const retry = $('playground-retry');
    if (retry) retry.hidden = false;
    this.setMessage('The canvas remains editable. Retry when the runtime is available.', true);
    this.sync();
  }

  private workerScene() {
    return {
      start: this.snapshot.scene.start,
      end: this.snapshot.scene.end,
      obstacles: this.snapshot.scene.gaussians,
    };
  }

  private schedule(kind: 'field' | 'initialize' | 'step' | 'evaluate') {
    if (!this.ready) return;
    window.clearTimeout(this.runTimer);
    const revision = this.revision;
    if (kind === 'field' || this.fieldRevision < 0) {
      this.queue.push({
        revision,
        run: () =>
          this.client.request('playgroundField', {
            revision,
            strokes:
              this.gesture?.kind === 'paint'
                ? [...this.snapshot.scene.strokes, this.gesture.stroke]
                : this.snapshot.scene.strokes,
            bounds: this.snapshot.scene.bounds,
            width: 120,
            height: 80,
            paths: [this.snapshot.path, ...this.snapshot.ghosts.map((ghost) => ghost.path)],
          }),
      });
      return;
    }
    if (kind === 'evaluate') {
      this.queue.push({
        revision,
        run: () =>
          this.client.request('playgroundEvaluate', {
            revision,
            paths: [this.snapshot.path, ...this.snapshot.ghosts.map((ghost) => ghost.path)],
          }),
      });
      return;
    }
    const action =
      kind === 'initialize' || !this.solverState ? 'playgroundInitialize' : 'playgroundStep';
    this.queue.push({
      revision,
      run: () =>
        this.client.request(action, {
          revision,
          scene: this.workerScene(),
          path: this.snapshot.path,
          settings: this.snapshot.settings,
          pin: this.pin,
          state: this.solverState,
          iterations: this.snapshot.settings.batch_iterations,
        }),
    });
  }

  private accept(result: WorkerResult, revision: number) {
    if (revision !== this.revision || !('revision' in result)) return;
    if ('field' in result) {
      const field = result as FieldResult & { obstacles?: FieldResult['gaussians'] };
      this.snapshot.scene.gaussians = field.gaussians ?? field.obstacles ?? [];
      this.fieldRollbacks.delete(revision);
      if (this.pendingImportRevision === revision) {
        this.history.commit(true);
        this.pendingImportRevision = null;
      }
      this.fieldRevision = revision;
      this.canvas.setField(field.field, field.width, field.height);
      field.ghost_costs?.forEach((cost, index) => {
        const ghost = index === 0 ? null : this.snapshot.ghosts[index - 1];
        if (ghost) ghost.route_cost = cost;
        else if (index === 0 && field.metrics) this.snapshot.metrics = field.metrics;
      });
      this.render();
      this.schedule('initialize');
      return;
    }
    if ('ghost_costs' in result && !('field' in result)) {
      result.ghost_costs.forEach((cost, index) => {
        if (index > 0 && this.snapshot.ghosts[index - 1])
          this.snapshot.ghosts[index - 1].route_cost = cost;
      });
      this.render();
      if (!this.paused && this.snapshot.metrics?.status === 'running') this.schedule('step');
      return;
    }
    if ('path' in result) {
      const step = result as StepResult & { state?: unknown };
      this.previousPath = this.snapshot.path.map(([x, y]) => [x, y]);
      this.snapshot.path = step.path;
      this.snapshot.metrics = step.metrics;
      this.solverState = step.state ?? null;
      this.render();
      if (this.paused) this.setStatus('Solver paused. Editing stays available.');
      else if (this.gesture?.kind === 'grab')
        this.setStatus('Adjusting the route around the held waypoint.');
      else if (step.metrics.status === 'running')
        this.setStatus(`Adjusting route · travel cost ${formatMetric(step.metrics.route_cost)}`);
      else if (step.metrics.status === 'converged')
        this.setStatus(`Settled · travel cost ${formatMetric(step.metrics.route_cost)}`);
      else this.setStatus(`Solver stopped: ${step.metrics.status.replaceAll('_', ' ')}.`, true);
      const canRunDuringGesture = !this.gesture || ['grab', 'paint'].includes(this.gesture.kind);
      if (!this.paused && step.metrics.status === 'running' && canRunDuringGesture)
        this.runTimer = window.setTimeout(() => this.schedule('step'), 16);
    }
  }

  private reject(error: Error, revision: number) {
    if (revision !== this.revision) return;
    if (this.pendingImportRevision === revision) {
      const before = this.fieldRollbacks.get(revision);
      this.pendingImportRevision = null;
      this.fieldRollbacks.delete(revision);
      this.history.cancel();
      if (before) this.restore(before);
      this.setMessage(`Import rejected. ${error.message}`, true);
      return;
    }
    if (/256|Gaussian|bump/i.test(error.message) && this.snapshot.scene.strokes.length) {
      const before = this.fieldRollbacks.get(revision) ?? null;
      if (before) {
        this.fieldRollbacks.delete(revision);
        this.history.cancel();
        this.gesture = null;
        this.canvas.activeStroke = null;
        this.restore(before);
      }
      this.setMessage('That stroke would exceed the 256-bump limit, so it was not added.', true);
      return;
    }
    this.setMessage(error.message, true);
  }

  private render() {
    this.canvas.scene = this.snapshot.scene;
    this.canvas.path = this.snapshot.path;
    this.canvas.ghosts = this.snapshot.ghosts;
    this.canvas.previousPath = this.previousPath;
    this.canvas.draw();
    const metrics = this.snapshot.metrics;
    if ($('playground-travel-cost'))
      $('playground-travel-cost')!.textContent = formatMetric(metrics?.route_cost);
    if ($('playground-energy')) $('playground-energy')!.textContent = formatMetric(metrics?.energy);
    if ($('playground-gradient'))
      $('playground-gradient')!.textContent = formatMetric(metrics?.free_gradient_norm, 5);
    if ($('playground-residual'))
      $('playground-residual')!.textContent = formatMetric(metrics?.ode_residual_norm, 5);
    if ($('playground-iteration'))
      $('playground-iteration')!.textContent = String(metrics?.iteration ?? 0);
    if ($('playground-ghost-count'))
      $('playground-ghost-count')!.textContent = `${this.snapshot.ghosts.length} / 3`;
    const ghostList = $('playground-ghosts');
    if (ghostList)
      ghostList.textContent = this.snapshot.ghosts.length
        ? this.snapshot.ghosts
            .map(
              (ghost, index) =>
                `Route ${index + 1}: ${formatMetric(ghost.route_cost ?? undefined)}`,
            )
            .join(' · ')
        : 'No retained routes yet.';
    this.sync();
  }

  private sync() {
    const undo = $<HTMLButtonElement>('playground-undo'),
      redo = $<HTMLButtonElement>('playground-redo'),
      toggle = $<HTMLButtonElement>('playground-toggle');
    if (undo) undo.disabled = !this.history.canUndo;
    if (redo) redo.disabled = !this.history.canRedo;
    if (toggle) {
      toggle.disabled = !this.ready;
      toggle.textContent = this.paused ? '▶ Resume' : 'Ⅱ Pause';
    }
    const resolution = $<HTMLSelectElement>('playground-resolution');
    if (resolution) resolution.value = String(this.snapshot.settings.interior_points);
    const method = $<HTMLSelectElement>('playground-method');
    if (method) method.value = this.snapshot.settings.method;
    const keyboardTarget = $<HTMLSelectElement>('playground-keyboard-target');
    if (keyboardTarget) {
      const midpoint = Math.floor(this.snapshot.path.length / 2);
      keyboardTarget.replaceChildren(
        new Option('Start point', 'path:0'),
        new Option('Path midpoint', `path:${midpoint}`),
        new Option('Destination', `path:${this.snapshot.path.length - 1}`),
        ...this.snapshot.scene.strokes.map(
          (stroke, index) => new Option(`Painted stroke ${index + 1}`, `stroke:${stroke.id}`),
        ),
      );
      if ([...keyboardTarget.options].some((option) => option.value === this.keyboardTarget))
        keyboardTarget.value = this.keyboardTarget;
      else {
        this.keyboardTarget = `path:${midpoint}`;
        keyboardTarget.value = this.keyboardTarget;
      }
    }
    const radius = $<HTMLInputElement>('playground-brush-radius'),
      strength = $<HTMLInputElement>('playground-brush-strength');
    if (radius)
      radius.value = String((this.snapshot.settings.brush_radius / initialViewWidth) * 100);
    if (strength) strength.value = String(this.snapshot.settings.brush_strength);
    if ($('playground-brush-radius-value'))
      $('playground-brush-radius-value')!.textContent =
        `${((this.snapshot.settings.brush_radius / initialViewWidth) * 100).toFixed(0)}%`;
    if ($('playground-brush-strength-value'))
      $('playground-brush-strength-value')!.textContent =
        `${this.snapshot.settings.brush_strength.toFixed(0)}×`;
  }

  private setRuntime(message: string) {
    const element = $('playground-runtime');
    if (element) element.textContent = message;
  }

  private setStatus(message: string, error = false) {
    const element = $('playground-status');
    if (!element) return;
    element.textContent = message;
    element.classList.toggle('error', error);
  }

  private setMessage(message: string, error = false) {
    const element = $('playground-message');
    if (!element) return;
    element.textContent = message;
    element.hidden = !message;
    element.classList.toggle('error', error);
  }
}

const canvas = $<HTMLCanvasElement>('playground-canvas');
if (canvas) new PlaygroundApp(canvas);
