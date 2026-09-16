import './playground.css';
import { PlaygroundCanvas } from './canvas';
import {
  clampToBounds,
  distance,
  movePathNeighborhood,
  resamplePath,
  straightPath,
  strokeHit,
} from './geometry';
import { GestureHistory } from './history';
import {
  DEFAULT_BOUNDS,
  DEFAULT_SETTINGS,
  UNIFORM_BASE,
  cloneSnapshot,
  isPoint,
  validatePlaygroundV2,
  type FieldResult,
  type Gaussian,
  type GhostRoute,
  type PlaygroundFileV2,
  type PlaygroundScene,
  type PlaygroundSettings,
  type PlaygroundSnapshot,
  type Point,
  type PresetName,
  type PresetResult,
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

function localDefault(): PlaygroundSnapshot {
  const settings = { ...DEFAULT_SETTINGS },
    start: Point = [-4.8, 0],
    end: Point = [4.8, 0];
  return {
    preset: 'random_hills',
    seed: 0,
    scene: {
      bounds: [...DEFAULT_BOUNDS],
      start,
      end,
      base_field: UNIFORM_BASE,
      gaussians: [
        {
          x: 0.9861241487144707,
          y: -1.2431517456751005,
          weight: 2.80730142952146,
          width: 0.459090199540691,
        },
        {
          x: 2.2555457222419615,
          y: 2.2288801172996973,
          weight: 7.049768318253849,
          width: 0.8512231085411992,
        },
        {
          x: 0.31409993855104457,
          y: 2.349391088453949,
          weight: 8.618901655911491,
          width: 0.4515061750935815,
        },
        {
          x: 2.5733107914304996,
          y: -2.5186378933504927,
          weight: 7.97241584822458,
          width: 0.5466105913314074,
        },
        {
          x: 2.6148882409191834,
          y: 0.22389058934509531,
          weight: 4.747839179030386,
          width: 0.6824779716587122,
        },
        {
          x: -3.396098367752667,
          y: -2.0288703069023546,
          weight: 7.529683110202227,
          width: 0.8059542313658377,
        },
        {
          x: 0.8307728026650278,
          y: -0.6281412069858296,
          weight: 9.979074518419083,
          width: 0.9894594363269267,
        },
      ],
      strokes: [],
      metadata: { kind: 'gaussian', preset: 'random_hills', seed: 0 },
    },
    path: straightPath(start, end, settings.interior_points),
    settings,
    ghosts: [],
    metrics: null,
  };
}

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

type Gesture =
  | { kind: 'grab'; pointerId: number; index: number; changed: boolean }
  | { kind: 'gaussian'; pointerId: number; index: number; changed: boolean }
  | {
      kind: 'paint';
      pointerId: number;
      stroke: Stroke;
      changed: boolean;
      before: PlaygroundSnapshot;
    }
  | { kind: 'erase'; pointerId: number; changed: boolean }
  | null;

type PendingPreset = {
  before: PlaygroundSnapshot;
  name: PresetName | 'imported';
  seed: number;
  message?: string;
};
type PendingImport = { before: PlaygroundSnapshot; candidate: PlaygroundSnapshot; message: string };

class PlaygroundApp {
  private snapshot = localDefault();
  private readonly history = new GestureHistory(50);
  private readonly canvas: PlaygroundCanvas;
  private readonly client: PlaygroundWorkerClient;
  private readonly queue: LatestTaskQueue<WorkerResult>;
  private revision = 0;
  private fieldReady = false;
  private ready = false;
  private paused = false;
  private tool: Tool = 'grab';
  private solverState: unknown = null;
  private previousPath: Point[] | null = null;
  private pin: { index: number; position: Point } | null = null;
  private runTimer = 0;
  private gesture: Gesture = null;
  private strokeSerial = 0;
  private keyboardTarget = 'path:17';
  private readonly fieldRollbacks = new Map<number, PlaygroundSnapshot>();
  private readonly pendingPresets = new Map<number, PendingPreset>();
  private readonly pendingImports = new Map<number, PendingImport>();
  private fieldRange: [number, number] = [1, 1];
  private elevationRange: [number, number] = [0, 0];

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = new PlaygroundCanvas(canvas, this.snapshot.scene, this.snapshot.path);
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
    const revision = this.revision;
    try {
      const response = await fetch(new URL('playground-preview.json', new URL('.', location.href)));
      if (!response.ok) return;
      const preview = (await response.json()) as {
        field?: number[];
        elevation?: number[];
        width?: number;
        height?: number;
        preset?: PresetResult['preset'];
        scene?: PresetResult['preset'];
      };
      if (revision !== this.revision || this.fieldReady) return;
      const scene = preview.scene ?? preview.preset;
      if (scene) this.snapshot = this.snapshotFromPreset(scene, 'random_hills', 0);
      if (preview.field && preview.width && preview.height) {
        this.canvas.setField(preview.field, preview.width, preview.height, preview.elevation);
        this.fieldRange = [Math.min(...preview.field), Math.max(...preview.field)];
      }
      if (!this.ready) this.setRuntime('Precomputed preview · starting solver…');
      this.render();
    } catch {
      /* The local placeholder remains interactive. */
    }
  }

  private bind() {
    for (const button of all<HTMLElement>('[data-playground-tool]'))
      button.onclick = () => this.setTool(button.dataset.playgroundTool as Tool);
    for (const button of all<HTMLElement>('[data-playground-preset]'))
      button.onclick = () =>
        this.requestPreset(button.dataset.playgroundPreset as PresetName, this.snapshot.seed);
    for (const button of all<HTMLElement>('[data-playground-brush]'))
      button.onclick = () =>
        this.setBrush(button.dataset.playgroundBrush === 'wall' ? 'wall' : 'hill');
    $('playground-randomize')?.addEventListener('click', () => {
      const next = this.snapshot.seed >= 2_147_483_646 ? 0 : this.snapshot.seed + 1;
      const name = this.snapshot.preset === 'imported' ? 'random_hills' : this.snapshot.preset;
      this.requestPreset(name, next);
    });
    const seed = $<HTMLInputElement>('playground-seed');
    seed?.addEventListener('change', () => {
      if (!Number.isSafeInteger(seed.valueAsNumber)) {
        this.setMessage('Seed must be an integer.', true);
        return;
      }
      const name = this.snapshot.preset === 'imported' ? 'random_hills' : this.snapshot.preset;
      this.requestPreset(name, seed.valueAsNumber);
    });
    $('playground-retry')?.addEventListener('click', () => void this.boot());
    $('playground-undo')?.addEventListener('click', () => this.undo());
    $('playground-redo')?.addEventListener('click', () => this.redo());
    $('playground-toggle')?.addEventListener('click', () => this.togglePause());
    $('playground-keep')?.addEventListener('click', () => this.keepPath());
    $('playground-reset')?.addEventListener('click', () => this.resetStraight());
    $('playground-fit')?.addEventListener('click', () => {
      this.canvas.draw();
      this.setStatus(this.paused ? 'Paused' : 'Adjusting');
    });
    $('playground-focus-canvas')?.addEventListener('click', () => this.canvas.canvas.focus());
    $('playground-save')?.addEventListener('click', () => this.save());
    $('playground-load')?.addEventListener('click', () => $('playground-file')?.click());
    $('playground-file')?.addEventListener('change', () => void this.loadFile());

    const points = $<HTMLInputElement>('playground-show-points'),
      elevation = $<HTMLInputElement>('playground-show-elevation'),
      previous = $<HTMLInputElement>('playground-show-previous');
    if (points)
      points.onchange = () => {
        this.canvas.showPoints = points.checked;
        this.canvas.draw();
      };
    if (elevation)
      elevation.onchange = () => {
        this.canvas.showElevation = elevation.checked;
        this.canvas.draw();
        this.updateLayerCaption();
      };
    if (previous)
      previous.onchange = () => {
        this.canvas.showPrevious = previous.checked;
        this.canvas.draw();
      };
    const resolution = $<HTMLSelectElement>('playground-resolution');
    if (resolution)
      resolution.onchange = () =>
        this.commitControlChange((draft) => {
          const count = Number(resolution.value) as 32 | 64 | 128;
          draft.settings.interior_points = count;
          draft.path = resamplePath(draft.path, count);
        });
    const keyboard = $<HTMLSelectElement>('playground-keyboard-target');
    if (keyboard)
      keyboard.onchange = () => {
        this.keyboardTarget = keyboard.value;
      };
    const radius = $<HTMLInputElement>('playground-brush-radius'),
      strength = $<HTMLInputElement>('playground-brush-strength');
    if (radius)
      radius.oninput = () => {
        this.snapshot.settings.brush_radius = (radius.valueAsNumber / 100) * this.sceneWidth();
        this.syncBrushValues();
      };
    if (strength)
      strength.oninput = () => {
        this.snapshot.settings.brush_strength = strength.valueAsNumber;
        this.syncBrushValues();
      };

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

  private sceneWidth() {
    return this.snapshot.scene.bounds[1] - this.snapshot.scene.bounds[0];
  }
  private clearActionMessage() {
    this.setMessage('');
  }

  private setTool(tool: Tool) {
    this.clearActionMessage();
    this.tool = tool;
    all<HTMLElement>('[data-playground-tool]').forEach((button) => {
      const active = button.dataset.playgroundTool === tool;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    const brush = $('playground-brush-section');
    if (brush) brush.hidden = tool !== 'paint';
    this.setStatus(
      this.paused
        ? 'Paused'
        : tool === 'grab'
          ? 'Adjusting'
          : tool === 'paint'
            ? 'Paint cost onto the landscape.'
            : 'Erase a hill or painted stroke.',
    );
  }

  private setBrush(brush: 'hill' | 'wall') {
    this.clearActionMessage();
    this.snapshot.settings.brush = brush;
    this.snapshot.settings.brush_radius = this.sceneWidth() * (brush === 'hill' ? 0.07 : 0.03);
    this.snapshot.settings.brush_strength = brush === 'hill' ? 6 : 40;
    all<HTMLElement>('[data-playground-brush]').forEach((button) => {
      const active = button.dataset.playgroundBrush === brush;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    this.syncBrushValues();
  }

  private snapshotFromPreset(
    preset: PresetResult['preset'],
    name: PresetName | 'imported',
    seed: number,
  ) {
    const oldWidth = this.sceneWidth(),
      newWidth = preset.bounds[1] - preset.bounds[0],
      radiusFraction = Math.max(
        0.02,
        Math.min(0.15, this.snapshot.settings.brush_radius / oldWidth),
      ),
      settings = {
        ...this.snapshot.settings,
        brush_radius: radiusFraction * newWidth,
      },
      start = preset.start as Point,
      end = preset.end as Point;
    return this.snapshotFromFile(
      validatePlaygroundV2({
        kind: 'path-playground',
        version: 2,
        bounds: preset.bounds,
        start,
        end,
        base_field: preset.base_field,
        gaussians: preset.gaussians,
        strokes: preset.strokes,
        path: straightPath(start, end, settings.interior_points),
        settings,
        ghosts: [],
        seed,
        metadata: preset.metadata,
      }),
      name,
    );
  }

  private snapshotFromFile(
    file: PlaygroundFileV2,
    preset: PresetName | 'imported' = 'imported',
  ): PlaygroundSnapshot {
    return {
      preset,
      seed: file.seed,
      scene: {
        bounds: file.bounds,
        start: file.start,
        end: file.end,
        base_field: file.base_field,
        gaussians: file.gaussians,
        strokes: file.strokes,
        metadata: file.metadata,
      },
      path: file.path,
      settings: file.settings,
      ghosts: file.ghosts,
      metrics: null,
    };
  }

  private requestPreset(name: PresetName, seed: number) {
    this.clearActionMessage();
    if (!Number.isSafeInteger(seed)) {
      this.setMessage('Seed must be an integer.', true);
      return;
    }
    if (!this.ready) {
      this.setMessage('The solver is still loading. You can edit the current scene meanwhile.');
      return;
    }
    const revision = ++this.revision;
    this.pendingPresets.clear();
    this.pendingPresets.set(revision, { before: cloneSnapshot(this.snapshot), name, seed });
    this.queue.clear();
    this.solverState = null;
    this.setStatus('Adjusting');
    this.queue.push({
      revision,
      run: () => this.client.request('playgroundPreset', { revision, name, seed }),
    });
  }

  private commitControlChange(change: (draft: PlaygroundSnapshot) => void) {
    this.clearActionMessage();
    const before = cloneSnapshot(this.snapshot);
    this.history.begin(before);
    change(this.snapshot);
    this.history.commit(true);
    this.previousPath = null;
    this.solverState = null;
    this.revision++;
    this.render();
    this.schedule('initialize');
  }

  private effectiveScene(): PlaygroundScene {
    if (this.gesture?.kind !== 'paint') return this.snapshot.scene;
    return {
      ...this.snapshot.scene,
      strokes: [...this.snapshot.scene.strokes, this.gesture.stroke],
    };
  }

  private workerScene(scene = this.effectiveScene()) {
    return {
      bounds: scene.bounds,
      start: scene.start,
      end: scene.end,
      base_field: scene.base_field,
      gaussians: scene.gaussians,
      strokes: scene.strokes,
      metadata: scene.metadata,
    };
  }

  private pointerDown(event: PointerEvent) {
    if (event.button !== 0) return;
    this.clearActionMessage();
    this.previousPath = null;
    const point = clampToBounds(
        this.canvas.fromClient(event.clientX, event.clientY),
        this.snapshot.scene.bounds,
      ),
      hitRadius = this.canvas.modelRadiusFromPixels(event.pointerType === 'touch' ? 28 : 18);
    if (this.tool === 'grab') {
      const route = this.canvas.nearestVisiblePoint(point),
        hill = this.canvas.nearestGaussian(point),
        hillHit =
          hill.index >= 0 &&
          hill.distance <= Math.max(hitRadius, this.snapshot.scene.gaussians[hill.index].width);
      this.history.begin(cloneSnapshot(this.snapshot));
      if (hillHit && hill.distance < route.distance) {
        this.gesture = {
          kind: 'gaussian',
          pointerId: event.pointerId,
          index: hill.index,
          changed: false,
        };
        this.keyboardTarget = `gaussian:${hill.index}`;
        this.canvas.heldGaussian = hill.index;
      } else if (route.index >= 0 && route.distance <= hitRadius) {
        this.gesture = {
          kind: 'grab',
          pointerId: event.pointerId,
          index: route.index,
          changed: false,
        };
        this.keyboardTarget = `path:${route.index}`;
        this.canvas.heldIndex = route.index;
        this.pin =
          route.index > 0 && route.index < this.snapshot.path.length - 1
            ? { index: route.index, position: this.snapshot.path[route.index] }
            : null;
        this.solverState = null;
        this.revision++;
        this.schedule('initialize');
      } else {
        this.history.commit(false);
        this.setStatus('Grab closer to the route or a hill.');
        return;
      }
    } else if (this.tool === 'paint') {
      const before = cloneSnapshot(this.snapshot),
        stroke: Stroke = {
          id: this.nextStrokeId(),
          points: [point],
          width: this.snapshot.settings.brush_radius,
          strength: this.snapshot.settings.brush_strength,
        };
      this.history.begin(before);
      this.gesture = { kind: 'paint', pointerId: event.pointerId, stroke, changed: true, before };
      this.canvas.activeStroke = stroke;
      this.fieldReady = false;
      this.solverState = null;
      this.revision++;
      this.fieldRollbacks.clear();
      this.fieldRollbacks.set(this.revision, before);
      this.schedule('field');
    } else {
      this.history.begin(cloneSnapshot(this.snapshot));
      let strokeIndex = -1;
      for (let i = this.snapshot.scene.strokes.length - 1; i >= 0; i--)
        if (strokeHit(this.snapshot.scene.strokes[i], point, hitRadius)) {
          strokeIndex = i;
          break;
        }
      const hill = this.canvas.nearestGaussian(point),
        hillHit =
          hill.index >= 0 &&
          hill.distance <= Math.max(hitRadius, this.snapshot.scene.gaussians[hill.index].width);
      let changed = false;
      if (strokeIndex >= 0) {
        this.snapshot.scene.strokes.splice(strokeIndex, 1);
        changed = true;
      } else if (hillHit) {
        this.snapshot.scene.gaussians.splice(hill.index, 1);
        changed = true;
      }
      this.gesture = { kind: 'erase', pointerId: event.pointerId, changed };
      if (changed) {
        this.fieldReady = false;
        this.solverState = null;
        this.revision++;
        this.render();
      }
    }
    this.canvas.canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
    this.render();
  }

  private pointerMove(event: PointerEvent) {
    if (!this.gesture || this.gesture.pointerId !== event.pointerId) return;
    const point = clampToBounds(
      this.canvas.fromClient(event.clientX, event.clientY),
      this.snapshot.scene.bounds,
    );
    if (this.gesture.kind === 'grab') {
      const index = this.gesture.index;
      this.snapshot.path = movePathNeighborhood(
        this.snapshot.path,
        index,
        point,
        this.snapshot.scene.bounds,
      );
      if (index === 0) this.snapshot.scene.start = point;
      if (index === this.snapshot.path.length - 1) this.snapshot.scene.end = point;
      this.pin =
        index > 0 && index < this.snapshot.path.length - 1 ? { index, position: point } : null;
      this.gesture.changed = true;
      this.solverState = null;
      this.revision++;
      this.schedule('initialize');
    } else if (this.gesture.kind === 'gaussian') {
      const hill = this.snapshot.scene.gaussians[this.gesture.index];
      hill.x = point[0];
      hill.y = point[1];
      this.gesture.changed = true;
      this.fieldReady = false;
      this.solverState = null;
      this.revision++;
      this.schedule('field');
    } else if (this.gesture.kind === 'paint') {
      const points = this.gesture.stroke.points,
        minimum = this.gesture.stroke.width * 0.12;
      if (distance(point, points.at(-1)!) >= minimum) points.push(point);
      this.canvas.activeStroke = this.gesture.stroke;
      this.fieldReady = false;
      this.solverState = null;
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
      this.fieldReady = false;
      this.solverState = null;
      this.revision++;
      this.fieldRollbacks.clear();
      this.fieldRollbacks.set(this.revision, gesture.before);
    } else if (gesture.kind === 'grab') {
      this.pin = null;
      this.canvas.heldIndex = null;
      this.solverState = null;
      this.revision++;
    } else if (gesture.kind === 'gaussian') {
      this.canvas.heldGaussian = null;
    }
    this.history.commit(gesture.changed);
    this.gesture = null;
    this.render();
    this.schedule(
      gesture.kind === 'paint' || gesture.kind === 'erase' || gesture.kind === 'gaussian'
        ? 'field'
        : 'initialize',
    );
  }

  private pointerCancel(event: PointerEvent) {
    if (!this.gesture || this.gesture.pointerId !== event.pointerId) return;
    const before = this.history.cancel();
    this.gesture = null;
    this.pin = null;
    this.canvas.activeStroke = null;
    this.canvas.heldIndex = null;
    this.canvas.heldGaussian = null;
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
    if (
      (event.key === 'Delete' || event.key === 'Backspace') &&
      /^(stroke|gaussian):/.test(this.keyboardTarget)
    ) {
      event.preventDefault();
      this.deleteKeyboardTarget();
      return;
    }
    if (!direction[event.key]) return;
    event.preventDefault();
    this.clearActionMessage();
    this.previousPath = null;
    const step = this.sceneWidth() * (event.shiftKey ? 0.01 : 0.0025),
      delta = direction[event.key],
      before = cloneSnapshot(this.snapshot);
    this.history.begin(before);
    if (this.keyboardTarget.startsWith('stroke:')) {
      const id = this.keyboardTarget.slice(7),
        stroke = this.snapshot.scene.strokes.find((item) => item.id === id);
      if (!stroke) return;
      stroke.points = stroke.points.map((point) =>
        clampToBounds(
          [point[0] + delta[0] * step, point[1] + delta[1] * step],
          this.snapshot.scene.bounds,
        ),
      );
      this.history.commit(true);
      this.invalidateField();
    } else if (this.keyboardTarget.startsWith('gaussian:')) {
      const index = Number(this.keyboardTarget.slice(9)),
        hill = this.snapshot.scene.gaussians[index];
      if (!hill) return;
      [hill.x, hill.y] = clampToBounds(
        [hill.x + delta[0] * step, hill.y + delta[1] * step],
        this.snapshot.scene.bounds,
      );
      this.history.commit(true);
      this.invalidateField();
    } else {
      const index = Math.max(
          0,
          Math.min(this.snapshot.path.length - 1, Number(this.keyboardTarget.slice(5)) || 0),
        ),
        old = this.snapshot.path[index],
        next = clampToBounds(
          [old[0] + delta[0] * step, old[1] + delta[1] * step],
          this.snapshot.scene.bounds,
        );
      this.snapshot.path = movePathNeighborhood(
        this.snapshot.path,
        index,
        next,
        this.snapshot.scene.bounds,
      );
      if (index === 0) this.snapshot.scene.start = next;
      if (index === this.snapshot.path.length - 1) this.snapshot.scene.end = next;
      this.history.commit(true);
      this.solverState = null;
      this.revision++;
      this.render();
      this.schedule('initialize');
    }
  }

  private deleteKeyboardTarget() {
    this.clearActionMessage();
    this.history.begin(cloneSnapshot(this.snapshot));
    let changed = false;
    if (this.keyboardTarget.startsWith('stroke:')) {
      const id = this.keyboardTarget.slice(7),
        index = this.snapshot.scene.strokes.findIndex((item) => item.id === id);
      if (index >= 0) {
        this.snapshot.scene.strokes.splice(index, 1);
        changed = true;
      }
    } else {
      const index = Number(this.keyboardTarget.slice(9));
      if (this.snapshot.scene.gaussians[index]) {
        this.snapshot.scene.gaussians.splice(index, 1);
        changed = true;
      }
    }
    this.history.commit(changed);
    if (changed) {
      this.keyboardTarget = `path:${Math.floor(this.snapshot.path.length / 2)}`;
      this.invalidateField();
    }
  }

  private invalidateField() {
    this.fieldReady = false;
    this.solverState = null;
    this.revision++;
    this.render();
    this.schedule('field');
  }

  private restore(snapshot: PlaygroundSnapshot) {
    this.snapshot = cloneSnapshot(snapshot);
    this.pin = null;
    this.solverState = null;
    this.previousPath = null;
    this.fieldReady = false;
    this.revision++;
    this.sync();
    this.render();
    this.schedule('field');
  }
  private undo() {
    this.clearActionMessage();
    const value = this.history.undo(cloneSnapshot(this.snapshot));
    if (value) this.restore(value);
  }
  private redo() {
    this.clearActionMessage();
    const value = this.history.redo(cloneSnapshot(this.snapshot));
    if (value) this.restore(value);
  }

  private resetStraight() {
    this.clearActionMessage();
    this.history.begin(cloneSnapshot(this.snapshot));
    this.snapshot.path = straightPath(
      this.snapshot.scene.start,
      this.snapshot.scene.end,
      this.snapshot.settings.interior_points,
    );
    this.snapshot.metrics = null;
    this.history.commit(true);
    this.previousPath = null;
    this.solverState = null;
    this.revision++;
    this.render();
    this.schedule('initialize');
  }

  private togglePause() {
    this.clearActionMessage();
    this.paused = !this.paused;
    window.clearTimeout(this.runTimer);
    this.queue.clear();
    this.solverState = null;
    this.revision++;
    this.render();
    if (!this.paused) this.schedule(this.fieldReady ? 'initialize' : 'field');
  }

  private keepPath() {
    this.clearActionMessage();
    const ghost: GhostRoute = {
      id: `path-${Date.now()}-${this.revision}`,
      path: this.snapshot.path.map(([x, y]) => [x, y]),
      route_cost: this.snapshot.metrics?.route_cost ?? null,
    };
    this.snapshot.ghosts = [...this.snapshot.ghosts, ghost].slice(-3);
    this.revision++;
    this.render();
    this.schedule('evaluate');
  }

  private save() {
    const file: PlaygroundFileV2 = {
      kind: 'path-playground',
      version: 2,
      bounds: this.snapshot.scene.bounds,
      start: this.snapshot.scene.start,
      end: this.snapshot.scene.end,
      base_field: this.snapshot.scene.base_field,
      gaussians: this.snapshot.scene.gaussians,
      strokes: this.snapshot.scene.strokes,
      path: this.snapshot.path,
      settings: this.snapshot.settings,
      ghosts: this.snapshot.ghosts,
      seed: this.snapshot.seed,
      metadata: this.snapshot.scene.metadata,
    };
    download('path-playground-v2.json', JSON.stringify(file, null, 2));
    this.setMessage('Scene saved.');
  }

  private async loadFile() {
    const input = $<HTMLInputElement>('playground-file'),
      file = input?.files?.[0];
    if (!file) return;
    this.clearActionMessage();
    try {
      if (file.size > 12_000_000) throw new Error('The file exceeds the 12 MB browser limit.');
      const raw = JSON.parse(await file.text());
      if (raw?.kind === 'path-playground' && raw?.version === 2)
        this.validateImport(this.snapshotFromFile(validatePlaygroundV2(raw)), 'Playground loaded.');
      else if (raw?.kind === 'path-playground' && raw?.version === 1)
        this.validateImport(this.adaptPlaygroundV1(raw), 'Playground v1 scene adapted.');
      else if (raw?.version === 2 && (raw?.scenario || raw?.field_x_m || raw?.bounds_m))
        this.adaptTerrainV2(raw?.scenario ?? raw);
      else this.validateImport(this.adaptGaussianV1(raw), 'Gaussian v1 scene adapted.');
    } catch (error) {
      this.setMessage(error instanceof Error ? error.message : String(error), true);
    } finally {
      if (input) input.value = '';
    }
  }

  private validateImport(candidate: PlaygroundSnapshot, message: string) {
    if (!this.ready) {
      this.setMessage('Wait for the solver, then load the file again.', true);
      return;
    }
    const revision = ++this.revision,
      before = cloneSnapshot(this.snapshot);
    this.pendingImports.clear();
    this.pendingImports.set(revision, { before, candidate, message });
    this.queue.clear();
    this.queue.push({
      revision,
      run: () => this.fieldRequest(revision, candidate.scene, candidate.path, candidate.ghosts),
    });
  }

  private adaptTerrainV2(scenario: unknown) {
    if (!this.ready) {
      this.setMessage('Wait for the solver, then load the terrain again.', true);
      return;
    }
    const revision = ++this.revision;
    this.pendingPresets.clear();
    this.pendingPresets.set(revision, {
      before: cloneSnapshot(this.snapshot),
      name: 'imported',
      seed: this.snapshot.seed,
    });
    this.queue.clear();
    this.queue.push({
      revision,
      run: () => this.client.request('playgroundAdaptTerrain', { revision, scenario }),
    });
  }

  private adaptPlaygroundV1(raw: any): PlaygroundSnapshot {
    if (
      !raw?.endpoints ||
      !isPoint(raw.endpoints.start) ||
      !isPoint(raw.endpoints.end) ||
      !Array.isArray(raw.path)
    )
      throw new Error('The playground v1 file is invalid.');
    const interior = raw.path.length - 2;
    if (![32, 64, 128].includes(interior))
      throw new Error('The playground v1 path resolution is unsupported.');
    const file = validatePlaygroundV2({
      kind: 'path-playground',
      version: 2,
      bounds: raw.bounds,
      start: raw.endpoints.start,
      end: raw.endpoints.end,
      base_field: UNIFORM_BASE,
      gaussians: [],
      strokes: raw.strokes ?? [],
      path: raw.path,
      settings: {
        ...DEFAULT_SETTINGS,
        interior_points: interior,
        brush_radius: raw.settings?.brush_radius ?? 0.84,
        brush_strength: raw.settings?.brush_strength ?? 6,
      },
      ghosts: raw.ghosts ?? [],
      seed: 0,
      metadata: { imported_from: 'path-playground-v1' },
    });
    return this.snapshotFromFile(file);
  }

  private adaptGaussianV1(value: unknown): PlaygroundSnapshot {
    const legacy = validateScene(value) as Scene,
      start = legacy.start as Point,
      end = legacy.end as Point;
    const points: Point[] = [
        start,
        end,
        ...legacy.obstacles.map((item) => [item.x, item.y] as Point),
      ],
      xs = points.map((p) => p[0]),
      ys = points.map((p) => p[1]);
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1),
      padding = span * 0.15;
    const bounds: [number, number, number, number] = [
      Math.min(...xs) - padding,
      Math.max(...xs) + padding,
      Math.min(...ys) - padding,
      Math.max(...ys) + padding,
    ];
    const settings = { ...DEFAULT_SETTINGS };
    return this.snapshotFromFile(
      validatePlaygroundV2({
        kind: 'path-playground',
        version: 2,
        bounds,
        start,
        end,
        base_field: UNIFORM_BASE,
        gaussians: legacy.obstacles.map(({ x, y, weight, width }) => ({ x, y, weight, width })),
        strokes: [],
        path: straightPath(start, end, settings.interior_points),
        settings,
        ghosts: [],
        seed: 0,
        metadata: { imported_from: 'gaussian-v1' },
      }),
    );
  }

  private nextStrokeId() {
    const ids = new Set(this.snapshot.scene.strokes.map(({ id }) => id));
    let id: string;
    do id = `stroke-${++this.strokeSerial}`;
    while (ids.has(id));
    return id;
  }

  private async boot() {
    window.clearTimeout(this.runTimer);
    this.ready = false;
    this.queue.clear();
    this.client.start();
    this.setRuntime('Starting solver…');
    const retry = $('playground-retry');
    if (retry) retry.hidden = true;
    this.sync();
    const revisionAtBoot = this.revision;
    try {
      await Promise.race([
        this.client.request('playgroundBoot', { base: new URL('.', location.href).href }),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Loading timed out.')), 90_000),
        ),
      ]);
      this.ready = true;
      this.setRuntime('Ready · runs on this device');
      this.sync();
      if (this.revision === revisionAtBoot && revisionAtBoot === 0)
        this.requestPreset('random_hills', 0);
      else this.schedule('field');
    } catch (error) {
      this.failWorker(error instanceof Error ? error : new Error(String(error)));
    }
  }

  private failWorker(error: Error) {
    this.ready = false;
    this.client.stop(error);
    this.setRuntime(`Solver unavailable. ${error.message}`);
    const retry = $('playground-retry');
    if (retry) retry.hidden = false;
    this.setMessage('The canvas remains editable. Retry when the runtime is available.', true);
    this.sync();
  }

  private fieldRequest(
    revision: number,
    scene = this.effectiveScene(),
    path = this.snapshot.path,
    ghosts = this.snapshot.ghosts,
  ) {
    return this.client.request('playgroundField', {
      revision,
      scene: this.workerScene(scene),
      width: 144,
      height: 96,
      paths: [path, ...ghosts.map((ghost) => ghost.path)],
    });
  }

  private schedule(kind: 'field' | 'initialize' | 'step' | 'evaluate') {
    if (!this.ready) return;
    window.clearTimeout(this.runTimer);
    const revision = this.revision;
    if (kind === 'field' || !this.fieldReady) {
      this.queue.push({ revision, run: () => this.fieldRequest(revision) });
      return;
    }
    if (kind === 'evaluate') {
      this.queue.push({
        revision,
        run: () =>
          this.client.request('playgroundEvaluate', {
            revision,
            scene: this.workerScene(),
            paths: [this.snapshot.path, ...this.snapshot.ghosts.map((ghost) => ghost.path)],
            state: this.solverState,
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
          iterations: 1,
        }),
    });
  }

  private accept(result: WorkerResult, revision: number) {
    if (revision !== this.revision || !('revision' in result)) return;
    if ('preset' in result) {
      const pending = this.pendingPresets.get(revision);
      if (!pending) return;
      this.pendingPresets.delete(revision);
      this.history.begin(pending.before);
      this.snapshot = this.snapshotFromPreset(result.preset, pending.name, pending.seed);
      this.history.commit(true);
      this.fieldReady = false;
      this.solverState = null;
      this.previousPath = null;
      this.sync();
      this.render();
      const resultMessage =
        'message' in result && typeof result.message === 'string'
          ? result.message
          : pending.message;
      if (resultMessage) this.setMessage(resultMessage);
      this.fieldRollbacks.set(revision, pending.before);
      this.schedule('field');
      return;
    }
    if ('field' in result) {
      const field = result as FieldResult,
        pending = this.pendingImports.get(revision);
      if (pending) {
        this.pendingImports.delete(revision);
        this.history.begin(pending.before);
        this.snapshot = pending.candidate;
        this.history.commit(true);
        this.previousPath = null;
        this.setMessage(pending.message);
        this.sync();
      }
      this.fieldRollbacks.delete(revision);
      this.fieldReady = true;
      this.canvas.setField(field.field, field.width, field.height, field.elevation);
      this.fieldRange = [field.field_min, field.field_max];
      this.elevationRange = [field.elevation_min, field.elevation_max];
      field.ghost_costs?.forEach((cost, index) => {
        if (index === 0 && field.metrics) this.snapshot.metrics = field.metrics;
        else if (index > 0 && this.snapshot.ghosts[index - 1])
          this.snapshot.ghosts[index - 1].route_cost = cost;
      });
      const elevationToggle = $<HTMLInputElement>('playground-show-elevation');
      if (elevationToggle)
        elevationToggle.disabled =
          !field.elevation?.length || field.elevation_min === field.elevation_max;
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
      if (!this.paused) this.schedule('initialize');
      return;
    }
    if ('path' in result) {
      const step = result as StepResult;
      this.previousPath = this.snapshot.path.map(([x, y]) => [x, y]);
      this.snapshot.path = step.path;
      this.snapshot.metrics = step.metrics;
      this.solverState = step.state ?? null;
      this.render();
      const canRun = !this.gesture || ['grab', 'paint', 'gaussian'].includes(this.gesture.kind);
      if (!this.paused && step.metrics.status === 'running' && canRun)
        this.runTimer = window.setTimeout(() => this.schedule('step'), 16);
    }
  }

  private reject(error: Error, revision: number) {
    if (revision !== this.revision) return;
    const pendingPreset = this.pendingPresets.get(revision);
    if (pendingPreset) {
      this.pendingPresets.delete(revision);
      this.setMessage(error.message, true);
      this.setStatus('Stopped — try reshaping');
      return;
    }
    const pendingImport = this.pendingImports.get(revision);
    if (pendingImport) {
      this.pendingImports.delete(revision);
      this.setMessage(`Import rejected. ${error.message}`, true);
      return;
    }
    const before = this.fieldRollbacks.get(revision);
    if (before) {
      this.fieldRollbacks.delete(revision);
      this.history.cancel();
      this.gesture = null;
      this.canvas.activeStroke = null;
      this.restore(before);
      this.setMessage(`Edit rejected. ${error.message}`, true);
      return;
    }
    this.setMessage(error.message, true);
    this.setStatus('Stopped — try reshaping');
  }

  private render() {
    this.canvas.scene = this.snapshot.scene;
    this.canvas.path = this.snapshot.path;
    this.canvas.ghosts = this.snapshot.ghosts;
    this.canvas.previousPath = this.previousPath;
    this.canvas.draw();
    const metric = $('playground-travel-cost');
    if (metric)
      metric.textContent = `${formatMetric(this.snapshot.metrics?.route_cost)}${this.isTerrain() && this.snapshot.metrics ? ' s' : ''}`;
    if ($('playground-ghost-count'))
      $('playground-ghost-count')!.textContent = `${this.snapshot.ghosts.length} / 3`;
    const list = $('playground-ghosts');
    if (list)
      list.textContent = this.snapshot.ghosts.length
        ? this.snapshot.ghosts
            .map(
              (ghost, i) =>
                `Path ${i + 1}: ${formatMetric(ghost.route_cost ?? undefined)}${this.isTerrain() && ghost.route_cost !== null ? ' s' : ''}`,
            )
            .join(' · ')
        : 'No retained paths yet.';
    if (this.paused) this.setStatus('Paused');
    else if (!this.snapshot.metrics || this.snapshot.metrics.status === 'running')
      this.setStatus('Adjusting');
    else if (this.snapshot.metrics.status === 'converged') this.setStatus('Settled');
    else this.setStatus('Stopped — try reshaping');
    this.updateLayerCaption();
    this.updateAttribution();
    this.sync();
  }

  private isTerrain() {
    return this.snapshot.scene.base_field.kind === 'terrain';
  }

  private updateLayerCaption() {
    const terrain = this.isTerrain(),
      elevation = this.canvas.showElevation && this.canvas.elevation.length > 0,
      name = $('playground-layer-name'),
      range = $('playground-layer-range'),
      scale = $('playground-map-scale');
    if (name) name.textContent = elevation ? 'Elevation' : 'Cost';
    if (range) {
      const values = elevation ? this.elevationRange : this.fieldRange,
        suffix = elevation ? ' m' : terrain ? ' s/m' : '×';
      range.textContent = `${formatMetric(values[0], 2)}–${formatMetric(values[1], 2)}${suffix}`;
    }
    if (scale)
      scale.textContent = terrain
        ? `${formatMetric(this.sceneWidth(), 0)} m wide`
        : `${formatMetric(this.sceneWidth(), 1)} illustrative units wide`;
  }

  private updateAttribution() {
    const element = $('playground-attribution');
    if (!element) return;
    element.replaceChildren();
    if (this.snapshot.scene.base_field.kind !== 'terrain') {
      element.hidden = true;
      return;
    }
    const scenario = this.snapshot.scene.base_field.scenario as Record<string, any>,
      provenance = scenario.provenance as Record<string, unknown> | undefined,
      attribution = typeof provenance?.attribution === 'string' ? provenance.attribution : '',
      source =
        typeof provenance?.registry_url === 'string'
          ? provenance.registry_url
          : typeof provenance?.attribution_requirements_url === 'string'
            ? provenance.attribution_requirements_url
            : '';
    let safeSource = '';
    if (source) {
      try {
        const parsed = new URL(source);
        if (parsed.protocol === 'http:' || parsed.protocol === 'https:') safeSource = parsed.href;
      } catch {
        // Imported provenance may omit or contain an invalid source URL.
      }
    }
    if (!attribution && !safeSource) {
      element.hidden = true;
      return;
    }
    element.append(
      document.createTextNode(
        `Illustrative travel-cost model${attribution ? ` · ${attribution}` : ''} `,
      ),
    );
    if (safeSource) {
      const link = document.createElement('a');
      link.href = safeSource;
      link.target = '_blank';
      link.rel = 'noreferrer';
      link.textContent = 'Source ↗';
      element.append(link);
    }
    element.hidden = false;
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
    const seed = $<HTMLInputElement>('playground-seed');
    const randomize = $<HTMLButtonElement>('playground-randomize'),
      seeded = new Set([
        'random_hills',
        'ridge_pass',
        'competing_corridors',
        'dead_ends',
        'correlated_roughness',
      ]).has(this.snapshot.preset);
    if (seed) {
      seed.value = String(this.snapshot.seed);
      seed.disabled = !seeded;
    }
    if (randomize) randomize.disabled = !seeded;
    all<HTMLElement>('[data-playground-preset]').forEach((button) =>
      button.classList.toggle('active', button.dataset.playgroundPreset === this.snapshot.preset),
    );
    all<HTMLElement>('[data-playground-brush]').forEach((button) => {
      const active = button.dataset.playgroundBrush === this.snapshot.settings.brush;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    const target = $<HTMLSelectElement>('playground-keyboard-target');
    if (target) {
      const middle = Math.floor(this.snapshot.path.length / 2);
      target.replaceChildren(
        new Option('Start', 'path:0'),
        new Option('Path midpoint', `path:${middle}`),
        new Option('Destination', `path:${this.snapshot.path.length - 1}`),
        ...this.snapshot.scene.gaussians.map(
          (_, index) => new Option(`Hill ${index + 1}`, `gaussian:${index}`),
        ),
        ...this.snapshot.scene.strokes.map(
          (stroke, index) => new Option(`Painted stroke ${index + 1}`, `stroke:${stroke.id}`),
        ),
      );
      if ([...target.options].some((option) => option.value === this.keyboardTarget))
        target.value = this.keyboardTarget;
      else {
        this.keyboardTarget = `path:${middle}`;
        target.value = this.keyboardTarget;
      }
    }
    this.syncBrushValues();
  }

  private syncBrushValues() {
    const radius = $<HTMLInputElement>('playground-brush-radius'),
      strength = $<HTMLInputElement>('playground-brush-strength'),
      radiusValue = $('playground-brush-radius-value'),
      strengthValue = $('playground-brush-strength-value');
    const percent = (100 * this.snapshot.settings.brush_radius) / this.sceneWidth();
    if (radius) radius.value = String(Math.max(2, Math.min(15, percent)));
    if (strength) strength.value = String(this.snapshot.settings.brush_strength);
    if (radiusValue) radiusValue.textContent = `${percent.toFixed(0)}%`;
    if (strengthValue)
      strengthValue.textContent = `${this.snapshot.settings.brush_strength.toFixed(0)}×`;
  }

  private setRuntime(message: string) {
    const element = $('playground-runtime');
    if (element) element.textContent = message;
  }
  private setStatus(message: string) {
    const element = $('playground-status');
    if (element) element.textContent = message;
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
