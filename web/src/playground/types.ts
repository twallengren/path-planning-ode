export type Point = [number, number];
export type Bounds = [number, number, number, number];
export type Tool = 'grab' | 'paint' | 'erase';
export type Method = 'descent' | 'newton';

export type Stroke = {
  id: string;
  points: Point[];
  width: number;
  strength: number;
};

export type Gaussian = {
  x: number;
  y: number;
  weight: number;
  width: number;
};

export type PlaygroundSettings = {
  interior_points: 32 | 64 | 128;
  method: Method;
  batch_iterations: number;
  tolerance: number;
  brush_radius: number;
  brush_strength: number;
};

export type Metrics = {
  energy: number;
  route_cost: number;
  free_gradient_norm: number;
  scaled_free_gradient_norm?: number;
  ode_residual_norm: number;
  iteration: number;
  status: string;
  elapsed_seconds?: number;
};

export type GhostRoute = {
  id: string;
  path: Point[];
  route_cost: number | null;
};

export type PlaygroundScene = {
  bounds: Bounds;
  start: Point;
  end: Point;
  strokes: Stroke[];
  gaussians: Gaussian[];
};

export type PlaygroundSnapshot = {
  scene: PlaygroundScene;
  path: Point[];
  settings: PlaygroundSettings;
  ghosts: GhostRoute[];
  metrics: Metrics | null;
};

export type PlaygroundFile = {
  kind: 'path-playground';
  version: 1;
  strokes: Stroke[];
  endpoints: { start: Point; end: Point };
  bounds: Bounds;
  path: Point[];
  settings: PlaygroundSettings;
  ghosts: GhostRoute[];
};

export type Pin = { index: number; position: Point } | null;

export type FieldResult = {
  revision: number;
  bounds: Bounds;
  width: number;
  height: number;
  field: number[];
  gaussians: Gaussian[];
  metrics?: Metrics;
  ghost_costs?: number[];
};

export type StepResult = {
  revision: number;
  path: Point[];
  metrics: Metrics;
};

export type EvaluationResult = { revision: number; ghost_costs: number[] };

export type WorkerResult = FieldResult | StepResult | EvaluationResult | Record<string, never>;

export const DEFAULT_BOUNDS: Bounds = [-6, 6, -4, 4];

export const DEFAULT_SETTINGS: PlaygroundSettings = {
  interior_points: 32,
  method: 'descent',
  batch_iterations: 1,
  tolerance: 1e-6,
  brush_radius: 0.72,
  brush_strength: 10,
};

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
const isPoint = (value: unknown): value is Point =>
  Array.isArray(value) && value.length === 2 && value.every(finite);

export function clonePoint(point: Point): Point {
  return [point[0], point[1]];
}

export function cloneSnapshot(snapshot: PlaygroundSnapshot): PlaygroundSnapshot {
  return structuredClone(snapshot);
}

function validateBounds(value: unknown): Bounds {
  if (!Array.isArray(value) || value.length !== 4 || !value.every(finite))
    throw new Error('The playground bounds are invalid.');
  const bounds = value as Bounds;
  if (!(bounds[0] < bounds[1] && bounds[2] < bounds[3]))
    throw new Error('The playground bounds must have positive width and height.');
  return [...bounds];
}

function validatePath(value: unknown, label = 'path'): Point[] {
  if (!Array.isArray(value) || value.length < 2 || value.length > 130 || !value.every(isPoint))
    throw new Error(`The ${label} is invalid.`);
  return value.map(clonePoint);
}

function validateSettings(value: unknown): PlaygroundSettings {
  if (!value || typeof value !== 'object') throw new Error('Solver settings are missing.');
  const settings = value as Partial<PlaygroundSettings>;
  if (![32, 64, 128].includes(Number(settings.interior_points)))
    throw new Error('Path resolution must be 32, 64, or 128 interior points.');
  if (settings.method !== 'descent' && settings.method !== 'newton')
    throw new Error('The solver method is invalid.');
  if (
    !Number.isInteger(settings.batch_iterations) ||
    Number(settings.batch_iterations) < 1 ||
    Number(settings.batch_iterations) > 20 ||
    !finite(settings.tolerance) ||
    settings.tolerance <= 0 ||
    !finite(settings.brush_radius) ||
    settings.brush_radius <= 0 ||
    !finite(settings.brush_strength) ||
    settings.brush_strength < 2 ||
    settings.brush_strength > 100
  )
    throw new Error('One or more solver settings are outside the supported range.');
  return { ...settings } as PlaygroundSettings;
}

function validateStrokes(value: unknown): Stroke[] {
  if (!Array.isArray(value) || value.length > 1_000) throw new Error('The stroke list is invalid.');
  const strokes = value.map((candidate, index) => {
    if (!candidate || typeof candidate !== 'object') throw new Error('A stroke is invalid.');
    const stroke = candidate as Partial<Stroke>;
    if (
      !Array.isArray(stroke.points) ||
      !stroke.points.length ||
      stroke.points.length > 10_000 ||
      !stroke.points.every(isPoint) ||
      !finite(stroke.width) ||
      stroke.width <= 0 ||
      !finite(stroke.strength) ||
      stroke.strength < 1 ||
      stroke.strength > 101
    )
      throw new Error('A stroke contains invalid points or brush settings.');
    return {
      id: typeof stroke.id === 'string' && stroke.id ? stroke.id.slice(0, 100) : `stroke-${index}`,
      points: stroke.points.map(clonePoint),
      width: stroke.width,
      strength: stroke.strength,
    };
  });
  if (new Set(strokes.map((stroke) => stroke.id)).size !== strokes.length)
    throw new Error('Stroke ids must be unique.');
  return strokes;
}

export function validatePlaygroundFile(value: unknown): PlaygroundFile {
  if (!value || typeof value !== 'object') throw new Error('Expected a playground file.');
  const file = value as Partial<PlaygroundFile>;
  if (file.kind !== 'path-playground' || file.version !== 1)
    throw new Error('Expected a kind:path-playground version 1 file.');
  if (!file.endpoints || !isPoint(file.endpoints.start) || !isPoint(file.endpoints.end))
    throw new Error('The playground endpoints are invalid.');
  const path = validatePath(file.path);
  const settings = validateSettings(file.settings);
  if (path.length !== settings.interior_points + 2)
    throw new Error('The saved path resolution does not match its solver settings.');
  if (
    path[0][0] !== file.endpoints.start[0] ||
    path[0][1] !== file.endpoints.start[1] ||
    path.at(-1)![0] !== file.endpoints.end[0] ||
    path.at(-1)![1] !== file.endpoints.end[1]
  )
    throw new Error('The saved path endpoints do not match the scene endpoints.');
  const ghosts = !Array.isArray(file.ghosts)
    ? []
    : file.ghosts.slice(-3).map((candidate, index) => {
        if (!candidate || typeof candidate !== 'object')
          throw new Error('A ghost route is invalid.');
        const ghost = candidate as Partial<GhostRoute>;
        return {
          id: typeof ghost.id === 'string' && ghost.id ? ghost.id.slice(0, 100) : `ghost-${index}`,
          path: validatePath(ghost.path, 'ghost route'),
          route_cost: finite(ghost.route_cost) ? ghost.route_cost : null,
        };
      });
  return {
    kind: 'path-playground',
    version: 1,
    strokes: validateStrokes(file.strokes),
    endpoints: { start: clonePoint(file.endpoints.start), end: clonePoint(file.endpoints.end) },
    bounds: validateBounds(file.bounds),
    path,
    settings,
    ghosts,
  };
}
