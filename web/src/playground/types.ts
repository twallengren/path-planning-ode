export type Point = [number, number];
/** Canvas order: xmin, xmax, ymin, ymax. */
export type Bounds = [number, number, number, number];
/** Terrain order: xmin, ymin, xmax, ymax. */
export type TerrainBounds = [number, number, number, number];
export type Tool = 'grab' | 'paint' | 'erase';
export type BrushPreset = 'hill' | 'wall';
export type PresetName =
  | 'blank'
  | 'random_hills'
  | 'corridor'
  | 'slalom'
  | 'ridge_pass'
  | 'competing_corridors'
  | 'dead_ends'
  | 'correlated_roughness'
  | 'mount_tamalpais';

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

export type UniformBaseField = { kind: 'uniform'; cost: number; elevation: number };
export type TerrainBaseField = {
  kind: 'terrain';
  scenario: Readonly<Record<string, unknown>>;
  scenario_hash: string;
};
export type BaseField = UniformBaseField | TerrainBaseField;

export type PlaygroundSettings = {
  interior_points: 32 | 64 | 128;
  method: 'auto';
  batch_iterations: 1;
  tolerance: number;
  max_iterations: number;
  brush: BrushPreset;
  brush_radius: number;
  brush_strength: number;
};

export type Metrics = {
  energy: number;
  normalized_energy?: number;
  route_cost: number;
  free_gradient_norm: number;
  scaled_free_gradient_norm?: number;
  ode_residual_norm: number;
  scaled_ode_residual_norm?: number;
  iteration: number;
  status: string;
  phase?: string;
  phase_reason?: string;
  elapsed_seconds?: number;
};

export type GhostRoute = { id: string; path: Point[]; route_cost: number | null };

export type PlaygroundScene = {
  bounds: Bounds;
  start: Point;
  end: Point;
  base_field: BaseField;
  gaussians: Gaussian[];
  strokes: Stroke[];
  metadata: Record<string, unknown>;
};

export type PlaygroundSnapshot = {
  preset: PresetName | 'imported';
  seed: number;
  scene: PlaygroundScene;
  path: Point[];
  settings: PlaygroundSettings;
  ghosts: GhostRoute[];
  metrics: Metrics | null;
};

export type PlaygroundFileV2 = {
  kind: 'path-playground';
  version: 2;
  bounds: Bounds;
  start: Point;
  end: Point;
  base_field: BaseField;
  gaussians: Gaussian[];
  strokes: Stroke[];
  path: Point[];
  settings: PlaygroundSettings;
  ghosts: GhostRoute[];
  seed: number;
  metadata: Record<string, unknown>;
};

export type Pin = { index: number; position: Point } | null;

export type PresetResult = {
  revision: number;
  preset: {
    name: PresetName;
    bounds: Bounds;
    start: Point;
    end: Point;
    base_field: BaseField;
    gaussians: Gaussian[];
    strokes: Stroke[];
    metadata: Record<string, unknown>;
  };
};

export type FieldResult = {
  revision: number;
  bounds: Bounds;
  width: number;
  height: number;
  field: number[];
  field_min: number;
  field_max: number;
  elevation?: number[];
  elevation_min: number;
  elevation_max: number;
  gaussians?: Gaussian[];
  obstacles?: Gaussian[];
  metrics?: Metrics;
  ghost_costs?: number[];
  field_hash?: string;
};

export type StepResult = {
  revision: number;
  path: Point[];
  metrics: Metrics;
  state?: unknown;
};
export type EvaluationResult = { revision: number; ghost_costs: number[] };
export type TerrainAdaptResult = {
  revision: number;
  preset: PresetResult['preset'];
  message: string;
};
export type WorkerResult =
  | FieldResult
  | StepResult
  | EvaluationResult
  | PresetResult
  | TerrainAdaptResult
  | Record<string, never>;

export const DEFAULT_BOUNDS: Bounds = [-6, 6, -4, 4];
export const DEFAULT_SETTINGS: PlaygroundSettings = {
  interior_points: 32,
  method: 'auto',
  batch_iterations: 1,
  tolerance: 1e-5,
  max_iterations: 2000,
  brush: 'hill',
  brush_radius: 0.84,
  brush_strength: 6,
};
export const UNIFORM_BASE: UniformBaseField = Object.freeze({
  kind: 'uniform',
  cost: 1,
  elevation: 0,
});

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
export const isPoint = (value: unknown): value is Point =>
  Array.isArray(value) && value.length === 2 && value.every(finite);

export function canvasBoundsFromTerrain(bounds: unknown): Bounds {
  if (!Array.isArray(bounds) || bounds.length !== 4 || !bounds.every(finite))
    throw new Error('Terrain bounds are invalid.');
  const [xmin, ymin, xmax, ymax] = bounds as TerrainBounds;
  if (!(xmin < xmax && ymin < ymax)) throw new Error('Terrain bounds must have positive area.');
  return [xmin, xmax, ymin, ymax];
}

export function terrainBoundsFromCanvas([xmin, xmax, ymin, ymax]: Bounds): TerrainBounds {
  return [xmin, ymin, xmax, ymax];
}

export function clonePoint([x, y]: Point): Point {
  return [x, y];
}

function cloneScene(scene: PlaygroundScene): PlaygroundScene {
  return {
    bounds: [...scene.bounds],
    start: clonePoint(scene.start),
    end: clonePoint(scene.end),
    // Base terrain arrays are immutable and structurally shared across 50-step history.
    base_field: scene.base_field,
    gaussians: scene.gaussians.map((item) => ({ ...item })),
    strokes: scene.strokes.map((stroke) => ({
      ...stroke,
      points: stroke.points.map(clonePoint),
    })),
    metadata: structuredClone(scene.metadata),
  };
}

export function cloneSnapshot(snapshot: PlaygroundSnapshot): PlaygroundSnapshot {
  return {
    preset: snapshot.preset,
    seed: snapshot.seed,
    scene: cloneScene(snapshot.scene),
    path: snapshot.path.map(clonePoint),
    settings: { ...snapshot.settings },
    ghosts: snapshot.ghosts.map((ghost) => ({
      ...ghost,
      path: ghost.path.map(clonePoint),
    })),
    metrics: snapshot.metrics ? { ...snapshot.metrics } : null,
  };
}

function validateBounds(value: unknown): Bounds {
  if (!Array.isArray(value) || value.length !== 4 || !value.every(finite))
    throw new Error('The playground bounds are invalid.');
  const bounds = value as Bounds;
  if (!(bounds[0] < bounds[1] && bounds[2] < bounds[3]))
    throw new Error('The playground bounds must have positive width and height.');
  return [...bounds];
}

export function validatePath(value: unknown, label = 'path'): Point[] {
  if (!Array.isArray(value) || value.length < 2 || value.length > 130 || !value.every(isPoint))
    throw new Error(`The ${label} is invalid.`);
  return value.map(clonePoint);
}

function validateSettings(value: unknown): PlaygroundSettings {
  if (!value || typeof value !== 'object') throw new Error('Solver settings are missing.');
  const input = value as Record<string, unknown>;
  const interior = Number(input.interior_points);
  if (![32, 64, 128].includes(interior))
    throw new Error('Path resolution must be 32, 64, or 128 interior points.');
  const tolerance = finite(input.tolerance) && input.tolerance > 0 ? input.tolerance : 1e-5;
  const maxIterations = Number(input.max_iterations ?? 2000);
  if (!Number.isInteger(maxIterations) || maxIterations < 1 || maxIterations > 2000)
    throw new Error('The iteration limit is invalid.');
  return {
    interior_points: interior as 32 | 64 | 128,
    method: 'auto',
    batch_iterations: 1,
    tolerance,
    max_iterations: maxIterations,
    brush: input.brush === 'wall' ? 'wall' : 'hill',
    brush_radius: finite(input.brush_radius) && input.brush_radius > 0 ? input.brush_radius : 0.84,
    brush_strength:
      finite(input.brush_strength) && input.brush_strength >= 1 && input.brush_strength <= 101
        ? input.brush_strength
        : 6,
  };
}

function validateStrokes(value: unknown): Stroke[] {
  if (!Array.isArray(value) || value.length > 1000) throw new Error('The stroke list is invalid.');
  const result = value.map((candidate, index) => {
    if (!candidate || typeof candidate !== 'object') throw new Error('A stroke is invalid.');
    const stroke = candidate as Record<string, unknown>;
    if (
      !Array.isArray(stroke.points) ||
      !stroke.points.length ||
      stroke.points.length > 10000 ||
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
  if (new Set(result.map(({ id }) => id)).size !== result.length)
    throw new Error('Stroke ids must be unique.');
  return result;
}

function validateGaussians(value: unknown): Gaussian[] {
  if (!Array.isArray(value) || value.length > 256) throw new Error('The hill list is invalid.');
  return value.map((candidate) => {
    if (!candidate || typeof candidate !== 'object') throw new Error('A hill is invalid.');
    const item = candidate as Record<string, unknown>;
    if (
      !finite(item.x) ||
      !finite(item.y) ||
      !finite(item.weight) ||
      item.weight < 0 ||
      !finite(item.width) ||
      item.width <= 0
    )
      throw new Error('A hill has invalid geometry or cost.');
    return { x: item.x, y: item.y, weight: item.weight, width: item.width };
  });
}

function validateBaseField(value: unknown): BaseField {
  if (!value || typeof value !== 'object') throw new Error('The base field is missing.');
  const base = value as Record<string, unknown>;
  if (base.kind === 'uniform') {
    if (
      !finite(base.cost) ||
      base.cost <= 0 ||
      !finite(base.elevation) ||
      Object.keys(base).some((key) => !['kind', 'cost', 'elevation'].includes(key))
    )
      throw new Error('The uniform base field is invalid.');
    return Object.freeze({ kind: 'uniform', cost: base.cost, elevation: base.elevation });
  }
  if (
    base.kind !== 'terrain' ||
    !base.scenario ||
    typeof base.scenario !== 'object' ||
    typeof base.scenario_hash !== 'string' ||
    !base.scenario_hash
  )
    throw new Error('The terrain base field is invalid.');
  return Object.freeze({
    kind: 'terrain',
    scenario: Object.freeze(structuredClone(base.scenario as Record<string, unknown>)),
    scenario_hash: base.scenario_hash,
  });
}

function validateGhosts(value: unknown): GhostRoute[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error('The retained paths are invalid.');
  const ghosts = value.slice(-3).map((candidate, index) => {
    if (!candidate || typeof candidate !== 'object') throw new Error('A retained path is invalid.');
    const ghost = candidate as Record<string, unknown>;
    return {
      id: typeof ghost.id === 'string' && ghost.id ? ghost.id.slice(0, 100) : `ghost-${index}`,
      path: validatePath(ghost.path, 'retained path'),
      route_cost: finite(ghost.route_cost) ? ghost.route_cost : null,
    };
  });
  if (new Set(ghosts.map(({ id }) => id)).size !== ghosts.length)
    throw new Error('Retained path ids must be unique.');
  return ghosts;
}

export function validatePlaygroundV2(value: unknown): PlaygroundFileV2 {
  if (!value || typeof value !== 'object') throw new Error('Expected a playground file.');
  const file = value as Record<string, unknown>;
  if (file.kind !== 'path-playground' || file.version !== 2)
    throw new Error('Expected a kind:path-playground version 2 file.');
  if (!isPoint(file.start) || !isPoint(file.end)) throw new Error('The endpoints are invalid.');
  const settings = validateSettings(file.settings),
    path = validatePath(file.path);
  if (path.length !== settings.interior_points + 2)
    throw new Error('The saved path resolution does not match its settings.');
  if (
    path[0][0] !== file.start[0] ||
    path[0][1] !== file.start[1] ||
    path.at(-1)![0] !== file.end[0] ||
    path.at(-1)![1] !== file.end[1]
  )
    throw new Error('The saved path endpoints do not match the scene endpoints.');
  if (!Number.isSafeInteger(file.seed)) throw new Error('The random seed must be an integer.');
  return {
    kind: 'path-playground',
    version: 2,
    bounds: validateBounds(file.bounds),
    start: clonePoint(file.start),
    end: clonePoint(file.end),
    base_field: validateBaseField(file.base_field),
    gaussians: validateGaussians(file.gaussians),
    strokes: validateStrokes(file.strokes),
    path,
    settings,
    ghosts: validateGhosts(file.ghosts),
    seed: file.seed as number,
    metadata:
      file.metadata && typeof file.metadata === 'object'
        ? structuredClone(file.metadata as Record<string, unknown>)
        : {},
  };
}
