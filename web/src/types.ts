export type Point = [number, number];
export type Guess = 'straight' | 'bend-x' | 'bend-y';
export type Obstacle = { x: number; y: number; weight: number; width: number };
export type Scene = {
  version: 1;
  start: Point;
  end: Point;
  obstacles: Obstacle[];
  guesses: Guess[];
  options: {
    interior_points: number;
    max_iterations: number;
    tolerance: number;
    mode: 'damped' | 'undamped';
  };
};
export type State = {
  path: Point[];
  iteration: number;
  residual_norm: number;
  energy: number;
  cost: number;
  length: number;
  status: 'running' | 'converged' | 'stagnated' | 'singular' | 'nonfinite' | 'iteration_limit';
  damping: number;
};
export type Bounds = [number, number, number, number];
export type Frame = {
  states: Record<string, State>;
  field?: number[];
  bounds?: Bounds;
  straight_cost?: number;
};

export type TerrainFamily =
  'ridge_pass' | 'competing_corridors' | 'dead_ends' | 'correlated_roughness';
export type TerrainLayer = 'elevation' | 'cost' | 'arrival';
export type TerrainMethod = 'euler_lagrange' | 'slsqp' | 'fast_marching';
export type TerrainScenario = {
  version: 2;
  name: string;
  bounds_m: [number, number, number, number];
  start_m: Point;
  goal_m: Point;
  field_x_m: number[];
  field_y_m: number[];
  elevation_m: number[][];
  log_slowness: number[][];
  barriers_geojson: Array<{ type: 'Polygon' | 'MultiPolygon'; coordinates: unknown }>;
  provenance: Record<string, unknown>;
  metadata: Record<string, unknown>;
};
export type TerrainConfig = {
  version: 2;
  method: TerrainMethod;
  initialization: string;
  interior_points: number;
  reference_grid_size: number;
  tolerance: number;
  max_iterations: number;
  time_limit_s: number;
  profile_samples: number;
  options: Record<string, unknown>;
};
export type RouteEvaluation = {
  cost_s: number | null;
  length_m: number;
  feasible: boolean;
  minimum_clearance_m: number | null;
  violations: string[];
  profile: {
    distance_m: number[];
    elevation_m: number[];
    slowness_s_per_m: number[];
    accumulated_cost_s: number[];
  };
  evaluation_time_s: number;
};
export type TerrainResult = {
  version: 2;
  method: TerrainMethod;
  initialization: string;
  route_m: Point[] | null;
  evaluated_cost_s: number | null;
  feasible: boolean;
  solver_success: boolean;
  termination_reason: string;
  diagnostics: Record<string, unknown>;
  timing_s: Record<string, number>;
  scenario_hash: string;
  config_hash: string;
  evaluation: RouteEvaluation | null;
};
export type TerrainBundle = {
  version: 2;
  scenario: TerrainScenario;
  configs?: TerrainConfig[];
  results?: TerrainResult[];
};

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
const point = (value: unknown): value is Point =>
  Array.isArray(value) && value.length === 2 && value.every(finite);
const digest = (value: unknown): value is string =>
  typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);

function finiteJson(value: unknown, depth = 0): boolean {
  if (depth > 12) return false;
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (finite(value)) return true;
  if (Array.isArray(value))
    return value.length <= 300_000 && value.every((v) => finiteJson(v, depth + 1));
  return (
    object(value) &&
    Object.entries(value).every(([key, v]) => key.length <= 200 && finiteJson(v, depth + 1))
  );
}

function samePoint(a: Point, b: Point, scale: number) {
  const tolerance = 1e-9 * Math.max(scale, 1);
  return Math.abs(a[0] - b[0]) <= tolerance && Math.abs(a[1] - b[1]) <= tolerance;
}

export function validateTerrainScenario(value: unknown): TerrainScenario {
  if (!value || typeof value !== 'object') throw new Error('Expected a terrain scenario object.');
  const s = value as TerrainScenario;
  if (s.version !== 2) throw new Error('Expected a version 2 terrain scenario.');
  if (
    typeof s.name !== 'string' ||
    !s.name ||
    s.name.length > 500 ||
    !Array.isArray(s.bounds_m) ||
    s.bounds_m.length !== 4 ||
    !s.bounds_m.every(finite)
  )
    throw new Error('Terrain name and bounds are required.');
  if (!point(s.start_m) || !point(s.goal_m)) throw new Error('Terrain endpoints are invalid.');
  const [xmin, ymin, xmax, ymax] = s.bounds_m;
  if (!(xmin < xmax && ymin < ymax)) throw new Error('Terrain bounds are invalid.');
  if (
    [s.start_m, s.goal_m].some(([x, y]) => x < xmin || x > xmax || y < ymin || y > ymax) ||
    samePoint(s.start_m, s.goal_m, Math.max(xmax - xmin, ymax - ymin))
  )
    throw new Error('Terrain endpoints must be distinct and inside the bounds.');
  const width = Array.isArray(s.field_x_m) ? s.field_x_m.length : 0;
  const height = Array.isArray(s.field_y_m) ? s.field_y_m.length : 0;
  if (!width || !height || width < 4 || height < 4 || width > 513 || height > 513)
    throw new Error('Terrain fields must be between 4 × 4 and 513 × 513.');
  for (const [axis, low, high] of [
    [s.field_x_m, xmin, xmax],
    [s.field_y_m, ymin, ymax],
  ] as const) {
    if (
      !axis.every(finite) ||
      axis.some((v, i) => i > 0 && v <= axis[i - 1]) ||
      axis[0] !== low ||
      axis[axis.length - 1] !== high
    )
      throw new Error('Terrain coordinates must increase and cover the bounds.');
  }
  for (const field of [s.elevation_m, s.log_slowness]) {
    if (
      !Array.isArray(field) ||
      field.length !== height ||
      field.some((row) => !Array.isArray(row) || row.length !== width || !row.every(finite))
    )
      throw new Error('Terrain field dimensions or values are invalid.');
  }
  if (!Array.isArray(s.barriers_geojson) || s.barriers_geojson.length > 100)
    throw new Error('Terrain barriers are invalid.');
  const coordinate = (candidate: unknown) =>
    point(candidate) &&
    candidate[0] >= xmin &&
    candidate[0] <= xmax &&
    candidate[1] >= ymin &&
    candidate[1] <= ymax;
  for (const barrier of s.barriers_geojson) {
    if (!object(barrier) || !['Polygon', 'MultiPolygon'].includes(String(barrier.type)))
      throw new Error('Terrain barriers are invalid.');
    const polygons = barrier.type === 'Polygon' ? [barrier.coordinates] : barrier.coordinates;
    if (
      !Array.isArray(polygons) ||
      !polygons.length ||
      polygons.some(
        (polygon) =>
          !Array.isArray(polygon) ||
          !polygon.length ||
          polygon.some(
            (ring) =>
              !Array.isArray(ring) ||
              ring.length < 4 ||
              ring.length > 20_000 ||
              !ring.every(coordinate),
          ),
      )
    )
      throw new Error('Terrain barrier coordinates are invalid.');
  }
  if (
    !object(s.provenance) ||
    !object(s.metadata) ||
    !finiteJson(s.provenance) ||
    !finiteJson(s.metadata)
  )
    throw new Error('Terrain metadata is invalid.');
  return structuredClone(s);
}

export function validateTerrainConfig(value: unknown, browserLimits = true): TerrainConfig {
  if (!object(value)) throw new Error('Planner configuration is invalid.');
  const c = value as TerrainConfig;
  if (
    c.version !== 2 ||
    !['euler_lagrange', 'slsqp', 'fast_marching'].includes(c.method) ||
    typeof c.initialization !== 'string' ||
    !c.initialization ||
    !Number.isInteger(c.interior_points) ||
    c.interior_points < 1 ||
    c.interior_points > (browserLimits ? 128 : 1024) ||
    (browserLimits && ![32, 64, 128].includes(c.interior_points)) ||
    !Number.isInteger(c.reference_grid_size) ||
    c.reference_grid_size < 3 ||
    c.reference_grid_size > (browserLimits ? 257 : 2049) ||
    c.reference_grid_size % 2 === 0 ||
    (browserLimits && ![129, 257].includes(c.reference_grid_size)) ||
    !finite(c.tolerance) ||
    c.tolerance < 1e-12 ||
    c.tolerance > 1e-2 ||
    !Number.isInteger(c.max_iterations) ||
    c.max_iterations < 1 ||
    c.max_iterations > (browserLimits ? 10_000 : 1_000_000) ||
    !finite(c.time_limit_s) ||
    c.time_limit_s <= 0 ||
    c.time_limit_s > (browserLimits ? 300 : 86_400) ||
    !Number.isInteger(c.profile_samples) ||
    c.profile_samples < 2 ||
    c.profile_samples > (browserLimits ? 4097 : 16_385) ||
    !object(c.options) ||
    !finiteJson(c.options)
  )
    throw new Error('Planner configuration is invalid or exceeds browser limits.');
  return structuredClone(c);
}

export function validateTerrainResult(value: unknown, scenario: TerrainScenario): TerrainResult {
  if (!object(value)) throw new Error('Planner result is invalid.');
  const r = value as TerrainResult;
  const [xmin, ymin, xmax, ymax] = scenario.bounds_m;
  const scale = Math.max(xmax - xmin, ymax - ymin);
  const route = r.route_m;
  if (
    r.version !== 2 ||
    !['euler_lagrange', 'slsqp', 'fast_marching'].includes(r.method) ||
    typeof r.initialization !== 'string' ||
    !r.initialization ||
    typeof r.feasible !== 'boolean' ||
    typeof r.solver_success !== 'boolean' ||
    typeof r.termination_reason !== 'string' ||
    !r.termination_reason ||
    !digest(r.scenario_hash) ||
    !digest(r.config_hash) ||
    !object(r.diagnostics) ||
    !finiteJson(r.diagnostics) ||
    !object(r.timing_s) ||
    Object.values(r.timing_s).some((v) => !finite(v) || v < 0) ||
    (r.evaluated_cost_s !== null && (!finite(r.evaluated_cost_s) || r.evaluated_cost_s < 0))
  )
    throw new Error('Planner result fields are invalid.');
  if (
    route !== null &&
    (!Array.isArray(route) || route.length < 2 || route.length > 16_385 || !route.every(point))
  )
    throw new Error('Planner route is invalid.');
  if (
    r.feasible &&
    (!route ||
      route.some(([x, y]) => x < xmin || x > xmax || y < ymin || y > ymax) ||
      !samePoint(route[0], scenario.start_m, scale) ||
      !samePoint(route[route.length - 1], scenario.goal_m, scale))
  )
    throw new Error('A feasible planner route must match the scenario bounds and endpoints.');
  if (r.evaluation === undefined || (r.evaluation !== null && !object(r.evaluation)))
    throw new Error('Planner evaluation or profile is invalid.');
  if (r.evaluation === null) {
    if (r.feasible || r.evaluated_cost_s !== null)
      throw new Error('A result without evaluation cannot be feasible or have a cost.');
  } else {
    const e = r.evaluation;
    const p = e.profile;
    const arrays = p && [p.distance_m, p.elevation_m, p.slowness_s_per_m, p.accumulated_cost_s];
    if (
      !object(e) ||
      !finite(e.length_m) ||
      e.length_m < 0 ||
      typeof e.feasible !== 'boolean' ||
      (e.cost_s !== null && (!finite(e.cost_s) || e.cost_s < 0)) ||
      (e.minimum_clearance_m !== null &&
        (!finite(e.minimum_clearance_m) || e.minimum_clearance_m < 0)) ||
      !Array.isArray(e.violations) ||
      e.violations.some((v) => typeof v !== 'string') ||
      !finite(e.evaluation_time_s) ||
      e.evaluation_time_s < 0 ||
      !arrays ||
      arrays.some((a) => !Array.isArray(a) || a.length > 16_385 || !a.every(finite)) ||
      new Set(arrays.map((a) => a.length)).size !== 1 ||
      (e.cost_s === null ? arrays[0].length !== 0 : arrays[0].length < 2) ||
      p.distance_m.some((v, i) => v < 0 || (i > 0 && v < p.distance_m[i - 1])) ||
      p.slowness_s_per_m.some((v) => v <= 0) ||
      p.accumulated_cost_s.some((v, i) => v < 0 || (i > 0 && v < p.accumulated_cost_s[i - 1])) ||
      e.feasible !== (e.violations.length === 0) ||
      r.feasible !== e.feasible ||
      (r.evaluated_cost_s === null) !== (e.cost_s === null) ||
      (r.evaluated_cost_s !== null &&
        Math.abs(r.evaluated_cost_s - (e.cost_s as number)) > 1e-10) ||
      (r.feasible && e.cost_s === null)
    )
      throw new Error('Planner evaluation or profile is invalid.');
  }
  return structuredClone(r);
}

export function validateTerrainBundle(value: unknown): TerrainBundle {
  if (!value || typeof value !== 'object') throw new Error('Expected a JSON object.');
  const raw = value as Partial<TerrainBundle> & Partial<Scene>;
  if (raw.version === 1) throw new Error('VERSION_1');
  if (raw.version !== 2) throw new Error('Only version 1 and version 2 files are supported.');
  const scenario = validateTerrainScenario('scenario' in raw ? raw.scenario : raw);
  if (raw.configs !== undefined && !Array.isArray(raw.configs))
    throw new Error('Planner configurations must be an array.');
  if (raw.results !== undefined && !Array.isArray(raw.results))
    throw new Error('Planner results must be an array.');
  const configs = raw.configs?.map((config) => validateTerrainConfig(config));
  const results = raw.results?.map((result) => validateTerrainResult(result, scenario));
  if (results?.length) {
    if (!configs?.length || configs.length !== results.length)
      throw new Error('Every imported result must have its original configuration.');
    const hashes = new Set(results.map((result) => result.scenario_hash));
    if (hashes.size !== 1) throw new Error('Imported results refer to different scenarios.');
    results.forEach((result, index) => {
      const config = configs[index];
      if (result.method !== config.method || result.initialization !== config.initialization)
        throw new Error('Imported result does not match its planner configuration.');
    });
  }
  return { version: 2, scenario, configs, results };
}

export function validateScene(value: unknown): Scene {
  if (!value || typeof value !== 'object') throw new Error('Expected a scene object.');
  const scene = value as Scene;
  const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
  const coordinate = (v: unknown) => finite(v) && Math.abs(v) <= 100;
  const point = (v: unknown) => Array.isArray(v) && v.length === 2 && v.every(coordinate);
  if (scene.version !== 1) throw new Error('Only version 1 scenes are supported.');
  if (!point(scene.start) || !point(scene.end))
    throw new Error('Endpoints must be between −100 and 100.');
  if (!Array.isArray(scene.obstacles) || scene.obstacles.length > 20)
    throw new Error('Use up to 20 obstacles.');
  for (const o of scene.obstacles) {
    if (
      !o ||
      !coordinate(o.x) ||
      !coordinate(o.y) ||
      !finite(o.weight) ||
      o.weight < 0 ||
      o.weight > 100 ||
      !finite(o.width) ||
      o.width < 0.1 ||
      o.width > 10
    ) {
      throw new Error('Obstacles need finite coordinates, weight 0–100, and width 0.1–10.');
    }
  }
  const opts = scene.options;
  if (
    !opts ||
    !Number.isInteger(opts.interior_points) ||
    opts.interior_points < 1 ||
    opts.interior_points > 100 ||
    !Number.isInteger(opts.max_iterations) ||
    opts.max_iterations < 1 ||
    opts.max_iterations > 100 ||
    !finite(opts.tolerance) ||
    opts.tolerance < 1e-12 ||
    opts.tolerance > 1e-2 ||
    !['damped', 'undamped'].includes(opts.mode)
  )
    throw new Error('Invalid solver settings or browser limits exceeded.');
  if (
    !Array.isArray(scene.guesses) ||
    !scene.guesses.length ||
    new Set(scene.guesses).size !== scene.guesses.length ||
    scene.guesses.some((g) => !['straight', 'bend-x', 'bend-y'].includes(g))
  )
    throw new Error('Select valid initial guesses.');
  return structuredClone(scene);
}
