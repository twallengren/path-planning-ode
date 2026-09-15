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
