import type { Bounds, Point, Stroke } from './types';

export const distance = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function clampToBounds(point: Point, bounds: Bounds): Point {
  return [
    Math.max(bounds[0], Math.min(bounds[1], point[0])),
    Math.max(bounds[2], Math.min(bounds[3], point[1])),
  ];
}

export function straightPath(start: Point, end: Point, interiorPoints: number): Point[] {
  return Array.from({ length: interiorPoints + 2 }, (_, index) => {
    const t = index / (interiorPoints + 1);
    return [start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])];
  });
}

export function resamplePath(path: Point[], interiorPoints: number): Point[] {
  const count = interiorPoints + 2;
  if (path.length === count) return path.map(([x, y]) => [x, y]);
  const accumulated = [0];
  for (let i = 1; i < path.length; i++)
    accumulated.push(accumulated[i - 1] + distance(path[i - 1], path[i]));
  const total = accumulated.at(-1)!;
  if (total <= Number.EPSILON) return straightPath(path[0], path.at(-1)!, interiorPoints);
  const result: Point[] = [];
  let segment = 1;
  for (let i = 0; i < count; i++) {
    const target = (i / (count - 1)) * total;
    while (segment < accumulated.length - 1 && accumulated[segment] < target) segment++;
    const low = accumulated[segment - 1],
      high = accumulated[segment],
      t = high === low ? 0 : (target - low) / (high - low),
      a = path[segment - 1],
      b = path[segment];
    result.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
  }
  return result;
}

export function nearestPathPoint(path: Point[], point: Point, includeEndpoints = true) {
  let best = -1,
    bestDistance = Infinity;
  const first = includeEndpoints ? 0 : 1,
    last = includeEndpoints ? path.length : path.length - 1;
  for (let index = first; index < last; index++) {
    const candidate = distance(path[index], point);
    if (candidate < bestDistance) {
      best = index;
      bestDistance = candidate;
    }
  }
  return { index: best, distance: bestDistance };
}

function pointSegmentDistance(point: Point, a: Point, b: Point) {
  const dx = b[0] - a[0],
    dy = b[1] - a[1],
    denominator = dx * dx + dy * dy;
  if (!denominator) return distance(point, a);
  const t = Math.max(
    0,
    Math.min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denominator),
  );
  return distance(point, [a[0] + t * dx, a[1] + t * dy]);
}

export function strokeHit(stroke: Stroke, point: Point, padding = 0) {
  if (stroke.points.length === 1)
    return distance(stroke.points[0], point) <= stroke.width + padding;
  return stroke.points.some(
    (candidate, index) =>
      index > 0 &&
      pointSegmentDistance(point, stroke.points[index - 1], candidate) <= stroke.width + padding,
  );
}

export function movePathNeighborhood(path: Point[], index: number, position: Point): Point[] {
  const result = path.map(([x, y]) => [x, y] as Point),
    delta: Point = [position[0] - result[index][0], position[1] - result[index][1]],
    radius = Math.max(2, Math.round((path.length - 2) * 0.1));
  for (let offset = -radius; offset <= radius; offset++) {
    const candidate = index + offset;
    if (candidate <= 0 || candidate >= result.length - 1) continue;
    const weight = 0.5 + 0.5 * Math.cos((Math.PI * Math.abs(offset)) / (radius + 1));
    result[candidate] = [
      result[candidate][0] + weight * delta[0],
      result[candidate][1] + weight * delta[1],
    ];
  }
  result[index] = position;
  return result;
}
