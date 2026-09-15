import type { Bounds, Point, Scene, State } from './types';

export const colors = ['#17695e', '#bd592f', '#7566b0'];
export const names: Record<string, string> = {
  straight: 'Direct',
  'bend-x': 'Right arc',
  'bend-y': 'Left arc',
};
export function sceneBounds(scene: Scene): Bounds {
  const points = [scene.start, scene.end, ...scene.obstacles.map((o) => [o.x, o.y])];
  const xs = points.map((p) => p[0]),
    ys = points.map((p) => p[1]);
  const span =
    Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 8) + 4;
  const cx = (Math.max(...xs) + Math.min(...xs)) / 2,
    cy = (Math.max(...ys) + Math.min(...ys)) / 2;
  return [cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2];
}

export class Plot {
  bounds: Bounds = [-4, 14, -4, 14];
  field: number[] = [];
  scene!: Scene;
  states: Record<string, State> = {};
  selected = -1;
  samples = false;
  heatmap = true;
  contours = true;
  rover: number | null = null;
  private background = document.createElement('canvas');
  constructor(public canvas: HTMLCanvasElement) {
    new ResizeObserver(() => this.draw()).observe(canvas);
  }
  fromPixel(x: number, y: number): Point {
    const rect = this.canvas.getBoundingClientRect(),
      margin = 36;
    const size = rect.width - 2 * margin;
    return [
      this.bounds[0] + ((x - rect.left - margin) / size) * (this.bounds[1] - this.bounds[0]),
      this.bounds[3] - ((y - rect.top - margin) / size) * (this.bounds[3] - this.bounds[2]),
    ];
  }
  setField(field: number[]) {
    this.field = field;
    this.background.width = this.background.height = 90;
    const ctx = this.background.getContext('2d')!;
    const pixels = ctx.createImageData(90, 90);
    const max = Math.max(...field, 2);
    for (let y = 0; y < 90; y++)
      for (let x = 0; x < 90; x++) {
        const alpha = Math.pow(Math.max(0, (field[y * 90 + x] - 1) / (max - 1)), 0.6);
        const index = ((89 - y) * 90 + x) * 4;
        [248 - 46 * alpha, 247 - 109 * alpha, 242 - 110 * alpha, 255].forEach(
          (v, k) => (pixels.data[index + k] = v),
        );
      }
    ctx.putImageData(pixels, 0, 0);
  }
  draw() {
    if (!this.scene) return;
    const width = this.canvas.getBoundingClientRect().width || 600;
    const ratio = window.devicePixelRatio || 1;
    this.canvas.width = width * ratio;
    this.canvas.height = width * ratio;
    const ctx = this.canvas.getContext('2d')!;
    ctx.scale(ratio, ratio);
    const m = 36,
      size = width - 2 * m;
    const point = (p: Point): Point => [
      m + ((p[0] - this.bounds[0]) / (this.bounds[1] - this.bounds[0])) * size,
      m + ((this.bounds[3] - p[1]) / (this.bounds[3] - this.bounds[2])) * size,
    ];
    ctx.fillStyle = '#f8f7f2';
    ctx.fillRect(0, 0, width, width);
    if (this.field.length && this.heatmap) ctx.drawImage(this.background, m, m, size, size);
    ctx.strokeStyle = '#263e3410';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#777e74';
    ctx.font = '10px monospace';
    for (let i = 0; i <= 8; i++) {
      const p = m + (i * size) / 8;
      ctx.beginPath();
      ctx.moveTo(p, m);
      ctx.lineTo(p, width - m);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(m, p);
      ctx.lineTo(width - m, p);
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.fillText(
        (this.bounds[0] + (i * (this.bounds[1] - this.bounds[0])) / 8).toFixed(0),
        p,
        width - 15,
      );
      ctx.textAlign = 'right';
      ctx.fillText(
        (this.bounds[3] - (i * (this.bounds[3] - this.bounds[2])) / 8).toFixed(0),
        m - 10,
        p + 3,
      );
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(m, m, size, size);
    ctx.clip();
    if (this.contours && this.field.length) {
      const maximum = Math.max(...this.field);
      ctx.strokeStyle = '#985c3333';
      ctx.lineWidth = 0.8;
      for (const fraction of [0.08, 0.2, 0.4, 0.65, 0.85]) {
        const level = 1 + (maximum - 1) * fraction;
        if (maximum <= 1) break;
        ctx.beginPath();
        for (let y = 0; y < 89; y++)
          for (let x = 0; x < 89; x++) {
            const corners = [
              [x, y],
              [x + 1, y],
              [x + 1, y + 1],
              [x, y + 1],
            ];
            const crossings: Point[] = [];
            for (let edge = 0; edge < 4; edge++) {
              const a = corners[edge],
                b = corners[(edge + 1) % 4];
              const va = this.field[a[1] * 90 + a[0]],
                vb = this.field[b[1] * 90 + b[0]];
              if (va < level !== vb < level) {
                const t = (level - va) / (vb - va);
                crossings.push([
                  m + ((a[0] + t * (b[0] - a[0])) / 89) * size,
                  m + (1 - (a[1] + t * (b[1] - a[1])) / 89) * size,
                ]);
              }
            }
            for (let i = 0; i + 1 < crossings.length; i += 2) {
              ctx.moveTo(...crossings[i]);
              ctx.lineTo(...crossings[i + 1]);
            }
          }
        ctx.stroke();
      }
    }
    Object.entries(this.states).forEach(([name, state]) => {
      const index = ['straight', 'bend-x', 'bend-y'].indexOf(name);
      ctx.strokeStyle = colors[index];
      ctx.lineWidth = 2.6;
      ctx.lineJoin = 'round';
      ctx.setLineDash(index === 1 ? [8, 3] : index === 2 ? [3, 3] : []);
      ctx.beginPath();
      state.path.forEach((p, i) => (i ? ctx.lineTo(...point(p)) : ctx.moveTo(...point(p))));
      ctx.stroke();
      ctx.setLineDash([]);
      if (this.samples)
        for (const p of state.path) {
          ctx.fillStyle = colors[index];
          ctx.beginPath();
          ctx.arc(...point(p), 2.4, 0, 2 * Math.PI);
          ctx.fill();
        }
      if (this.rover !== null) {
        const distances = state.path
          .slice(1)
          .map((p, i) => Math.hypot(p[0] - state.path[i][0], p[1] - state.path[i][1]));
        let target = this.rover * distances.reduce((a, b) => a + b, 0),
          i = 0;
        while (i < distances.length - 1 && target > distances[i]) target -= distances[i++];
        const t = distances[i] ? target / distances[i] : 0,
          a = state.path[i],
          b = state.path[i + 1];
        ctx.fillStyle = colors[index];
        ctx.beginPath();
        ctx.arc(...point([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]), 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
    this.scene.obstacles.forEach((o, index) => {
      const p = point([o.x, o.y]);
      if (index === this.selected) {
        ctx.beginPath();
        ctx.arc(...p, 12, 0, 2 * Math.PI);
        ctx.strokeStyle = '#8d553d';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      ctx.strokeStyle = '#8d553d';
      ctx.lineWidth = 1.7;
      ctx.beginPath();
      ctx.moveTo(p[0] - 4, p[1] - 4);
      ctx.lineTo(p[0] + 4, p[1] + 4);
      ctx.moveTo(p[0] - 4, p[1] + 4);
      ctx.lineTo(p[0] + 4, p[1] - 4);
      ctx.stroke();
    });
    ctx.restore();
    [this.scene.start, this.scene.end].forEach((p, i) => {
      const xy = point(p);
      ctx.fillStyle = '#263e34';
      ctx.beginPath();
      ctx.arc(...xy, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#263e34';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = i ? 'right' : 'left';
      ctx.fillText(i ? 'END' : 'START', xy[0] + (i ? -10 : 10), xy[1] + 3);
    });
  }
}

export function chart(
  canvas: HTMLCanvasElement,
  histories: Record<string, State[]>,
  metric: 'cost' | 'residual_norm',
  cursor: number,
) {
  const w = canvas.clientWidth || 300,
    h = 104,
    ratio = window.devicePixelRatio || 1;
  canvas.width = w * ratio;
  canvas.height = h * ratio;
  const ctx = canvas.getContext('2d')!;
  ctx.scale(ratio, ratio);
  const transform = (v: number) =>
    metric === 'residual_norm' ? Math.log10(Math.max(v, 1e-12)) : v;
  const values = Object.values(histories)
    .flat()
    .map((s) => transform(s[metric]))
    .filter(Number.isFinite);
  if (!values.length) return;
  const min = Math.min(...values),
    max = Math.max(...values),
    span = Math.max(max - min, 1e-5);
  const count = Math.max(...Object.values(histories).map((v) => v.length), 2) - 1;
  ctx.fillStyle = '#7f837c';
  ctx.font = '10px monospace';
  const label = (v: number) => (metric === 'residual_norm' ? `10^${v.toFixed(0)}` : v.toFixed(1));
  ctx.fillText(label(max), 0, 14);
  ctx.fillText(label(min), 0, h - 12);
  ctx.strokeStyle = '#e6e6df';
  ctx.beginPath();
  ctx.moveTo(44, 8);
  ctx.lineTo(44, h - 18);
  ctx.lineTo(w, h - 18);
  ctx.stroke();
  for (const [name, states] of Object.entries(histories)) {
    const index = ['straight', 'bend-x', 'bend-y'].indexOf(name);
    ctx.strokeStyle = colors[index];
    ctx.lineWidth = 1.7;
    ctx.setLineDash(index ? [5, 3] : []);
    ctx.beginPath();
    states.forEach((state, i) => {
      const x = 44 + (i / count) * (w - 50),
        y = 8 + (1 - (transform(state[metric]) - min) / span) * (h - 26);
      if (i) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.strokeStyle = '#263e3450';
  const x = 44 + (cursor / count) * (w - 50);
  ctx.beginPath();
  ctx.moveTo(x, 8);
  ctx.lineTo(x, h - 18);
  ctx.stroke();
}
