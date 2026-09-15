import { distance } from './geometry';
import type { Bounds, GhostRoute, PlaygroundScene, Point, Stroke } from './types';

const COLORS = {
  ink: '#203930',
  route: '#125f56',
  routeHalo: '#f7f4ec',
  ghost: '#8b7665',
  start: '#17695e',
  end: '#bf5b35',
};

export class PlaygroundCanvas {
  field: number[] = [];
  fieldWidth = 0;
  fieldHeight = 0;
  scene: PlaygroundScene;
  path: Point[];
  ghosts: GhostRoute[] = [];
  previousPath: Point[] | null = null;
  showPrevious = false;
  showPoints = false;
  activeStroke: Stroke | null = null;
  heldIndex: number | null = null;
  private background = document.createElement('canvas');

  constructor(
    readonly canvas: HTMLCanvasElement,
    scene: PlaygroundScene,
    path: Point[],
  ) {
    this.scene = scene;
    this.path = path;
    canvas.tabIndex = 0;
    canvas.setAttribute('role', 'application');
    canvas.setAttribute(
      'aria-label',
      'Path playground. Choose grab, paint, or erase, then drag on the canvas. Arrow keys move the focused path point.',
    );
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  setField(values: number[], width: number, height: number) {
    this.field = values;
    this.fieldWidth = width;
    this.fieldHeight = height;
    this.paintField();
  }

  private resize() {
    const rect = this.canvas.getBoundingClientRect(),
      ratio = devicePixelRatio || 1,
      width = Math.max(1, Math.round(rect.width * ratio)),
      height = Math.max(1, Math.round(rect.height * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    return { ctx: this.canvas.getContext('2d')!, width, height, ratio };
  }

  fromClient(clientX: number, clientY: number): Point {
    const rect = this.canvas.getBoundingClientRect(),
      [xmin, xmax, ymin, ymax] = this.scene.bounds;
    return [
      xmin + ((clientX - rect.left) / rect.width) * (xmax - xmin),
      ymax - ((clientY - rect.top) / rect.height) * (ymax - ymin),
    ];
  }

  private toPixel(point: Point, width: number, height: number): Point {
    const [xmin, xmax, ymin, ymax] = this.scene.bounds;
    return [
      ((point[0] - xmin) / (xmax - xmin)) * width,
      ((ymax - point[1]) / (ymax - ymin)) * height,
    ];
  }

  modelRadiusFromPixels(pixels: number) {
    const rect = this.canvas.getBoundingClientRect();
    return (pixels / Math.max(1, rect.width)) * (this.scene.bounds[1] - this.scene.bounds[0]);
  }

  private paintField() {
    if (!this.field.length || !this.fieldWidth || !this.fieldHeight) return;
    this.background.width = this.fieldWidth;
    this.background.height = this.fieldHeight;
    const ctx = this.background.getContext('2d')!,
      image = ctx.createImageData(this.fieldWidth, this.fieldHeight),
      max = Math.max(1, ...this.field),
      logMax = Math.log1p(max - 1);
    for (let index = 0; index < this.field.length; index++) {
      const t = logMax ? Math.log1p(Math.max(0, this.field[index] - 1)) / logMax : 0;
      image.data[index * 4] = 246 - 35 * t;
      image.data[index * 4 + 1] = 244 - 91 * t;
      image.data[index * 4 + 2] = 235 - 119 * t;
      image.data[index * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }

  draw() {
    const { ctx, width, height, ratio } = this.resize();
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#f3f1e9';
    ctx.fillRect(0, 0, width, height);
    if (this.field.length)
      ctx.drawImage(this.background, 0, 0, this.fieldWidth, this.fieldHeight, 0, 0, width, height);
    this.drawGrid(ctx, width, height, ratio);
    for (const ghost of this.ghosts)
      this.drawPath(ctx, ghost.path, width, height, COLORS.ghost, 1.5 * ratio, [
        6 * ratio,
        6 * ratio,
      ]);
    if (this.showPrevious && this.previousPath)
      this.drawPath(ctx, this.previousPath, width, height, '#5d8178', 1.5 * ratio, [
        3 * ratio,
        4 * ratio,
      ]);
    this.drawPath(ctx, this.path, width, height, COLORS.routeHalo, 6 * ratio);
    this.drawPath(ctx, this.path, width, height, COLORS.route, 2.5 * ratio);
    if (this.showPoints) {
      ctx.fillStyle = COLORS.route;
      for (let index = 1; index < this.path.length - 1; index++) {
        const [x, y] = this.toPixel(this.path[index], width, height);
        ctx.beginPath();
        ctx.arc(x, y, 2.25 * ratio, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (this.activeStroke) this.drawStrokeOverlay(ctx, this.activeStroke, width, height, ratio);
    this.drawEndpoints(ctx, width, height, ratio);
    if (this.heldIndex !== null) {
      const [x, y] = this.toPixel(this.path[this.heldIndex], width, height);
      ctx.strokeStyle = COLORS.ink;
      ctx.lineWidth = 1.5 * ratio;
      ctx.beginPath();
      ctx.arc(x, y, 8 * ratio, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  private drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number, ratio: number) {
    ctx.strokeStyle = 'rgba(50,70,60,.13)';
    ctx.lineWidth = ratio;
    for (let i = 1; i < 12; i++) {
      ctx.beginPath();
      ctx.moveTo((i * width) / 12, 0);
      ctx.lineTo((i * width) / 12, height);
      ctx.stroke();
    }
    for (let i = 1; i < 8; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (i * height) / 8);
      ctx.lineTo(width, (i * height) / 8);
      ctx.stroke();
    }
  }

  private drawPath(
    ctx: CanvasRenderingContext2D,
    path: Point[],
    width: number,
    height: number,
    color: string,
    lineWidth: number,
    dash: number[] = [],
  ) {
    if (!path.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.setLineDash(dash);
    ctx.beginPath();
    path.forEach((point, index) => {
      const [x, y] = this.toPixel(point, width, height);
      if (index) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  private drawStrokeOverlay(
    ctx: CanvasRenderingContext2D,
    stroke: Stroke,
    width: number,
    height: number,
    ratio: number,
  ) {
    const pixelWidth = (stroke.width / (this.scene.bounds[1] - this.scene.bounds[0])) * width;
    ctx.globalAlpha = 0.22;
    this.drawPath(
      ctx,
      stroke.points,
      width,
      height,
      '#d56b36',
      Math.max(2 * ratio, pixelWidth * 2),
    );
    if (stroke.points.length === 1) {
      const [x, y] = this.toPixel(stroke.points[0], width, height);
      ctx.fillStyle = '#d56b36';
      ctx.beginPath();
      ctx.arc(x, y, pixelWidth, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  private drawEndpoints(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    ratio: number,
  ) {
    for (const [point, color, label] of [
      [this.scene.start, COLORS.start, 'A'],
      [this.scene.end, COLORS.end, 'B'],
    ] as const) {
      const [x, y] = this.toPixel(point, width, height);
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(x, y, 8 * ratio, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 5 * ratio, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = COLORS.ink;
      ctx.font = `${10 * ratio}px ui-monospace, monospace`;
      ctx.fillText(label, x + 11 * ratio, y - 9 * ratio);
    }
  }

  nearestVisiblePoint(point: Point, includeEndpoints = true) {
    let index = -1,
      nearest = Infinity;
    this.path.forEach((candidate, candidateIndex) => {
      if (!includeEndpoints && (candidateIndex === 0 || candidateIndex === this.path.length - 1))
        return;
      const d = distance(candidate, point);
      if (d < nearest) {
        index = candidateIndex;
        nearest = d;
      }
    });
    return { index, distance: nearest };
  }
}
