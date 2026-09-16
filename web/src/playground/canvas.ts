import { distance } from './geometry';
import type { GhostRoute, PlaygroundScene, Point, Stroke } from './types';

const COLORS = {
  ink: '#203930',
  route: '#125f56',
  routeHalo: '#fffdf7',
  ghost: '#8b7665',
  start: '#17695e',
  end: '#bf5b35',
  hill: '#9c5d32',
};

type View = {
  scale: number;
  offsetX: number;
  offsetY: number;
  drawWidth: number;
  drawHeight: number;
};

export class PlaygroundCanvas {
  field: number[] = [];
  elevation: number[] = [];
  fieldWidth = 0;
  fieldHeight = 0;
  scene: PlaygroundScene;
  path: Point[];
  ghosts: GhostRoute[] = [];
  previousPath: Point[] | null = null;
  showPrevious = false;
  showPoints = false;
  showElevation = false;
  activeStroke: Stroke | null = null;
  heldIndex: number | null = null;
  heldGaussian: number | null = null;
  private readonly costBackground = document.createElement('canvas');
  private readonly elevationBackground = document.createElement('canvas');

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
      'Path playground. Choose grab, paint, or erase, then drag on the canvas. Arrow keys move the selected target.',
    );
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  setField(values: number[], width: number, height: number, elevation: number[] = []) {
    this.field = values;
    this.elevation = elevation;
    this.fieldWidth = width;
    this.fieldHeight = height;
    this.paintRaster(this.costBackground, values, width, height, false);
    if (elevation.length)
      this.paintRaster(this.elevationBackground, elevation, width, height, true);
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

  private view(width: number, height: number): View {
    const [xmin, xmax, ymin, ymax] = this.scene.bounds,
      worldWidth = xmax - xmin,
      worldHeight = ymax - ymin,
      scale = Math.min(width / worldWidth, height / worldHeight),
      drawWidth = worldWidth * scale,
      drawHeight = worldHeight * scale;
    return {
      scale,
      drawWidth,
      drawHeight,
      offsetX: (width - drawWidth) / 2,
      offsetY: (height - drawHeight) / 2,
    };
  }

  fromClient(clientX: number, clientY: number): Point {
    const rect = this.canvas.getBoundingClientRect(),
      ratioX = this.canvas.width / Math.max(1, rect.width),
      ratioY = this.canvas.height / Math.max(1, rect.height),
      pixelX = (clientX - rect.left) * ratioX,
      pixelY = (clientY - rect.top) * ratioY,
      view = this.view(this.canvas.width, this.canvas.height),
      [xmin, , , ymax] = this.scene.bounds;
    return [
      xmin + (pixelX - view.offsetX) / view.scale,
      ymax - (pixelY - view.offsetY) / view.scale,
    ];
  }

  private toPixel(point: Point, width: number, height: number): Point {
    const view = this.view(width, height),
      [xmin, , , ymax] = this.scene.bounds;
    return [
      view.offsetX + (point[0] - xmin) * view.scale,
      view.offsetY + (ymax - point[1]) * view.scale,
    ];
  }

  modelRadiusFromPixels(pixels: number) {
    const rect = this.canvas.getBoundingClientRect(),
      view = this.view(rect.width, rect.height);
    return pixels / view.scale;
  }

  private paintRaster(
    target: HTMLCanvasElement,
    values: number[],
    width: number,
    height: number,
    elevation: boolean,
  ) {
    if (!values.length || !width || !height) return;
    target.width = width;
    target.height = height;
    const sorted = values.filter(Number.isFinite).sort((a, b) => a - b),
      low = sorted[Math.floor(sorted.length * 0.02)] ?? 0,
      high = sorted[Math.floor(sorted.length * 0.98)] ?? low + 1,
      span = Math.max(Number.EPSILON, high - low),
      ctx = target.getContext('2d')!,
      image = ctx.createImageData(width, height);
    for (let index = 0; index < values.length; index++) {
      const raw = Math.max(0, Math.min(1, (values[index] - low) / span)),
        t = elevation ? raw : Math.log1p(9 * raw) / Math.log(10);
      if (elevation) {
        image.data[index * 4] = 228 - 91 * t;
        image.data[index * 4 + 1] = 230 - 64 * t;
        image.data[index * 4 + 2] = 211 - 94 * t;
      } else {
        image.data[index * 4] = 247 - 44 * t;
        image.data[index * 4 + 1] = 244 - 105 * t;
        image.data[index * 4 + 2] = 233 - 126 * t;
      }
      image.data[index * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }

  draw() {
    const { ctx, width, height, ratio } = this.resize(),
      view = this.view(width, height);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#e8e5dc';
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = '#f3f1e9';
    ctx.fillRect(view.offsetX, view.offsetY, view.drawWidth, view.drawHeight);
    const raster =
      this.showElevation && this.elevation.length ? this.elevationBackground : this.costBackground;
    if (this.field.length)
      ctx.drawImage(
        raster,
        0,
        0,
        this.fieldWidth,
        this.fieldHeight,
        view.offsetX,
        view.offsetY,
        view.drawWidth,
        view.drawHeight,
      );
    this.drawGrid(ctx, view, ratio);
    this.drawGaussians(ctx, width, height, ratio, view.scale);
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
    if (this.activeStroke)
      this.drawStrokeOverlay(ctx, this.activeStroke, width, height, ratio, view.scale);
    this.drawEndpoints(ctx, width, height, ratio);
    if (this.heldIndex !== null)
      this.drawHeld(ctx, this.path[this.heldIndex], width, height, ratio);
    if (this.heldGaussian !== null) {
      const hill = this.scene.gaussians[this.heldGaussian];
      if (hill) this.drawHeld(ctx, [hill.x, hill.y], width, height, ratio);
    }
  }

  private drawGrid(ctx: CanvasRenderingContext2D, view: View, ratio: number) {
    ctx.save();
    ctx.strokeStyle = 'rgba(50,70,60,.11)';
    ctx.lineWidth = ratio;
    for (let i = 1; i < 12; i++) {
      const x = view.offsetX + (i * view.drawWidth) / 12;
      ctx.beginPath();
      ctx.moveTo(x, view.offsetY);
      ctx.lineTo(x, view.offsetY + view.drawHeight);
      ctx.stroke();
    }
    for (let i = 1; i < 8; i++) {
      const y = view.offsetY + (i * view.drawHeight) / 8;
      ctx.beginPath();
      ctx.moveTo(view.offsetX, y);
      ctx.lineTo(view.offsetX + view.drawWidth, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  private drawGaussians(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    ratio: number,
    scale: number,
  ) {
    ctx.save();
    ctx.strokeStyle = 'rgba(106,66,39,.45)';
    ctx.setLineDash([3 * ratio, 4 * ratio]);
    ctx.lineWidth = ratio;
    for (const hill of this.scene.gaussians) {
      const [x, y] = this.toPixel([hill.x, hill.y], width, height);
      ctx.beginPath();
      ctx.arc(x, y, Math.max(4 * ratio, hill.width * scale), 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = COLORS.hill;
      ctx.beginPath();
      ctx.arc(x, y, 2.5 * ratio, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
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
    ctx.save();
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
    ctx.restore();
  }

  private drawStrokeOverlay(
    ctx: CanvasRenderingContext2D,
    stroke: Stroke,
    width: number,
    height: number,
    ratio: number,
    scale: number,
  ) {
    const pixelWidth = stroke.width * scale;
    ctx.save();
    ctx.globalAlpha = 0.25;
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
    ctx.restore();
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

  private drawHeld(
    ctx: CanvasRenderingContext2D,
    point: Point,
    width: number,
    height: number,
    ratio: number,
  ) {
    const [x, y] = this.toPixel(point, width, height);
    ctx.strokeStyle = COLORS.ink;
    ctx.lineWidth = 1.5 * ratio;
    ctx.beginPath();
    ctx.arc(x, y, 9 * ratio, 0, Math.PI * 2);
    ctx.stroke();
  }

  nearestVisiblePoint(point: Point) {
    let index = -1,
      nearest = Infinity;
    this.path.forEach((candidate, candidateIndex) => {
      const d = distance(candidate, point);
      if (d < nearest) {
        index = candidateIndex;
        nearest = d;
      }
    });
    return { index, distance: nearest };
  }

  nearestGaussian(point: Point) {
    let index = -1,
      nearest = Infinity;
    this.scene.gaussians.forEach((hill, candidateIndex) => {
      const d = distance([hill.x, hill.y], point);
      if (d < nearest) {
        index = candidateIndex;
        nearest = d;
      }
    });
    return { index, distance: nearest };
  }
}
