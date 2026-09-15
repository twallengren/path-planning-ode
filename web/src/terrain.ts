import './terrain.css';
import {
  validateScene,
  validateTerrainBundle,
  validateTerrainConfig,
  validateTerrainResult,
  validateTerrainScenario,
  type Scene,
  type TerrainBundle,
  type TerrainConfig,
  type TerrainLayer,
  type TerrainResult,
  type TerrainScenario,
} from './types';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const base = new URL('.', location.href);
const colors = ['#17695e', '#d76e3d', '#7163a6', '#187da0', '#b48b2b', '#202c27'];
const labels: Record<string, string> = {
  straight: 'Local · straight',
  arc_left: 'Local · left arc',
  arc_right: 'Local · right arc',
  barrier: 'Local · barrier only',
  fast_marching: 'Local · FMM warm',
};
let scenario: TerrainScenario;
let results: TerrainResult[] = [];
let displayedConfigs: TerrainConfig[] = [];
let resultSource: 'computed' | 'imported' | 'preview' | 'published' = 'preview';
let recordedComparison: {
  difference: number | null;
  unresolved: boolean;
  referenceGrid: number | null;
  declaredGrid: number | null;
  refinementChange: number | null;
  reason: string | null;
} | null = null;
let recordedConfig: TerrainConfig | null = null;
let recordedFailure: string | null = null;
let layer: TerrainLayer = 'elevation';
let arrival: number[][] | null = null;
let controlsDirty = false;
let worker: Worker | null = null;
let ready = false;
let serial = 0;
let generation = 0;
const waiting = new Map<
  number,
  { generation: number; resolve: (value: any) => void; reject: (error: Error) => void }
>();

function message(text: string) {
  $('message').textContent = text;
}
function status(text: string) {
  $('runtime').textContent = text;
}
function errorText(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}
function request(action: string, extra: Record<string, unknown> = {}) {
  const id = ++serial,
    token = generation;
  return new Promise<any>((resolve, reject) => {
    waiting.set(id, { generation: token, resolve, reject });
    worker!.postMessage({ id, action, ...extra });
  });
}
function terminate(reason = 'Cancelled.') {
  generation++;
  worker?.terminate();
  worker = null;
  ready = false;
  for (const item of waiting.values()) item.reject(new Error(reason));
  waiting.clear();
}
async function boot() {
  if (ready && worker) return;
  if (worker) terminate('Worker restarted.');
  const token = generation;
  worker = new Worker(new URL('./solver.worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => {
    if (token !== generation) return;
    if (data.progress) {
      status(data.progress);
      return;
    }
    const pending = waiting.get(data.id);
    if (!pending || pending.generation !== generation) return;
    waiting.delete(data.id);
    data.error ? pending.reject(new Error(data.error)) : pending.resolve(data.result);
  };
  worker.onerror = (event) => {
    if (token !== generation) return;
    const error = new Error(event.message || 'Terrain worker failed.');
    for (const item of waiting.values()) item.reject(error);
    waiting.clear();
    worker?.terminate();
    worker = null;
    ready = false;
    status('Python runtime unavailable · Run comparison to retry');
  };
  await request('terrainBoot', { base: base.href });
  if (token !== generation) throw new Error('Cancelled.');
  ready = true;
  status('Python + NumPy + SciPy + Shapely ready · computation stays in your browser');
}

function palette(value: number, min: number, max: number, kind: TerrainLayer) {
  const t = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  const stops =
    kind === 'elevation'
      ? [
          [231, 238, 222],
          [116, 159, 119],
          [71, 78, 63],
        ]
      : kind === 'cost'
        ? [
            [241, 239, 218],
            [220, 148, 82],
            [118, 48, 48],
          ]
        : [
            [235, 241, 237],
            [76, 148, 157],
            [45, 48, 91],
          ];
  const x = t * (stops.length - 1),
    i = Math.min(stops.length - 2, Math.floor(x)),
    f = x - i;
  return stops[i].map((v, j) => Math.round(v + (stops[i + 1][j] - v) * f));
}
function resize(canvas: HTMLCanvasElement) {
  const ratio = devicePixelRatio || 1,
    w = Math.max(1, Math.round(canvas.clientWidth * ratio)),
    h = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return { ctx: canvas.getContext('2d')!, w, h, ratio };
}
function fieldForLayer(): number[][] {
  if (layer === 'arrival' && arrival) return arrival;
  if (layer === 'cost') return scenario.log_slowness.map((row) => row.map(Math.exp));
  return scenario.elevation_m;
}
function drawMap() {
  if (!scenario) return;
  const canvas = $<HTMLCanvasElement>('terrain-map'),
    { ctx, w, h, ratio } = resize(canvas);
  const pad = 32 * ratio,
    pw = w - pad * 2,
    ph = h - pad * 2;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#f5f6f0';
  ctx.fillRect(0, 0, w, h);
  const field = fieldForLayer(),
    finite = field.flat().filter((v) => Number.isFinite(v));
  const min = Math.min(...finite),
    max = Math.max(...finite);
  const image = ctx.createImageData(Math.max(1, Math.round(pw)), Math.max(1, Math.round(ph)));
  const rows = field.length,
    cols = field[0].length;
  for (let iy = 0; iy < image.height; iy++)
    for (let ix = 0; ix < image.width; ix++) {
      const row = Math.min(rows - 1, Math.floor((1 - iy / image.height) * rows));
      const col = Math.min(cols - 1, Math.floor((ix / image.width) * cols));
      const v = field[row][col],
        c = Number.isFinite(v) ? palette(v, min, max, layer) : [49, 54, 51];
      const q = (iy * image.width + ix) * 4;
      image.data[q] = c[0];
      image.data[q + 1] = c[1];
      image.data[q + 2] = c[2];
      image.data[q + 3] = 255;
    }
  ctx.putImageData(image, pad, pad);
  const [xmin, ymin, xmax, ymax] = scenario.bounds_m;
  const xy = ([x, y]: number[]) => [
    pad + ((x - xmin) / (xmax - xmin)) * pw,
    pad + (1 - (y - ymin) / (ymax - ymin)) * ph,
  ];
  ctx.save();
  ctx.beginPath();
  ctx.rect(pad, pad, pw, ph);
  ctx.clip();
  ctx.fillStyle = 'rgba(37,43,40,.83)';
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = ratio;
  for (const geometry of scenario.barriers_geojson) {
    const polygons: any[] =
      geometry.type === 'Polygon' ? [geometry.coordinates] : (geometry.coordinates as any[]);
    for (const polygon of polygons)
      for (const ring of polygon) {
        ctx.beginPath();
        ring.forEach((point: number[], i: number) => {
          const p = xy(point);
          i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
        });
        ctx.fill('evenodd');
        ctx.stroke();
      }
  }
  results.forEach((result, index) => {
    if (!result.route_m) return;
    ctx.beginPath();
    result.route_m.forEach((point, i) => {
      const p = xy(point);
      i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
    });
    ctx.strokeStyle = colors[index % colors.length];
    ctx.lineWidth = 3 * ratio;
    ctx.setLineDash(result.method === 'fast_marching' ? [7 * ratio, 4 * ratio] : []);
    ctx.stroke();
  });
  ctx.restore();
  ctx.setLineDash([]);
  for (const [name, point] of [
    ['START', scenario.start_m],
    ['GOAL', scenario.goal_m],
  ] as const) {
    const p = xy(point);
    ctx.beginPath();
    ctx.arc(p[0], p[1], 5 * ratio, 0, Math.PI * 2);
    ctx.fillStyle = '#1e3028';
    ctx.fill();
    ctx.font = `${9 * ratio}px IBM Plex Mono`;
    ctx.fillText(name, p[0] + 8 * ratio, p[1] - 7 * ratio);
  }
  ctx.strokeStyle = '#a9b1a7';
  ctx.strokeRect(pad, pad, pw, ph);
  ctx.fillStyle = '#657168';
  ctx.font = `${9 * ratio}px IBM Plex Mono`;
  ctx.fillText(`${xmin.toFixed(0)} m`, pad, h - 9 * ratio);
  ctx.fillText(`${xmax.toFixed(0)} m`, w - pad - 40 * ratio, h - 9 * ratio);
  const units =
    layer === 'elevation' ? 'metres' : layer === 'arrival' ? 'seconds' : 'seconds / metre';
  $('map-readout').textContent =
    `${layer === 'cost' ? 'Travel cost' : layer[0].toUpperCase() + layer.slice(1)} · ${min.toFixed(2)}–${max.toFixed(2)} ${units}`;
  draw3d();
}
function draw3d() {
  const canvas = $<HTMLCanvasElement>('terrain-3d');
  if (canvas.hidden || !scenario) return;
  const { ctx, w, h, ratio } = resize(canvas),
    z = scenario.elevation_m,
    rows = z.length,
    cols = z[0].length,
    flat = z.flat(),
    min = Math.min(...flat),
    max = Math.max(...flat);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#eef0e9';
  ctx.fillRect(0, 0, w, h);
  ctx.lineWidth = 0.7 * ratio;
  const project = (x: number, y: number, v: number) => [
    w * 0.5 + (x - y) * w * 0.34,
    h * 0.78 - (x + y) * h * 0.22 - ((v - min) / (max - min || 1)) * h * 0.35,
  ];
  for (let r = 0; r < rows; r += 3) {
    ctx.beginPath();
    for (let c = 0; c < cols; c++) {
      const p = project(c / (cols - 1), r / (rows - 1), z[r][c]);
      c ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
    }
    ctx.strokeStyle = `rgba(39,94,77,${0.25 + (0.5 * r) / rows})`;
    ctx.stroke();
  }
  for (let c = 0; c < cols; c += 4) {
    ctx.beginPath();
    for (let r = 0; r < rows; r++) {
      const p = project(c / (cols - 1), r / (rows - 1), z[r][c]);
      r ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
    }
    ctx.strokeStyle = 'rgba(66,89,72,.22)';
    ctx.stroke();
  }
}
function resultName(result: TerrainResult) {
  if (result.method === 'fast_marching') return 'Fast marching reference';
  const method = result.method === 'slsqp' ? 'SLSQP' : 'Euler–Lagrange';
  return `${method} · ${(labels[result.initialization] || result.initialization).replace('Local · ', '')}`;
}
function savedConfigText(config: TerrainConfig) {
  return config.method === 'fast_marching'
    ? `grid=${config.reference_grid_size}`
    : `N=${config.interior_points}, grid=${config.reference_grid_size}`;
}
function extractArrival() {
  arrival = null;
  const reference = results.find((r) => r.method === 'fast_marching');
  const raw = reference?.diagnostics?.arrival_time_s;
  if (Array.isArray(raw) && Array.isArray(raw[0])) arrival = raw as number[][];
}
function renderResults() {
  extractArrival();
  const ref = results.find((r) => r.method === 'fast_marching' && r.evaluated_cost_s !== null);
  $('cards').innerHTML = results.length
    ? results
        .map((r, i) => {
          const difference = recordedComparison
            ? recordedComparison.unresolved
              ? null
              : recordedComparison.difference
            : r.method !== 'fast_marching' && r.evaluated_cost_s !== null && ref?.evaluated_cost_s
              ? (100 * (r.evaluated_cost_s - ref.evaluated_cost_s)) / ref.evaluated_cost_s
              : null;
          const referenceLabel = recordedComparison
            ? recordedComparison.unresolved
              ? `unresolved · ${recordedComparison.reason || (recordedComparison.referenceGrid ? `${recordedComparison.referenceGrid} fallback; ${recordedComparison.declaredGrid || 1025} required` : `${recordedComparison.declaredGrid || 1025} required`)}`
              : `${difference === null ? '—' : `${difference >= 0 ? '+' : ''}${difference.toFixed(2)}%`} vs ${recordedComparison.referenceGrid || recordedComparison.declaredGrid || '—'} grid`
            : difference === null
              ? '—'
              : `${difference >= 0 ? '+' : ''}${difference.toFixed(2)}%`;
          const source =
            resultSource === 'imported'
              ? '<span class="pill bad">cached · unverified</span>'
              : resultSource === 'published'
                ? '<span class="pill">published record</span>'
                : '';
          const violations = r.evaluation?.violations.length
            ? ` · violations: ${escapeHtml(r.evaluation.violations.join(', '))}`
            : '';
          return `<article class="result-card"><h3><span style="color:${colors[i % colors.length]}">${resultName(r)}</span><span class="pill ${r.feasible ? '' : 'bad'}">${r.feasible ? 'feasible' : 'infeasible'}</span>${source}</h3><p class="cost-number">${r.evaluated_cost_s === null ? '—' : r.evaluated_cost_s.toFixed(1)} <span>s evaluated cost</span></p><div class="facts"><div>Route length<strong>${r.evaluation ? r.evaluation.length_m.toFixed(1) : '—'} m</strong></div><div>Grid difference<strong>${escapeHtml(referenceLabel)}</strong></div><div>Solver success<strong>${r.solver_success ? 'yes' : 'no'}</strong></div><div>Clearance<strong>${r.evaluation?.minimum_clearance_m == null ? '—' : `${r.evaluation.minimum_clearance_m.toFixed(1)} m`}</strong></div></div><p class="termination">Termination: ${escapeHtml(r.termination_reason)}${violations}${recordedComparison?.refinementChange == null ? '' : ` · reference refinement ${recordedComparison.refinementChange.toFixed(2)}%`}${recordedConfig ? ` · saved native config ${savedConfigText(recordedConfig)}` : ''}</p></article>`;
        })
        .join('')
    : resultSource === 'published' && recordedFailure
      ? `<article class="result-card"><h3>Recorded run · no planner result</h3><p>${escapeHtml(recordedFailure)}</p>${recordedConfig ? `<p class="termination">Saved native config ${savedConfigText(recordedConfig)}. Live controls were not changed.</p>` : ''}</article>`
      : '<p class="empty">Choose routes and run the comparison.</p>';
  $('legend').innerHTML = results
    .map(
      (r, i) =>
        `<span><i style="background:${colors[i % colors.length]}"></i>${escapeHtml(resultName(r))}</span>`,
    )
    .join('');
  drawProfile($<HTMLCanvasElement>('elevation-profile'), 'elevation_m');
  drawProfile($<HTMLCanvasElement>('cost-profile'), 'accumulated_cost_s');
  renderTerrainMeta();
  drawMap();
}
function renderTerrainMeta() {
  if (!scenario) return;
  const observed = scenario.metadata.kind === 'observed_elevation_with_illustrative_cost_model';
  $('terrain-meta').textContent = observed
    ? `${scenario.provenance.attribution || 'Bundled observed elevation.'} Projection: ${scenario.provenance.local_projection || 'local metric coordinates'}. Model: ${scenario.metadata.model_assumptions || scenario.metadata.cost_model}. Offline packaged data; no live elevation request.`
    : `Synthetic benchmark · ${scenario.metadata.cost_model || 'static isotropic positive travel cost'} · source field ${scenario.field_x_m.length} × ${scenario.field_y_m.length}.`;
}
function drawProfile(canvas: HTMLCanvasElement, key: 'elevation_m' | 'accumulated_cost_s') {
  const { ctx, w, h, ratio } = resize(canvas);
  ctx.clearRect(0, 0, w, h);
  const sets = results
    .map((r, i) => ({
      i,
      x: r.evaluation?.profile.distance_m || [],
      y: r.evaluation?.profile[key] || [],
    }))
    .filter((s) => s.x.length && s.y.length);
  if (!sets.length) {
    ctx.fillStyle = '#7b847e';
    ctx.font = `${11 * ratio}px DM Sans`;
    ctx.fillText('Run routes to see profiles.', 12 * ratio, 25 * ratio);
    return;
  }
  const values = sets.flatMap((s) => s.y),
    maxX = Math.max(...sets.flatMap((s) => s.x)),
    minY = Math.min(...values),
    maxY = Math.max(...values),
    pad = 28 * ratio;
  ctx.strokeStyle = '#d1d7cd';
  ctx.beginPath();
  ctx.moveTo(pad, 8 * ratio);
  ctx.lineTo(pad, h - pad);
  ctx.lineTo(w - 8 * ratio, h - pad);
  ctx.stroke();
  for (const s of sets) {
    ctx.beginPath();
    s.x.forEach((x, j) => {
      const px = pad + (x / (maxX || 1)) * (w - pad - 10 * ratio),
        py = h - pad - ((s.y[j] - minY) / (maxY - minY || 1)) * (h - pad - 12 * ratio);
      j ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.strokeStyle = colors[s.i % colors.length];
    ctx.lineWidth = 2 * ratio;
    ctx.stroke();
  }
  ctx.fillStyle = '#68756d';
  ctx.font = `${8 * ratio}px IBM Plex Mono`;
  ctx.fillText(minY.toFixed(1), 2 * ratio, h - pad);
  ctx.fillText(maxY.toFixed(1), 2 * ratio, 12 * ratio);
  ctx.fillText(`${maxX.toFixed(0)} m`, w - 48 * ratio, h - 8 * ratio);
}
function escapeHtml(value: unknown) {
  const div = document.createElement('div');
  div.textContent = String(value);
  return div.innerHTML;
}

function configs(): TerrainConfig[] {
  const interior_points = Number($<HTMLSelectElement>('local-n').value) as 32 | 64 | 128,
    reference_grid_size = Number($<HTMLSelectElement>('reference-n').value) as 129 | 257;
  const common = {
    version: 2 as const,
    interior_points,
    reference_grid_size,
    tolerance: 1e-7,
    max_iterations: 1000,
    time_limit_s: 60,
    profile_samples: 129,
  };
  const local: TerrainConfig[] = [];
  for (const method of document.querySelectorAll<HTMLInputElement>('[data-method]:checked'))
    for (const input of document.querySelectorAll<HTMLInputElement>('[data-init]:checked'))
      local.push({
        ...common,
        method: method.dataset.method as 'slsqp' | 'euler_lagrange',
        initialization: input.dataset.init!,
        options: {},
      });
  if ($<HTMLInputElement>('run-reference').checked)
    local.push({
      ...common,
      method: 'fast_marching',
      initialization: 'fast_marching',
      options: { include_arrival: $<HTMLInputElement>('include-arrival').checked },
    });
  return local;
}
function restoreConfigControls(imported: TerrainConfig[]) {
  if (!imported.length) return;
  const local = imported.filter((config) => config.method !== 'fast_marching');
  for (const input of document.querySelectorAll<HTMLInputElement>('[data-method]'))
    input.checked = local.some((config) => config.method === input.dataset.method);
  for (const input of document.querySelectorAll<HTMLInputElement>('[data-init]'))
    input.checked = local.some((config) => config.initialization === input.dataset.init);
  $<HTMLInputElement>('run-reference').checked = imported.some(
    (config) => config.method === 'fast_marching',
  );
  const interior = new Set(imported.map((config) => config.interior_points));
  const grids = new Set(imported.map((config) => config.reference_grid_size));
  if (interior.size === 1)
    $<HTMLSelectElement>('local-n').value = String(imported[0].interior_points);
  if (grids.size === 1)
    $<HTMLSelectElement>('reference-n').value = String(imported[0].reference_grid_size);
  $<HTMLInputElement>('include-arrival').checked = imported.some(
    (config) => config.method === 'fast_marching' && config.options.include_arrival === true,
  );
}
async function generateScenario() {
  scenario = validateTerrainScenario(
    await request('terrainGenerate', {
      family: $<HTMLSelectElement>('family').value,
      seed: $<HTMLInputElement>('seed').valueAsNumber,
      contrast: $<HTMLInputElement>('contrast').valueAsNumber,
      barriers: $<HTMLInputElement>('barriers').checked,
    }),
  );
  results = [];
  displayedConfigs = [];
  recordedComparison = null;
  recordedConfig = null;
  recordedFailure = null;
  arrival = null;
  controlsDirty = false;
  renderResults();
}
async function run() {
  terminate('Worker restarted.');
  $<HTMLButtonElement>('run').disabled = true;
  $<HTMLButtonElement>('cancel').disabled = false;
  message('');
  try {
    await boot();
    const token = generation;
    if (controlsDirty) await generateScenario();
    const selected = configs();
    if (!selected.length) throw new Error('Select at least one route.');
    status(`Solving ${selected.length} route${selected.length === 1 ? '' : 's'}…`);
    const solved = await request('terrainSolve', { scene: scenario, configs: selected });
    if (generation !== token) return;
    const checked = validateTerrainBundle({
      version: 2,
      scenario,
      configs: selected,
      results: solved,
    });
    results = checked.results || [];
    displayedConfigs = selected;
    resultSource = 'computed';
    recordedComparison = null;
    recordedConfig = null;
    recordedFailure = null;
    status('Comparison complete · results independently evaluated');
    renderResults();
  } catch (error) {
    if (!/Cancelled|restarted|Settings changed/.test(errorText(error))) {
      message(errorText(error));
      status('Run failed · use Run comparison to retry');
    }
  } finally {
    $<HTMLButtonElement>('run').disabled = false;
    $<HTMLButtonElement>('cancel').disabled = true;
  }
}

document.querySelectorAll<HTMLButtonElement>('[data-layer]').forEach(
  (button) =>
    (button.onclick = () => {
      layer = button.dataset.layer as TerrainLayer;
      document
        .querySelectorAll('[data-layer]')
        .forEach((el) => el.classList.toggle('active', el === button));
      if (layer === 'arrival' && !arrival)
        message('Run fast marching with “Keep full arrival-time layer” to display arrival time.');
      else message('');
      drawMap();
    }),
);
$('show-3d').onchange = () => {
  $('terrain-3d').hidden = !$<HTMLInputElement>('show-3d').checked;
  draw3d();
};
$('contrast').oninput = () => {
  $('contrast-value').textContent = `${$<HTMLInputElement>('contrast').valueAsNumber.toFixed(1)}×`;
  settingsChanged(true);
};
function settingsChanged(changesScenario: boolean) {
  if (changesScenario) controlsDirty = true;
  if (!$<HTMLButtonElement>('cancel').disabled) {
    terminate('Settings changed.');
    $<HTMLButtonElement>('run').disabled = false;
    $<HTMLButtonElement>('cancel').disabled = true;
    status('Settings changed · previous results retained for comparison');
    message('Run stopped because its settings changed.');
  } else if (results.length) status('Previous results · run to apply changed settings');
}
function syncFamilyControls() {
  const observed = $<HTMLSelectElement>('family').value === 'mount_tamalpais';
  for (const id of ['seed', 'contrast', 'barriers']) $<HTMLInputElement>(id).disabled = observed;
  if (observed)
    $('terrain-meta').textContent =
      'Mount Tamalpais elevation: Mapzen Terrain Tiles; USGS 3DEP and GMTED2010/SRTM. Local WGS84 metric projection; travel cost is an illustrative static slope model. Bundled offline.';
}
for (const id of ['family', 'seed', 'barriers'])
  $(id).onchange = () => {
    settingsChanged(true);
    if (id === 'family') syncFamilyControls();
  };
for (const element of document.querySelectorAll<HTMLInputElement | HTMLSelectElement>(
  '[data-method], [data-init], #run-reference, #local-n, #reference-n, #include-arrival',
))
  element.addEventListener('change', () => settingsChanged(false));
$('run').onclick = () => void run();
$('cancel').onclick = () => {
  terminate();
  status('Cancelled · validated preview remains available');
  $<HTMLButtonElement>('run').disabled = false;
  $<HTMLButtonElement>('cancel').disabled = true;
  message('Run cancelled.');
};
$('export').onclick = () => {
  const payload: TerrainBundle = {
    version: 2,
    scenario,
    configs: results.length ? displayedConfigs : configs(),
    results,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }),
    a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'terrain-scenario-v2.json';
  a.click();
  URL.revokeObjectURL(a.href);
};
$('import').onclick = () => $<HTMLInputElement>('file').click();
$('file').onchange = async () => {
  const input = $<HTMLInputElement>('file'),
    file = input.files?.[0];
  if (!file) return;
  terminate('Import started.');
  $<HTMLButtonElement>('run').disabled = false;
  $<HTMLButtonElement>('cancel').disabled = true;
  status('Importing scenario…');
  try {
    if (file.size > 8_000_000) throw new Error('Terrain file exceeds 8 MB.');
    const raw = JSON.parse(await file.text());
    try {
      const bundle = validateTerrainBundle(raw);
      scenario = bundle.scenario;
      results = bundle.results || [];
      displayedConfigs = bundle.configs || [];
      restoreConfigControls(displayedConfigs);
      resultSource = 'imported';
      recordedComparison = null;
      recordedConfig = null;
      recordedFailure = null;
    } catch (error) {
      if (errorText(error) !== 'VERSION_1') throw error;
      await boot();
      scenario = validateTerrainScenario(
        await request('terrainAdaptV1', { scene: validateScene(raw) as Scene }),
      );
      results = [];
      displayedConfigs = [];
      resultSource = 'imported';
      recordedComparison = null;
      recordedConfig = null;
      recordedFailure = null;
    }
    controlsDirty = false;
    arrival = null;
    renderResults();
    message(
      results.length
        ? 'Scenario and cached results imported. Run to independently evaluate them.'
        : 'Scenario imported.',
    );
    status(
      results.length
        ? 'Imported cached results · unverified until rerun'
        : 'Imported scenario · ready to run',
    );
  } catch (error) {
    message(errorText(error));
  } finally {
    input.value = '';
  }
};
new ResizeObserver(() => {
  drawMap();
  renderResults();
}).observe($('terrain-map'));

async function loadStudy() {
  try {
    const response = await fetch(new URL('study/index.json', base));
    if (!response.ok) throw new Error();
    const study = await response.json();
    const scope = study.scope || {};
    $('study-status').textContent =
      `${study.title || 'Recorded study'} · ${study.runtime_label || 'native'} · ${scope.statement || scope.profile || 'published scope'}`;
    const assetResponse = await fetch(new URL('study/assets.json', base));
    const assets: string[] = assetResponse.ok ? await assetResponse.json() : [];
    const artifactLabels: Record<string, string> = {
      'summary.csv': 'Summary CSV',
      'paired.csv': 'Paired comparisons',
      'initialization-sensitivity.csv': 'Initialization sensitivity',
      'local-refinement.csv': 'Local refinement',
      'paired-outcomes.csv': 'Paired outcomes',
    };
    $('study-artifacts').innerHTML = assets
      .filter(
        (name) =>
          name === 'study-report.md' || name in artifactLabels || /\.(png|svg|pdf)$/.test(name),
      )
      .map((name) => {
        if (name === 'study-report.md') {
          const rendered =
            'https://github.com/twallengren/path-planning-ode/blob/master/docs/study-report.md';
          return `<a href="${rendered}">Study report</a><a href="${escapeHtml(new URL(`study/${name}`, base).href)}" download>Markdown source</a>`;
        }
        return `<a href="${escapeHtml(new URL(`study/${name}`, base).href)}">${escapeHtml(artifactLabels[name] || name)}</a>`;
      })
      .join('');
    const rows = study.aggregate || study.summary || [];
    $('aggregate').innerHTML = rows.length
      ? `<table class="study-table"><thead><tr>${Object.keys(rows[0])
          .map((k) => `<th>${escapeHtml(k.replaceAll('_', ' '))}</th>`)
          .join('')}</tr></thead><tbody>${rows
          .map(
            (row: Record<string, unknown>) =>
              `<tr>${Object.values(row)
                .map((v) => `<td>${v == null ? '—' : escapeHtml(v)}</td>`)
                .join('')}</tr>`,
          )
          .join('')}</tbody></table>`
      : '';
    $('runs').innerHTML = (study.runs || [])
      .map(
        (run: unknown, i: number) =>
          `<details class="study-runs"><summary>Run ${i + 1} · ${escapeHtml((run as any).case_id || 'record')}</summary><button class="study-load" data-study-run="${i}">Load scenario and result</button><pre>${escapeHtml(JSON.stringify(run, null, 2))}</pre></details>`,
      )
      .join('');
    document
      .querySelectorAll<HTMLButtonElement>('[data-study-run]')
      .forEach(
        (button) =>
          (button.onclick = () => void loadStudyRun(study.runs[Number(button.dataset.studyRun)])),
      );
  } catch {
    $('study-status').textContent =
      'No recorded study is bundled in this build yet. The live laboratory remains available.';
  }
}
async function loadStudyRun(run: any) {
  terminate('Recorded run selected.');
  $<HTMLButtonElement>('run').disabled = false;
  $<HTMLButtonElement>('cancel').disabled = true;
  try {
    let recordedResult = run.result;
    if (run.result_ref) {
      const response = await fetch(new URL(`study/${run.result_ref}`, base));
      if (!response.ok || !response.body)
        throw new Error('The detailed recorded result is unavailable.');
      const body = response.body.pipeThrough(new DecompressionStream('gzip'));
      recordedResult = await new Response(body).json();
    }
    let rawScenario = run.scenario;
    const scenarioHash = run.scenario_hash || recordedResult?.scenario_hash;
    if (!rawScenario && scenarioHash) {
      const response = await fetch(new URL(`study/scenarios/${scenarioHash}.json.gz`, base));
      if (response.ok && response.body) {
        const body = response.body.pipeThrough(new DecompressionStream('gzip'));
        rawScenario = await new Response(body).json();
      }
    }
    if (!rawScenario && run.scenario_ref) {
      let response = await fetch(new URL(String(run.scenario_ref), base));
      if (!response.ok)
        response = await fetch(new URL(String(run.scenario_ref), new URL('study/', base)));
      if (!response.ok) throw new Error('The recorded scenario asset is unavailable.');
      rawScenario = await response.json();
    }
    scenario = validateTerrainScenario(rawScenario);
    const rawResults = run.results || (recordedResult ? [recordedResult] : []);
    results = rawResults.map((result: unknown) => validateTerrainResult(result, scenario));
    displayedConfigs = [];
    recordedConfig = null;
    if (run.config) {
      try {
        recordedConfig = validateTerrainConfig(run.config, false);
        displayedConfigs = [recordedConfig];
      } catch {
        recordedConfig = null;
      }
    }
    resultSource = 'published';
    const finiteNumber = (value: unknown): value is number =>
      typeof value === 'number' && Number.isFinite(value);
    recordedComparison = {
      difference: finiteNumber(run.reference_difference_percent)
        ? run.reference_difference_percent
        : null,
      unresolved: run.reference_unresolved === true,
      referenceGrid: finiteNumber(run.reference_grid_size) ? run.reference_grid_size : null,
      declaredGrid: finiteNumber(run.declared_finest_reference_grid_size)
        ? run.declared_finest_reference_grid_size
        : null,
      refinementChange: finiteNumber(run.reference_refinement_change_percent)
        ? run.reference_refinement_change_percent
        : null,
      reason:
        typeof run.reference_unresolved_reason === 'string'
          ? run.reference_unresolved_reason
          : null,
    };
    const failure = run.error;
    recordedFailure = results.length
      ? null
      : typeof failure === 'string'
        ? failure
        : failure
          ? JSON.stringify(failure)
          : String(run.record_status || 'The native worker produced no planner result.');
    controlsDirty = false;
    renderResults();
    status(
      `Recorded native run · ${run.case_id || 'study case'}${recordedConfig ? ` · saved ${savedConfigText(recordedConfig)}; live controls unchanged` : ''}`,
    );
    $('comparison-title').scrollIntoView({ behavior: 'smooth' });
  } catch (error) {
    message(errorText(error));
  }
}
async function start() {
  try {
    const response = await fetch(new URL('terrain/preview.json', base));
    if (!response.ok) throw new Error('Preview unavailable.');
    const bundle = validateTerrainBundle(await response.json());
    scenario = bundle.scenario;
    results = bundle.results || [];
    displayedConfigs = bundle.configs || [];
    resultSource = 'preview';
    recordedComparison = null;
    recordedConfig = null;
    recordedFailure = null;
    renderResults();
    status('Precomputed validated preview · Python loads when you run');
    syncFamilyControls();
  } catch (error) {
    message(errorText(error));
  }
  void loadStudy();
}
void start();
