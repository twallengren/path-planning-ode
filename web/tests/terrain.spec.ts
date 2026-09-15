import { test, expect, type Page } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';

const python = existsSync('.venv/bin/python') ? '.venv/bin/python' : 'python';

function fixture(name = 'uniform') {
  return execFileSync(
    python,
    [
      '-c',
      `import json; from path_planning_ode.terrain_generators import validation_terrain; print(json.dumps(validation_terrain(${JSON.stringify(name)}).to_dict()))`,
    ],
    { encoding: 'utf8' },
  );
}

function softWallFixture() {
  return execFileSync(
    python,
    [
      '-c',
      'import json; from path_planning_ode.soft_walls import soften_walls; from path_planning_ode.terrain_generators import obstacle_detour_fixture; print(json.dumps(soften_walls(obstacle_detour_fixture()).to_dict()))',
    ],
    { encoding: 'utf8' },
  );
}

async function captureTerrainResults(page: Page) {
  await page.addInitScript(() => {
    const OriginalWorker = window.Worker;
    (window as any).terrainResults = [];
    window.Worker = class extends OriginalWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener('message', ({ data }) => {
          if (Array.isArray(data.result) && data.result[0]?.version === 2)
            (window as any).terrainResults = data.result;
        });
      }
    };
  });
}

async function openTerrain(page: Page) {
  await page.goto('./terrain.html');
  await expect(page.locator('#runtime')).toContainText('Precomputed validated preview');
  await expect(page.locator('#terrain-map')).toBeVisible();
  await expect(page.locator('#cards .result-card')).toHaveCount(1);
}

test('terrain preview, native parity, profiles, layers, and v2 export', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.addInitScript(() => {
    const OriginalWorker = window.Worker;
    (window as any).terrainResults = [];
    window.Worker = class extends OriginalWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener('message', ({ data }) => {
          if (Array.isArray(data.result) && data.result[0]?.version === 2)
            (window as any).terrainResults = data.result;
        });
      }
    };
  });
  await openTerrain(page);
  await expect(page.locator('#map-readout')).toHaveText(/Elevation .* metres$/);
  await page.locator('[data-layer="cost"]').click();
  await expect(page.locator('#map-readout')).toHaveText(/Travel cost .* seconds \/ metre$/);
  await page.locator('[data-layer="elevation"]').click();
  await page.locator('#file').setInputFiles({
    name: 'uniform-v2.json',
    mimeType: 'application/json',
    buffer: Buffer.from(fixture()),
  });
  await expect(page.locator('#message')).toHaveText('Scenario imported.');
  await page.locator('[data-init="arc_left"]').uncheck();
  await page.locator('[data-init="arc_right"]').uncheck();
  await page.locator('[data-method="euler_lagrange"]').check();
  await page.locator('#include-arrival').check();
  await page.locator('#run').click();
  await expect(page.locator('#runtime')).toContainText('Comparison complete', { timeout: 90_000 });
  await expect(page.locator('#cards .result-card')).toHaveCount(3);
  await expect(page.locator('#cards')).toContainText('Solver success');
  await expect(page.locator('#cards')).toContainText('Grid difference');
  await page.locator('[data-layer="arrival"]').click();
  await expect(page.locator('#map-readout')).toHaveText(/Arrival .* seconds$/);
  await page.locator('#show-3d').check();
  await expect(page.locator('#terrain-3d')).toBeVisible();

  const browser = await page.evaluate(() => (window as any).terrainResults);
  const native = JSON.parse(
    execFileSync(
      python,
      [
        '-c',
        "import json; from path_planning_ode.terrain_generators import validation_terrain; from path_planning_ode.terrain import PlannerConfig; from path_planning_ode.planners import plan; s=validation_terrain('uniform'); cs=[PlannerConfig(method=m, initialization='straight', interior_points=32, reference_grid_size=129, options=({'include_arrival':True} if m=='fast_marching' else {})) for m in ('slsqp','euler_lagrange','fast_marching')]; print(json.dumps([plan(s,c).to_dict() for c in cs]))",
      ],
      { encoding: 'utf8' },
    ),
  );
  for (let i = 0; i < native.length; i++) {
    expect(browser[i].feasible).toBe(native[i].feasible);
    expect(browser[i].evaluated_cost_s).toBeCloseTo(native[i].evaluated_cost_s, 7);
  }
  await page.locator('summary').filter({ hasText: 'Import & export' }).click();
  const download = page.waitForEvent('download');
  await page.locator('#export').click();
  expect((await download).suggestedFilename()).toBe('terrain-scenario-v2.json');

  await page.locator('#file').setInputFiles({
    name: 'heterogeneous-v2.json',
    mimeType: 'application/json',
    buffer: Buffer.from(fixture('symmetry')),
  });
  await page.locator('#run-reference').uncheck();
  await page.locator('#run').click();
  await expect(page.locator('#runtime')).toContainText('Comparison complete', { timeout: 90_000 });
  const heterogeneousBrowser = await page.evaluate(() => (window as any).terrainResults);
  const heterogeneousNative = JSON.parse(
    execFileSync(
      python,
      [
        '-c',
        "import json; from path_planning_ode.terrain_generators import validation_terrain; from path_planning_ode.terrain import PlannerConfig; from path_planning_ode.planners import plan; s=validation_terrain('symmetry'); cs=[PlannerConfig(method=m, initialization='straight', interior_points=32, reference_grid_size=129) for m in ('slsqp','euler_lagrange')]; print(json.dumps([plan(s,c).to_dict() for c in cs]))",
      ],
      { encoding: 'utf8' },
    ),
  );
  for (let i = 0; i < heterogeneousNative.length; i++) {
    expect(heterogeneousBrowser[i].feasible).toBe(heterogeneousNative[i].feasible);
    expect(heterogeneousBrowser[i].evaluated_cost_s).toBeCloseTo(
      heterogeneousNative[i].evaluated_cost_s,
      6,
    );
  }
  expect(errors).toEqual([]);
});

test('soft walls are the live default and share finite costs across browser planners', async ({
  page,
}) => {
  await captureTerrainResults(page);
  await openTerrain(page);
  await expect(page.locator('#wall-model')).toHaveValue('soft');
  await expect(page.locator('#wall-strength')).toHaveValue('100');
  await expect(page.locator('#model-indicator')).toContainText(
    'finite high-cost walls · 100× nominal strength',
  );
  await expect(page.locator('#wall-model-help')).toHaveText(
    'Crossings are allowed and charged through the shared cost field.',
  );
  await expect(page.locator('[data-init="barrier"]')).toBeDisabled();

  await page.locator('#file').setInputFiles({
    name: 'soft-walls-v2.json',
    mimeType: 'application/json',
    buffer: Buffer.from(softWallFixture()),
  });
  await page.locator('[data-init="arc_left"]').uncheck();
  await page.locator('[data-init="arc_right"]').uncheck();
  await page.locator('[data-method="euler_lagrange"]').check();
  await page.locator('#run').click();
  await expect(page.locator('#runtime')).toContainText('Comparison complete', { timeout: 90_000 });
  const browser = await page.evaluate(() => (window as any).terrainResults);
  const native = JSON.parse(
    execFileSync(
      python,
      [
        '-c',
        "import json; from path_planning_ode.soft_walls import soften_walls; from path_planning_ode.terrain_generators import obstacle_detour_fixture; from path_planning_ode.terrain import PlannerConfig; from path_planning_ode.planners import plan; s=soften_walls(obstacle_detour_fixture()); cs=[PlannerConfig(method=m, initialization=('fast_marching' if m=='fast_marching' else 'straight'), interior_points=32, reference_grid_size=129) for m in ('slsqp','euler_lagrange','fast_marching')]; print(json.dumps([plan(s,c).to_dict() for c in cs]))",
      ],
      { encoding: 'utf8' },
    ),
  );
  expect(browser).toHaveLength(3);
  for (let i = 0; i < native.length; i++) {
    expect(browser[i].feasible).toBe(native[i].feasible);
    expect(browser[i].evaluation.violations).not.toContain('barrier_collision');
    expect(browser[i].evaluated_cost_s).toBeCloseTo(native[i].evaluated_cost_s, 6);
  }

  await page.locator('#wall-model').selectOption('hard');
  await expect(page.locator('#barriers-label')).toHaveText('Include impassable barriers');
  await expect(page.locator('[data-init="barrier"]')).toBeEnabled();
  await expect(page.locator('#model-indicator')).toContainText('finite high-cost walls');
  await page.locator('[data-method="slsqp"]').uncheck();
  await page.locator('[data-method="euler_lagrange"]').uncheck();
  await page.locator('#run').click();
  await expect(page.locator('#runtime')).toContainText('Comparison complete', { timeout: 90_000 });
  await expect(page.locator('#model-indicator')).toContainText('hard barriers · impassable');

  await page.locator('#wall-model').selectOption('soft');
  await page.locator('#wall-strength').fill('25');
  await page.locator('#run').click();
  await expect(page.locator('#runtime')).toContainText('Comparison complete', { timeout: 90_000 });
  await expect(page.locator('#model-indicator')).toContainText('25× nominal strength');
  await page.locator('summary').filter({ hasText: 'Import & export' }).click();
  const download = page.waitForEvent('download');
  await page.locator('#export').click();
  const downloaded = await download;
  const exported = JSON.parse(readFileSync((await downloaded.path())!, 'utf8'));
  expect(exported.scenario.barriers_geojson).toEqual([]);
  expect(exported.scenario.metadata.soft_walls.multiplier).toBe(25);
  expect(exported.scenario.metadata.soft_walls.geometry_geojson.length).toBeGreaterThan(0);
  expect(exported.results).toHaveLength(1);
  expect(exported.results[0].method).toBe('fast_marching');

  const nativeGenerated = JSON.parse(
    execFileSync(
      python,
      [
        '-c',
        "import json; from path_planning_ode.soft_walls import soften_walls; from path_planning_ode.terrain_generators import synthetic_terrain; print(json.dumps(soften_walls(synthetic_terrain('ridge_pass', seed=0, contrast=1.0, barriers=True), multiplier=25).to_dict()))",
      ],
      { encoding: 'utf8' },
    ),
  );
  for (const [row, column] of [
    [0, 0],
    [16, 32],
    [32, 32],
    [48, 20],
    [64, 64],
  ])
    expect(exported.scenario.log_slowness[row][column]).toBeCloseTo(
      nativeGenerated.log_slowness[row][column],
      12,
    );
});

test('lazy terrain dependency failure keeps preview and retry recovers', async ({ page }) => {
  await page.route('**/runtime/scipy-*.whl', (route) => route.abort());
  await openTerrain(page);
  await page.locator('[data-init="arc_left"]').uncheck();
  await page.locator('[data-init="arc_right"]').uncheck();
  await page.locator('[data-init="straight"]').uncheck();
  await page.locator('#run').click();
  await expect(page.locator('#runtime')).toContainText('Run failed', { timeout: 60_000 });
  await expect(page.locator('#cards .result-card')).toHaveCount(1);
  await page.unroute('**/runtime/scipy-*.whl');
  await page.locator('#run').click();
  await expect(page.locator('#runtime')).toContainText('Comparison complete', { timeout: 90_000 });
  await expect(page.locator('#cards .result-card')).toHaveCount(1);
});

test('cancellation rejects stale work, retry recovers, and v1 imports adapt', async ({ page }) => {
  await openTerrain(page);
  await page.locator('#run').click();
  await expect(page.locator('#cancel')).toBeEnabled();
  await page.locator('#seed').fill('3');
  await page.locator('#seed').press('Tab');
  await expect(page.locator('#runtime')).toContainText('Settings changed');
  await expect(page.locator('#run')).toBeEnabled();
  await page.locator('#file').setInputFiles({
    name: 'legacy.json',
    mimeType: 'application/json',
    buffer: Buffer.from(
      JSON.stringify({
        version: 1,
        start: [0, 0],
        end: [10, 10],
        obstacles: [],
        guesses: ['straight'],
        options: { interior_points: 12, max_iterations: 20, tolerance: 1e-7, mode: 'damped' },
      }),
    ),
  });
  await expect(page.locator('#message')).toHaveText('Scenario imported.', { timeout: 90_000 });
  await expect(page.locator('#runtime')).toContainText('ready');
});

test('import stops active work and keeps imported cached results unverified', async ({ page }) => {
  await openTerrain(page);
  const bundle = await page.evaluate(async () => (await fetch('./terrain/preview.json')).json());
  await page.locator('#run').click();
  await expect(page.locator('#cancel')).toBeEnabled();
  await page.locator('#file').setInputFiles({
    name: 'cached-v2.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(bundle)),
  });
  await expect(page.locator('#runtime')).toContainText('Imported cached results');
  await expect(page.locator('#runtime')).toContainText('unverified until rerun');
  await expect(page.locator('#cards')).toContainText('cached · unverified');
  await expect(page.locator('#cards .result-card')).toHaveCount(1);
  await page.waitForTimeout(1_000);
  await expect(page.locator('#runtime')).toContainText('Imported cached results');
  await expect(page.locator('#runtime')).toContainText('unverified until rerun');
  await expect(page.locator('#cards .result-card')).toHaveCount(1);
});

test('v2 import rejects malformed results and exports their original configuration', async ({
  page,
}) => {
  await openTerrain(page);
  const bundle = await page.evaluate(async () => (await fetch('./terrain/preview.json')).json());
  const malformed = structuredClone(bundle);
  malformed.results[0].route_m[0] = ['not-a-coordinate', 0];
  await page.locator('#file').setInputFiles({
    name: 'malformed-v2.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(malformed)),
  });
  await expect(page.locator('#message')).toContainText('Planner route is invalid');

  const outsideFailure = structuredClone(bundle);
  const [xmin] = outsideFailure.scenario.bounds_m;
  outsideFailure.results[0].route_m[0][0] = xmin - 1;
  outsideFailure.results[0].evaluated_cost_s = null;
  outsideFailure.results[0].feasible = false;
  outsideFailure.results[0].solver_success = false;
  outsideFailure.results[0].termination_reason = 'evaluation_failed';
  outsideFailure.results[0].evaluation.cost_s = null;
  outsideFailure.results[0].evaluation.feasible = false;
  outsideFailure.results[0].evaluation.minimum_clearance_m = null;
  outsideFailure.results[0].evaluation.violations = ['outside_domain'];
  for (const values of Object.values(outsideFailure.results[0].evaluation.profile))
    (values as unknown[]).length = 0;
  await page.locator('#file').setInputFiles({
    name: 'outside-failure-v2.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(outsideFailure)),
  });
  await expect(page.locator('#cards')).toContainText('cached · unverified');
  await expect(page.locator('#cards')).toContainText('infeasible');

  const forgedFeasible = structuredClone(outsideFailure);
  forgedFeasible.results[0].feasible = true;
  forgedFeasible.results[0].evaluation.feasible = true;
  forgedFeasible.results[0].evaluation.violations = [];
  await page.locator('#file').setInputFiles({
    name: 'forged-feasible-v2.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(forgedFeasible)),
  });
  await expect(page.locator('#message')).toContainText('feasible planner route');

  await page.locator('#local-n').selectOption('128');
  await page.locator('#reference-n').selectOption('257');
  await page.locator('#file').setInputFiles({
    name: 'valid-v2.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(bundle)),
  });
  await expect(page.locator('#local-n')).toHaveValue(String(bundle.configs[0].interior_points));
  await expect(page.locator('#reference-n')).toHaveValue(
    String(bundle.configs[0].reference_grid_size),
  );
  await page.locator('#local-n').selectOption('128');
  await page.locator('#reference-n').selectOption('257');
  await page.locator('summary').filter({ hasText: 'Import & export' }).click();
  const download = page.waitForEvent('download');
  await page.locator('#export').click();
  const downloaded = await download;
  const exported = JSON.parse(readFileSync((await downloaded.path())!, 'utf8'));
  expect(exported.configs).toEqual(bundle.configs);
});

test('terrain controls and results do not overflow a narrow screen', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openTerrain(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator('#terrain-map').scrollIntoViewIfNeeded();
  await expect(page.locator('.control-panel')).toBeVisible();
  await expect(page.locator('#aggregate tbody tr')).toHaveCount(308);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('published study artifacts and a real recorded run load into the explorer', async ({
  page,
}) => {
  await openTerrain(page);
  await expect(page.locator('#study-status')).toContainText('Complete frozen study matrix');
  await expect(page.locator('#study-artifacts')).toContainText('Study report');
  await expect(page.locator('#study-artifacts')).toContainText('Summary CSV');
  await expect(page.locator('.study-runs')).toHaveCount(2816);
  await page.locator('.study-runs').first().locator('summary').click();
  await page.locator('.study-load').first().click();
  await expect(page.locator('#runtime')).toContainText('Recorded native run');
  await expect(page.locator('#runtime')).toContainText('hard barriers · impassable');
  await expect(page.locator('#model-indicator')).toContainText('hard barriers · impassable');
  await expect(page.locator('#wall-model')).toHaveValue('soft');
  await expect(page.locator('#cards .result-card')).toHaveCount(1);
  await expect(page.locator('#cards')).toContainText('evaluated cost');
  await expect(page.locator('#cards')).toContainText(/unresolved|vs \d+ grid/);
});

test('full published records lazily load native-grid, nonconverged, collision, and timeout cases', async ({
  page,
}) => {
  await openTerrain(page);
  const selectors = await page.evaluate(async () => {
    const study = await (await fetch('./study/index.json')).json();
    const find = (predicate: (run: any) => boolean) => study.runs.findIndex(predicate);
    return {
      fmm1025: find(
        (run: any) =>
          run.config?.method === 'fast_marching' &&
          run.config?.reference_grid_size === 1025 &&
          run.result_ref,
      ),
      nonconverged: find(
        (run: any) => run.result?.feasible && !run.result?.solver_success && run.result_ref,
      ),
      collision: find(
        (run: any) =>
          run.result?.evaluation?.violations?.includes('barrier_collision') && run.result_ref,
      ),
      timeout: find((run: any) => run.record_status === 'timeout' && !run.result_ref),
    };
  });
  expect(Object.values(selectors).every((index) => index >= 0)).toBe(true);
  const load = async (index: number) => {
    const record = page.locator('.study-runs').nth(index);
    await record.locator('summary').click();
    await record.locator('.study-load').click();
    await expect(page.locator('#runtime')).toContainText('Recorded native run');
  };

  await load(selectors.fmm1025);
  await expect(page.locator('#cards')).toContainText('saved native config');
  await expect(page.locator('#cards')).toContainText('grid=1025');
  await expect(page.locator('#reference-n')).toHaveValue('129');

  await load(selectors.nonconverged);
  await expect(page.locator('#cards')).toContainText('feasible');
  await expect(
    page.locator('#cards .facts div').filter({ hasText: 'Solver success' }),
  ).toContainText('no');

  await load(selectors.collision);
  await expect(page.locator('#cards')).toContainText('barrier_collision');
  await expect(page.locator('#cards')).toContainText('infeasible');

  await load(selectors.timeout);
  await expect(page.locator('#cards')).toContainText('no planner result');
  await expect(page.locator('#cards')).toContainText('HardTimeout');
});
