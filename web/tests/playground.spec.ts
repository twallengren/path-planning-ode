import { expect, test, type Page } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';

const python = existsSync('.venv/bin/python') ? '.venv/bin/python' : 'python';

async function instrument(page: Page) {
  await page.addInitScript(() => {
    const OriginalWorker = window.Worker;
    (window as any).playgroundRequests = [];
    (window as any).playgroundResults = [];
    window.Worker = class extends OriginalWorker {
      postMessage(message: unknown, options?: StructuredSerializeOptions | Transferable[]) {
        (window as any).playgroundRequests.push(structuredClone(message));
        super.postMessage(message, options as StructuredSerializeOptions);
      }
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener('message', ({ data }) => {
          if (data.result?.revision !== undefined)
            (window as any).playgroundResults.push(structuredClone(data.result));
        });
      }
    };
  });
}

async function ready(page: Page) {
  await page.goto('./');
  await expect(page.locator('#playground-runtime')).toContainText('Ready', { timeout: 60_000 });
  await expect(page.locator('#playground-travel-cost')).not.toHaveText('—', { timeout: 60_000 });
}

async function saveScene(page: Page) {
  const download = page.waitForEvent('download');
  await page.locator('#playground-save').evaluate((button: HTMLButtonElement) => button.click());
  const path = await (await download).path();
  return JSON.parse(readFileSync(path!, 'utf8'));
}

async function upload(page: Page, name: string, value: unknown) {
  await page.locator('#playground-file').setInputFiles({
    name,
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(value)),
  });
}

async function modelToClient(page: Page, point: number[], bounds: number[]) {
  const canvas = page.locator('#playground-canvas'),
    box = (await canvas.boundingBox())!;
  const worldWidth = bounds[1] - bounds[0],
    worldHeight = bounds[3] - bounds[2],
    scale = Math.min(box.width / worldWidth, box.height / worldHeight),
    drawWidth = worldWidth * scale,
    drawHeight = worldHeight * scale,
    offsetX = (box.width - drawWidth) / 2,
    offsetY = (box.height - drawHeight) / 2;
  return {
    x: box.x + offsetX + (point[0] - bounds[0]) * scale,
    y: box.y + offsetY + (bounds[3] - point[1]) * scale,
  };
}

const terrainFixture = () =>
  JSON.parse(
    execFileSync(
      python,
      [
        '-c',
        `import json
from path_planning_ode.terrain import TerrainScenario
x=(0.,50.,100.,200.); y=(0.,25.,50.,100.)
e=[[10.+i+j for i in range(4)] for j in range(4)]
l=[[0.0 for i in range(4)] for j in range(4)]
s=TerrainScenario(version=2,name='nonsquare',bounds_m=(0.,0.,200.,100.),start_m=(10.,10.),goal_m=(190.,90.),field_x_m=x,field_y_m=y,elevation_m=e,log_slowness=l,barriers_geojson=({'type':'Polygon','coordinates':[[[90.,0.],[110.,0.],[110.,45.],[90.,45.],[90.,0.]]]},),provenance={'attribution':'Fixture source','registry_url':'javascript:alert(1)'},metadata={})
print(json.dumps(s.to_dict(),allow_nan=False))`,
      ],
      { encoding: 'utf8', env: { ...process.env, PYTHONPATH: 'src' } },
    ),
  );

test('automatic worker converges with native parity and pause rejects an in-flight result', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await instrument(page);
  await ready(page);
  expect(await page.locator('#playground-method').count()).toBe(0);
  await expect(page.locator('#playground-status')).toHaveText('Settled', { timeout: 60_000 });
  const browser = await page.evaluate(() =>
    (window as any).playgroundResults.filter((item: any) => item.path).at(-1),
  );
  expect(
    await page.evaluate(() =>
      (window as any).playgroundResults.some((item: any) => item.metrics?.phase === 'newton'),
    ),
  ).toBe(true);
  const native = JSON.parse(
    execFileSync(
      python,
      [
        '-c',
        `import json
from path_planning_ode.field_adapter import build_playground_preset,PlaygroundField
from path_planning_ode.playground import PlaygroundOptions,initialize_playground,advance_playground
p=build_playground_preset('random_hills',0); f=PlaygroundField.from_spec(p['base_field'],p['gaussians'],bounds=(-6.,-4.,6.,4.)); o=PlaygroundOptions(interior_points=32,method='auto',tolerance=1e-5,max_iterations=2000); s=initialize_playground(p['start'],p['end'],field=f,options=o)
while s.status=='running': s=advance_playground(s)
d=s.to_dict(); print(json.dumps({'path':d['path'],'metrics':d['metrics'],'iteration':d['iteration'],'status':d['status']}))`,
      ],
      { encoding: 'utf8', env: { ...process.env, PYTHONPATH: 'src' } },
    ),
  );
  expect(browser.metrics.route_cost).toBeCloseTo(native.metrics.route_cost, 8);
  expect(browser.metrics.energy).toBeCloseTo(native.metrics.energy, 8);
  expect(browser.path[17][0]).toBeCloseTo(native.path[17][0], 8);
  expect(browser.path[17][1]).toBeCloseTo(native.path[17][1], 8);
  await page.locator('#playground-reset').click();
  await page.locator('#playground-toggle').click();
  await expect(page.locator('#playground-status')).toHaveText('Paused');
  const paused = await saveScene(page);
  await page.waitForTimeout(500);
  expect((await saveScene(page)).path).toEqual(paused.path);
  await page.locator('#playground-toggle').click();
  await expect(page.locator('#playground-status')).not.toHaveText('Paused');

  const beforeKeep = await page.evaluate(
    () =>
      (window as any).playgroundRequests.filter(
        (item: any) => item.action === 'playgroundInitialize',
      ).length,
  );
  await page.locator('#playground-keep').click();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as any).playgroundRequests.filter(
            (item: any) => item.action === 'playgroundInitialize',
          ).length,
      ),
    )
    .toBeGreaterThan(beforeKeep);
  expect(errors).toEqual([]);
});

test('grab pin, draggable hills, live paint, erase, undo, and straight reset preserve edits', async ({
  page,
}) => {
  await instrument(page);
  await ready(page);
  await expect(page.locator('#playground-status')).toHaveText('Settled', { timeout: 60_000 });
  const original = await saveScene(page),
    index = Math.floor(original.path.length / 2);
  const at = await modelToClient(page, original.path[index], original.bounds);
  await page.mouse.move(at.x, at.y);
  await page.mouse.down();
  await page.mouse.move(at.x, at.y - 45, { steps: 6 });
  await expect
    .poll(() =>
      page.evaluate(() => {
        const pinned = (window as any).playgroundResults.filter(
            (item: any) => item.state?.pin_index !== null && item.path,
          ),
          revision = pinned.at(-1)?.revision;
        return pinned.filter((item: any) => item.revision === revision).length;
      }),
    )
    .toBeGreaterThanOrEqual(2);
  const held = await page.evaluate(() => {
    const pinned = (window as any).playgroundResults.filter(
        (item: any) => item.state?.pin_index !== null && item.path,
      ),
      revision = pinned.at(-1).revision;
    return pinned.filter((item: any) => item.revision === revision).slice(-2);
  });
  expect(held[0].path[index]).toEqual(held[1].path[index]);
  expect(held[0].path[index - 1]).not.toEqual(held[1].path[index - 1]);
  await page.mouse.up();

  const afterRoute = await saveScene(page),
    hill = afterRoute.gaussians[0],
    hillAt = await modelToClient(page, [hill.x, hill.y], afterRoute.bounds);
  await page.mouse.move(hillAt.x, hillAt.y);
  await page.mouse.down();
  await page.mouse.move(hillAt.x + 35, hillAt.y + 10, { steps: 4 });
  await page.mouse.up();
  await expect.poll(async () => (await saveScene(page)).gaussians[0].x).not.toBe(hill.x);

  await page.locator('[data-playground-tool="paint"]').click();
  await page.locator('[data-playground-brush="wall"]').click();
  const canvas = page.locator('#playground-canvas'),
    box = (await canvas.boundingBox())!,
    beforePaint = await saveScene(page);
  await page.mouse.move(box.x + box.width * 0.48, box.y + box.height * 0.25);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.52, box.y + box.height * 0.75, { steps: 8 });
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as any).playgroundRequests
            .filter((item: any) => item.action === 'playgroundField')
            .at(-1)?.scene.strokes.length,
      ),
    )
    .toBe(beforePaint.strokes.length + 1);
  await page.mouse.up();
  await expect
    .poll(async () => (await saveScene(page)).strokes.length)
    .toBe(beforePaint.strokes.length + 1);
  await page.locator('#playground-undo').click();
  await expect
    .poll(async () => (await saveScene(page)).strokes.length)
    .toBe(beforePaint.strokes.length);
  await page.locator('#playground-redo').click();
  await expect
    .poll(async () => (await saveScene(page)).strokes.length)
    .toBe(beforePaint.strokes.length + 1);

  await page.locator('#playground-reset').click();
  const reset = await saveScene(page);
  expect(reset.strokes).toHaveLength(beforePaint.strokes.length + 1);
  reset.path.forEach((point: number[], i: number) => {
    const t = i / (reset.path.length - 1);
    expect(point[0]).toBeCloseTo(reset.start[0] + t * (reset.end[0] - reset.start[0]), 10);
    expect(point[1]).toBeCloseTo(reset.start[1] + t * (reset.end[1] - reset.start[1]), 10);
  });

  await page.locator('[data-playground-tool="erase"]').click();
  await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await expect
    .poll(async () => (await saveScene(page)).strokes.length)
    .toBe(beforePaint.strokes.length);
});

test('v2 and legacy files transact safely; nonsquare terrain hard walls become finite', async ({
  page,
}) => {
  await instrument(page);
  await ready(page);
  const saved = await saveScene(page);
  expect(saved).toMatchObject({ kind: 'path-playground', version: 2, seed: 0 });
  const uniform = structuredClone(saved);
  uniform.base_field = { kind: 'uniform', cost: 2, elevation: 3 };
  await upload(page, 'uniform-v2.json', uniform);
  await expect(page.locator('#playground-message')).toHaveText('Playground loaded.');
  expect((await saveScene(page)).base_field).toEqual(uniform.base_field);

  await upload(page, 'gaussian-v1.json', {
    version: 1,
    start: [-2, -2],
    end: [12, 12],
    obstacles: [
      { x: 5, y: 5, weight: 0, width: 2 },
      { x: 7, y: 7, weight: 100, width: 1 },
    ],
    guesses: ['straight'],
    options: { interior_points: 30, max_iterations: 50, tolerance: 1e-7, mode: 'damped' },
  });
  await expect(page.locator('#playground-message')).toHaveText('Gaussian v1 scene adapted.');
  const gaussian = await saveScene(page);
  expect(gaussian.gaussians.map((item: any) => item.weight)).toEqual([0, 100]);
  expect(gaussian.settings.method).toBe('auto');
  expect(gaussian.bounds[1]).toBeGreaterThan(12);

  await upload(page, 'terrain-v2.json', terrainFixture());
  await expect(page.locator('#playground-message')).toContainText('Hard barriers were converted', {
    timeout: 60_000,
  });
  const terrain = await saveScene(page);
  expect(terrain.bounds).toEqual([0, 200, 0, 100]);
  expect(terrain.start).toEqual([10, 10]);
  expect(terrain.end).toEqual([190, 90]);
  expect(terrain.base_field.scenario.bounds_m).toEqual([0, 0, 200, 100]);
  expect(terrain.base_field.scenario.barriers_geojson).toEqual([]);
  expect(terrain.base_field.scenario.metadata.soft_walls).toBeTruthy();
  expect(page.locator('#playground-attribution a')).toHaveCount(0);

  const corrupt = structuredClone(terrain);
  corrupt.base_field.scenario_hash = 'bad-hash';
  await upload(page, 'bad-hash.json', corrupt);
  await expect(page.locator('#playground-message')).toContainText('Import rejected');
  const afterReject = await saveScene(page);
  expect(afterReject.base_field.scenario_hash).toBe(terrain.base_field.scenario_hash);
  expect(afterReject.bounds).toEqual(terrain.bounds);
});

test('terrain cost/elevation units, seeded controls, Mount Tam attribution, and mobile keyboard work', async ({
  page,
}) => {
  await instrument(page);
  await ready(page);
  await page.locator('[data-playground-preset="ridge_pass"]').click();
  await expect(page.locator('#playground-map-scale')).toContainText('m wide', { timeout: 60_000 });
  await expect(page.locator('#playground-travel-cost')).toContainText('s', { timeout: 60_000 });
  await expect(page.locator('#playground-runtime')).toContainText('Ready');
  const ridge = await saveScene(page),
    ridgeBase = structuredClone(ridge.base_field),
    fieldBefore = await page.evaluate(
      () => (window as any).playgroundResults.filter((item: any) => item.field).at(-1).field,
    );
  expect(ridge.settings.brush_radius / (ridge.bounds[1] - ridge.bounds[0])).toBeCloseTo(0.07, 8);
  await page.locator('[data-playground-tool="paint"]').click();
  const terrainCanvas = page.locator('#playground-canvas'),
    terrainBox = (await terrainCanvas.boundingBox())!;
  await page.mouse.click(
    terrainBox.x + terrainBox.width * 0.35,
    terrainBox.y + terrainBox.height * 0.32,
  );
  await expect.poll(async () => (await saveScene(page)).strokes.length).toBe(1);
  const paintedTerrain = await saveScene(page);
  expect(paintedTerrain.base_field).toEqual(ridgeBase);
  await expect
    .poll(() =>
      page.evaluate((before: number[]) => {
        const after = (window as any).playgroundResults
          .filter((item: any) => item.field)
          .at(-1).field;
        return after.some((value: number, index: number) => value !== before[index]);
      }, fieldBefore),
    )
    .toBe(true);
  await page.locator('summary').filter({ hasText: 'Details' }).click();
  await expect(page.locator('#playground-show-elevation')).toBeEnabled();
  await page.locator('#playground-show-elevation').check();
  await expect(page.locator('#playground-layer-name')).toHaveText('Elevation');
  await expect(page.locator('#playground-layer-range')).toContainText('m');
  await page.locator('[data-playground-preset="blank"]').click();
  await expect(page.locator('#playground-seed')).toBeDisabled();
  await expect(page.locator('#playground-randomize')).toBeDisabled();
  await page.locator('[data-playground-preset="random_hills"]').click();
  await expect(page.locator('#playground-seed')).toBeEnabled();

  await page.locator('[data-playground-preset="mount_tamalpais"]').click();
  await expect(page.locator('#playground-attribution')).toBeVisible({ timeout: 60_000 });
  const source = page.locator('#playground-attribution a');
  await expect(source).toHaveAttribute('href', /^https?:/);
  const mount = await saveScene(page);
  await upload(page, 'mount-roundtrip.json', mount);
  await expect(page.locator('#playground-message')).toHaveText('Playground loaded.');
  await expect(page.locator('#playground-attribution')).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  const canvas = page.locator('#playground-canvas');
  await page.locator('summary').filter({ hasText: 'Scene file & keyboard' }).click();
  const before = await saveScene(page);
  await page.locator('#playground-keyboard-target').selectOption(`path:${before.path.length - 1}`);
  await page.locator('#playground-focus-canvas').click();
  await expect(canvas).toBeFocused();
  await page.keyboard.press('Shift+ArrowLeft');
  await expect.poll(async () => (await saveScene(page)).end[0]).toBeLessThan(before.end[0]);
  await page.screenshot({ path: 'test-results/unified-mobile.png', fullPage: true });
});

test('runtime failure keeps edits and retry restores the worker', async ({ page }) => {
  await page.route('**/runtime/pyodide.asm.wasm', (route) => route.abort());
  await page.goto('./');
  await expect(page.locator('#playground-retry')).toBeVisible({ timeout: 60_000 });
  await page.locator('[data-playground-tool="paint"]').click();
  const canvas = page.locator('#playground-canvas'),
    box = (await canvas.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.35, box.y + box.height * 0.4);
  await page.unroute('**/runtime/pyodide.asm.wasm');
  await page.locator('#playground-retry').click();
  await expect(page.locator('#playground-runtime')).toContainText('Ready', { timeout: 60_000 });
  await expect.poll(async () => (await saveScene(page)).strokes.length).toBe(1);
});
