import { test, expect, type Page } from '@playwright/test';
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
  await expect(page.locator('#playground-runtime')).toContainText('Python + NumPy ready', {
    timeout: 60_000,
  });
  await expect(page.locator('#playground-travel-cost')).not.toHaveText('—');
}

async function saveScene(page: Page) {
  const download = page.waitForEvent('download');
  await page.locator('#playground-save').evaluate((button: HTMLButtonElement) => button.click());
  const path = await (await download).path();
  return JSON.parse(readFileSync(path!, 'utf8'));
}

test('live Python solver matches native, pauses exactly, and switches methods', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await instrument(page);
  await ready(page);
  await expect(page.locator('#playground-status')).toContainText(/Adjusting|Settled/);
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as any).playgroundResults.filter((item: any) => item.path).at(-1)?.metrics.status,
      ),
    )
    .toBe('converged');
  const browser = await page.evaluate(() =>
    (window as any).playgroundResults.filter((item: any) => item.path).at(-1),
  );
  const native = JSON.parse(
    execFileSync(
      python,
      [
        '-c',
        "import json; from path_planning_ode.playground import *; s=[{'id':'seed-1','points':[[0,.15]],'width':.85,'strength':18},{'id':'seed-2','points':[[2.25,-2.25],[2.55,-1.8],[2.8,-1.25]],'width':.5,'strength':8}]; o=strokes_to_obstacles(s); a=(-4.8,-1.8); b=(4.8,1.8); n=32; p=[[a[0]+i/(n+1)*(b[0]-a[0]),a[1]+i/(n+1)*(b[1]-a[1])] for i in range(n+2)]; x=initialize_playground(a,b,o,path=p);\nwhile x.status=='running': x=advance_playground(x);\nd=x.to_dict(); print(json.dumps({'path':d['path'],'metrics':d['metrics'],'iteration':d['iteration'],'status':d['status']}))",
      ],
      { encoding: 'utf8' },
    ),
  );
  expect(browser.metrics.route_cost).toBeCloseTo(native.metrics.route_cost, 9);
  expect(browser.metrics.energy).toBeCloseTo(native.metrics.energy, 9);
  expect(browser.path[16][0]).toBeCloseTo(native.path[16][0], 8);
  expect(browser.path[16][1]).toBeCloseTo(native.path[16][1], 8);
  await page.screenshot({ path: 'test-results/playground-desktop.png', fullPage: true });

  const stepCount = await page.evaluate(
    () =>
      (window as any).playgroundRequests.filter((item: any) => item.action === 'playgroundStep')
        .length,
  );
  await page.locator('#playground-reset').click();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as any).playgroundRequests.filter((item: any) => item.action === 'playgroundStep')
            .length,
      ),
    )
    .toBeGreaterThan(stepCount);
  await page.locator('#playground-keep').click();
  const keptStepCount = await page.evaluate(
    () =>
      (window as any).playgroundRequests.filter((item: any) => item.action === 'playgroundStep')
        .length,
  );
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as any).playgroundRequests.filter((item: any) => item.action === 'playgroundStep')
            .length,
      ),
    )
    .toBeGreaterThan(keptStepCount);
  await page.locator('#playground-toggle').click();
  const paused = await saveScene(page);
  await page.waitForTimeout(150);
  expect((await saveScene(page)).path).toEqual(paused.path);
  await page.locator('summary').filter({ hasText: 'Math & solver' }).click();
  await page.locator('#playground-method').selectOption('newton');
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as any).playgroundRequests
            .filter((item: any) => item.action === 'playgroundInitialize')
            .at(-1)?.settings.method,
      ),
    )
    .toBe('newton');
  expect(errors).toEqual([]);
});

test('grab pinning, live painting, erase, and complete-gesture undo/redo', async ({ page }) => {
  await instrument(page);
  await ready(page);
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as any).playgroundResults.filter((item: any) => item.path).at(-1)?.metrics.status,
      ),
    )
    .toBe('converged');
  const canvas = page.locator('#playground-canvas');
  await canvas.scrollIntoViewIfNeeded();
  const box = (await canvas.boundingBox())!;
  await page.locator('[data-playground-tool="grab"]').click();
  const path = await page.evaluate(
      () => (window as any).playgroundResults.filter((item: any) => item.path).at(-1).path,
    ),
    index = Math.floor(path.length / 2),
    point = path[index],
    x = box.x + ((point[0] + 6) / 12) * box.width,
    y = box.y + ((4 - point[1]) / 8) * box.height;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x, y - 35, { steps: 5 });
  await expect
    .poll(
      () =>
        page.evaluate(() => {
          const pinned = (window as any).playgroundResults.filter(
              (item: any) => item.state?.pin_index !== null && item.path,
            ),
            revision = pinned.at(-1)?.revision;
          return pinned.filter((item: any) => item.revision === revision).length;
        }),
      { timeout: 30_000 },
    )
    .toBeGreaterThanOrEqual(2);
  const held = await page.evaluate(() => {
      const pinned = (window as any).playgroundResults.filter(
        (item: any) => item.state?.pin_index !== null && item.path,
      );
      return pinned.filter((item: any) => item.revision === pinned.at(-1).revision);
    }),
    heldA = held.at(-2),
    heldB = held.at(-1),
    pin = heldB.state.pin_index;
  expect(heldA.path[pin]).toEqual(heldB.path[pin]);
  expect(heldA.path[pin - 1]).not.toEqual(heldB.path[pin - 1]);
  await page.mouse.up();
  await expect
    .poll(() =>
      page.evaluate(
        ({ pin, heldPoint }: any) => {
          const free = (window as any).playgroundResults
            .filter((item: any) => item.state?.pin_index === null && item.path)
            .at(-1)?.path[pin];
          return free ? Math.hypot(free[0] - heldPoint[0], free[1] - heldPoint[1]) : 0;
        },
        { pin, heldPoint: heldB.path[pin] },
      ),
    )
    .toBeGreaterThan(1e-8);

  await page.locator('[data-playground-tool="paint"]').click();
  await expect(page.locator('#playground-brush-section')).toBeVisible();
  const beforePaint = await page.evaluate(
    () => (window as any).playgroundResults.filter((item: any) => item.path).at(-1).path,
  );
  await page.mouse.move(box.x + box.width * 0.42, box.y + box.height * 0.25);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.58, box.y + box.height * 0.75, { steps: 8 });
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as any).playgroundRequests
            .filter((item: any) => item.action === 'playgroundField')
            .at(-1)?.strokes.length,
      ),
    )
    .toBe(3);
  await expect
    .poll(() =>
      page.evaluate((before: number[][]) => {
        const after = (window as any).playgroundResults
          .filter((item: any) => item.path)
          .at(-1)?.path;
        return after
          ? Math.max(
              ...after.map((candidate: number[], i: number) =>
                Math.hypot(candidate[0] - before[i][0], candidate[1] - before[i][1]),
              ),
            )
          : 0;
      }, beforePaint),
    )
    .toBeGreaterThan(1e-8);
  await page.mouse.up();
  await page.locator('#playground-undo').click();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as any).playgroundRequests
            .filter((item: any) => item.action === 'playgroundField')
            .at(-1)?.strokes.length,
      ),
    )
    .toBe(2);
  await page.locator('#playground-redo').click();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as any).playgroundRequests
            .filter((item: any) => item.action === 'playgroundField')
            .at(-1)?.strokes.length,
      ),
    )
    .toBe(3);
  await page.locator('[data-playground-tool="erase"]').click();
  await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await expect(page.locator('#playground-undo')).toBeEnabled();
});

test('save/load, legacy adaptation, terrain guidance, keyboard, and mobile layout', async ({
  page,
}) => {
  await instrument(page);
  await ready(page);
  const saved = await saveScene(page);
  expect(saved.kind).toBe('path-playground');
  expect(saved.version).toBe(1);
  expect(saved.path).toHaveLength(34);
  await page.locator('[data-playground-preset="blank"]').click();
  await page.locator('#playground-file').setInputFiles({
    name: 'roundtrip.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(saved)),
  });
  await expect(page.locator('#playground-message')).toHaveText('Playground loaded.');
  const roundtrip = await saveScene(page);
  expect(roundtrip.strokes).toEqual(saved.strokes);
  expect(roundtrip.endpoints).toEqual(saved.endpoints);
  expect(roundtrip.settings).toEqual(saved.settings);
  const roundtripCanvas = page.locator('#playground-canvas');
  await roundtripCanvas.scrollIntoViewIfNeeded();
  const roundtripBox = (await roundtripCanvas.boundingBox())!;
  await page.locator('[data-playground-tool="paint"]').click();
  await page.mouse.click(
    roundtripBox.x + roundtripBox.width * 0.2,
    roundtripBox.y + roundtripBox.height * 0.2,
  );
  await expect
    .poll(async () => (await saveScene(page)).strokes.length)
    .toBe(saved.strokes.length + 1);
  const paintedRoundtrip = await saveScene(page);
  expect(new Set(paintedRoundtrip.strokes.map((stroke: any) => stroke.id)).size).toBe(
    paintedRoundtrip.strokes.length,
  );
  await page.locator('#playground-undo').click();
  await expect.poll(async () => (await saveScene(page)).strokes.length).toBe(saved.strokes.length);

  const overCap = structuredClone(saved);
  overCap.strokes = [
    {
      id: 'too-many-bumps',
      points: [
        [-6, 0],
        [6, 0],
      ],
      width: 0.01,
      strength: 10,
    },
  ];
  await page.locator('#playground-file').setInputFiles({
    name: 'over-cap.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(overCap)),
  });
  await expect(page.locator('#playground-message')).toContainText('Import rejected');
  const afterRejectedImport = await saveScene(page);
  expect(afterRejectedImport.strokes).toEqual(saved.strokes);
  expect(afterRejectedImport.endpoints).toEqual(saved.endpoints);

  await page.locator('#playground-file').setInputFiles({
    name: 'legacy.json',
    mimeType: 'application/json',
    buffer: Buffer.from(
      JSON.stringify({
        version: 1,
        start: [-2, -2],
        end: [12, 12],
        obstacles: [
          { x: 5, y: 5, weight: 0, width: 2 },
          { x: 7, y: 7, weight: 100, width: 1 },
        ],
        guesses: ['straight'],
        options: { interior_points: 30, max_iterations: 50, tolerance: 1e-7, mode: 'damped' },
      }),
    ),
  });
  await expect(page.locator('#playground-message')).toHaveText('Gaussian explorer scene adapted.');
  const adapted = await saveScene(page);
  expect(adapted.strokes.map((stroke: any) => stroke.strength)).toEqual([1, 101]);
  expect(adapted.bounds[1]).toBeGreaterThan(12);

  const beforeGuidance = await page.evaluate(
    () => (window as any).playgroundResults.filter((item: any) => item.path).length,
  );
  await page.locator('#playground-file').setInputFiles({
    name: 'terrain.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({ version: 2, field_x_m: [] })),
  });
  await expect(page.locator('#playground-message')).toContainText('Terrain lab');
  await expect
    .poll(
      () =>
        page.evaluate(
          () => (window as any).playgroundResults.filter((item: any) => item.path).length,
        ),
      { timeout: 5_000 },
    )
    .toBeGreaterThan(beforeGuidance);
  await expect(page.locator('#playground-message')).toContainText('Terrain lab');

  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  const canvas = page.locator('#playground-canvas');
  await page.locator('summary').filter({ hasText: 'Keyboard controls' }).click();
  const beforeKeyboard = await saveScene(page);
  await page
    .locator('#playground-keyboard-target')
    .selectOption(`path:${beforeKeyboard.path.length - 1}`);
  await expect(page.locator('#playground-keyboard-target')).toHaveValue(
    `path:${beforeKeyboard.path.length - 1}`,
  );
  await page.locator('#playground-focus-canvas').click();
  await expect(canvas).toBeFocused();
  await page.keyboard.press('Shift+ArrowDown');
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as any).playgroundRequests
            ?.filter((item: any) => item.action === 'playgroundInitialize')
            .at(-1)?.scene.end[1],
      ),
    )
    .toBeLessThan(beforeKeyboard.endpoints.end[1]);
  await expect
    .poll(async () => (await saveScene(page)).endpoints.end[1])
    .toBeLessThan(beforeKeyboard.endpoints.end[1]);
  const beforeDelete = await saveScene(page);
  await page
    .locator('#playground-keyboard-target')
    .selectOption(`stroke:${beforeDelete.strokes[0].id}`);
  await page.locator('#playground-focus-canvas').click();
  await expect(canvas).toBeFocused();
  await page.keyboard.press('Delete');
  await expect
    .poll(async () => (await saveScene(page)).strokes.length)
    .toBe(beforeDelete.strokes.length - 1);
  await page.screenshot({ path: 'test-results/playground-mobile.png', fullPage: true });
});

test('runtime failure keeps editing available and retry recovers', async ({ page }) => {
  await page.route('**/runtime/pyodide.asm.wasm', (route) => route.abort());
  await page.goto('./');
  await expect(page.locator('#playground-retry')).toBeVisible({ timeout: 60_000 });
  await page.locator('[data-playground-tool="paint"]').click();
  await expect(page.locator('#playground-brush-section')).toBeVisible();
  const canvas = page.locator('#playground-canvas'),
    box = (await canvas.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.25, box.y + box.height * 0.25);
  await page.unroute('**/runtime/pyodide.asm.wasm');
  await page.locator('#playground-retry').click();
  await expect(page.locator('#playground-runtime')).toContainText('Python + NumPy ready', {
    timeout: 60_000,
  });
  await expect.poll(async () => (await saveScene(page)).strokes.length).toBe(3);
});
