import { test, expect, type Page } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';

const python = existsSync('.venv/bin/python') ? '.venv/bin/python' : 'python';
async function ready(page: Page) {
  await page.goto('./');
  await expect(page.locator('#runtime')).toContainText('Python + NumPy ready', { timeout: 60_000 });
  await expect(page.locator('#view-label')).toHaveText('Live Python solver');
}

test('browser worker matches native Python; editing, playback and exports work', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.addInitScript(() => {
    const OriginalWorker = window.Worker;
    (window as any).framesFromSolver = [];
    window.Worker = class extends OriginalWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener('message', ({ data }) => {
          if (data.result?.states) (window as any).framesFromSolver.push(data.result);
        });
      }
    };
  });
  await ready(page);
  await page.locator('#step').click();
  await expect(page.locator('#timeline-value')).toHaveText('1 / 1');
  const frame = await page.evaluate(() => (window as any).framesFromSolver.at(-1));
  const native = JSON.parse(
    execFileSync(
      python,
      [
        '-c',
        'import json; from path_planning_ode import presets, initialize, step; s=presets()["asymmetric"]; print(json.dumps({g:step(s,initialize(s,g)).to_dict() for g in s.guesses}))',
      ],
      { encoding: 'utf8' },
    ),
  );
  for (const [name, state] of Object.entries<any>(frame.states)) {
    expect(state.residual_norm).toBeCloseTo(native[name].residual_norm, 7);
    expect(state.cost).toBeCloseTo(native[name].cost, 10);
    expect(state.energy).toBeCloseTo(native[name].energy, 7);
    expect(state.length).toBeCloseTo(native[name].length, 10);
    for (let i = 0; i < state.path.length; i++)
      for (let axis = 0; axis < 2; axis++)
        expect(state.path[i][axis]).toBeCloseTo(native[name].path[i][axis], 8);
  }
  await page.locator('#empty').click();
  await expect(page.locator('#step')).toBeEnabled();
  await page.locator('#step').click();
  await expect(page.locator('#metrics .converged')).toHaveCount(3);
  await page.locator('#timeline').fill('0');
  await expect(page.locator('#timeline-value')).toHaveText('0 / 1');
  await page.locator('#step').click();
  await expect(page.locator('#timeline-value')).toHaveText('1 / 1');
  await page.locator('#add').click();
  await expect(page.locator('#obstacle-count')).toHaveText('01');
  await page.locator('#obstacle-x').fill('4');
  await page.locator('#obstacle-x').press('Tab');
  await expect(page.locator('#step')).toBeEnabled();
  await page.getByText('Display & export', { exact: true }).click();
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#export').click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('path-scene.json');
  const file = await download.path();
  await page.locator('#empty').click();
  await page.locator('#file').setInputFiles(file!);
  await expect(page.locator('#message')).toHaveText('Scene imported.');
  await expect(page.locator('#obstacle-x')).toHaveValue('4');
  const imagePromise = page.waitForEvent('download');
  await page.locator('#png').click();
  expect((await imagePromise).suggestedFilename()).toBe('path-planning.png');
  await page.locator('#rover').click();
  await expect(page.locator('#rover')).toHaveText('Stop rover');
  await page.locator('#rover').click();
  await expect(page.locator('#rover')).toHaveText('Play rover');
  await page.screenshot({ path: 'test-results/desktop.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('scene links, invalid imports, and editing during a run', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await ready(page);
  await page.locator('#play').click();
  await page.locator('#start-x').fill('-4');
  await page.locator('#start-x').press('Tab');
  await expect(page.locator('#timeline-value')).toHaveText('0 / 0');
  await expect(page.locator('#view-label')).toHaveText('Live Python solver');
  await page.getByText('Display & export', { exact: true }).click();
  await page.locator('#share').click();
  const link = await page.evaluate(() => navigator.clipboard.readText());
  expect(link).toContain('#scene=');
  await page.goto(link);
  await expect(page.locator('#runtime')).toContainText('Python + NumPy ready');
  await expect(page.locator('#start-x')).toHaveValue('-4');
  await page.locator('#file').setInputFiles({
    name: 'invalid.json',
    mimeType: 'application/json',
    buffer: Buffer.from('{"version":99}'),
  });
  await expect(page.locator('#message')).toContainText('Only version 1');
  await expect(page.locator('#start-x')).toHaveValue('-4');
});

test('loading failure preserves essay and precomputed history; retry recovers', async ({
  page,
}) => {
  await page.route('**/runtime/pyodide.asm.wasm', (route) => route.abort());
  await page.goto('./');
  await expect(page.locator('#retry')).toBeVisible({ timeout: 60_000 });
  await expect(page.locator('#view-label')).toContainText('Precomputed example');
  await expect(page.locator('h1')).toContainText('The shape');
  await page.locator('#timeline').fill('0');
  await expect(page.locator('#timeline-value')).toContainText('0 /');
  await page.unroute('**/runtime/pyodide.asm.wasm');
  await page.locator('#retry').click();
  await expect(page.locator('#runtime')).toContainText('Python + NumPy ready');
});

test('narrow screen, touch dragging, keyboard controls, reduced motion', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await ready(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator('#landscape').scrollIntoViewIfNeeded();
  const box = (await page.locator('#landscape').boundingBox())!;
  const size = box.width - 72;
  const x = box.x + 36 + (7 / 18) * size,
    y = box.y + 36 + (10 / 18) * size;
  const touch = await page.context().newCDPSession(page);
  await touch.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] });
  await touch.send('Input.dispatchTouchEvent', {
    type: 'touchMove',
    touchPoints: [{ x: x + 15, y: y - 10 }],
  });
  await touch.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await expect(page.locator('#obstacle-x')).not.toHaveValue('3');
  await expect(page.locator('#view-label')).toHaveText('Live Python solver');
  await page.locator('#start-x').focus();
  await page.keyboard.press('ArrowUp');
  await page.keyboard.press('Tab');
  await expect(page.locator('#start-x')).toHaveValue('-1.5');
  await expect(page.locator('#step')).toBeEnabled();
  await page.locator('#step').click();
  await expect(page.locator('#timeline-value')).toHaveText('1 / 1');
  await page.screenshot({ path: 'test-results/mobile.png', fullPage: true });
});

test('a longer detour is cheaper; scene changes refresh the baseline cost', async ({ page }) => {
  await ready(page);
  await page.locator('.experiment-tabs [data-preset="central"]').click();
  await expect(page.locator('#view-label')).toHaveText('Live Python solver');
  await expect(page.locator('#cost-comparison')).toContainText('Direct route cost: 41.07');
  await page.locator('#play').click();
  await expect(page.locator('#metrics .converged')).toHaveCount(3);
  await expect(page.locator('#cost-comparison')).toContainText('22.09');
  await expect(page.locator('#cost-comparison')).toContainText('46.2% less');
  await expect(page.locator('#metrics')).toContainText('total cost');
  await expect(page.locator('#metrics')).not.toContainText('energy');
  const detour = page.locator('#metrics .metric').nth(1);
  await expect(detour).toContainText('21.6');
  await page.getByText('Solver convergence', { exact: true }).click();
  await expect(page.locator('#residual-chart')).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page
        .locator('#cost-chart')
        .evaluate((node) =>
          Math.abs((node as HTMLCanvasElement).width - node.clientWidth * devicePixelRatio),
        ),
    )
    .toBeLessThanOrEqual(1);
  await page.getByText('Solver settings', { exact: true }).click();
  await expect(page.locator('#resolution')).toBeVisible();
  await page.locator('#weight').fill('0');
  await expect(page.locator('#cost-comparison')).toContainText('Direct route cost: 19.80');
  await page.locator('#play').click();
  await expect(page.locator('#metrics .converged')).toHaveCount(3);
  await expect(page.locator('#cost-comparison')).toContainText('No cheaper detour shown');
});
