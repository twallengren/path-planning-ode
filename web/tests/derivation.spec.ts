import { test, expect } from '@playwright/test';

test('derivation typesets every equation without loading Python and works under the Pages prefix', async ({
  page,
}) => {
  const errors: string[] = [];
  const runtimeRequests: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (request) => {
    if (request.url().includes('/runtime/')) runtimeRequests.push(request.url());
  });
  await page.goto('./derivation.html');
  await expect(page.locator('h1')).toContainText('From a cost');
  const equations = await page.locator('[data-tex]').count();
  expect(equations).toBeGreaterThan(30);
  await expect(page.locator('[data-tex] .katex')).toHaveCount(equations);
  await expect(page.locator('.katex-error')).toHaveCount(0);
  await expect(page.locator('.derivation-notice')).toContainText('does not switch the playground');
  await page.getByRole('link', { name: '6. The two ODE systems', exact: true }).click();
  await expect(page).toHaveURL(/derivation\.html#two-equations$/);
  await expect(page.locator('#two-equations h2')).toBeInViewport();
  expect(runtimeRequests).toEqual([]);
  expect(errors).toEqual([]);
  await page.screenshot({ path: 'test-results/derivation-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByText('The analytic formulas for', { exact: false }).click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/derivation-mobile.png', fullPage: true });
  await page.getByRole('link', { name: 'Back to the playground', exact: false }).click();
  await expect(page).toHaveURL(/path-planning-ode\/#playground$/);
  await page.locator('.derivation-link').click();
  await expect(page.locator('h1')).toContainText('From a cost');
});

test('the static article remains readable when JavaScript is disabled', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:4173/path-planning-ode/derivation.html');
  await expect(page.locator('#example')).toContainText('The new midpoint is (2, 0)');
  await expect(page.locator('[data-tex]').first()).toContainText('q(t) = (x(t), y(t))');
  await context.close();
});
