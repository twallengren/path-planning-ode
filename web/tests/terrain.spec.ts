import { expect, test } from '@playwright/test';
test('terrain lab redirects to the playground', async ({ page }) => {
  await page.goto('./terrain.html');
  await expect(page).toHaveURL(/path-planning-ode\/index\.html$/);
  await expect(page.locator('#playground-title')).toBeVisible();
});
