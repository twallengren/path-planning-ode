import { expect, test } from '@playwright/test';
test('Gaussian explorer redirects to the playground', async ({ page }) => {
  await page.goto('./explorer.html');
  await expect(page).toHaveURL(/path-planning-ode\/index\.html$/);
  await expect(page.locator('#playground-title')).toBeVisible();
});
