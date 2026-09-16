import { expect, test } from '@playwright/test';

test('math page presents the route equations and links to the playground', async ({ page }) => {
  await page.goto('./derivation.html');
  await expect(page.getByRole('heading', { name: 'The path equations.' })).toBeVisible();
  await expect(page.locator('#objective')).toContainText('J[q]');
  await expect(page.locator('#euler-lagrange .katex')).toHaveCount(4);
  await expect(page.locator('.katex-error')).toHaveCount(0);
  await page.getByRole('link', { name: 'Open the playground' }).click();
  await expect(page).toHaveURL(/#playground-workspace$/);
});
