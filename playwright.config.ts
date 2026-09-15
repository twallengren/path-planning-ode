import { defineConfig, devices } from '@playwright/test';
import { existsSync } from 'node:fs';

export default defineConfig({
  testDir: './web/tests',
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:4173/path-planning-ode/', trace: 'retain-on-failure' },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 1000 } },
    },
  ],
  webServer: {
    command: `${existsSync('.venv/bin/python') ? '.venv/bin/python' : 'python'} scripts/serve_site.py`,
    url: 'http://127.0.0.1:4173/path-planning-ode/',
    reuseExistingServer: !process.env.CI,
  },
});
