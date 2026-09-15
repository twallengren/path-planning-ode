import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import { constants, copyFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { platform } from 'node:process';
import { spawnSync } from 'node:child_process';

function cloneOrCopy(source: string, destination: string) {
  if (platform === 'darwin') {
    const clone = spawnSync('/bin/cp', ['-c', '-f', source, destination]);
    if (clone.status === 0) return;
  }
  copyFileSync(source, destination, constants.COPYFILE_FICLONE);
}

function copyPublic(source: string, destination: string) {
  mkdirSync(destination, { recursive: true });
  for (const entry of readdirSync(source, { withFileTypes: true })) {
    const from = join(source, entry.name),
      to = join(destination, entry.name);
    if (entry.name === 'runtime' && source.endsWith(join('web', 'public'))) continue;
    if (entry.isDirectory()) copyPublic(from, to);
    else cloneOrCopy(from, to);
  }
}
const portableRuntime = join(tmpdir(), 'path-planning-ode-pyodide-314.0.7');
const macRuntime = '/private/tmp/path-planning-ode-pyodide-314.0.7';
const runtimeCache = existsSync(macRuntime) ? macRuntime : portableRuntime;

export default defineConfig(({ command }) => ({
  base: './',
  // Runtime wheels are large. During builds, copy them as filesystem clones
  // where supported; COPYFILE_FICLONE falls back to a normal copy elsewhere.
  publicDir: command === 'build' ? false : 'public',
  plugins:
    command === 'build'
      ? [
          {
            name: 'copy-public-efficiently',
            closeBundle() {
              copyPublic(
                fileURLToPath(new URL('./public', import.meta.url)),
                fileURLToPath(new URL('./dist', import.meta.url)),
              );
              copyPublic(runtimeCache, fileURLToPath(new URL('./dist/runtime', import.meta.url)));
            },
          },
        ]
      : [],
  worker: { format: 'es' },
  build: {
    rolldownOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        explorer: fileURLToPath(new URL('./explorer.html', import.meta.url)),
        derivation: fileURLToPath(new URL('./derivation.html', import.meta.url)),
        terrain: fileURLToPath(new URL('./terrain.html', import.meta.url)),
      },
    },
  },
}));
