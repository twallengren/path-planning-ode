import { execFileSync } from 'node:child_process';
import { cp, mkdir, readFile, access, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';

const python =
  process.env.PYTHON ??
  (existsSync('.venv/bin/python')
    ? '.venv/bin/python'
    : existsSync('.venv/Scripts/python.exe')
      ? '.venv/Scripts/python.exe'
      : 'python');
execFileSync(python, ['scripts/build_assets.py'], { stdio: 'inherit' });
const destination = resolve('web/public/runtime');
await mkdir(destination, { recursive: true });
await cp('node_modules/pyodide', destination, { recursive: true });
const lock = JSON.parse(await readFile(`${destination}/pyodide-lock.json`, 'utf8'));
const packages = new Set();
function collect(name) {
  if (packages.has(name)) return;
  packages.add(name);
  for (const dependency of lock.packages[name].depends ?? []) collect(dependency);
}
collect('numpy');
for (const name of packages) {
  const filename = lock.packages[name].file_name;
  const target = `${destination}/${filename}`;
  const verified = async () => {
    try {
      await access(target);
      return (
        createHash('sha256')
          .update(await readFile(target))
          .digest('hex') === lock.packages[name].sha256
      );
    } catch {
      return false;
    }
  };
  if (await verified()) continue;
  const response = await fetch(`https://cdn.jsdelivr.net/pyodide/v314.0.7/full/${filename}`);
  if (!response.ok) throw new Error(`Cannot download ${filename}: ${response.status}`);
  await writeFile(target, new Uint8Array(await response.arrayBuffer()));
  if (!(await verified())) throw new Error(`Checksum mismatch for ${filename}`);
}
console.log('Prepared Python package, deterministic examples, and local browser runtime.');
