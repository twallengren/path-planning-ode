import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  base: './',
  worker: { format: 'es' },
  build: {
    rolldownOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        derivation: fileURLToPath(new URL('./derivation.html', import.meta.url)),
      },
    },
  },
});
