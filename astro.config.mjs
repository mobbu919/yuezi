// Astro configuration for static site generation (SSG).
// Deploy target: Cloudflare Pages.

import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  site: 'https://your-site.pages.dev',
  output: 'static',
  build: {
    assets: '_assets',
  },
  server: {
    port: 3000,
  },
});
