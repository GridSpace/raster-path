import fs from 'node:fs';
import path from 'node:path';
import { glob, cp, mkdir, rm } from 'node:fs/promises';

const BUILD_DIR = path.join(import.meta.dirname, '../build');
const BUILD_WEB_DIR = path.join(import.meta.dirname, '../build-web');

// Ensure build directories exists
if (!fs.existsSync(BUILD_DIR)) {
  await mkdir(BUILD_DIR, { recursive: true });
} else {
  await rm(BUILD_DIR, { recursive: true, force: true });
}

if (!fs.existsSync(BUILD_WEB_DIR)) {
  await mkdir(BUILD_WEB_DIR, { recursive: true });
} else {
  await rm(BUILD_WEB_DIR, { recursive: true, force: true });
}

// Copy source files
await cp('src/core/raster-path.js', 'build/raster-path.js');
await cp('src/core/raster-path.js', 'build-web/raster-path.js');
await cp('src/etc/serve.json', 'build-web/serve.json');

// Copy web files
for await (const entry of glob('src/web/*.{html,css,js}')) {
  cp(entry, `build-web/${entry.split(/\/|\\/).pop()}`);
}
