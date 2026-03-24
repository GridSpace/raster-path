#!/usr/bin/env node
/**
 * Build script: Bundle worker modules with esbuild, then inject shader code
 *
 * 1. Bundle all worker modules (raster-worker.js + imports) into single file
 * 2. Replace shader placeholders like 'SHADER:radial-raster' with shader file contents
 */

import fs from 'fs';
import path from 'path';
import * as esbuild from 'esbuild';

const __dirname = import.meta.dirname;

const SHADER_DIR = path.join(__dirname, '../src/shaders');
const WORKER_SRC = path.join(__dirname, '../src/core/raster-worker.js');
const BUILD_DIR = path.join(__dirname, '../build');
const WORKER_BUNDLED = path.join(BUILD_DIR, 'raster-worker.bundled.js');
const WORKER_DEST = path.join(BUILD_DIR, 'raster-worker.js');

// Ensure build directory exists
if (!fs.existsSync(BUILD_DIR)) {
  fs.mkdirSync(BUILD_DIR, { recursive: true });
}

// Step 1: Bundle worker modules with esbuild
console.log('📦 Bundling worker modules with esbuild...');
await esbuild.build({
  entryPoints: [WORKER_SRC],
  bundle: true,
  format: 'esm',
  outfile: WORKER_BUNDLED,
  platform: 'browser',
  target: 'es2020',
});
console.log(`✅ Bundled: ${WORKER_BUNDLED}`);

// Step 2: Read bundled code and inject shaders
let workerCode = fs.readFileSync(WORKER_BUNDLED, 'utf8');

// Find all shader placeholders (handle both single and double quotes)
const shaderRegex = /['"]SHADER:([a-z0-9-]+)['"]/g;
let match;
const replacements = [];

while ((match = shaderRegex.exec(workerCode)) !== null) {
  const shaderName = match[1];
  const placeholder = match[0];
  replacements.push({ shaderName, placeholder });
}

// Replace each placeholder with shader code
for (const { shaderName, placeholder } of replacements) {
  const shaderPath = path.join(SHADER_DIR, `${shaderName}.wgsl`);

  if (!fs.existsSync(shaderPath)) {
    console.error(`❌ Shader file not found: ${shaderPath}`);
    process.exit(1);
  }

  const shaderCode = fs.readFileSync(shaderPath, 'utf8');

  // Wrap in template literal and escape backticks
  const escapedShader = shaderCode.replace(/`/g, '\\`').replace(/\$/g, '\\$');
  const wrapped = '`' + escapedShader + '`';

  workerCode = workerCode.replace(placeholder, wrapped);
  console.log(`✅ Injected shader: ${shaderName}.wgsl`);
}

// Inject unique build ID
const buildId = Math.random().toString(36).substring(2, 10).toUpperCase();
workerCode = workerCode.replace(/BUILD_ID_PLACEHOLDER/g, buildId);
console.log(`🔨 Build ID: ${buildId}`);

// Write output
fs.writeFileSync(WORKER_DEST, workerCode, 'utf8');
console.log(`✅ Built: ${WORKER_DEST}`);

// Clean up intermediate bundled file
fs.unlinkSync(WORKER_BUNDLED);
console.log('🧹 Cleaned up intermediate bundle');
