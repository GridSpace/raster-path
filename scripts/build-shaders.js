#!/usr/bin/env node
/**
 * Build script: Injects shader code into webgpu-worker.js
 *
 * Replaces placeholders like SHADER-radial-cull with shader file contents
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SHADER_DIR = path.join(__dirname, '../src/shaders');
const WORKER_SRC = path.join(__dirname, '../src/web/webgpu-worker.js');
const BUILD_DIR = path.join(__dirname, '../build');
const WORKER_DEST = path.join(BUILD_DIR, 'webgpu-worker.js');

// Read worker source
let workerCode = fs.readFileSync(WORKER_SRC, 'utf8');

// Find all shader placeholders
const shaderRegex = /\/\*SHADER:([a-z0-9-]+)\*\//g;
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

// Ensure build directory exists
if (!fs.existsSync(BUILD_DIR)) {
    fs.mkdirSync(BUILD_DIR, { recursive: true });
}

// Write output
fs.writeFileSync(WORKER_DEST, workerCode, 'utf8');
console.log(`✅ Built: ${WORKER_DEST}`);
