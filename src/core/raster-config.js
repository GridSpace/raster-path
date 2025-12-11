/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Raster Config - Shared WebGPU State and Configuration
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Central module providing shared state, configuration, and utilities for all
 * worker modules. All WebGPU resources and pipelines are initialized here.
 *
 * EXPORTS:
 * ────────
 * State Variables:
 *   - config              - Worker configuration (memory limits, tiling, etc.)
 *   - device              - WebGPU device handle
 *   - deviceCapabilities  - Device limits and capabilities
 *   - isInitialized       - Initialization status flag
 *
 * Cached Pipelines:
 *   - cachedRasterizePipeline      - Planar rasterization compute pipeline
 *   - cachedToolpathPipeline       - Planar toolpath generation pipeline
 *   - cachedRadialBatchPipeline    - Radial rasterization pipeline
 *   - cachedRasterizeShaderModule  - Compiled planar rasterize shader
 *   - cachedToolpathShaderModule   - Compiled planar toolpath shader
 *   - cachedRadialBatchShaderModule - Compiled radial shader
 *
 * Constants:
 *   - EMPTY_CELL    - Sentinel value for empty raster cells (-1e10)
 *   - log_pre       - Log prefix string "[Worker]"
 *   - diagnostic    - Debug mode flag
 *
 * Utilities:
 *   - debug         - Logging object (error, warn, log, ok)
 *   - round(v, d)   - Round number to d decimal places
 *
 * Functions:
 *   - initWebGPU()         - Initialize WebGPU device and compile pipelines
 *   - setConfig(obj)       - Replace entire config object
 *   - updateConfig(obj)    - Merge updates into existing config
 *
 * ARCHITECTURE:
 * ─────────────
 * This module acts as a singleton state container. All worker modules import
 * from here to access GPU resources. The initWebGPU() function is called once
 * during worker initialization and pre-compiles all compute pipelines to avoid
 * runtime compilation overhead.
 *
 * Shader code placeholders ('SHADER:xxx') are replaced during build by the
 * build-shaders.js script, which bundles modules with esbuild then injects
 * WGSL shader source code.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

// WebGPU state
export let config = {};
export let device = null;
export let deviceCapabilities = null;
export let isInitialized = false;

// Cached pipelines (created during initialization)
export let cachedRasterizePipeline = null;
export let cachedRasterizeShaderModule = null;
export let cachedToolpathPipeline = null;
export let cachedToolpathShaderModule = null;
export let cachedRadialBatchPipeline = null;
export let cachedRadialBatchShaderModule = null;
export let cachedTracingPipeline = null;
export let cachedTracingShaderModule = null;
export let cachedRadialV3RotatePipeline = null;
export let cachedRadialV3RotateShaderModule = null;
export let cachedRadialV3RasterizePipeline = null;
export let cachedRadialV3RasterizeShaderModule = null;
export let cachedRadialV3BatchedRasterizePipeline = null;
export let cachedRadialV3BatchedRasterizeShaderModule = null;

// Constants
export const EMPTY_CELL = -1e10;
export const log_pre = '[Worker]';
export const diagnostic = false;

// Logging utilities
let lastlog;
export const debug = {
    error: function() { console.error(log_pre, ...arguments) },
    warn: function() { console.warn(log_pre, ...arguments) },
    log: function() {
        if (!config.quiet) {
            let now = performance.now();
            let since = ((now - (lastlog ?? now)) | 0).toString().padStart(4,' ');
            console.log(log_pre, `[${since}]`, ...arguments);
            lastlog = now;
        }
    },
    ok: function() { console.log(log_pre, '✅', ...arguments) },
};

// Utility functions
export function round(v, d = 1) {
    return parseFloat(v.toFixed(d));
}

// Shader code placeholders (replaced by build script)
const rasterizeShaderCode = 'SHADER:planar-rasterize';
const toolpathShaderCode = 'SHADER:planar-toolpath';
const radialRasterizeShaderCode = 'SHADER:radial-raster';
const tracingShaderCode = 'SHADER:tracing-toolpath';
const radialV3RotateShaderCode = 'SHADER:radial-rotate-triangles';
const radialV3RasterizeShaderCode = 'SHADER:radial-rasterize-filtered';
const radialV3BatchedRasterizeShaderCode = 'SHADER:radial-rasterize-batched';

// Initialize WebGPU device in worker context
export async function initWebGPU() {
    if (isInitialized) return true;

    if (!navigator.gpu) {
        debug.warn('WebGPU not supported');
        return false;
    }

    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            debug.warn('WebGPU adapter not available');
            return false;
        }

        // Request device with higher limits for large meshes
        const adapterLimits = adapter.limits;
        debug.log('Adapter limits:', adapterLimits);

        device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: Math.min(
                    adapterLimits.maxStorageBufferBindingSize,
                    1024 * 1024 * 1024 // Request up to 1GB
                ),
                maxBufferSize: Math.min(
                    adapterLimits.maxBufferSize,
                    1024 * 1024 * 1024 // Request up to 1GB
                )
            }
        });

        // Pre-compile rasterize shader module (expensive operation)
        cachedRasterizeShaderModule = device.createShaderModule({ code: rasterizeShaderCode });

        // Pre-create rasterize pipeline (very expensive operation)
        cachedRasterizePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: cachedRasterizeShaderModule, entryPoint: 'main' },
        });

        // Pre-compile toolpath shader module
        cachedToolpathShaderModule = device.createShaderModule({ code: toolpathShaderCode });

        // Pre-create toolpath pipeline
        cachedToolpathPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: cachedToolpathShaderModule, entryPoint: 'main' },
        });

        // Pre-compile radial batch shader module
        cachedRadialBatchShaderModule = device.createShaderModule({ code: radialRasterizeShaderCode });

        // Pre-create radial batch pipeline
        cachedRadialBatchPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: cachedRadialBatchShaderModule, entryPoint: 'main' },
        });

        // Pre-compile tracing shader module
        cachedTracingShaderModule = device.createShaderModule({ code: tracingShaderCode });

        // Pre-create tracing pipeline
        cachedTracingPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: cachedTracingShaderModule, entryPoint: 'main' },
        });

        // Pre-compile radial V3 rotate shader module
        cachedRadialV3RotateShaderModule = device.createShaderModule({ code: radialV3RotateShaderCode });

        // Pre-create radial V3 rotate pipeline
        cachedRadialV3RotatePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: cachedRadialV3RotateShaderModule, entryPoint: 'main' },
        });

        // Pre-compile radial V3 rasterize shader module
        cachedRadialV3RasterizeShaderModule = device.createShaderModule({ code: radialV3RasterizeShaderCode });

        // Pre-create radial V3 rasterize pipeline
        cachedRadialV3RasterizePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: cachedRadialV3RasterizeShaderModule, entryPoint: 'main' },
        });

        // Pre-compile radial V3 batched rasterize shader module
        cachedRadialV3BatchedRasterizeShaderModule = device.createShaderModule({ code: radialV3BatchedRasterizeShaderCode });

        // Pre-create radial V3 batched rasterize pipeline
        cachedRadialV3BatchedRasterizePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: cachedRadialV3BatchedRasterizeShaderModule, entryPoint: 'main' },
        });

        // Store device capabilities
        deviceCapabilities = {
            maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
            maxBufferSize: device.limits.maxBufferSize,
            maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
            maxComputeWorkgroupSizeY: device.limits.maxComputeWorkgroupSizeY,
        };

        isInitialized = true;
        debug.log('Initialized (pipelines cached)');
        return true;
    } catch (error) {
        debug.error('Failed to initialize:', error);
        return false;
    }
}

// Mutators for config (called from worker main)
export function setConfig(newConfig) {
    config = newConfig;
}

export function updateConfig(updates) {
    Object.assign(config, updates);
}
