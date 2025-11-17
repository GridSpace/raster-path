/**
 * ═══════════════════════════════════════════════════════════════════════════
 * WebGPU Worker - GPU Compute Operations
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Offloads all WebGPU compute operations to a worker thread to prevent UI blocking.
 * Handles both planar (XY grid) and radial (cylindrical) rasterization modes.
 *
 * MESSAGE PROTOCOL:
 * ─────────────────
 * Main Thread → Worker:
 *   'init'                     - Initialize WebGPU device
 *   'rasterize-planar'         - Rasterize geometry to XY grid
 *   'generate-toolpath-planar' - Generate planar toolpath from rasters
 *   'radial-generate-toolpaths'- Generate radial toolpaths (does rasterization + toolpath)
 *   'calibrate'                - Run GPU workload calibration
 *
 * Worker → Main Thread:
 *   'webgpu-ready'            - Initialization complete
 *   'rasterize-complete'      - Planar rasterization complete
 *   'rasterize-progress'      - Progress update (0-1)
 *   'toolpath-complete'       - Planar toolpath complete
 *   'toolpath-progress'       - Progress update (0-1)
 *   'radial-toolpaths-complete' - Radial toolpaths complete
 *   'calibrate-complete'      - Calibration results
 *
 * ARCHITECTURE:
 * ─────────────
 * 1. PLANAR MODE:
 *    - Rasterize terrain: XY grid, keep max Z per cell
 *    - Rasterize tool: XY grid, keep min Z per cell
 *    - Generate toolpath: Scan tool over terrain, compute Z-heights
 *
 * 2. RADIAL MODE:
 *    - Batched processing: 360 angles per batch
 *    - X-bucketing: Spatial partitioning to reduce triangle tests
 *    - For each angle:
 *        * Cast ray from origin
 *        * Rasterize terrain triangles along ray
 *        * Calculate tool-terrain collision
 *        * Output Z-heights along X-axis
 *
 * MEMORY MANAGEMENT:
 * ──────────────────
 * - All GPU buffers are preallocated to known maximum sizes
 * - Triangle data transferred once per operation
 * - Output buffers mapped asynchronously to avoid blocking
 * - Worker maintains pipeline cache to avoid recompilation
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

import { initWebGPU, setConfig, updateConfig, deviceCapabilities, debug, device } from './raster-config.js';
import { rasterizeMesh } from './raster-planar.js';
import { generateToolpath } from './path-planar.js';
import { generateRadialToolpaths } from './path-radial.js';
import { generateTracingToolpaths, createReusableTracingBuffers, destroyReusableTracingBuffers } from './path-tracing.js';
import { calibrateGPU } from './workload-calibrate.js';

// Global error handler for uncaught errors in worker
self.addEventListener('error', (event) => {
    debug.error('Uncaught error:', event.error || event.message);
    debug.error('Stack:', event.error?.stack);
});

self.addEventListener('unhandledrejection', (event) => {
    debug.error('Unhandled promise rejection:', event.reason);
});

// Handle messages from main thread
self.onmessage = async function(e) {
    const { type, data } = e.data;

    try {
        switch (type) {
            case 'init':
                // Store config
                setConfig(data?.config || {
                    maxGPUMemoryMB: 256,
                    gpuMemorySafetyMargin: 0.8,
                    tileOverlapMM: 10,
                    autoTiling: true,
                    batchDivisor: 1,  // For testing batching overhead: 1=optimal, 2=2x batches, 4=4x batches, etc.
                    maxConcurrentThreads: 32768  // GPU watchdog limit: max threads across all workgroups in a dispatch
                });
                const success = await initWebGPU();
                self.postMessage({
                    type: 'webgpu-ready',
                    data: {
                        success,
                        capabilities: deviceCapabilities
                    }
                });
                break;

            case 'update-config':
                updateConfig(data.config);
                debug.log('Config updated');
                break;

            case 'rasterize':
                const { triangles, stepSize, filterMode, boundsOverride } = data;
                const rasterOptions = boundsOverride || {};
                const rasterResult = await rasterizeMesh(triangles, stepSize, filterMode, rasterOptions);
                self.postMessage({
                    type: 'rasterize-complete',
                    data: rasterResult,
                }, [rasterResult.positions.buffer]);
                break;

            case 'generate-toolpath':
                const { terrainPositions, toolPositions, xStep, yStep, zFloor, gridStep, terrainBounds, singleScanline } = data;
                const toolpathResult = await generateToolpath(
                    terrainPositions, toolPositions, xStep, yStep, zFloor, gridStep, terrainBounds, singleScanline
                );
                self.postMessage({
                    type: 'toolpath-complete',
                    data: toolpathResult
                }, [toolpathResult.pathData.buffer]);
                break;

            case 'radial-generate-toolpaths':
                const radialToolpathResult = await generateRadialToolpaths(data);
                const toolpathTransferBuffers = radialToolpathResult.strips.map(strip => strip.pathData.buffer);
                self.postMessage({
                    type: 'radial-toolpaths-complete',
                    data: radialToolpathResult
                }, toolpathTransferBuffers);
                break;

            case 'tracing-generate-toolpaths':
                const tracingResult = await generateTracingToolpaths({
                    paths: data.paths,
                    terrainPositions: data.terrainPositions,
                    terrainData: data.terrainData,
                    toolPositions: data.toolPositions,
                    step: data.step,
                    gridStep: data.gridStep,
                    terrainBounds: data.terrainBounds,
                    zFloor: data.zFloor,
                    onProgress: (progressData) => {
                        self.postMessage({
                            type: 'tracing-progress',
                            data: progressData.data
                        });
                    }
                });
                const tracingTransferBuffers = tracingResult.paths.map(p => p.buffer);
                self.postMessage({
                    type: 'tracing-toolpaths-complete',
                    data: tracingResult
                }, tracingTransferBuffers);
                break;

            case 'create-tracing-buffers':
                createReusableTracingBuffers(data.terrainPositions, data.toolPositions);
                self.postMessage({
                    type: 'tracing-buffers-created',
                    data: { success: true }
                });
                break;

            case 'destroy-tracing-buffers':
                destroyReusableTracingBuffers();
                self.postMessage({
                    type: 'tracing-buffers-destroyed',
                    data: { success: true }
                });
                break;

            case 'calibrate':
                const calibrationResult = await calibrateGPU(device, data?.options || {});
                self.postMessage({
                    type: 'calibrate-complete',
                    data: calibrationResult
                });
                break;

            default:
                self.postMessage({
                    type: 'error',
                    message: 'Unknown message type: ' + type
                });
        }
    } catch (error) {
        debug.error('Error:', error);
        self.postMessage({
            type: 'error',
            message: error.message,
            stack: error.stack
        });
    }
};
