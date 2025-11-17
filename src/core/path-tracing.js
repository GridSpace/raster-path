/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Path Tracing - Toolpath Z-Depth Tracing for Input Polylines
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Generates toolpath Z-coordinates by following input polylines and calculating
 * tool contact depth at each sampled point along the path.
 *
 * EXPORTS:
 * ────────
 * Functions:
 *   - generateTracingToolpaths(options)
 *       Main API - processes array of input paths, returns array of XYZ paths
 *
 * ALGORITHM:
 * ──────────
 * For each input polyline:
 *   1. Sample path segments at 'step' resolution (densification)
 *   2. For each sampled point (X, Y):
 *      - Convert world coordinates to terrain grid coordinates
 *      - Test tool collision with terrain at that position
 *      - Calculate maximum collision Z (same algorithm as planar mode)
 *   3. Build output array with X, Y, Z triplets
 *
 * PATH SAMPLING:
 * ──────────────
 * Input paths are arrays of XY coordinate pairs. The 'step' parameter controls
 * how densely segments are sampled:
 *   - Vertices are always included
 *   - Segments longer than 'step' are subdivided
 *   - Output maintains original vertex positions + interpolated points
 *
 * OUTPUT FORMAT:
 * ──────────────
 * Array of Float32Array buffers, each containing XYZ triplets:
 *   [x1, y1, z1, x2, y2, z2, x3, y3, z3, ...]
 *
 * MEMORY SAFETY:
 * ──────────────
 * Validates that sampled path points will fit in GPU buffers before processing.
 * Throws error if estimated memory exceeds safe limits.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

import {
    device, deviceCapabilities, isInitialized, config,
    cachedTracingPipeline, debug, initWebGPU
} from './raster-config.js';
import { createSparseToolFromPoints } from './raster-tool.js';

/**
 * Sample a path at specified step resolution
 * @param {Float32Array} pathXY - Input path as XY coordinate pairs
 * @param {number} step - Maximum distance between sampled points (world units)
 * @returns {Float32Array} - Sampled XY coordinates
 */
function samplePath(pathXY, step) {
    if (pathXY.length < 2) {
        // Empty or single-point path
        return new Float32Array(pathXY);
    }

    const numVertices = pathXY.length / 2;
    const sampledPoints = [];

    // Always include first vertex
    sampledPoints.push(pathXY[0], pathXY[1]);

    // Process each segment
    for (let i = 0; i < numVertices - 1; i++) {
        const x1 = pathXY[i * 2];
        const y1 = pathXY[i * 2 + 1];
        const x2 = pathXY[(i + 1) * 2];
        const y2 = pathXY[(i + 1) * 2 + 1];

        const dx = x2 - x1;
        const dy = y2 - y1;
        const segmentLength = Math.sqrt(dx * dx + dy * dy);

        // If segment is longer than step, subdivide it
        if (segmentLength > step) {
            const numSubdivisions = Math.ceil(segmentLength / step);
            const subdivisionStep = 1.0 / numSubdivisions;

            // Add interpolated points (skip t=0 since it's already added, skip t=1 since it's the next vertex)
            for (let j = 1; j < numSubdivisions; j++) {
                const t = j * subdivisionStep;
                const x = x1 + t * dx;
                const y = y1 + t * dy;
                sampledPoints.push(x, y);
            }
        }

        // Add next vertex (except for last iteration where it's already the end)
        if (i < numVertices - 1) {
            sampledPoints.push(x2, y2);
        }
    }

    return new Float32Array(sampledPoints);
}

/**
 * Generate tracing toolpaths for input polylines
 * @param {Object} options - Configuration
 * @param {Float32Array[]} options.paths - Array of input paths (XY coordinate pairs)
 * @param {Float32Array} options.terrainPositions - Dense terrain Z-only grid
 * @param {Object} options.terrainData - Terrain metadata (width, height, bounds)
 * @param {Float32Array} options.toolPositions - Tool points (XYZ triplets)
 * @param {number} options.step - Sampling resolution along paths (world units)
 * @param {number} options.gridStep - Terrain rasterization resolution
 * @param {Object} options.terrainBounds - Terrain bounding box
 * @param {number} options.zFloor - Minimum Z depth for out-of-bounds points
 * @param {Function} options.onProgress - Progress callback
 * @returns {Promise<Object>} - Result with array of XYZ paths
 */
export async function generateTracingToolpaths({
    paths,
    terrainPositions,
    terrainData,
    toolPositions,
    step,
    gridStep,
    terrainBounds,
    zFloor,
    onProgress
}) {
    const startTime = performance.now();
    debug.log('Generating tracing toolpaths...');
    debug.log(`Input: ${paths.length} paths, step=${step}, gridStep=${gridStep}, zFloor=${zFloor}`);
    debug.log(`Terrain: ${terrainData.width}×${terrainData.height}, bounds: min(${terrainBounds.min.x.toFixed(2)}, ${terrainBounds.min.y.toFixed(2)}) max(${terrainBounds.max.x.toFixed(2)}, ${terrainBounds.max.y.toFixed(2)})`);

    // Initialize WebGPU if needed
    if (!isInitialized) {
        const success = await initWebGPU();
        if (!success) {
            throw new Error('WebGPU not available');
        }
    }

    // Create sparse tool representation (reuse for all paths)
    const sparseToolData = createSparseToolFromPoints(toolPositions);
    debug.log(`Created sparse tool: ${sparseToolData.count} points`);

    // Create terrain buffer (shared across all paths)
    const terrainBuffer = device.createBuffer({
        size: terrainPositions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(terrainBuffer, 0, terrainPositions);

    // Create tool buffer (shared across all paths)
    const toolBufferData = new ArrayBuffer(sparseToolData.count * 16);
    const toolBufferI32 = new Int32Array(toolBufferData);
    const toolBufferF32 = new Float32Array(toolBufferData);

    for (let i = 0; i < sparseToolData.count; i++) {
        toolBufferI32[i * 4 + 0] = sparseToolData.xOffsets[i];
        toolBufferI32[i * 4 + 1] = sparseToolData.yOffsets[i];
        toolBufferF32[i * 4 + 2] = sparseToolData.zValues[i];
        toolBufferF32[i * 4 + 3] = 0;
    }

    const toolBuffer = device.createBuffer({
        size: toolBufferData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(toolBuffer, 0, toolBufferData);

    // Wait for buffer uploads to complete
    await device.queue.onSubmittedWorkDone();

    // Process each path
    const outputPaths = [];
    let totalSampledPoints = 0;

    for (let pathIdx = 0; pathIdx < paths.length; pathIdx++) {
        const pathStartTime = performance.now();
        const inputPath = paths[pathIdx];

        debug.log(`Processing path ${pathIdx + 1}/${paths.length}: ${inputPath.length / 2} input vertices`);

        // Sample path at specified resolution
        const sampledPath = samplePath(inputPath, step);
        const numSampledPoints = sampledPath.length / 2;
        totalSampledPoints += numSampledPoints;

        debug.log(`  Sampled to ${numSampledPoints} points`);

        // Check GPU memory limits
        const inputBufferSize = sampledPath.byteLength;
        const outputBufferSize = numSampledPoints * 4; // 4 bytes per float (Z only)
        const estimatedMemory = inputBufferSize + outputBufferSize;
        const configuredLimit = config.maxGPUMemoryMB * 1024 * 1024;
        const deviceLimit = deviceCapabilities.maxStorageBufferBindingSize;
        const maxSafeSize = Math.min(configuredLimit, deviceLimit) * config.gpuMemorySafetyMargin;

        if (estimatedMemory > maxSafeSize) {
            terrainBuffer.destroy();
            toolBuffer.destroy();
            throw new Error(
                `Path ${pathIdx + 1} exceeds GPU memory limits: ` +
                `${(estimatedMemory / 1024 / 1024).toFixed(1)}MB > ` +
                `${(maxSafeSize / 1024 / 1024).toFixed(1)}MB safe limit. ` +
                `Consider reducing step parameter or splitting path.`
            );
        }

        // Create GPU buffers for this path
        const inputBuffer = device.createBuffer({
            size: sampledPath.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(inputBuffer, 0, sampledPath);

        const outputBuffer = device.createBuffer({
            size: outputBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Create uniforms
        const uniformData = new Uint32Array([
            terrainData.width,
            terrainData.height,
            sparseToolData.count,
            numSampledPoints,
            0, // padding for alignment
            0, // padding for alignment
            0, // padding for alignment
            0, // padding for alignment
        ]);
        const uniformDataFloat = new Float32Array(uniformData.buffer);
        uniformDataFloat[4] = terrainBounds.min.x;
        uniformDataFloat[5] = terrainBounds.min.y;
        uniformDataFloat[6] = gridStep;
        uniformDataFloat[7] = zFloor;

        const uniformBuffer = device.createBuffer({
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        // Wait for buffer uploads
        await device.queue.onSubmittedWorkDone();

        // Create bind group
        const bindGroup = device.createBindGroup({
            layout: cachedTracingPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: terrainBuffer } },
                { binding: 1, resource: { buffer: toolBuffer } },
                { binding: 2, resource: { buffer: inputBuffer } },
                { binding: 3, resource: { buffer: outputBuffer } },
                { binding: 4, resource: { buffer: uniformBuffer } },
            ],
        });

        // Dispatch compute shader
        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(cachedTracingPipeline);
        passEncoder.setBindGroup(0, bindGroup);

        const workgroupsX = Math.ceil(numSampledPoints / 64);
        passEncoder.dispatchWorkgroups(workgroupsX);
        passEncoder.end();

        // Copy output to staging buffer
        const stagingBuffer = device.createBuffer({
            size: outputBufferSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputBufferSize);
        device.queue.submit([commandEncoder.finish()]);

        // Wait for GPU to finish
        await device.queue.onSubmittedWorkDone();

        // Read back results
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const outputDepths = new Float32Array(stagingBuffer.getMappedRange());
        const depthsCopy = new Float32Array(outputDepths);
        stagingBuffer.unmap();

        // Build XYZ output array
        const outputXYZ = new Float32Array(numSampledPoints * 3);
        for (let i = 0; i < numSampledPoints; i++) {
            outputXYZ[i * 3 + 0] = sampledPath[i * 2 + 0]; // X
            outputXYZ[i * 3 + 1] = sampledPath[i * 2 + 1]; // Y
            outputXYZ[i * 3 + 2] = depthsCopy[i];           // Z
        }

        outputPaths.push(outputXYZ);

        // Cleanup path-specific buffers
        inputBuffer.destroy();
        outputBuffer.destroy();
        uniformBuffer.destroy();
        stagingBuffer.destroy();

        const pathTime = performance.now() - pathStartTime;
        debug.log(`  Path ${pathIdx + 1} complete: ${numSampledPoints} points in ${pathTime.toFixed(1)}ms`);

        // Report progress
        if (onProgress) {
            onProgress({
                type: 'tracing-progress',
                data: {
                    percent: Math.round(((pathIdx + 1) / paths.length) * 100),
                    current: pathIdx + 1,
                    total: paths.length,
                    pathIndex: pathIdx
                }
            });
        }
    }

    // Cleanup shared buffers
    terrainBuffer.destroy();
    toolBuffer.destroy();

    const endTime = performance.now();
    debug.log(`Tracing complete: ${paths.length} paths, ${totalSampledPoints} total points in ${(endTime - startTime).toFixed(1)}ms`);

    return {
        paths: outputPaths,
        generationTime: endTime - startTime
    };
}

/**
 * TODO: Batched path processing
 *
 * OPTIMIZATION OPPORTUNITY:
 * Currently processes one path at a time. For better GPU utilization:
 *
 * 1. Concatenate all sampled paths into single input buffer
 * 2. Create offset table: [path1Start, path1End, path2Start, path2End, ...]
 * 3. Single GPU dispatch processes all paths
 * 4. Split output buffer back into individual path arrays
 *
 * BENEFITS:
 * - Reduce GPU dispatch overhead (N dispatches → 1 dispatch)
 * - Better GPU occupancy (more threads active)
 * - Fewer buffer create/destroy cycles
 *
 * COMPLEXITY:
 * - Need offset management in shader or CPU-side splitting
 * - Memory limit checking becomes more complex
 * - Progress reporting granularity reduced (can still report workgroup completion)
 *
 * ESTIMATE: 2-5x speedup for many small paths, minimal benefit for few large paths
 */
