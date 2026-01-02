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
 * Reusable GPU buffers for iterative tracing
 * Stored globally in worker to be reused across multiple generateTracingToolpaths calls
 */
let cachedTracingBuffers = null;

/**
 * Create reusable GPU buffers for tracing (terrain and tool buffers)
 * These persist across multiple generateTracingToolpaths calls
 * @param {Float32Array} terrainPositions - Dense terrain Z-only grid
 * @param {Float32Array} toolPositions - Tool points (XYZ triplets)
 * @returns {Object} - Buffer handles and metadata
 */
export function createReusableTracingBuffers(terrainPositions, toolPositions) {
    if (!isInitialized) {
        throw new Error('WebGPU not initialized');
    }

    // Destroy existing buffers if any
    if (cachedTracingBuffers) {
        destroyReusableTracingBuffers();
    }

    // Create sparse tool representation
    const sparseToolData = createSparseToolFromPoints(toolPositions);
    debug.log(`Created reusable tracing buffers: terrain ${terrainPositions.length} floats, tool ${sparseToolData.count} points`);

    // Create terrain buffer
    const terrainBuffer = device.createBuffer({
        size: terrainPositions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(terrainBuffer, 0, terrainPositions);

    // Create tool buffer
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

    cachedTracingBuffers = {
        terrainBuffer,
        toolBuffer,
        sparseToolData
    };

    return cachedTracingBuffers;
}

/**
 * Destroy reusable tracing buffers
 */
export function destroyReusableTracingBuffers() {
    if (cachedTracingBuffers) {
        cachedTracingBuffers.terrainBuffer.destroy();
        cachedTracingBuffers.toolBuffer.destroy();
        cachedTracingBuffers = null;
        debug.log('Destroyed reusable tracing buffers');
    }
}

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

    // Use cached buffers if available, otherwise create them for this call
    let terrainBuffer, toolBuffer, sparseToolData;
    let shouldCleanupBuffers = false;

    if (cachedTracingBuffers) {
        // Reuse existing buffers (optimized for iterative tracing)
        debug.log('Using cached tracing buffers');
        terrainBuffer = cachedTracingBuffers.terrainBuffer;
        toolBuffer = cachedTracingBuffers.toolBuffer;
        sparseToolData = cachedTracingBuffers.sparseToolData;
    } else {
        // Create temporary buffers (will be cleaned up at end)
        debug.log('Creating temporary tracing buffers');
        sparseToolData = createSparseToolFromPoints(toolPositions);
        debug.log(`Created sparse tool: ${sparseToolData.count} points`);

        // Create terrain buffer
        terrainBuffer = device.createBuffer({
            size: terrainPositions.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(terrainBuffer, 0, terrainPositions);

        // Create tool buffer
        const toolBufferData = new ArrayBuffer(sparseToolData.count * 16);
        const toolBufferI32 = new Int32Array(toolBufferData);
        const toolBufferF32 = new Float32Array(toolBufferData);

        for (let i = 0; i < sparseToolData.count; i++) {
            toolBufferI32[i * 4 + 0] = sparseToolData.xOffsets[i];
            toolBufferI32[i * 4 + 1] = sparseToolData.yOffsets[i];
            toolBufferF32[i * 4 + 2] = sparseToolData.zValues[i];
            toolBufferF32[i * 4 + 3] = 0;
        }

        toolBuffer = device.createBuffer({
            size: toolBufferData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(toolBuffer, 0, toolBufferData);

        // Wait for buffer uploads to complete
        await device.queue.onSubmittedWorkDone();

        shouldCleanupBuffers = true;
    }

    // Create maxZ buffer (one i32 per path for atomic operations)
    // Initialize to sentinel value (bitcast of -1e30)
    const SENTINEL_Z = -1e30;
    const sentinelBits = new Float32Array([SENTINEL_Z]);
    const sentinelI32 = new Int32Array(sentinelBits.buffer)[0];
    const maxZInitData = new Int32Array(paths.length).fill(sentinelI32);

    const maxZBuffer = device.createBuffer({
        size: maxZInitData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(maxZBuffer, 0, maxZInitData);

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 1: Sample all paths and build unified buffer
    // ═══════════════════════════════════════════════════════════════════════
    debug.log('PHASE 1: Sampling all paths...');
    const pathIndex = [];  // Maps path ID → unified buffer offsets
    const sampledSegments = [];
    let totalSampledPoints = 0;

    for (let pathIdx = 0; pathIdx < paths.length; pathIdx++) {
        const inputPath = paths[pathIdx];
        debug.log(`Path ${pathIdx + 1}/${paths.length}: ${inputPath.length / 2} input vertices`);

        // Sample path at specified resolution
        const sampledPath = samplePath(inputPath, step);
        const numPoints = sampledPath.length / 2;

        pathIndex.push({
            startOffset: totalSampledPoints,
            endOffset: totalSampledPoints + numPoints,
            numPoints: numPoints
        });

        sampledSegments.push(sampledPath);
        totalSampledPoints += numPoints;

        debug.log(`  Sampled to ${numPoints} points`);
    }

    // Concatenate all sampled paths into unified buffer
    const unifiedSampledXY = new Float32Array(totalSampledPoints * 2);
    let writeOffset = 0;

    for (let pathIdx = 0; pathIdx < sampledSegments.length; pathIdx++) {
        const sampledPath = sampledSegments[pathIdx];
        unifiedSampledXY.set(sampledPath, writeOffset * 2);
        writeOffset += sampledPath.length / 2;
    }

    debug.log(`Unified buffer: ${totalSampledPoints} total points from ${paths.length} paths`);

    // Debug: Log first sampled point
    if (totalSampledPoints > 0) {
        const firstX = unifiedSampledXY[0];
        const firstY = unifiedSampledXY[1];
        const gridX = (firstX - terrainBounds.min.x) / gridStep;
        const gridY = (firstY - terrainBounds.min.y) / gridStep;
        debug.log(`First point: world(${firstX.toFixed(2)}, ${firstY.toFixed(2)}) -> grid(${gridX.toFixed(2)}, ${gridY.toFixed(2)})`);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 2: Calculate memory budget and create chunks
    // ═══════════════════════════════════════════════════════════════════════
    debug.log('PHASE 2: Calculating memory budget and chunking...');
    const bytesPerPoint = 8 + 4;  // XY input (2 floats) + Z output (1 float)
    const configuredLimit = config.maxGPUMemoryMB * 1024 * 1024;
    const deviceLimit = deviceCapabilities.maxStorageBufferBindingSize;
    const maxSafeSize = Math.min(configuredLimit, deviceLimit) * config.gpuMemorySafetyMargin;

    // Fixed overhead: terrain, tool, maxZ, uniforms
    const fixedOverhead = terrainPositions.byteLength +
                          (sparseToolData.count * 16) +
                          (paths.length * 4) +
                          48;

    if (fixedOverhead > maxSafeSize) {
        if (shouldCleanupBuffers) {
            terrainBuffer.destroy();
            toolBuffer.destroy();
        }
        throw new Error(
            `Fixed buffers (terrain + tool) exceed GPU memory: ` +
            `${(fixedOverhead / 1024 / 1024).toFixed(1)}MB > ` +
            `${(maxSafeSize / 1024 / 1024).toFixed(1)}MB. ` +
            `Try reducing terrain resolution or tool density.`
        );
    }

    const availableForPaths = maxSafeSize - fixedOverhead;
    const maxPointsPerChunk = Math.floor(availableForPaths / bytesPerPoint);

    debug.log(`Memory budget: ${(maxSafeSize / 1024 / 1024).toFixed(1)}MB safe, ${(availableForPaths / 1024 / 1024).toFixed(1)}MB available for paths`);
    debug.log(`Max points per chunk: ${maxPointsPerChunk.toLocaleString()}`);

    // Create chunks
    const chunks = [];
    let currentStart = 0;
    while (currentStart < totalSampledPoints) {
        const currentEnd = Math.min(currentStart + maxPointsPerChunk, totalSampledPoints);
        chunks.push({
            startPoint: currentStart,
            endPoint: currentEnd,
            numPoints: currentEnd - currentStart
        });
        currentStart = currentEnd;
    }

    debug.log(`Created ${chunks.length} chunk(s) for processing`);

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 3: Create reusable GPU buffers (buffer pool pattern)
    // ═══════════════════════════════════════════════════════════════════════
    debug.log('PHASE 3: Creating reusable GPU buffers...');

    // Input buffer: XY pairs for sampled points
    const inputBuffer = device.createBuffer({
        size: maxPointsPerChunk * 8,  // 2 floats per point
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Output buffer: Z depths
    const outputBuffer = device.createBuffer({
        size: maxPointsPerChunk * 4,  // 1 float per point
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Uniform buffer
    const uniformBuffer = device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Staging buffer for readback
    const stagingBuffer = device.createBuffer({
        size: maxPointsPerChunk * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Unified output array (filled chunk-by-chunk)
    const unifiedOutputZ = new Float32Array(totalSampledPoints);

    debug.log(`Buffers created for ${maxPointsPerChunk.toLocaleString()} points per chunk`);

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 4: Process each chunk with single GPU dispatch
    // ═══════════════════════════════════════════════════════════════════════
    debug.log('PHASE 4: Processing chunks...');

    for (let chunkIdx = 0; chunkIdx < chunks.length; chunkIdx++) {
        const chunk = chunks[chunkIdx];
        const { startPoint, endPoint, numPoints } = chunk;

        debug.log(`Processing chunk ${chunkIdx + 1}/${chunks.length}: points ${startPoint}-${endPoint} (${numPoints} points)`);

        // Extract chunk slice from unified buffer
        const chunkInputXY = unifiedSampledXY.subarray(startPoint * 2, endPoint * 2);

        // Upload to GPU (reuse same buffers)
        device.queue.writeBuffer(inputBuffer, 0, chunkInputXY);

        // Update uniforms for this chunk
        const uniformData = new Uint32Array(12); // 48 bytes
        uniformData[0] = terrainData.width;
        uniformData[1] = terrainData.height;
        uniformData[2] = sparseToolData.count;
        uniformData[3] = numPoints;  // point_count for THIS CHUNK
        uniformData[4] = 0;  // path_index (unused, maxZ computed on CPU)

        const uniformDataFloat = new Float32Array(uniformData.buffer);
        uniformDataFloat[5] = terrainBounds.min.x;
        uniformDataFloat[6] = terrainBounds.min.y;
        uniformDataFloat[7] = gridStep;
        uniformDataFloat[8] = zFloor;

        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        // Wait for uploads
        await device.queue.onSubmittedWorkDone();

        // Create bind group (same bindings as before)
        const bindGroup = device.createBindGroup({
            layout: cachedTracingPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: terrainBuffer } },
                { binding: 1, resource: { buffer: toolBuffer } },
                { binding: 2, resource: { buffer: inputBuffer } },
                { binding: 3, resource: { buffer: outputBuffer } },
                { binding: 4, resource: { buffer: maxZBuffer } },  // Keep for shader compatibility
                { binding: 5, resource: { buffer: uniformBuffer } },
            ],
        });

        // Single GPU dispatch for entire chunk
        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(cachedTracingPipeline);
        passEncoder.setBindGroup(0, bindGroup);

        const workgroupsX = Math.ceil(numPoints / 64);
        passEncoder.dispatchWorkgroups(workgroupsX);
        passEncoder.end();

        // Copy output to staging buffer
        commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, numPoints * 4);
        device.queue.submit([commandEncoder.finish()]);

        // Wait for GPU to finish
        await device.queue.onSubmittedWorkDone();

        // Read back results
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const chunkOutputZ = new Float32Array(stagingBuffer.getMappedRange(), 0, numPoints);

        // Copy to unified output array
        unifiedOutputZ.set(chunkOutputZ, startPoint);

        stagingBuffer.unmap();

        debug.log(`  Chunk ${chunkIdx + 1} complete: ${numPoints} points processed`);

        // Report progress (point-based, not path-based)
        if (onProgress) {
            onProgress({
                type: 'tracing-progress',
                data: {
                    percent: Math.round((endPoint / totalSampledPoints) * 100),
                    current: endPoint,
                    total: totalSampledPoints,
                    chunkIndex: chunkIdx + 1,
                    totalChunks: chunks.length
                }
            });
        }
    }

    // Cleanup reusable buffers
    inputBuffer.destroy();
    outputBuffer.destroy();
    uniformBuffer.destroy();
    stagingBuffer.destroy();

    debug.log('All chunks processed');

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 5: Remap unified output back to individual paths & compute maxZ
    // ═══════════════════════════════════════════════════════════════════════
    debug.log('PHASE 5: Remapping to individual paths and computing maxZ...');

    const outputPaths = [];
    const maxZValues = new Array(paths.length).fill(zFloor);

    for (let pathIdx = 0; pathIdx < pathIndex.length; pathIdx++) {
        const { startOffset, numPoints } = pathIndex[pathIdx];

        if (numPoints === 0) {
            outputPaths.push(new Float32Array(0));  // Empty path
            debug.log(`Path ${pathIdx + 1}: empty`);
            continue;
        }

        // Allocate XYZ output
        const pathXYZ = new Float32Array(numPoints * 3);

        // Copy from unified buffers + compute maxZ
        for (let i = 0; i < numPoints; i++) {
            const unifiedIdx = startOffset + i;
            const x = unifiedSampledXY[unifiedIdx * 2 + 0];
            const y = unifiedSampledXY[unifiedIdx * 2 + 1];
            const z = unifiedOutputZ[unifiedIdx];

            pathXYZ[i * 3 + 0] = x;
            pathXYZ[i * 3 + 1] = y;
            pathXYZ[i * 3 + 2] = z;

            // Track max Z for this path (CPU-side)
            maxZValues[pathIdx] = Math.max(maxZValues[pathIdx], z);
        }

        outputPaths.push(pathXYZ);
        debug.log(`Path ${pathIdx + 1}: ${numPoints} points, maxZ=${maxZValues[pathIdx].toFixed(2)}`);
    }

    // Cleanup maxZ buffer (was only used for shader compatibility)
    maxZBuffer.destroy();

    // Cleanup temporary buffers only (don't destroy cached buffers)
    if (shouldCleanupBuffers) {
        terrainBuffer.destroy();
        toolBuffer.destroy();
        debug.log('Cleaned up temporary tracing buffers');
    }

    const endTime = performance.now();
    debug.log(`Tracing complete: ${paths.length} paths, ${totalSampledPoints} total points in ${(endTime - startTime).toFixed(1)}ms`);
    debug.log(`Max Z values: [${Array.from(maxZValues).map(z => z.toFixed(2)).join(', ')}]`);

    return {
        paths: outputPaths,
        maxZ: Array.from(maxZValues),
        generationTime: endTime - startTime
    };
}

/**
 * IMPLEMENTATION NOTE: Unified Batching System
 *
 * This function uses a unified batching approach for optimal performance:
 *
 * 1. All paths are sampled and concatenated into a single unified buffer
 * 2. Paths are chunked based on GPU memory limits (handles giant paths)
 * 3. Each chunk is processed with a single GPU dispatch (reduces overhead)
 * 4. Output is remapped back to individual path arrays
 * 5. MaxZ is computed on CPU (avoids complex GPU atomic coordination)
 *
 * BENEFITS:
 * - Handles paths that exceed GPU memory limits (automatic chunking)
 * - Reduces GPU dispatch overhead (10-100x for many small paths)
 * - Better progress tracking (point-based instead of path-based)
 * - Buffer pool pattern reduces allocation overhead
 */
