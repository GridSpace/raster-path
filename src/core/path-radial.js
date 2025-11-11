/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Path Radial - Cylindrical Toolpath Generation
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Generates toolpaths for cylindrical (lathe-like) parts by rasterizing at
 * multiple rotation angles. Uses X-bucketing spatial partitioning and bucket
 * batching to handle large models efficiently.
 *
 * EXPORTS:
 * ────────
 * Functions:
 *   - radialRasterize({triangles, bucketData, resolution, angleStep, ...})
 *       Rasterize model at multiple angles with X-bucketing
 *   - generateRadialToolpaths({triangles, bucketData, toolData, ...})
 *       Complete pipeline: rasterize + generate toolpaths for all angles
 *
 * ALGORITHM:
 * ──────────
 * For each angle θ:
 *   1. Rotate coordinate frame by θ around Z-axis
 *   2. For each X-bucket (spatial partition along X-axis):
 *      - Cast rays in YZ plane at this rotated angle
 *      - Test only triangles in this X-bucket (spatial culling)
 *      - Record max Z hit per (X, Y) grid cell
 *   3. Stitch bucket results into complete angle strip
 *   4. Generate toolpath by scanning sparse tool over strip
 *
 * X-BUCKETING:
 * ────────────
 * Triangles are pre-sorted into X-axis buckets (computed on main thread):
 * - Each bucket covers a slice of X-range (e.g., 5mm wide)
 * - Bucket contains indices of triangles overlapping that X-range
 * - GPU processes one bucket at a time, testing only relevant triangles
 * - Reduces triangle tests by ~10-100x depending on model geometry
 *
 * BUCKET BATCHING:
 * ────────────────
 * To avoid GPU timeouts on complex models:
 * - Estimate work per bucket: numTriangles × numAngles × cellsPerBucket
 * - Batch buckets to keep work under ~1M ray-triangle tests per dispatch
 * - Typical: 4-20 buckets per batch, multiple dispatches per angle set
 *
 * ANGLE BATCHING:
 * ───────────────
 * To avoid GPU memory allocation failures:
 * - Calculate memory needed: numAngles × gridWidth × gridHeight × 4 bytes
 * - If exceeds limit (1.8GB), split into angle batches
 * - Reuse triangle/bucket GPU buffers across angle batches
 * - Process angle ranges sequentially (e.g., 0-180°, 180-360°)
 *
 * OUTPUT FORMAT:
 * ──────────────
 * Array of strips, one per angle:
 *   [{
 *     angle: number,              // Rotation angle (degrees)
 *     pathData: Float32Array,     // Z-heights, row-major
 *     numScanlines: number,       // Number of Y scanlines
 *     pointsPerLine: number,      // Number of X samples per scanline
 *     terrainBounds: {...}        // Bounding box for this strip
 *   }, ...]
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

import {
    device, config, cachedRadialBatchPipeline, debug, diagnostic
} from './raster-config.js';
import { createSparseToolFromPoints } from './raster-tool.js';
import {
    createReusableToolpathBuffers,
    destroyReusableToolpathBuffers,
    runToolpathComputeWithBuffers
} from './path-planar.js';

// Radial: Rasterize model with rotating ray planes and X-bucketing
export async function radialRasterize({
    triangles,
    bucketData,
    resolution,
    angleStep,
    numAngles,
    maxRadius,
    toolWidth,
    zFloor,
    bounds,
    startAngle = 0,
    reusableBuffers = null,
    returnBuffersForReuse = false,
    batchInfo = {}
}) {
    if (!device) {
        throw new Error('WebGPU not initialized');
    }

    const timings = {
        start: performance.now(),
        prep: 0,
        gpu: 0,
        stitch: 0
    };

    // Calculate grid dimensions based on BUCKET range (not model bounds)
    // Buckets may extend slightly beyond model bounds due to rounding
    const bucketMinX = bucketData.buckets[0].minX;
    const bucketMaxX = bucketData.buckets[bucketData.numBuckets - 1].maxX;
    const gridWidth = Math.ceil((bucketMaxX - bucketMinX) / resolution);
    const gridYHeight = Math.ceil(toolWidth / resolution);
    const bucketGridWidth = Math.ceil((bucketData.buckets[0].maxX - bucketData.buckets[0].minX) / resolution);

    // Calculate workgroup load distribution for timeout analysis
    const bucketTriangleCounts = bucketData.buckets.map(b => b.count);
    const minTriangles = Math.min(...bucketTriangleCounts);
    const maxTriangles = Math.max(...bucketTriangleCounts);
    const avgTriangles = bucketTriangleCounts.reduce((a, b) => a + b, 0) / bucketTriangleCounts.length;
    const workPerWorkgroup = maxTriangles * numAngles * bucketGridWidth * gridYHeight;

    // Determine bucket batching to avoid GPU timeouts AND respect thread limits
    // Constraint 1: Keep work per batch under ~10B ray-triangle tests (work-based limit)
    // Constraint 2: Keep concurrent threads under GPU watchdog limit (thread-based limit)
    const maxWorkPerBatch = 1e10;
    const estimatedWorkPerBucket = avgTriangles * numAngles * bucketGridWidth * gridYHeight;

    // Calculate thread-based limit
    // Each bucket batch dispatches: dispatchX × dispatchY × bucketsInBatch workgroups
    // Workgroup size is 8×8×1 = 64 threads
    // Total threads = (numAngles/8) × (gridYHeight/8) × bucketsInBatch × 64
    const THREADS_PER_WORKGROUP = 64;
    const maxConcurrentThreads = config.maxConcurrentThreads || 32768;
    const dispatchX = Math.ceil(numAngles / 8);
    const dispatchY = Math.ceil(gridYHeight / 8);
    const threadsPerBucket = dispatchX * dispatchY * THREADS_PER_WORKGROUP;
    const threadLimitBuckets = Math.max(1, Math.floor(maxConcurrentThreads / threadsPerBucket));

    // Calculate buckets per batch, enforcing both work and thread limits
    let maxBucketsPerBatch;
    if (estimatedWorkPerBucket === 0) {
        maxBucketsPerBatch = Math.min(threadLimitBuckets, bucketData.numBuckets); // Empty model
    } else {
        const workBasedLimit = Math.floor(maxWorkPerBatch / estimatedWorkPerBucket);

        // Use the MORE RESTRICTIVE of work-based or thread-based limits
        const idealBucketsPerBatch = Math.min(workBasedLimit, threadLimitBuckets);

        // Apply minimum only if it doesn't violate thread limit
        const minBucketsPerBatch = Math.min(4, bucketData.numBuckets, threadLimitBuckets);
        maxBucketsPerBatch = Math.max(minBucketsPerBatch, idealBucketsPerBatch);

        // Cap at total buckets
        maxBucketsPerBatch = Math.min(maxBucketsPerBatch, bucketData.numBuckets);
    }

    const numBucketBatches = Math.ceil(bucketData.numBuckets / maxBucketsPerBatch);

    if (diagnostic) {
        debug.log(`Radial: ${gridWidth}x${gridYHeight} grid, ${numAngles} angles, ${bucketData.buckets.length} buckets`);
        debug.log(`Load: min=${minTriangles} max=${maxTriangles} avg=${avgTriangles.toFixed(0)} (${(maxTriangles/avgTriangles).toFixed(2)}x imbalance, worst=${(workPerWorkgroup/1e6).toFixed(1)}M tests)`);
        debug.log(`Thread limits: ${threadsPerBucket} threads/bucket, max ${threadLimitBuckets} buckets/dispatch (${maxConcurrentThreads} thread limit)`);
        debug.log(`Estimated work/bucket: ${(estimatedWorkPerBucket/1e6).toFixed(1)}M tests`);
        debug.log(`Bucket batching: ${numBucketBatches} batches of ${maxBucketsPerBatch} buckets (work limit: ${Math.floor(maxWorkPerBatch / estimatedWorkPerBucket)}, thread limit: ${threadLimitBuckets})`);
    }

    // Reuse buffers if provided, otherwise create new ones
    let triangleBuffer, triangleIndicesBuffer;
    let shouldCleanupBuffers = false;

    if (reusableBuffers) {
        // Reuse cached buffers from previous angle batch
        triangleBuffer = reusableBuffers.triangleBuffer;
        triangleIndicesBuffer = reusableBuffers.triangleIndicesBuffer;
    } else {
        // Create new GPU buffers (first batch or non-batched operation)
        shouldCleanupBuffers = true;

        triangleBuffer = device.createBuffer({
            size: triangles.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(triangleBuffer.getMappedRange()).set(triangles);
        triangleBuffer.unmap();

        // Create triangle indices buffer
        triangleIndicesBuffer = device.createBuffer({
            size: bucketData.triangleIndices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Uint32Array(triangleIndicesBuffer.getMappedRange()).set(bucketData.triangleIndices);
        triangleIndicesBuffer.unmap();
    }

    // Create output buffer (all angles, all buckets)
    const outputSize = numAngles * bucketData.numBuckets * bucketGridWidth * gridYHeight * 4;
    const outputBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    // CRITICAL: Initialize output buffer with zFloor to avoid reading garbage data
    const initData = new Float32Array(outputSize / 4);
    initData.fill(zFloor);
    device.queue.writeBuffer(outputBuffer, 0, initData);
    // Note: No need to wait - GPU will execute writeBuffer before compute shader

    // Prep complete, GPU starting
    timings.prep = performance.now() - timings.start;
    const gpuStart = performance.now();

    // Use cached pipeline (created in initWebGPU)
    const pipeline = cachedRadialBatchPipeline;

    // Process buckets in batches to avoid GPU timeouts
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);

    // Collect buffers to destroy after GPU completes
    const batchBuffersToDestroy = [];

    for (let batchIdx = 0; batchIdx < numBucketBatches; batchIdx++) {
        const startBucket = batchIdx * maxBucketsPerBatch;
        const endBucket = Math.min(startBucket + maxBucketsPerBatch, bucketData.numBuckets);
        const bucketsInBatch = endBucket - startBucket;

        // Create bucket info buffer for this batch
        const bucketInfoSize = bucketsInBatch * 16; // 4 fields * 4 bytes per bucket
        const bucketInfoBuffer = device.createBuffer({
            size: bucketInfoSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });

        const bucketView = new ArrayBuffer(bucketInfoSize);
        const bucketFloatView = new Float32Array(bucketView);
        const bucketUintView = new Uint32Array(bucketView);

        for (let i = 0; i < bucketsInBatch; i++) {
            const bucket = bucketData.buckets[startBucket + i];
            const offset = i * 4;
            bucketFloatView[offset] = bucket.minX;           // f32
            bucketFloatView[offset + 1] = bucket.maxX;       // f32
            bucketUintView[offset + 2] = bucket.startIndex;  // u32
            bucketUintView[offset + 3] = bucket.count;       // u32
        }

        new Uint8Array(bucketInfoBuffer.getMappedRange()).set(new Uint8Array(bucketView));
        bucketInfoBuffer.unmap();

        // Create uniforms for this batch
        // Struct layout: f32, f32, u32, f32, f32, u32, f32, u32, f32, f32, u32, u32, f32, u32
        const uniformBuffer = device.createBuffer({
            size: 56,  // 14 fields * 4 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });

        const uniformView = new ArrayBuffer(56);
        const floatView = new Float32Array(uniformView);
        const uintView = new Uint32Array(uniformView);

        floatView[0] = resolution;                                          // f32
        floatView[1] = angleStep * (Math.PI / 180);                         // f32
        uintView[2] = numAngles;                                            // u32
        floatView[3] = maxRadius;                                           // f32
        floatView[4] = toolWidth;                                           // f32
        uintView[5] = gridYHeight;                                          // u32
        floatView[6] = bucketData.buckets[0].maxX - bucketData.buckets[0].minX;  // f32 bucketWidth
        uintView[7] = bucketGridWidth;                                      // u32
        floatView[8] = bucketMinX;                                          // f32 global_min_x
        floatView[9] = zFloor;                                              // f32
        uintView[10] = 0;                                                   // u32 filterMode
        uintView[11] = bucketData.numBuckets;                               // u32 (total buckets, for validation)
        floatView[12] = startAngle * (Math.PI / 180);                       // f32 start_angle (radians)
        uintView[13] = startBucket;                                         // u32 bucket_offset

        new Uint8Array(uniformBuffer.getMappedRange()).set(new Uint8Array(uniformView));
        uniformBuffer.unmap();

        // Create bind group for this batch
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: triangleBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } },
                { binding: 2, resource: { buffer: uniformBuffer } },
                { binding: 3, resource: { buffer: bucketInfoBuffer } },
                { binding: 4, resource: { buffer: triangleIndicesBuffer } }
            ]
        });

        // Dispatch this batch - just dispatch normally
        // The XY dimensions are fixed by the problem (angles × gridYHeight)
        // If this exceeds the thread limit, the real fix is to increase maxBucketsPerBatch
        // in the work estimation code (lines 118-137) to create more, smaller batches
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(dispatchX, dispatchY, bucketsInBatch);

        if (diagnostic) {
            const totalThreads = dispatchX * dispatchY * bucketsInBatch * THREADS_PER_WORKGROUP;
            debug.log(`  Batch ${batchIdx + 1}/${numBucketBatches}: (${dispatchX}, ${dispatchY}, ${bucketsInBatch}) = ${totalThreads} threads, buckets ${startBucket}-${endBucket - 1}`);
        }

        // Save buffers to destroy after GPU completes
        batchBuffersToDestroy.push(uniformBuffer, bucketInfoBuffer);
    }

    passEncoder.end();

    // Read back
    const stagingBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputSize);
    device.queue.submit([commandEncoder.finish()]);

    // Wait for GPU to finish before reading results
    await device.queue.onSubmittedWorkDone();
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    // const outputData = new Float32Array(stagingBuffer.getMappedRange());
    // const outputCopy = new Float32Array(outputData);
    const outputCopy = new Float32Array(stagingBuffer.getMappedRange().slice());
    stagingBuffer.unmap();

    // Now safe to destroy batch buffers (GPU has completed)
    for (const buffer of batchBuffersToDestroy) {
        buffer.destroy();
    }

    // Cleanup main buffers
    outputBuffer.destroy();
    stagingBuffer.destroy();

    timings.gpu = performance.now() - gpuStart;

    // Stitch strips
    const stitchStart = performance.now();
    const strips = [];

    for (let angleIdx = 0; angleIdx < numAngles; angleIdx++) {
        const stripData = new Float32Array(gridWidth * gridYHeight);
        stripData.fill(zFloor);  // Initialize with zFloor, not zeros!

        // Gather from each bucket
        for (let bucketIdx = 0; bucketIdx < bucketData.numBuckets; bucketIdx++) {
            const bucket = bucketData.buckets[bucketIdx];
            const bucketMinGridX = Math.floor((bucket.minX - bucketMinX) / resolution);

            for (let localX = 0; localX < bucketGridWidth; localX++) {
                const gridX = bucketMinGridX + localX;
                if (gridX >= gridWidth) continue;

                for (let gridY = 0; gridY < gridYHeight; gridY++) {
                    const srcIdx = bucketIdx * numAngles * bucketGridWidth * gridYHeight
                                 + angleIdx * bucketGridWidth * gridYHeight
                                 + gridY * bucketGridWidth
                                 + localX;
                    const dstIdx = gridY * gridWidth + gridX;
                    stripData[dstIdx] = outputCopy[srcIdx];
                }
            }
        }

        // Keep as DENSE Z-only format (toolpath generator expects this!)
        // Count valid points
        let validCount = 0;
        for (let i = 0; i < stripData.length; i++) {
            if (stripData[i] !== zFloor) validCount++;
        }

        strips.push({
            angle: startAngle + (angleIdx * angleStep),
            positions: stripData,  // DENSE Z-only format!
            gridWidth,
            gridHeight: gridYHeight,
            pointCount: validCount,  // Number of non-floor cells
            bounds: {
                min: { x: bucketMinX, y: 0, z: zFloor },
                max: { x: bucketMaxX, y: toolWidth, z: bounds.max.z }
            }
        });
    }

    timings.stitch = performance.now() - stitchStart;
    const totalTime = performance.now() - timings.start;

    Object.assign(batchInfo, {
        'prep': (timings.prep | 0),
        'raster': (timings.gpu | 0),
        'stitch': (timings.stitch | 0)
    });

    const result = { strips, timings };

    // Decide what to do with triangle/indices buffers
    // Note: bucketInfoBuffer is now created/destroyed per bucket batch within the loop
    if (returnBuffersForReuse && shouldCleanupBuffers) {
        // First batch in multi-batch operation: return buffers for subsequent batches to reuse
        result.reusableBuffers = {
            triangleBuffer,
            triangleIndicesBuffer
        };
    } else if (shouldCleanupBuffers) {
        // Single batch operation OR we're NOT supposed to return buffers: destroy them now
        triangleBuffer.destroy();
        triangleIndicesBuffer.destroy();
    }
    // else: we're reusing buffers from a previous angle batch, don't destroy them (caller will destroy after all angle batches)

    return result;
}

// Radial: Complete pipeline - rasterize model + generate toolpaths for all strips
export async function generateRadialToolpaths({
    triangles,
    bucketData,
    toolData,
    resolution,
    angleStep,
    numAngles,
    maxRadius,
    toolWidth,
    zFloor,
    bounds,
    xStep,
    yStep
}) {
    debug.log('radial-generate-toolpaths', { triangles: triangles.length, numAngles, resolution });

    // Batch processing: rasterize angle ranges to avoid memory allocation failure
    // Calculate safe batch size based on available GPU memory
    const MAX_BUFFER_SIZE_MB = 1800; // Stay under 2GB WebGPU limit with headroom
    const bytesPerCell = 4; // f32

    const xSize = bounds.max.x - bounds.min.x;
    const ySize = bounds.max.y - bounds.min.y;
    const gridXSize = Math.ceil(xSize / resolution);
    const gridYHeight = Math.ceil(ySize / resolution);

    // Calculate total memory requirement
    const cellsPerAngle = gridXSize * gridYHeight;
    const bytesPerAngle = cellsPerAngle * bytesPerCell;
    const totalMemoryMB = (numAngles * bytesPerAngle) / (1024 * 1024);

    // Only batch if total memory exceeds threshold
    const batchDivisor = config?.batchDivisor || 1;
    let ANGLES_PER_BATCH, numBatches;
    if (totalMemoryMB > MAX_BUFFER_SIZE_MB) {
        // Need to batch
        const maxAnglesPerBatch = Math.floor((MAX_BUFFER_SIZE_MB * 1024 * 1024) / bytesPerAngle);
        // Apply batch divisor for overhead testing
        const adjustedMaxAngles = Math.floor(maxAnglesPerBatch / batchDivisor);

        ANGLES_PER_BATCH = Math.max(1, Math.min(adjustedMaxAngles, numAngles));
        numBatches = Math.ceil(numAngles / ANGLES_PER_BATCH);
        const batchSizeMB = (ANGLES_PER_BATCH * bytesPerAngle / 1024 / 1024).toFixed(1);
        debug.log(`Grid: ${gridXSize} x ${gridYHeight}, ${cellsPerAngle.toLocaleString()} cells/angle`);
        debug.log(`Total memory: ${totalMemoryMB.toFixed(1)}MB exceeds limit, batching required`);
        if (batchDivisor > 1) {
            debug.log(`batchDivisor: ${batchDivisor}x (testing overhead: ${maxAnglesPerBatch} → ${adjustedMaxAngles} angles/batch)`);
        }
        debug.log(`Batch size: ${ANGLES_PER_BATCH} angles (~${batchSizeMB}MB per batch)`);
        debug.log(`Processing ${numAngles} angles in ${numBatches} batch(es)`);
    } else {
        // Process all angles at once (but still respect batchDivisor for testing)
        if (batchDivisor > 1) {
            ANGLES_PER_BATCH = Math.max(10, Math.floor(numAngles / batchDivisor));
            numBatches = Math.ceil(numAngles / ANGLES_PER_BATCH);
            debug.log(`Grid: ${gridXSize} x ${gridYHeight}, ${cellsPerAngle.toLocaleString()} cells/angle`);
            debug.log(`Total memory: ${totalMemoryMB.toFixed(1)}MB (fits in buffer normally)`);
            debug.log(`batchDivisor: ${batchDivisor}x (artificially creating ${numBatches} batches for overhead testing)`);
        } else {
            ANGLES_PER_BATCH = numAngles;
            numBatches = 1;
            debug.log(`Grid: ${gridXSize} x ${gridYHeight}, ${cellsPerAngle.toLocaleString()} cells/angle`);
            debug.log(`Total memory: ${totalMemoryMB.toFixed(1)}MB fits in buffer, processing all ${numAngles} angles in single batch`);
        }
    }

    const allStripToolpaths = [];
    let totalToolpathPoints = 0;
    const pipelineStartTime = performance.now();

    // Prepare sparse tool once
    const sparseToolData = createSparseToolFromPoints(toolData.positions);
    debug.log(`Created sparse tool: ${sparseToolData.count} points (reusing for all strips)`);

    // Create reusable rasterization buffers if batching (numBatches > 1)
    // These buffers (triangles, buckets, indices) don't change between batches
    let batchReuseBuffers = null;
    let batchTracking = [];

    for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        const batchStartTime = performance.now();
        const startAngleIdx = batchIdx * ANGLES_PER_BATCH;
        const endAngleIdx = Math.min(startAngleIdx + ANGLES_PER_BATCH, numAngles);
        const batchNumAngles = endAngleIdx - startAngleIdx;
        const batchStartAngle = startAngleIdx * angleStep;

        const batchInfo = {
            from: startAngleIdx,
            to: endAngleIdx
        };
        batchTracking.push(batchInfo);

        debug.log(`Batch ${batchIdx + 1}/${numBatches}: angles ${startAngleIdx}-${endAngleIdx - 1} (${batchNumAngles} angles), startAngle=${batchStartAngle.toFixed(1)}°`);

        // Rasterize this batch of strips
        const rasterStartTime = performance.now();
        const shouldReturnBuffers = (batchIdx === 0 && numBatches > 1);  // First batch of multi-batch operation
        const batchModelResult = await radialRasterize({
            triangles,
            bucketData,
            resolution,
            angleStep,
            numAngles: batchNumAngles,
            maxRadius,
            toolWidth,
            zFloor,
            bounds,
            startAngle: batchStartAngle,
            reusableBuffers: batchReuseBuffers,
            returnBuffersForReuse: shouldReturnBuffers,
            batchInfo
        });

        const rasterTime = performance.now() - rasterStartTime;

        // Capture buffers from first batch for reuse
        if (batchIdx === 0 && batchModelResult.reusableBuffers) {
            batchReuseBuffers = batchModelResult.reusableBuffers;
        }

        // Find max dimensions for this batch
        let maxStripWidth = 0;
        let maxStripHeight = 0;
        for (const strip of batchModelResult.strips) {
            maxStripWidth = Math.max(maxStripWidth, strip.gridWidth);
            maxStripHeight = Math.max(maxStripHeight, strip.gridHeight);
        }

        // Create reusable buffers for this batch
        const reusableBuffers = createReusableToolpathBuffers(maxStripWidth, maxStripHeight, sparseToolData, xStep, maxStripHeight);

        // Generate toolpaths for this batch
        const toolpathStartTime = performance.now();

        for (let i = 0; i < batchModelResult.strips.length; i++) {
            const strip = batchModelResult.strips[i];
            const globalStripIdx = startAngleIdx + i;

            if (globalStripIdx % 10 === 0 || globalStripIdx === numAngles - 1) {
                // Reserve final 2% for cleanup phase after strip processing
                const stripProgress = ((globalStripIdx + 1) / numAngles) * 98;
                self.postMessage({
                    type: 'toolpath-progress',
                    data: {
                        percent: Math.round(stripProgress),
                        current: globalStripIdx + 1,
                        total: numAngles,
                        layer: globalStripIdx + 1
                    }
                });
            }

            if (!strip.positions || strip.positions.length === 0) continue;

            // DEBUG: Diagnostic logging
            if (diagnostic && (globalStripIdx === 0 || globalStripIdx === 360)) {
                debug.log(`BUILD_ID_PLACEHOLDER | Strip ${globalStripIdx} (${strip.angle.toFixed(1)}°) INPUT terrain first 5 Z values: ${strip.positions.slice(0, 5).map(v => v.toFixed(3)).join(',')}`);
            }

            const stripToolpathResult = await runToolpathComputeWithBuffers(
                strip.positions,
                strip.gridWidth,
                strip.gridHeight,
                xStep,
                strip.gridHeight,
                zFloor,
                reusableBuffers,
                pipelineStartTime
            );

            // DEBUG: Verify toolpath generation output
            if (diagnostic && (globalStripIdx === 0 || globalStripIdx === 360)) {
                debug.log(`BUILD_ID_PLACEHOLDER | Strip ${globalStripIdx} (${strip.angle.toFixed(1)}°) OUTPUT toolpath first 5 Z values: ${stripToolpathResult.pathData.slice(0, 5).map(v => v.toFixed(3)).join(',')}`);
            }

            allStripToolpaths.push({
                angle: strip.angle,
                pathData: stripToolpathResult.pathData,
                numScanlines: stripToolpathResult.numScanlines,
                pointsPerLine: stripToolpathResult.pointsPerLine,
                terrainBounds: strip.bounds
            });

            totalToolpathPoints += stripToolpathResult.pathData.length;
        }
        const toolpathTime = performance.now() - toolpathStartTime;

        // Free batch terrain data
        for (const strip of batchModelResult.strips) {
            strip.positions = null;
        }
        destroyReusableToolpathBuffers(reusableBuffers);

        const batchTotalTime = performance.now() - batchStartTime;

        Object.assign(batchInfo, {
            'prep': batchInfo.prep || 0,
            'gpu': batchInfo.gpu || 0,
            'stitch': batchInfo.stitch || 0,
            'raster': batchInfo.raster || 0,
            'paths': (toolpathTime | 0),
            'strips': allStripToolpaths.length,
            'total': (batchTotalTime | 0)
        });
    }

    console.table(batchTracking);

    // Cleanup cached rasterization buffers after all batches complete
    if (batchReuseBuffers) {
        batchReuseBuffers.triangleBuffer.destroy();
        batchReuseBuffers.triangleIndicesBuffer.destroy();
        // Note: bucketInfoBuffer is no longer in reusableBuffers (created/destroyed per bucket batch)
        debug.log(`Destroyed cached GPU buffers after all batches`);
    }

    const pipelineTotalTime = performance.now() - pipelineStartTime;
    debug.log(`Complete radial toolpath: ${allStripToolpaths.length} strips, ${totalToolpathPoints} total points in ${pipelineTotalTime.toFixed(0)}ms`);

    // Send final 100% progress after cleanup completes
    self.postMessage({
        type: 'toolpath-progress',
        data: {
            percent: 100,
            current: numAngles,
            total: numAngles,
            layer: numAngles
        }
    });

    return {
        strips: allStripToolpaths,
        totalPoints: totalToolpathPoints,
        numStrips: allStripToolpaths.length
    };
}
