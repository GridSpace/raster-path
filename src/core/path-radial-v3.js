/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Path Radial V3 - Bucket-Angle Pipeline with Y-Filtering
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * ALGORITHM CHANGES FROM V2:
 * ──────────────────────────
 * V2 (current): Process all angles simultaneously
 *   - Rotate rays on the fly
 *   - Test all triangles in bucket (no Y-filtering)
 *   - Large memory footprint: numAngles × gridWidth × gridHeight
 *
 * V3 (this file): Process bucket-by-angle pipeline
 *   For each bucket:
 *     For each angle:
 *       1. Rotate triangles (parallel) → rotated tris + Y-bounds
 *       2. Rasterize with Y-filter (parallel) → dense terrain strip
 *       3. Toolpath generation (parallel) → sparse toolpath
 *
 * BENEFITS:
 * ─────────
 * - Lower memory: Only one angle's data in GPU at a time
 * - Y-axis filtering: Skip triangles outside tool radius
 * - Immediate toolpath generation: No need to store all strips
 * - Better cache locality: Process bucket completely before moving on
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

import {
    device, config, debug, diagnostic,
    cachedRadialV3RotatePipeline,
    cachedRadialV3RasterizePipeline,
    cachedRadialV3BatchedRasterizePipeline
} from './raster-config.js';
import {
    createReusableToolpathBuffers,
    destroyReusableToolpathBuffers,
    runToolpathComputeWithBuffers
} from './path-planar.js';
import { createSparseToolFromPoints } from './raster-tool.js';

/**
 * Rotate all triangles in a bucket by a single angle
 */
async function rotateTriangles({
    triangleBuffer,      // GPU buffer with original triangles
    numTriangles,
    angle                // Radians
}) {
    const rotatePipeline = cachedRadialV3RotatePipeline;
    if (!rotatePipeline) {
        throw new Error('Radial V3 pipelines not initialized');
    }

    // Create output buffer for rotated triangles + bounds
    // Layout: 11 floats per triangle (v0, v1, v2, y_min, y_max)
    const outputSize = numTriangles * 11 * 4;
    const rotatedBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Create uniforms
    const uniformBuffer = device.createBuffer({
        size: 8,  // f32 angle + u32 num_triangles
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });

    const uniformView = new ArrayBuffer(8);
    const floatView = new Float32Array(uniformView);
    const uintView = new Uint32Array(uniformView);
    floatView[0] = angle;
    uintView[1] = numTriangles;

    new Uint8Array(uniformBuffer.getMappedRange()).set(new Uint8Array(uniformView));
    uniformBuffer.unmap();

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: rotatePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: triangleBuffer } },
            { binding: 1, resource: { buffer: rotatedBuffer } },
            { binding: 2, resource: { buffer: uniformBuffer } }
        ]
    });

    // Dispatch
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(rotatePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(numTriangles / 64));
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    // Cleanup
    uniformBuffer.destroy();

    return rotatedBuffer;
}

/**
 * Rasterize ALL buckets in one dispatch (batched GPU processing)
 */
async function rasterizeAllBuckets({
    rotatedTrianglesBuffer,
    buckets,
    triangleIndices,
    resolution,
    toolRadius,
    fullGridWidth,
    gridHeight,
    globalMinX,
    bucketMinY,
    zFloor
}) {
    const rasterizePipeline = cachedRadialV3BatchedRasterizePipeline;
    if (!rasterizePipeline) {
        throw new Error('Radial V3 batched pipeline not initialized');
    }

    // Create bucket info buffer (all buckets)
    const bucketInfoSize = buckets.length * 16;  // 4 fields × 4 bytes per bucket
    const bucketInfoBuffer = device.createBuffer({
        size: bucketInfoSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });

    const bucketView = new ArrayBuffer(bucketInfoSize);
    const bucketFloatView = new Float32Array(bucketView);
    const bucketUintView = new Uint32Array(bucketView);

    for (let i = 0; i < buckets.length; i++) {
        const bucket = buckets[i];
        const offset = i * 4;
        bucketFloatView[offset] = bucket.minX;
        bucketFloatView[offset + 1] = bucket.maxX;
        bucketUintView[offset + 2] = bucket.startIndex;
        bucketUintView[offset + 3] = bucket.count;
    }

    new Uint8Array(bucketInfoBuffer.getMappedRange()).set(new Uint8Array(bucketView));
    bucketInfoBuffer.unmap();

    // Create triangle indices buffer
    const indicesBuffer = device.createBuffer({
        size: triangleIndices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Uint32Array(indicesBuffer.getMappedRange()).set(triangleIndices);
    indicesBuffer.unmap();

    // Create output buffer for full terrain strip
    const outputSize = fullGridWidth * gridHeight * 4;
    const outputBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    // Initialize with zFloor
    const initData = new Float32Array(fullGridWidth * gridHeight);
    initData.fill(zFloor);
    device.queue.writeBuffer(outputBuffer, 0, initData);

    // Create uniforms
    const uniformBuffer = device.createBuffer({
        size: 32,  // 8 fields × 4 bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });

    const uniformView = new ArrayBuffer(32);
    const floatView = new Float32Array(uniformView);
    const uintView = new Uint32Array(uniformView);

    floatView[0] = resolution;
    floatView[1] = toolRadius;
    uintView[2] = fullGridWidth;
    uintView[3] = gridHeight;
    floatView[4] = globalMinX;
    floatView[5] = bucketMinY;
    floatView[6] = zFloor;
    uintView[7] = buckets.length;

    new Uint8Array(uniformBuffer.getMappedRange()).set(new Uint8Array(uniformView));
    uniformBuffer.unmap();

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: rasterizePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: rotatedTrianglesBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
            { binding: 2, resource: { buffer: uniformBuffer } },
            { binding: 3, resource: { buffer: bucketInfoBuffer } },
            { binding: 4, resource: { buffer: indicesBuffer } }
        ]
    });

    // Dispatch - covers full grid width (all buckets)
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(rasterizePipeline);
    passEncoder.setBindGroup(0, bindGroup);

    const dispatchX = Math.ceil(fullGridWidth / 8);
    const dispatchY = Math.ceil(gridHeight / 8);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY);
    passEncoder.end();

    // Read back results
    const stagingBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputSize);
    device.queue.submit([commandEncoder.finish()]);

    await device.queue.onSubmittedWorkDone();
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const terrainData = new Float32Array(stagingBuffer.getMappedRange().slice());
    stagingBuffer.unmap();

    // Cleanup
    outputBuffer.destroy();
    stagingBuffer.destroy();
    uniformBuffer.destroy();
    bucketInfoBuffer.destroy();
    indicesBuffer.destroy();

    return terrainData;
}

/**
 * Rasterize a bucket using specific triangle indices from the full rotated buffer
 */
async function rasterizeStripWithIndices({
    rotatedTrianglesBuffer,
    triangleIndices,
    resolution,
    toolRadius,
    gridWidth,
    gridHeight,
    bucketMinX,
    bucketMinY,
    zFloor
}) {
    const rasterizePipeline = cachedRadialV3RasterizePipeline;
    if (!rasterizePipeline) {
        throw new Error('Radial V3 pipelines not initialized');
    }

    // Create triangle indices buffer
    const indicesBuffer = device.createBuffer({
        size: triangleIndices.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Uint32Array(indicesBuffer.getMappedRange()).set(triangleIndices);
    indicesBuffer.unmap();

    // Create output buffer for terrain strip
    const outputSize = gridWidth * gridHeight * 4;
    const outputBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    // Initialize with zFloor
    const initData = new Float32Array(gridWidth * gridHeight);
    initData.fill(zFloor);
    device.queue.writeBuffer(outputBuffer, 0, initData);

    // Create uniforms
    const uniformBuffer = device.createBuffer({
        size: 36,  // 9 fields × 4 bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });

    const uniformView = new ArrayBuffer(36);
    const floatView = new Float32Array(uniformView);
    const uintView = new Uint32Array(uniformView);

    floatView[0] = resolution;
    floatView[1] = toolRadius;
    uintView[2] = gridWidth;
    uintView[3] = gridHeight;
    floatView[4] = bucketMinX;
    floatView[5] = bucketMinY;
    floatView[6] = zFloor;
    uintView[7] = triangleIndices.length;  // number of triangles in this bucket
    uintView[8] = 0;  // padding for alignment

    new Uint8Array(uniformBuffer.getMappedRange()).set(new Uint8Array(uniformView));
    uniformBuffer.unmap();

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: rasterizePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: rotatedTrianglesBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
            { binding: 2, resource: { buffer: uniformBuffer } },
            { binding: 3, resource: { buffer: indicesBuffer } }
        ]
    });

    // Dispatch
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(rasterizePipeline);
    passEncoder.setBindGroup(0, bindGroup);

    const dispatchX = Math.ceil(gridWidth / 8);
    const dispatchY = Math.ceil(gridHeight / 8);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY);
    passEncoder.end();

    // Read back results
    const stagingBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputSize);
    device.queue.submit([commandEncoder.finish()]);

    await device.queue.onSubmittedWorkDone();
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const terrainData = new Float32Array(stagingBuffer.getMappedRange().slice());
    stagingBuffer.unmap();

    // Cleanup
    outputBuffer.destroy();
    stagingBuffer.destroy();
    uniformBuffer.destroy();
    indicesBuffer.destroy();

    return terrainData;
}

/**
 * Generate radial toolpaths using bucket-angle pipeline
 */
export async function generateRadialToolpathsV3({
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
    debug.log('radial-v3-generate-toolpaths', { triangles: triangles.length / 9, numAngles, resolution });

    const pipelineStartTime = performance.now();
    const allStripToolpaths = [];
    let totalToolpathPoints = 0;

    // Prepare sparse tool once
    const sparseToolData = createSparseToolFromPoints(toolData.positions);
    debug.log(`Created sparse tool: ${sparseToolData.count} points (reusing for all strips)`);

    const toolRadius = toolWidth / 2;

    // Calculate full grid dimensions (all buckets)
    const bucketMinX = bucketData.buckets[0].minX;
    const bucketMaxX = bucketData.buckets[bucketData.numBuckets - 1].maxX;
    const fullWidth = bucketMaxX - bucketMinX;
    const fullGridWidth = Math.ceil(fullWidth / resolution);
    const gridHeight = Math.ceil(toolWidth / resolution);

    // OPTIMIZATION: Upload all triangles to GPU ONCE (reused across all angles)
    debug.log(`Uploading ${triangles.length / 9} triangles to GPU (reused across all angles)...`);
    const allTrianglesBuffer = device.createBuffer({
        size: triangles.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(allTrianglesBuffer.getMappedRange()).set(triangles);
    allTrianglesBuffer.unmap();

    // Process angle-by-angle (outer loop)
    for (let angleIdx = 0; angleIdx < numAngles; angleIdx++) {
        const angle = -(angleIdx * angleStep * (Math.PI / 180));  // Convert to radians (negative: rotating terrain vs tool)
        const angleDegrees = angleIdx * angleStep;

        if (diagnostic) {
            debug.log(`Angle ${angleIdx + 1}/${numAngles}: ${angleDegrees.toFixed(1)}°`);
        }

        // Report progress
        if (angleIdx % 10 === 0 || angleIdx === numAngles - 1) {
            const stripProgress = ((angleIdx + 1) / numAngles) * 98;
            self.postMessage({
                type: 'toolpath-progress',
                data: {
                    percent: Math.round(stripProgress),
                    current: angleIdx + 1,
                    total: numAngles,
                    layer: angleIdx + 1
                }
            });
        }

        // OPTIMIZATION: Rotate ALL triangles once per angle (batch rotation)
        const numTotalTriangles = triangles.length / 9;
        const allRotatedTrianglesBuffer = await rotateTriangles({
            triangleBuffer: allTrianglesBuffer,
            numTriangles: numTotalTriangles,
            angle
        });

        // OPTIMIZATION: Rasterize ALL buckets in ONE dispatch (no CPU loop!)
        const fullTerrainStrip = await rasterizeAllBuckets({
            rotatedTrianglesBuffer: allRotatedTrianglesBuffer,
            buckets: bucketData.buckets,
            triangleIndices: bucketData.triangleIndices,
            resolution,
            toolRadius,
            fullGridWidth,
            gridHeight,
            globalMinX: bucketMinX,
            bucketMinY: -toolWidth / 2,
            zFloor
        });

        // Cleanup rotated buffer (created per angle)
        allRotatedTrianglesBuffer.destroy();

        // Step 3: Generate toolpath for this complete angle strip
        const reusableToolpathBuffers = createReusableToolpathBuffers(
            fullGridWidth,
            gridHeight,
            sparseToolData,
            xStep,
            gridHeight
        );

        const stripToolpathResult = await runToolpathComputeWithBuffers(
            fullTerrainStrip,
            fullGridWidth,
            gridHeight,
            xStep,
            gridHeight,
            zFloor,
            reusableToolpathBuffers,
            pipelineStartTime
        );

        destroyReusableToolpathBuffers(reusableToolpathBuffers);

        allStripToolpaths.push({
            angle: angleDegrees,
            pathData: stripToolpathResult.pathData,
            numScanlines: stripToolpathResult.numScanlines,
            pointsPerLine: stripToolpathResult.pointsPerLine,
            terrainBounds: {
                min: { x: bucketMinX, y: -toolWidth / 2, z: zFloor },
                max: { x: bucketMaxX, y: toolWidth / 2, z: bounds.max.z }
            }
        });

        totalToolpathPoints += stripToolpathResult.pathData.length;
    }

    // Cleanup triangles buffer (reused across all angles)
    allTrianglesBuffer.destroy();
    debug.log(`Destroyed reusable triangle buffer`);

    const pipelineTotalTime = performance.now() - pipelineStartTime;
    debug.log(`Complete radial V3 toolpath: ${allStripToolpaths.length} strips, ${totalToolpathPoints} total points in ${pipelineTotalTime.toFixed(0)}ms`);

    // Send final 100% progress
    self.postMessage({
        type: 'toolpath-progress',
        data: {
            percent: 100,
            current: bucketData.numBuckets * numAngles,
            total: bucketData.numBuckets * numAngles,
            layer: numAngles
        }
    });

    return {
        strips: allStripToolpaths,
        totalPoints: totalToolpathPoints,
        numStrips: allStripToolpaths.length
    };
}
