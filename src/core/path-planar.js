/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Path Planar - Planar Toolpath Generation
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Generates CNC toolpaths by scanning a sparse tool representation over a
 * dense terrain height map. Computes Z-heights where tool contacts terrain.
 *
 * EXPORTS:
 * ────────
 * Functions:
 *   - generateToolpath(terrainPoints, toolPoints, xStep, yStep, zFloor, ...)
 *       Main API - generates toolpath with automatic tiling
 *   - generateToolpathWithSparseTools(terrainPoints, sparseToolData, ...)
 *       Batch-optimized version with pre-created sparse tool
 *   - createReusableToolpathBuffers(width, height, sparseToolData, ...)
 *       Create GPU buffers for reuse across multiple tiles
 *   - destroyReusableToolpathBuffers(buffers)
 *       Cleanup GPU buffers
 *   - runToolpathComputeWithBuffers(terrainData, ...)
 *       Run toolpath compute shader with pre-allocated buffers
 *
 * ALGORITHM:
 * ──────────
 * For each output point (i, j) sampled at (xStep, yStep):
 *   1. Position tool center at terrain grid cell (i, j)
 *   2. For each point in sparse tool:
 *      - Calculate terrain sample position: terrain[i + xOffset, j + yOffset]
 *      - Calculate tool collision Z: terrainZ - toolZ
 *   3. Output maximum collision Z (highest point tool must be raised)
 *
 * TILING SUPPORT:
 * ───────────────
 * For large terrains exceeding GPU memory:
 * 1. Calculate tool dimensions to determine required overlap
 * 2. Subdivide terrain into tiles with tool-radius overlap
 * 3. Pre-generate all tile terrain arrays (CPU side)
 * 4. Create reusable GPU buffers sized for largest tile
 * 5. Process each tile, reusing buffers
 * 6. Stitch results, dropping overlap regions
 *
 * MEMORY OPTIMIZATION:
 * ────────────────────
 * - Tool buffer created once, reused for all tiles
 * - Terrain buffer updated per tile (GPU DMA)
 * - Output buffer reused, read back per tile
 * - Reduces GPU memory pressure by 10-100x for large models
 *
 * OUTPUT FORMAT:
 * ──────────────
 * Float32Array of Z-heights, row-major order:
 *   [z(0,0), z(1,0), z(2,0), ..., z(0,1), z(1,1), ...]
 * Dimensions: (terrain.width / xStep) × (terrain.height / yStep)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

import {
    device, deviceCapabilities, isInitialized, config,
    cachedToolpathPipeline, EMPTY_CELL, debug, initWebGPU
} from './raster-config.js';
import { createHeightMapFromPoints } from './raster-planar.js';
import { createSparseToolFromPoints } from './raster-tool.js';

// Generate toolpath with pre-created sparse tool (for batch operations)
export async function generateToolpathWithSparseTools(terrainPoints, sparseToolData, xStep, yStep, oobZ, gridStep, terrainBounds = null, singleScanline = false) {
    const startTime = performance.now();

    try {
        // Create height map from terrain points (use terrain bounds if provided)
        const terrainMapData = createHeightMapFromPoints(terrainPoints, gridStep, terrainBounds);

        // Run WebGPU compute with pre-created sparse tool
        const result = await runToolpathCompute(
            terrainMapData, sparseToolData, xStep, yStep, oobZ, startTime
        );

        return result;
    } catch (error) {
        debug.error('Error generating toolpath:', error);
        throw error;
    }
}

// Generate toolpath for a single region (internal)
async function generateToolpathSingle(terrainPoints, toolPoints, xStep, yStep, oobZ, gridStep, terrainBounds = null) {
    const startTime = performance.now();
    debug.log('Generating toolpath...');
    debug.log(`Input: terrain ${terrainPoints.length/3} points, tool ${toolPoints.length/3} points, steps (${xStep}, ${yStep}), oobZ ${oobZ}, gridStep ${gridStep}`);

    if (terrainBounds) {
        debug.log(`Using terrain bounds: min(${terrainBounds.min.x.toFixed(2)}, ${terrainBounds.min.y.toFixed(2)}, ${terrainBounds.min.z.toFixed(2)}) max(${terrainBounds.max.x.toFixed(2)}, ${terrainBounds.max.y.toFixed(2)}, ${terrainBounds.max.z.toFixed(2)})`);
    }

    try {
        // Create height map from terrain points (use terrain bounds if provided)
        const terrainMapData = createHeightMapFromPoints(terrainPoints, gridStep, terrainBounds);
        debug.log(`Created terrain map: ${terrainMapData.width}x${terrainMapData.height}`);

        // Create sparse tool representation
        const sparseToolData = createSparseToolFromPoints(toolPoints);
        debug.log(`Created sparse tool: ${sparseToolData.count} points`);

        // Run WebGPU compute
        const result = await runToolpathCompute(
            terrainMapData, sparseToolData, xStep, yStep, oobZ, startTime
        );

        return result;
    } catch (error) {
        debug.error('Error generating toolpath:', error);
        throw error;
    }
}

async function runToolpathCompute(terrainMapData, sparseToolData, xStep, yStep, oobZ, startTime) {
    if (!isInitialized) {
        const success = await initWebGPU();
        if (!success) {
            throw new Error('WebGPU not available');
        }
    }

    // Use WASM-generated terrain grid
    const terrainBuffer = device.createBuffer({
        size: terrainMapData.grid.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(terrainBuffer, 0, terrainMapData.grid);

    // Use WASM-generated sparse tool
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

    // Calculate output dimensions
    const pointsPerLine = Math.ceil(terrainMapData.width / xStep);
    const numScanlines = Math.ceil(terrainMapData.height / yStep);
    const outputSize = pointsPerLine * numScanlines;

    const outputBuffer = device.createBuffer({
        size: outputSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const uniformData = new Uint32Array([
        terrainMapData.width,
        terrainMapData.height,
        sparseToolData.count,
        xStep,
        yStep,
        0,
        pointsPerLine,
        numScanlines,
        0,  // y_offset (default 0 for planar mode)
    ]);
    const uniformDataFloat = new Float32Array(uniformData.buffer);
    uniformDataFloat[5] = oobZ;

    const uniformBuffer = device.createBuffer({
        size: uniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // CRITICAL: Wait for all writeBuffer operations to complete before compute dispatch
    await device.queue.onSubmittedWorkDone();

    // Use cached pipeline
    const bindGroup = device.createBindGroup({
        layout: cachedToolpathPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: terrainBuffer } },
            { binding: 1, resource: { buffer: toolBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
            { binding: 3, resource: { buffer: uniformBuffer } },
        ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(cachedToolpathPipeline);
    passEncoder.setBindGroup(0, bindGroup);

    const workgroupsX = Math.ceil(pointsPerLine / 16);
    const workgroupsY = Math.ceil(numScanlines / 16);
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
    passEncoder.end();

    const stagingBuffer = device.createBuffer({
        size: outputSize * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputSize * 4);

    device.queue.submit([commandEncoder.finish()]);

    // CRITICAL: Wait for GPU to finish before reading results
    await device.queue.onSubmittedWorkDone();

    await stagingBuffer.mapAsync(GPUMapMode.READ);

    const outputData = new Float32Array(stagingBuffer.getMappedRange());
    const result = new Float32Array(outputData);
    stagingBuffer.unmap();

    terrainBuffer.destroy();
    toolBuffer.destroy();
    outputBuffer.destroy();
    uniformBuffer.destroy();
    stagingBuffer.destroy();

    const endTime = performance.now();

    return {
        pathData: result,
        numScanlines,
        pointsPerLine,
        generationTime: endTime - startTime
    };
}

// Create reusable GPU buffers for tiled toolpath generation
export function createReusableToolpathBuffers(terrainWidth, terrainHeight, sparseToolData, xStep, yStep) {
    const pointsPerLine = Math.ceil(terrainWidth / xStep);
    const numScanlines = Math.ceil(terrainHeight / yStep);
    const outputSize = pointsPerLine * numScanlines;

    // Create terrain buffer (will be updated for each tile)
    const terrainBuffer = device.createBuffer({
        size: terrainWidth * terrainHeight * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Create tool buffer (STATIC - same for all tiles!)
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
    device.queue.writeBuffer(toolBuffer, 0, toolBufferData);  // Write once!

    // Create output buffer (will be read for each tile)
    const outputBuffer = device.createBuffer({
        size: outputSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Create uniform buffer (will be updated for each tile)
    const uniformBuffer = device.createBuffer({
        size: 36,  // 9 fields × 4 bytes (added y_offset field)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create staging buffer (will be reused for readback)
    const stagingBuffer = device.createBuffer({
        size: outputSize * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    return {
        terrainBuffer,
        toolBuffer,
        outputBuffer,
        uniformBuffer,
        stagingBuffer,
        maxOutputSize: outputSize,
        maxTerrainWidth: terrainWidth,
        maxTerrainHeight: terrainHeight,
        sparseToolData
    };
}

// Destroy reusable GPU buffers
export function destroyReusableToolpathBuffers(buffers) {
    buffers.terrainBuffer.destroy();
    buffers.toolBuffer.destroy();
    buffers.outputBuffer.destroy();
    buffers.uniformBuffer.destroy();
    buffers.stagingBuffer.destroy();
}

// Run toolpath compute using pre-created reusable buffers
export async function runToolpathComputeWithBuffers(terrainData, terrainWidth, terrainHeight, xStep, yStep, oobZ, buffers, startTime) {
    // Update terrain buffer with new tile data
    device.queue.writeBuffer(buffers.terrainBuffer, 0, terrainData);

    // Calculate output dimensions
    const pointsPerLine = Math.ceil(terrainWidth / xStep);
    const numScanlines = Math.ceil(terrainHeight / yStep);
    const outputSize = pointsPerLine * numScanlines;

    // Calculate Y offset for single-scanline radial mode
    // When numScanlines=1 and terrainHeight > 1, center the tool at the midline
    const yOffset = (numScanlines === 1 && terrainHeight > 1) ? Math.floor(terrainHeight / 2) : 0;

    // Update uniforms for this tile
    const uniformData = new Uint32Array([
        terrainWidth,
        terrainHeight,
        buffers.sparseToolData.count,
        xStep,
        yStep,
        0,
        pointsPerLine,
        numScanlines,
        yOffset,  // y_offset for radial single-scanline mode
    ]);
    const uniformDataFloat = new Float32Array(uniformData.buffer);
    uniformDataFloat[5] = oobZ;
    device.queue.writeBuffer(buffers.uniformBuffer, 0, uniformData);

    // CRITICAL: Wait for all writeBuffer operations to complete before compute dispatch
    // Without this, compute shader may read stale/incomplete buffer data
    await device.queue.onSubmittedWorkDone();

    // Create bind group (reusing cached pipeline)
    const bindGroup = device.createBindGroup({
        layout: cachedToolpathPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: buffers.terrainBuffer } },
            { binding: 1, resource: { buffer: buffers.toolBuffer } },
            { binding: 2, resource: { buffer: buffers.outputBuffer } },
            { binding: 3, resource: { buffer: buffers.uniformBuffer } },
        ],
    });

    // Dispatch compute shader
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(cachedToolpathPipeline);
    passEncoder.setBindGroup(0, bindGroup);

    const workgroupsX = Math.ceil(pointsPerLine / 16);
    const workgroupsY = Math.ceil(numScanlines / 16);
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
    passEncoder.end();

    // Copy to staging buffer
    commandEncoder.copyBufferToBuffer(buffers.outputBuffer, 0, buffers.stagingBuffer, 0, outputSize * 4);

    device.queue.submit([commandEncoder.finish()]);

    // CRITICAL: Wait for GPU to finish before reading results
    await device.queue.onSubmittedWorkDone();

    await buffers.stagingBuffer.mapAsync(GPUMapMode.READ);

    // Create a true copy using slice() - new Float32Array(typedArray) only creates a view!
    const outputData = new Float32Array(buffers.stagingBuffer.getMappedRange(), 0, outputSize);
    const result = outputData.slice();  // slice() creates a new ArrayBuffer with copied data
    buffers.stagingBuffer.unmap();

    const endTime = performance.now();

    // Debug: Log first few Z values to detect non-determinism
    if (result.length > 0) {
        const samples = [];
        for (let i = 0; i < Math.min(10, result.length); i++) {
            samples.push(result[i].toFixed(3));
        }
        // debug.log(`[Toolpath] Output samples (${result.length} total): ${samples.join(', ')}`);
    }

    return {
        pathData: result,
        numScanlines,
        pointsPerLine,
        generationTime: endTime - startTime
    };
}

// Generate toolpath with tiling support (public API)
export async function generateToolpath(terrainPoints, toolPoints, xStep, yStep, oobZ, gridStep, terrainBounds = null, singleScanline = false) {
    // Calculate bounds if not provided
    if (!terrainBounds) {
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        for (let i = 0; i < terrainPoints.length; i += 3) {
            minX = Math.min(minX, terrainPoints[i]);
            maxX = Math.max(maxX, terrainPoints[i]);
            minY = Math.min(minY, terrainPoints[i + 1]);
            maxY = Math.max(maxY, terrainPoints[i + 1]);
            minZ = Math.min(minZ, terrainPoints[i + 2]);
            maxZ = Math.max(maxZ, terrainPoints[i + 2]);
        }
        terrainBounds = {
            min: { x: minX, y: minY, z: minZ },
            max: { x: maxX, y: maxY, z: maxZ }
        };
    }

    // Note: singleScanline mode means OUTPUT only centerline, but terrain bounds stay full
    // This ensures all terrain Y values contribute to tool interference at the centerline

    // Debug tool bounds and center
    for (let i=0; i<toolPoints.length; i += 3) {
        if (toolPoints[i] === 0 && toolPoints[i+1] === 0) {
            debug.log('[WebGPU Worker]', { TOOL_CENTER: toolPoints[i+2] });
        }
    }
    debug.log('[WebGPU Worker]',
        'toolZMin:', ([...toolPoints].filter((_,i) => i % 3 === 2).reduce((a,b) => Math.min(a,b), Infinity)),
        'toolZMax:', ([...toolPoints].filter((_,i) => i % 3 === 2).reduce((a,b) => Math.max(a,b), -Infinity))
    );

    // Calculate tool dimensions for overlap
    // Tool points are [gridX, gridY, Z] where X/Y are grid indices (not mm)
    let toolMinX = Infinity, toolMaxX = -Infinity;
    let toolMinY = Infinity, toolMaxY = -Infinity;
    for (let i = 0; i < toolPoints.length; i += 3) {
        toolMinX = Math.min(toolMinX, toolPoints[i]);
        toolMaxX = Math.max(toolMaxX, toolPoints[i]);
        toolMinY = Math.min(toolMinY, toolPoints[i + 1]);
        toolMaxY = Math.max(toolMaxY, toolPoints[i + 1]);
    }
    // Tool dimensions in grid cells
    const toolWidthCells = toolMaxX - toolMinX;
    const toolHeightCells = toolMaxY - toolMinY;
    // Convert to mm for logging
    const toolWidthMm = toolWidthCells * gridStep;
    const toolHeightMm = toolHeightCells * gridStep;

    // Check if tiling is needed based on output grid size
    const outputWidth = Math.ceil((terrainBounds.max.x - terrainBounds.min.x) / gridStep) + 1;
    const outputHeight = Math.ceil((terrainBounds.max.y - terrainBounds.min.y) / gridStep) + 1;
    const outputPoints = Math.ceil(outputWidth / xStep) * Math.ceil(outputHeight / yStep);
    const outputMemory = outputPoints * 4; // 4 bytes per float

    const configuredLimit = config.maxGPUMemoryMB * 1024 * 1024;
    const deviceLimit = deviceCapabilities.maxStorageBufferBindingSize;
    const maxSafeSize = Math.min(configuredLimit, deviceLimit) * config.gpuMemorySafetyMargin;

    if (outputMemory <= maxSafeSize) {
        // No tiling needed
        return await generateToolpathSingle(terrainPoints, toolPoints, xStep, yStep, oobZ, gridStep, terrainBounds);
    }

    // Tiling needed (terrain is ALWAYS dense)
    const tilingStartTime = performance.now();
    debug.log('Using tiled toolpath generation');
    debug.log(`Terrain: DENSE (${terrainPoints.length} cells = ${outputWidth}x${outputHeight})`);
    debug.log(`Tool dimensions: ${toolWidthMm.toFixed(2)}mm × ${toolHeightMm.toFixed(2)}mm (${toolWidthCells}×${toolHeightCells} cells)`);

    // Create tiles with tool-size overlap (pass dimensions in grid cells)
    const { tiles, maxTileGridWidth, maxTileGridHeight } = createToolpathTiles(terrainBounds, gridStep, xStep, yStep, toolWidthCells, toolHeightCells, maxSafeSize);
    debug.log(`Created ${tiles.length} tiles`);

    // Pre-generate all tile terrain point arrays
    const pregenStartTime = performance.now();
    debug.log(`Pre-generating ${tiles.length} tile terrain arrays...`);
    const allTileTerrainPoints = [];

    for (let i = 0; i < tiles.length; i++) {
        const tile = tiles[i];

        // Extract terrain sub-grid for this tile (terrain is ALWAYS dense)
        const tileMinGridX = Math.floor((tile.bounds.min.x - terrainBounds.min.x) / gridStep);
        const tileMaxGridX = Math.ceil((tile.bounds.max.x - terrainBounds.min.x) / gridStep);
        const tileMinGridY = Math.floor((tile.bounds.min.y - terrainBounds.min.y) / gridStep);
        const tileMaxGridY = Math.ceil((tile.bounds.max.y - terrainBounds.min.y) / gridStep);

        const tileWidth = tileMaxGridX - tileMinGridX + 1;
        const tileHeight = tileMaxGridY - tileMinGridY + 1;

        // Pad to max dimensions for consistent buffer sizing
        const paddedTileTerrainPoints = new Float32Array(maxTileGridWidth * maxTileGridHeight);
        paddedTileTerrainPoints.fill(EMPTY_CELL);

        // Copy relevant sub-grid from full terrain into top-left of padded array
        for (let ty = 0; ty < tileHeight; ty++) {
            const globalY = tileMinGridY + ty;
            if (globalY < 0 || globalY >= outputHeight) continue;

            for (let tx = 0; tx < tileWidth; tx++) {
                const globalX = tileMinGridX + tx;
                if (globalX < 0 || globalX >= outputWidth) continue;

                const globalIdx = globalY * outputWidth + globalX;
                const tileIdx = ty * maxTileGridWidth + tx;  // Use maxTileGridWidth for stride
                paddedTileTerrainPoints[tileIdx] = terrainPoints[globalIdx];
            }
        }

        allTileTerrainPoints.push({
            data: paddedTileTerrainPoints,
            actualWidth: tileWidth,
            actualHeight: tileHeight
        });
    }

    const pregenTime = performance.now() - pregenStartTime;
    debug.log(`Pre-generation complete in ${pregenTime.toFixed(1)}ms`);

    // Create reusable GPU buffers (sized for maximum tile)
    if (!isInitialized) {
        const success = await initWebGPU();
        if (!success) {
            throw new Error('WebGPU not available');
        }
    }

    const sparseToolData = createSparseToolFromPoints(toolPoints);
    const reusableBuffers = createReusableToolpathBuffers(maxTileGridWidth, maxTileGridHeight, sparseToolData, xStep, yStep);
    debug.log(`Created reusable GPU buffers for ${maxTileGridWidth}x${maxTileGridHeight} tiles`);

    // Process each tile with reusable buffers
    const tileResults = [];
    let totalTileTime = 0;
    for (let i = 0; i < tiles.length; i++) {
        const tile = tiles[i];
        const tileStartTime = performance.now();
        debug.log(`Processing tile ${i + 1}/${tiles.length}...`);

        // Report progress
        const percent = Math.round(((i + 1) / tiles.length) * 100);
        self.postMessage({
            type: 'toolpath-progress',
            data: {
                percent,
                current: i + 1,
                total: tiles.length,
                layer: i + 1  // Using tile index as "layer" for consistency
            }
        });

        debug.log(`Tile ${i+1} using pre-generated terrain: ${allTileTerrainPoints[i].actualWidth}x${allTileTerrainPoints[i].actualHeight} (padded to ${maxTileGridWidth}x${maxTileGridHeight})`);

        // Generate toolpath for this tile using reusable buffers
        const tileToolpathResult = await runToolpathComputeWithBuffers(
            allTileTerrainPoints[i].data,
            maxTileGridWidth,
            maxTileGridHeight,
            xStep,
            yStep,
            oobZ,
            reusableBuffers,
            tileStartTime
        );

        const tileTime = performance.now() - tileStartTime;
        totalTileTime += tileTime;

        tileResults.push({
            pathData: tileToolpathResult.pathData,
            numScanlines: tileToolpathResult.numScanlines,
            pointsPerLine: tileToolpathResult.pointsPerLine,
            tile: tile
        });

        debug.log(`Tile ${i + 1}/${tiles.length} complete: ${tileToolpathResult.numScanlines}×${tileToolpathResult.pointsPerLine} in ${tileTime.toFixed(1)}ms`);
    }

    // Cleanup reusable buffers
    destroyReusableToolpathBuffers(reusableBuffers);

    debug.log(`All tiles processed in ${totalTileTime.toFixed(1)}ms (avg ${(totalTileTime/tiles.length).toFixed(1)}ms per tile)`);

    // Stitch tiles together, dropping overlap regions
    const stitchStartTime = performance.now();
    const stitchedResult = stitchToolpathTiles(tileResults, terrainBounds, gridStep, xStep, yStep);
    const stitchTime = performance.now() - stitchStartTime;

    const totalTime = performance.now() - tilingStartTime;
    debug.log(`Stitching took ${stitchTime.toFixed(1)}ms`);
    debug.log(`Tiled toolpath complete: ${stitchedResult.numScanlines}×${stitchedResult.pointsPerLine} in ${totalTime.toFixed(1)}ms total`);

    // Update generation time to reflect total tiled time
    stitchedResult.generationTime = totalTime;

    return stitchedResult;
}

// Create tiles for toolpath generation with overlap (using integer grid coordinates)
// toolWidth and toolHeight are in grid cells (not mm)
function createToolpathTiles(bounds, gridStep, xStep, yStep, toolWidthCells, toolHeightCells, maxMemoryBytes) {
    // Calculate global grid dimensions
    const globalGridWidth = Math.ceil((bounds.max.x - bounds.min.x) / gridStep) + 1;
    const globalGridHeight = Math.ceil((bounds.max.y - bounds.min.y) / gridStep) + 1;

    // Calculate tool overlap in grid cells (use radius = half diameter)
    // Tool centered at tile boundary needs terrain extending half tool width beyond boundary
    const toolOverlapX = Math.ceil(toolWidthCells / 2);
    const toolOverlapY = Math.ceil(toolHeightCells / 2);

    // Binary search for optimal tile size in grid cells
    let low = Math.max(toolOverlapX, toolOverlapY) * 2; // At least 2x tool size
    let high = Math.max(globalGridWidth, globalGridHeight);
    let bestTileGridSize = high;

    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        const outputW = Math.ceil(mid / xStep);
        const outputH = Math.ceil(mid / yStep);
        const memoryNeeded = outputW * outputH * 4;

        if (memoryNeeded <= maxMemoryBytes) {
            bestTileGridSize = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    const tilesX = Math.ceil(globalGridWidth / bestTileGridSize);
    const tilesY = Math.ceil(globalGridHeight / bestTileGridSize);
    const coreGridWidth = Math.ceil(globalGridWidth / tilesX);
    const coreGridHeight = Math.ceil(globalGridHeight / tilesY);

    // Calculate maximum tile dimensions (for buffer sizing)
    const maxTileGridWidth = coreGridWidth + 2 * toolOverlapX;
    const maxTileGridHeight = coreGridHeight + 2 * toolOverlapY;

    debug.log(`Creating ${tilesX}×${tilesY} tiles (${coreGridWidth}×${coreGridHeight} cells core + ${toolOverlapX}×${toolOverlapY} cells overlap)`);
    debug.log(`Max tile dimensions: ${maxTileGridWidth}×${maxTileGridHeight} cells (for buffer sizing)`);

    const tiles = [];
    for (let ty = 0; ty < tilesY; ty++) {
        for (let tx = 0; tx < tilesX; tx++) {
            // Core tile in grid coordinates
            const coreGridStartX = tx * coreGridWidth;
            const coreGridStartY = ty * coreGridHeight;
            const coreGridEndX = Math.min((tx + 1) * coreGridWidth, globalGridWidth) - 1;
            const coreGridEndY = Math.min((ty + 1) * coreGridHeight, globalGridHeight) - 1;

            // Extended tile with overlap in grid coordinates
            let extGridStartX = coreGridStartX;
            let extGridStartY = coreGridStartY;
            let extGridEndX = coreGridEndX;
            let extGridEndY = coreGridEndY;

            // Add overlap on sides that aren't at global boundary
            if (tx > 0) extGridStartX -= toolOverlapX;
            if (ty > 0) extGridStartY -= toolOverlapY;
            if (tx < tilesX - 1) extGridEndX += toolOverlapX;
            if (ty < tilesY - 1) extGridEndY += toolOverlapY;

            // Clamp to global bounds
            extGridStartX = Math.max(0, extGridStartX);
            extGridStartY = Math.max(0, extGridStartY);
            extGridEndX = Math.min(globalGridWidth - 1, extGridEndX);
            extGridEndY = Math.min(globalGridHeight - 1, extGridEndY);

            // Calculate actual dimensions for this tile
            const tileGridWidth = extGridEndX - extGridStartX + 1;
            const tileGridHeight = extGridEndY - extGridStartY + 1;

            // Convert grid coordinates to world coordinates
            const extMinX = bounds.min.x + extGridStartX * gridStep;
            const extMinY = bounds.min.y + extGridStartY * gridStep;
            const extMaxX = bounds.min.x + extGridEndX * gridStep;
            const extMaxY = bounds.min.y + extGridEndY * gridStep;

            const coreMinX = bounds.min.x + coreGridStartX * gridStep;
            const coreMinY = bounds.min.y + coreGridStartY * gridStep;
            const coreMaxX = bounds.min.x + coreGridEndX * gridStep;
            const coreMaxY = bounds.min.y + coreGridEndY * gridStep;

            tiles.push({
                id: `tile_${tx}_${ty}`,
                tx, ty,
                tilesX, tilesY,
                gridWidth: tileGridWidth,
                gridHeight: tileGridHeight,
                bounds: {
                    min: { x: extMinX, y: extMinY, z: bounds.min.z },
                    max: { x: extMaxX, y: extMaxY, z: bounds.max.z }
                },
                core: {
                    gridStart: { x: coreGridStartX, y: coreGridStartY },
                    gridEnd: { x: coreGridEndX, y: coreGridEndY },
                    min: { x: coreMinX, y: coreMinY },
                    max: { x: coreMaxX, y: coreMaxY }
                }
            });
        }
    }

    return { tiles, maxTileGridWidth, maxTileGridHeight };
}

// Stitch toolpath tiles together, dropping overlap regions (using integer grid coordinates)
function stitchToolpathTiles(tileResults, globalBounds, gridStep, xStep, yStep) {
    // Calculate global output dimensions
    const globalWidth = Math.ceil((globalBounds.max.x - globalBounds.min.x) / gridStep) + 1;
    const globalHeight = Math.ceil((globalBounds.max.y - globalBounds.min.y) / gridStep) + 1;
    const globalPointsPerLine = Math.ceil(globalWidth / xStep);
    const globalNumScanlines = Math.ceil(globalHeight / yStep);

    debug.log(`Stitching toolpath: global grid ${globalWidth}x${globalHeight}, output ${globalPointsPerLine}x${globalNumScanlines}`);

    const result = new Float32Array(globalPointsPerLine * globalNumScanlines);
    result.fill(NaN);

    // Fast path for 1x1 stepping: use bulk row copying
    const use1x1FastPath = (xStep === 1 && yStep === 1);

    for (const tileResult of tileResults) {
        const tile = tileResult.tile;
        const tileData = tileResult.pathData;

        // Use the pre-calculated integer grid coordinates from tile.core
        const coreGridStartX = tile.core.gridStart.x;
        const coreGridStartY = tile.core.gridStart.y;
        const coreGridEndX = tile.core.gridEnd.x;
        const coreGridEndY = tile.core.gridEnd.y;

        // Calculate tile's extended grid coordinates
        const extGridStartX = Math.round((tile.bounds.min.x - globalBounds.min.x) / gridStep);
        const extGridStartY = Math.round((tile.bounds.min.y - globalBounds.min.y) / gridStep);

        let copiedCount = 0;

        // Calculate output coordinate ranges for this tile's core
        // Core region in grid coordinates
        const coreGridWidth = coreGridEndX - coreGridStartX + 1;
        const coreGridHeight = coreGridEndY - coreGridStartY + 1;

        // Core region in output coordinates (sampled by xStep/yStep)
        const coreOutStartX = Math.floor(coreGridStartX / xStep);
        const coreOutStartY = Math.floor(coreGridStartY / yStep);
        const coreOutEndX = Math.floor(coreGridEndX / xStep);
        const coreOutEndY = Math.floor(coreGridEndY / yStep);
        const coreOutWidth = coreOutEndX - coreOutStartX + 1;
        const coreOutHeight = coreOutEndY - coreOutStartY + 1;

        // Tile's extended region start in grid coordinates
        const extOutStartX = Math.floor(extGridStartX / xStep);
        const extOutStartY = Math.floor(extGridStartY / yStep);

        // Copy entire rows at once (works for all stepping values)
        for (let outY = 0; outY < coreOutHeight; outY++) {
            const globalOutY = coreOutStartY + outY;
            const tileOutY = globalOutY - extOutStartY;

            if (globalOutY >= 0 && globalOutY < globalNumScanlines &&
                tileOutY >= 0 && tileOutY < tileResult.numScanlines) {

                const globalRowStart = globalOutY * globalPointsPerLine + coreOutStartX;
                const tileRowStart = tileOutY * tileResult.pointsPerLine + (coreOutStartX - extOutStartX);

                // Bulk copy entire row of output values
                result.set(tileData.subarray(tileRowStart, tileRowStart + coreOutWidth), globalRowStart);
                copiedCount += coreOutWidth;
            }
        }

        debug.log(`  Tile ${tile.id}: copied ${copiedCount} values`);
    }

    // Count how many output values are still NaN (gaps)
    let nanCount = 0;
    for (let i = 0; i < result.length; i++) {
        if (isNaN(result[i])) nanCount++;
    }
    debug.log(`Stitching complete: ${result.length} total values, ${nanCount} still NaN`);

    return {
        pathData: result,
        numScanlines: globalNumScanlines,
        pointsPerLine: globalPointsPerLine,
        generationTime: 0 // Sum from tiles if needed
    };
}
