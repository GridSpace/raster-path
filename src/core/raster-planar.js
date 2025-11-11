/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Raster Planar - XY Grid Rasterization
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Planar rasterization converts 3D triangle meshes into 2D height maps by
 * casting vertical rays through an XY grid. Used for both terrain (keep max Z)
 * and tool (keep min Z) rasterization.
 *
 * EXPORTS:
 * ────────
 * Functions:
 *   - calculateBounds(triangles)           - Compute mesh bounding box
 *   - buildSpatialGrid(triangles, bounds)  - Create spatial acceleration structure
 *   - rasterizeMesh(triangles, stepSize, filterMode, options)
 *       Main entry point with automatic tiling support
 *   - createHeightMapFromPoints(points, gridStep, bounds)
 *       Convert dense raster data to height map format
 *
 * RASTERIZATION MODES:
 * ────────────────────
 * filterMode = 0 (Terrain):
 *   - Output: Dense Z-only array (1 float per grid cell)
 *   - Keep maximum Z where ray hits geometry
 *   - Empty cells contain EMPTY_CELL sentinel value (-1e10)
 *
 * filterMode = 1 (Tool):
 *   - Output: Sparse [X, Y, Z] triplets
 *   - Keep minimum Z where ray hits geometry
 *   - Only valid points are returned (no empty cells)
 *
 * AUTOMATIC TILING:
 * ─────────────────
 * For large meshes exceeding GPU memory limits:
 * 1. Calculate if tiling is needed based on config.maxGPUMemoryMB
 * 2. Subdivide into overlapping tiles (2-cell overlap to avoid gaps)
 * 3. Rasterize each tile independently
 * 4. Stitch results back together, merging overlaps (keep max Z)
 *
 * SPATIAL ACCELERATION:
 * ─────────────────────
 * Builds a 2D spatial grid to cull triangles:
 * - Default cell size: 5mm (configurable)
 * - Each cell stores list of potentially intersecting triangles
 * - Reduces ray-triangle tests by ~100x for typical models
 *
 * DATA FLOW:
 * ──────────
 * Triangles → Spatial Grid → GPU Rasterize → Dense/Sparse Output → Height Map
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

import {
    device, deviceCapabilities, isInitialized, config,
    cachedRasterizePipeline, EMPTY_CELL, debug, initWebGPU, log_pre, round
} from './raster-config.js';

// Calculate bounding box from triangle vertices
export function calculateBounds(triangles) {
    let min_x = Infinity, min_y = Infinity, min_z = Infinity;
    let max_x = -Infinity, max_y = -Infinity, max_z = -Infinity;

    for (let i = 0; i < triangles.length; i += 3) {
        const x = triangles[i];
        const y = triangles[i + 1];
        const z = triangles[i + 2];

        if (x < min_x) min_x = x;
        if (y < min_y) min_y = y;
        if (z < min_z) min_z = z;
        if (x > max_x) max_x = x;
        if (y > max_y) max_y = y;
        if (z > max_z) max_z = z;
    }

    return {
        min: { x: min_x, y: min_y, z: min_z },
        max: { x: max_x, y: max_y, z: max_z }
    };
}

// Build spatial grid for efficient triangle culling
export function buildSpatialGrid(triangles, bounds, cellSize = 5.0) {
    const gridWidth = Math.max(1, Math.ceil((bounds.max.x - bounds.min.x) / cellSize));
    const gridHeight = Math.max(1, Math.ceil((bounds.max.y - bounds.min.y) / cellSize));
    const totalCells = gridWidth * gridHeight;

    const grid = new Array(totalCells);
    for (let i = 0; i < totalCells; i++) {
        grid[i] = [];
    }

    const triangleCount = triangles.length / 9;
    for (let t = 0; t < triangleCount; t++) {
        const base = t * 9;

        const v0x = triangles[base], v0y = triangles[base + 1];
        const v1x = triangles[base + 3], v1y = triangles[base + 4];
        const v2x = triangles[base + 6], v2y = triangles[base + 7];

        // Add small epsilon to catch triangles near cell boundaries
        const epsilon = cellSize * 0.01;  // 1% of cell size
        const minX = Math.min(v0x, v1x, v2x) - epsilon;
        const maxX = Math.max(v0x, v1x, v2x) + epsilon;
        const minY = Math.min(v0y, v1y, v2y) - epsilon;
        const maxY = Math.max(v0y, v1y, v2y) + epsilon;

        let minCellX = Math.floor((minX - bounds.min.x) / cellSize);
        let maxCellX = Math.floor((maxX - bounds.min.x) / cellSize);
        let minCellY = Math.floor((minY - bounds.min.y) / cellSize);
        let maxCellY = Math.floor((maxY - bounds.min.y) / cellSize);

        minCellX = Math.max(0, Math.min(gridWidth - 1, minCellX));
        maxCellX = Math.max(0, Math.min(gridWidth - 1, maxCellX));
        minCellY = Math.max(0, Math.min(gridHeight - 1, minCellY));
        maxCellY = Math.max(0, Math.min(gridHeight - 1, maxCellY));

        for (let cy = minCellY; cy <= maxCellY; cy++) {
            for (let cx = minCellX; cx <= maxCellX; cx++) {
                const cellIdx = cy * gridWidth + cx;
                grid[cellIdx].push(t);
            }
        }
    }

    let totalTriangleRefs = 0;
    for (let i = 0; i < totalCells; i++) {
        totalTriangleRefs += grid[i].length;
    }

    const cellOffsets = new Uint32Array(totalCells + 1);
    const triangleIndices = new Uint32Array(totalTriangleRefs);

    let currentOffset = 0;
    for (let i = 0; i < totalCells; i++) {
        cellOffsets[i] = currentOffset;
        for (let j = 0; j < grid[i].length; j++) {
            triangleIndices[currentOffset++] = grid[i][j];
        }
    }
    cellOffsets[totalCells] = currentOffset;

    const avgPerCell = totalTriangleRefs / totalCells;

    // Calculate actual tool diameter from bounds for logging
    const toolWidth = bounds.max.x - bounds.min.x;
    const toolHeight = bounds.max.y - bounds.min.y;
    const toolDiameter = Math.max(toolWidth, toolHeight);

    debug.log(`Spatial grid: ${gridWidth}x${gridHeight} ${totalTriangleRefs} tri-refs ~${avgPerCell.toFixed(0)}/${cellSize}mm cell (tool: ${toolDiameter.toFixed(2)}mm)`);

    return {
        gridWidth,
        gridHeight,
        cellSize,
        cellOffsets,
        triangleIndices,
        avgTrianglesPerCell: avgPerCell
    };
}

// Create reusable GPU buffers for multiple rasterization passes (e.g., radial rotations)

// Rasterize mesh to point cloud
// Internal function - rasterize without tiling (do not modify this function!)
async function rasterizeMeshSingle(triangles, stepSize, filterMode, options = {}) {
    const startTime = performance.now();

    if (!isInitialized) {
        const initStart = performance.now();
        const success = await initWebGPU();
        if (!success) {
            throw new Error('WebGPU not available');
        }
        const initEnd = performance.now();
        debug.log(`First-time init: ${(initEnd - initStart).toFixed(1)}ms`);
    }

    // debug.log(`Raster ${triangles.length / 9} triangles (step ${stepSize}mm, mode ${filterMode})...`);

    // Extract options
    // boundsOverride: Optional manual bounds to avoid recalculating from triangles
    // Useful when bounds are already known (e.g., from tiling operations)
    const boundsOverride = options.bounds || options.min ? options : null;

    // Use bounds override if provided, otherwise calculate from triangles
    const bounds = boundsOverride || calculateBounds(triangles);

    if (boundsOverride) {
        // debug.log(`Using bounds override: min(${bounds.min.x.toFixed(2)}, ${bounds.min.y.toFixed(2)}, ${bounds.min.z.toFixed(2)}) max(${bounds.max.x.toFixed(2)}, ${bounds.max.y.toFixed(2)}, ${bounds.max.z.toFixed(2)})`);

        // Validate bounds
        if (bounds.min.x >= bounds.max.x || bounds.min.y >= bounds.max.y || bounds.min.z >= bounds.max.z) {
            throw new Error(`Invalid bounds: min must be less than max. Got min(${bounds.min.x}, ${bounds.min.y}, ${bounds.min.z}) max(${bounds.max.x}, ${bounds.max.y}, ${bounds.max.z})`);
        }
    }

    const gridWidth = Math.ceil((bounds.max.x - bounds.min.x) / stepSize) + 1;
    const gridHeight = Math.ceil((bounds.max.y - bounds.min.y) / stepSize) + 1;
    const totalGridPoints = gridWidth * gridHeight;

    // debug.log(`Grid: ${gridWidth}x${gridHeight} = ${totalGridPoints.toLocaleString()} points`);

    // Calculate buffer size based on filter mode
    // filterMode 0 (terrain): Dense Z-only output (1 float per grid cell)
    // filterMode 1 (tool): Sparse X,Y,Z output (3 floats per grid cell)
    const floatsPerPoint = filterMode === 0 ? 1 : 3;
    const outputSize = totalGridPoints * floatsPerPoint * 4;
    const maxBufferSize = device.limits.maxBufferSize || 268435456; // 256MB default
    // const modeStr = filterMode === 0 ? 'terrain (dense Z-only)' : 'tool (sparse XYZ)';
    // debug.log(`Output buffer size: ${(outputSize / 1024 / 1024).toFixed(2)} MB for ${modeStr} (max: ${(maxBufferSize / 1024 / 1024).toFixed(2)} MB)`);

    if (outputSize > maxBufferSize) {
        throw new Error(`Output buffer too large: ${(outputSize / 1024 / 1024).toFixed(2)} MB exceeds device limit of ${(maxBufferSize / 1024 / 1024).toFixed(2)} MB. Try a larger step size.`);
    }

    console.time(`${log_pre} Build Spatial Grid`);
    const spatialGrid = buildSpatialGrid(triangles, bounds);
    console.timeEnd(`${log_pre} Build Spatial Grid`);

    // Create buffers
    const triangleBuffer = device.createBuffer({
        size: triangles.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(triangleBuffer, 0, triangles);

    // Create and INITIALIZE output buffer (GPU buffers contain garbage by default!)
    const outputBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Initialize output buffer with sentinel value for terrain, zeros for tool
    if (filterMode === 0) {
        // Terrain: initialize with EMPTY_CELL sentinel value
        const initData = new Float32Array(totalGridPoints);
        initData.fill(EMPTY_CELL);
        device.queue.writeBuffer(outputBuffer, 0, initData);
    }
    // Tool mode: zeros are fine (will check valid mask)

    const validMaskBuffer = device.createBuffer({
        size: totalGridPoints * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const spatialCellOffsetsBuffer = device.createBuffer({
        size: spatialGrid.cellOffsets.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(spatialCellOffsetsBuffer, 0, spatialGrid.cellOffsets);

    const spatialTriangleIndicesBuffer = device.createBuffer({
        size: spatialGrid.triangleIndices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(spatialTriangleIndicesBuffer, 0, spatialGrid.triangleIndices);

    // Uniforms
    const uniformData = new Float32Array([
        bounds.min.x, bounds.min.y, bounds.min.z,
        bounds.max.x, bounds.max.y, bounds.max.z,
        stepSize,
        0, 0, 0, 0, 0, 0, 0  // Padding for alignment
    ]);
    const uniformDataU32 = new Uint32Array(uniformData.buffer);
    uniformDataU32[7] = gridWidth;
    uniformDataU32[8] = gridHeight;
    uniformDataU32[9] = triangles.length / 9;
    uniformDataU32[10] = filterMode;
    uniformDataU32[11] = spatialGrid.gridWidth;
    uniformDataU32[12] = spatialGrid.gridHeight;
    const uniformDataF32 = new Float32Array(uniformData.buffer);
    uniformDataF32[13] = spatialGrid.cellSize;

    // Check for u32 overflow
    const maxU32 = 4294967295;
    if (gridWidth > maxU32 || gridHeight > maxU32) {
        throw new Error(`Grid dimensions exceed u32 max: ${gridWidth}x${gridHeight}`);
    }

    // debug.log(`Uniforms: gridWidth=${gridWidth}, gridHeight=${gridHeight}, triangles=${triangles.length / 9}`);

    const uniformBuffer = device.createBuffer({
        size: uniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // CRITICAL: Wait for all writeBuffer operations to complete before compute dispatch
    await device.queue.onSubmittedWorkDone();

    // Use cached pipeline
    const bindGroup = device.createBindGroup({
        layout: cachedRasterizePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: triangleBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
            { binding: 2, resource: { buffer: validMaskBuffer } },
            { binding: 3, resource: { buffer: uniformBuffer } },
            { binding: 4, resource: { buffer: spatialCellOffsetsBuffer } },
            { binding: 5, resource: { buffer: spatialTriangleIndicesBuffer } },
        ],
    });

    // Dispatch compute shader
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(cachedRasterizePipeline);
    passEncoder.setBindGroup(0, bindGroup);

    const workgroupsX = Math.ceil(gridWidth / 16);
    const workgroupsY = Math.ceil(gridHeight / 16);

    // Check dispatch limits
    const maxWorkgroupsPerDim = device.limits.maxComputeWorkgroupsPerDimension || 65535;

    if (workgroupsX > maxWorkgroupsPerDim || workgroupsY > maxWorkgroupsPerDim) {
        throw new Error(`Workgroup dispatch too large: ${workgroupsX}x${workgroupsY} exceeds limit of ${maxWorkgroupsPerDim}. Try a larger step size.`);
    }

    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
    passEncoder.end();

    // Create staging buffers for readback
    const stagingOutputBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const stagingValidMaskBuffer = device.createBuffer({
        size: totalGridPoints * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingOutputBuffer, 0, outputSize);
    commandEncoder.copyBufferToBuffer(validMaskBuffer, 0, stagingValidMaskBuffer, 0, totalGridPoints * 4);

    device.queue.submit([commandEncoder.finish()]);

    // Wait for GPU to finish
    await device.queue.onSubmittedWorkDone();

    // Read back results
    await stagingOutputBuffer.mapAsync(GPUMapMode.READ);
    await stagingValidMaskBuffer.mapAsync(GPUMapMode.READ);

    const outputData = new Float32Array(stagingOutputBuffer.getMappedRange());
    const validMaskData = new Uint32Array(stagingValidMaskBuffer.getMappedRange());

    let result, pointCount;

    if (filterMode === 0) {
        // Terrain: Dense output (Z-only), no compaction needed
        // Copy the full array (already has NaN for empty cells)
        result = new Float32Array(outputData);
        pointCount = totalGridPoints;

        if (config.debug) {
            // Count valid points for logging (sentinel value = -1e10)
            let zeroCount = 0;
            let validCount = 0;
            for (let i = 0; i < totalGridPoints; i++) {
                if (result[i] > EMPTY_CELL + 1) validCount++;  // Any value significantly above sentinel
                if (result[i] === 0) zeroCount++;
            }

            let percentHit = validCount/totalGridPoints;
            if (zeroCount > 0 || percentHit < 0.5 ) {
                debug.log(totalGridPoints, 'cells,', round(percentHit*100), '% coverage,', zeroCount, 'zeros');
            }
        }
    } else {
        // Tool: Sparse output (X,Y,Z triplets), compact to remove invalid points
        const validPoints = [];
        for (let i = 0; i < totalGridPoints; i++) {
            if (validMaskData[i] === 1) {
                validPoints.push(
                    outputData[i * 3],
                    outputData[i * 3 + 1],
                    outputData[i * 3 + 2]
                );
            }
        }
        result = new Float32Array(validPoints);
        pointCount = validPoints.length / 3;
    }

    stagingOutputBuffer.unmap();
    stagingValidMaskBuffer.unmap();

    // Cleanup
    triangleBuffer.destroy();
    outputBuffer.destroy();
    validMaskBuffer.destroy();
    uniformBuffer.destroy();
    spatialCellOffsetsBuffer.destroy();
    spatialTriangleIndicesBuffer.destroy();
    stagingOutputBuffer.destroy();
    stagingValidMaskBuffer.destroy();

    const endTime = performance.now();
    const conversionTime = endTime - startTime;
    // debug.log(`Rasterize complete: ${pointCount} points in ${conversionTime.toFixed(1)}ms`);
    // debug.log(`Bounds: min(${bounds.min.x.toFixed(2)}, ${bounds.min.y.toFixed(2)}, ${bounds.min.z.toFixed(2)}) max(${bounds.max.x.toFixed(2)}, ${bounds.max.y.toFixed(2)}, ${bounds.max.z.toFixed(2)})`);

    // Verify result data integrity
    if (filterMode === 0) {
        // Terrain: Dense Z-only format
        if (result.length > 0) {
            const firstZ = result[0] <= EMPTY_CELL + 1 ? 'EMPTY' : result[0].toFixed(3);
            const lastZ = result[result.length-1] <= EMPTY_CELL + 1 ? 'EMPTY' : result[result.length-1].toFixed(3);
            // debug.log(`First Z: ${firstZ}, Last Z: ${lastZ}`);
        }
    } else {
        // Tool: Sparse X,Y,Z format
        if (result.length > 0) {
            const firstPoint = `(${result[0].toFixed(3)}, ${result[1].toFixed(3)}, ${result[2].toFixed(3)})`;
            const lastIdx = result.length - 3;
            const lastPoint = `(${result[lastIdx].toFixed(3)}, ${result[lastIdx+1].toFixed(3)}, ${result[lastIdx+2].toFixed(3)})`;
            // debug.log(`First point: ${firstPoint}, Last point: ${lastPoint}`);
        }
    }

    return {
        positions: result,
        pointCount: pointCount,
        bounds: bounds,
        conversionTime: conversionTime,
        gridWidth: gridWidth,
        gridHeight: gridHeight,
        isDense: filterMode === 0  // True for terrain (dense), false for tool (sparse)
    };
}

// Create tiles for tiled rasterization
function createTiles(bounds, stepSize, maxMemoryBytes) {
    const width = bounds.max.x - bounds.min.x;
    const height = bounds.max.y - bounds.min.y;
    const aspectRatio = width / height;

    // Calculate how many grid points we can fit in one tile
    // Terrain uses dense Z-only format: (gridW * gridH * 1 * 4) for output
    // This is 4x more efficient than the old sparse format (16 bytes → 4 bytes per point)
    const bytesPerPoint = 1 * 4; // 4 bytes per grid point (Z-only)
    const maxPointsPerTile = Math.floor(maxMemoryBytes / bytesPerPoint);
    debug.log(`Dense terrain format: ${bytesPerPoint} bytes/point (was 16), can fit ${(maxPointsPerTile/1e6).toFixed(1)}M points per tile`);

    // Calculate optimal tile grid dimensions while respecting aspect ratio
    // We want: tileGridW * tileGridH <= maxPointsPerTile
    // And: tileGridW / tileGridH ≈ aspectRatio

    let tileGridW, tileGridH;
    if (aspectRatio >= 1) {
        // Width >= Height
        tileGridH = Math.floor(Math.sqrt(maxPointsPerTile / aspectRatio));
        tileGridW = Math.floor(tileGridH * aspectRatio);
    } else {
        // Height > Width
        tileGridW = Math.floor(Math.sqrt(maxPointsPerTile * aspectRatio));
        tileGridH = Math.floor(tileGridW / aspectRatio);
    }

    // Ensure we don't exceed limits
    while (tileGridW * tileGridH * bytesPerPoint > maxMemoryBytes) {
        if (tileGridW > tileGridH) {
            tileGridW--;
        } else {
            tileGridH--;
        }
    }

    // Convert grid dimensions to world dimensions
    const tileWidth = tileGridW * stepSize;
    const tileHeight = tileGridH * stepSize;

    // Calculate number of tiles needed
    const tilesX = Math.ceil(width / tileWidth);
    const tilesY = Math.ceil(height / tileHeight);

    // Calculate actual tile dimensions (distribute evenly)
    const actualTileWidth = width / tilesX;
    const actualTileHeight = height / tilesY;

    debug.log(`Creating ${tilesX}x${tilesY} = ${tilesX * tilesY} tiles (${actualTileWidth.toFixed(2)}mm × ${actualTileHeight.toFixed(2)}mm each)`);
    debug.log(`Tile grid: ${Math.ceil(actualTileWidth / stepSize)}x${Math.ceil(actualTileHeight / stepSize)} points per tile`);

    const tiles = [];
    const overlap = stepSize * 2; // Overlap by 2 grid cells to ensure no gaps

    for (let ty = 0; ty < tilesY; ty++) {
        for (let tx = 0; tx < tilesX; tx++) {
            // Calculate base tile bounds (no overlap)
            let tileMinX = bounds.min.x + (tx * actualTileWidth);
            let tileMinY = bounds.min.y + (ty * actualTileHeight);
            let tileMaxX = Math.min(bounds.max.x, tileMinX + actualTileWidth);
            let tileMaxY = Math.min(bounds.max.y, tileMinY + actualTileHeight);

            // Add overlap (except at outer edges) - but DON'T extend beyond global bounds
            if (tx > 0) tileMinX = Math.max(bounds.min.x, tileMinX - overlap);
            if (ty > 0) tileMinY = Math.max(bounds.min.y, tileMinY - overlap);
            if (tx < tilesX - 1) tileMaxX = Math.min(bounds.max.x, tileMaxX + overlap);
            if (ty < tilesY - 1) tileMaxY = Math.min(bounds.max.y, tileMaxY + overlap);

            tiles.push({
                id: `tile_${tx}_${ty}`,
                bounds: {
                    min: { x: tileMinX, y: tileMinY, z: bounds.min.z },
                    max: { x: tileMaxX, y: tileMaxY, z: bounds.max.z }
                }
            });
        }
    }

    return { tiles, tilesX, tilesY };
}

// Stitch tiles from multiple rasterization passes
function stitchTiles(tileResults, fullBounds, stepSize) {
    if (tileResults.length === 0) {
        throw new Error('No tile results to stitch');
    }

    // Check if results are dense (terrain) or sparse (tool)
    const isDense = tileResults[0].isDense;

    if (isDense) {
        // DENSE TERRAIN STITCHING: Simple array copying (Z-only format)
        debug.log(`Stitching ${tileResults.length} dense terrain tiles...`);

        // Calculate global grid dimensions
        const globalWidth = Math.ceil((fullBounds.max.x - fullBounds.min.x) / stepSize) + 1;
        const globalHeight = Math.ceil((fullBounds.max.y - fullBounds.min.y) / stepSize) + 1;
        const totalGridCells = globalWidth * globalHeight;

        // Allocate global dense grid (Z-only), initialize to sentinel value
        const globalGrid = new Float32Array(totalGridCells);
        globalGrid.fill(EMPTY_CELL);

        debug.log(`Global grid: ${globalWidth}x${globalHeight} = ${totalGridCells.toLocaleString()} cells`);

        // Copy each tile's Z-values to the correct position in global grid
        for (const tile of tileResults) {
            // Calculate tile's position in global grid
            const tileOffsetX = Math.round((tile.tileBounds.min.x - fullBounds.min.x) / stepSize);
            const tileOffsetY = Math.round((tile.tileBounds.min.y - fullBounds.min.y) / stepSize);

            const tileWidth = tile.gridWidth;
            const tileHeight = tile.gridHeight;

            // Copy Z-values row by row
            for (let ty = 0; ty < tileHeight; ty++) {
                const globalY = tileOffsetY + ty;
                if (globalY >= globalHeight) continue;

                for (let tx = 0; tx < tileWidth; tx++) {
                    const globalX = tileOffsetX + tx;
                    if (globalX >= globalWidth) continue;

                    const tileIdx = ty * tileWidth + tx;
                    const globalIdx = globalY * globalWidth + globalX;
                    const tileZ = tile.positions[tileIdx];

                    // For overlapping cells, keep max Z (terrain surface)
                    // Skip empty cells (sentinel value)
                    if (tileZ > EMPTY_CELL + 1) {
                        const existingZ = globalGrid[globalIdx];
                        if (existingZ <= EMPTY_CELL + 1 || tileZ > existingZ) {
                            globalGrid[globalIdx] = tileZ;
                        }
                    }
                }
            }
        }

        // Count valid cells (above sentinel value)
        let validCount = 0;
        for (let i = 0; i < totalGridCells; i++) {
            if (globalGrid[i] > EMPTY_CELL + 1) validCount++;
        }

        debug.log(`Stitched: ${totalGridCells} total cells, ${validCount} with geometry (${(validCount/totalGridCells*100).toFixed(1)}% coverage)`);

        return {
            positions: globalGrid,
            pointCount: totalGridCells,
            bounds: fullBounds,
            gridWidth: globalWidth,
            gridHeight: globalHeight,
            isDense: true,
            conversionTime: tileResults.reduce((sum, r) => sum + (r.conversionTime || 0), 0),
            tileCount: tileResults.length
        };

    } else {
        // SPARSE TOOL STITCHING: Keep existing deduplication logic (X,Y,Z triplets)
        debug.log(`Stitching ${tileResults.length} sparse tool tiles...`);

        const pointMap = new Map();

        for (const result of tileResults) {
            const positions = result.positions;

            // Calculate offset from tile origin to global origin (in grid cells)
            const tileOffsetX = Math.round((result.tileBounds.min.x - fullBounds.min.x) / stepSize);
            const tileOffsetY = Math.round((result.tileBounds.min.y - fullBounds.min.y) / stepSize);

            // Convert each point from tile-local to global grid coordinates
            for (let i = 0; i < positions.length; i += 3) {
                const localGridX = positions[i];
                const localGridY = positions[i + 1];
                const z = positions[i + 2];

                // Convert local grid indices to global grid indices
                const globalGridX = localGridX + tileOffsetX;
                const globalGridY = localGridY + tileOffsetY;

                const key = `${globalGridX},${globalGridY}`;
                const existing = pointMap.get(key);

                // Keep lowest Z value (for tool)
                if (!existing || z < existing.z) {
                    pointMap.set(key, { x: globalGridX, y: globalGridY, z });
                }
            }
        }

        // Convert Map to flat array
        const finalPointCount = pointMap.size;
        const allPositions = new Float32Array(finalPointCount * 3);
        let writeOffset = 0;

        for (const point of pointMap.values()) {
            allPositions[writeOffset++] = point.x;
            allPositions[writeOffset++] = point.y;
            allPositions[writeOffset++] = point.z;
        }

        debug.log(`Stitched: ${finalPointCount} unique sparse points`);

        return {
            positions: allPositions,
            pointCount: finalPointCount,
            bounds: fullBounds,
            isDense: false,
            conversionTime: tileResults.reduce((sum, r) => sum + (r.conversionTime || 0), 0),
            tileCount: tileResults.length
        };
    }
}

// Check if tiling is needed (only called for terrain, which uses dense format)
function shouldUseTiling(bounds, stepSize) {
    if (!config || !config.autoTiling) return false;
    if (!deviceCapabilities) return false;

    const gridWidth = Math.ceil((bounds.max.x - bounds.min.x) / stepSize) + 1;
    const gridHeight = Math.ceil((bounds.max.y - bounds.min.y) / stepSize) + 1;
    const totalPoints = gridWidth * gridHeight;

    // Terrain uses dense Z-only format: 1 float (4 bytes) per grid cell
    const gpuOutputBuffer = totalPoints * 1 * 4;
    const totalGPUMemory = gpuOutputBuffer;  // No mask needed for dense output

    // Use the smaller of configured limit or device capability
    const configuredLimit = config.maxGPUMemoryMB * 1024 * 1024;
    const deviceLimit = deviceCapabilities.maxStorageBufferBindingSize;
    const maxSafeSize = Math.min(configuredLimit, deviceLimit) * config.gpuMemorySafetyMargin;

    return totalGPUMemory > maxSafeSize;
}

// Rasterize mesh - wrapper that handles automatic tiling if needed
export async function rasterizeMesh(triangles, stepSize, filterMode, options = {}) {
    const boundsOverride = options.bounds || options.min ? options : null;  // Support old and new format
    const bounds = boundsOverride || calculateBounds(triangles);

    // Check if tiling is needed
    if (shouldUseTiling(bounds, stepSize)) {
        debug.log('Tiling required - switching to tiled rasterization');

        // Calculate max safe size per tile
        const configuredLimit = config.maxGPUMemoryMB * 1024 * 1024;
        const deviceLimit = deviceCapabilities.maxStorageBufferBindingSize;
        const maxSafeSize = Math.min(configuredLimit, deviceLimit) * config.gpuMemorySafetyMargin;

        // Create tiles
        const { tiles } = createTiles(bounds, stepSize, maxSafeSize);

        // Rasterize each tile
        const tileResults = [];
        for (let i = 0; i < tiles.length; i++) {
            const tileStart = performance.now();
            debug.log(`Processing tile ${i + 1}/${tiles.length}: ${tiles[i].id}`);
            debug.log(`  Tile bounds: min(${tiles[i].bounds.min.x.toFixed(2)}, ${tiles[i].bounds.min.y.toFixed(2)}) max(${tiles[i].bounds.max.x.toFixed(2)}, ${tiles[i].bounds.max.y.toFixed(2)})`);

            const tileResult = await rasterizeMeshSingle(triangles, stepSize, filterMode, {
                ...tiles[i].bounds,
            });

            const tileTime = performance.now() - tileStart;
            debug.log(`  Tile ${i + 1} complete: ${tileResult.pointCount} points in ${tileTime.toFixed(1)}ms`);

            // Store tile bounds with result for coordinate conversion during stitching
            tileResult.tileBounds = tiles[i].bounds;
            tileResults.push(tileResult);
        }

        // Stitch tiles together (pass full bounds and step size for coordinate conversion)
        return stitchTiles(tileResults, bounds, stepSize);
    } else {
        // Single-pass rasterization
        return await rasterizeMeshSingle(triangles, stepSize, filterMode, options);
    }
}

// Helper: Create height map from dense terrain points (Z-only array)
// Terrain is ALWAYS dense (Z-only), never sparse
export function createHeightMapFromPoints(points, gridStep, bounds = null) {
    if (!points || points.length === 0) {
        throw new Error('No points provided');
    }

    // Calculate dimensions from bounds
    if (!bounds) {
        throw new Error('Bounds required for height map creation');
    }

    const minX = bounds.min.x;
    const minY = bounds.min.y;
    const minZ = bounds.min.z;
    const maxX = bounds.max.x;
    const maxY = bounds.max.y;
    const maxZ = bounds.max.z;
    const width = Math.ceil((maxX - minX) / gridStep) + 1;
    const height = Math.ceil((maxY - minY) / gridStep) + 1;

    // Terrain is ALWAYS dense (Z-only format from GPU rasterizer)
    // debug.log(`Terrain dense format: ${width}x${height} = ${points.length} cells`);

    return {
        grid: points,  // Dense Z-only array
        width,
        height,
        minX,
        minY,
        minZ,
        maxX,
        maxY,
        maxZ
    };
}
