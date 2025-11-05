// webgpu-worker.js
// WebGPU worker for compute-only rasterization and toolpath generation
// Runs all GPU operations off the main thread to prevent UI blocking

let device = null;
let isInitialized = false;
let cachedRasterizePipeline = null;
let cachedRasterizeShaderModule = null;
let cachedToolpathPipeline = null;
let cachedToolpathShaderModule = null;
let config = null;
let deviceCapabilities = null;

const EMPTY_CELL = -1e10;
const log_pre = '[Raster Worker]';

// url params to control logging
let { search } = self.location;
let verbose = search.indexOf('debug') >= 0;
let quiet = search.indexOf('quiet') >= 0;

const debug = {
    error: function() { console.error(log_pre, ...arguments) },
    warn: function() { console.warn(log_pre, ...arguments) },
    log: function() { !quiet && console.log(log_pre, ...arguments) },
    ok: function() { console.log(log_pre, '✅', ...arguments) },
};

function round(v, d = 1) {
    return parseFloat(v.toFixed(d));
}

// Global error handler for uncaught errors in worker
self.addEventListener('error', (event) => {
    debug.error('Uncaught error:', event.error || event.message);
    debug.error('Stack:', event.error?.stack);
});

self.addEventListener('unhandledrejection', (event) => {
    debug.error('Unhandled promise rejection:', event.reason);
});

// Initialize WebGPU device in worker context
async function initWebGPU() {
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
        debug.log('Adapter limits:', {
            maxStorageBufferBindingSize: adapterLimits.maxStorageBufferBindingSize,
            maxBufferSize: adapterLimits.maxBufferSize
        });

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

// Planar rasterization with spatial partitioning
const rasterizeShaderCode = /*SHADER:planar-rasterize*/;

// Planar toolpath generation
const toolpathShaderCode = /*SHADER:planar-toolpath*/;

// Radial V2: Rasterization with rotating ray planes and X-bucketing
const radialRasterizeV2ShaderCode = /*SHADER:radial-raster-v2*/;

// Calculate bounding box from triangle vertices
function calculateBounds(triangles) {
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
function buildSpatialGrid(triangles, bounds, cellSize = 5.0) {
    const gridWidth = Math.max(1, Math.ceil((bounds.max.x - bounds.min.x) / cellSize));
    const gridHeight = Math.max(1, Math.ceil((bounds.max.y - bounds.min.y) / cellSize));
    const totalCells = gridWidth * gridHeight;

    // debug.log(`Building spatial grid ${gridWidth}x${gridHeight} (${cellSize}mm cells)`);

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
    // debug.log(`Spatial grid: ${totalTriangleRefs} refs (avg ${avgPerCell.toFixed(1)} per cell)`);

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

    // debug.log(`Rasterizing ${triangles.length / 9} triangles (step ${stepSize}mm, mode ${filterMode})...`);

    // Extract options
    const boundsOverride = options.bounds || options.min ? options : null;  // Support old and new format

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
    const modeStr = filterMode === 0 ? 'terrain (dense Z-only)' : 'tool (sparse XYZ)';
    // debug.log(`Output buffer size: ${(outputSize / 1024 / 1024).toFixed(2)} MB for ${modeStr} (max: ${(maxBufferSize / 1024 / 1024).toFixed(2)} MB)`);

    if (outputSize > maxBufferSize) {
        throw new Error(`Output buffer too large: ${(outputSize / 1024 / 1024).toFixed(2)} MB exceeds device limit of ${(maxBufferSize / 1024 / 1024).toFixed(2)} MB. Try a larger step size.`);
    }

    const spatialGrid = buildSpatialGrid(triangles, bounds);

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

        if (verbose) {
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
async function rasterizeMesh(triangles, stepSize, filterMode, options = {}) {
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
                rotationAngleDeg: options.rotationAngleDeg
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
function createHeightMapFromPoints(points, gridStep, bounds = null) {
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

// Helper: Create sparse tool representation
// Points come from GPU as [gridX, gridY, Z] - pure integer grid coordinates for X/Y
function createSparseToolFromPoints(points) {
    if (!points || points.length === 0) {
        throw new Error('No tool points provided');
    }

    // Points are [gridX, gridY, Z] where gridX/gridY are grid indices (floats but integer values)
    // Find bounds in grid space and tool tip Z
    let minGridX = Infinity, minGridY = Infinity, minZ = Infinity;
    let maxGridX = -Infinity, maxGridY = -Infinity;

    for (let i = 0; i < points.length; i += 3) {
        const gridX = points[i];      // Already a grid index
        const gridY = points[i + 1];  // Already a grid index
        const z = points[i + 2];

        minGridX = Math.min(minGridX, gridX);
        maxGridX = Math.max(maxGridX, gridX);
        minGridY = Math.min(minGridY, gridY);
        maxGridY = Math.max(maxGridY, gridY);
        minZ = Math.min(minZ, z);
    }

    // Calculate tool center in grid coordinates (pure integer)
    const width = Math.floor(maxGridX - minGridX) + 1;
    const height = Math.floor(maxGridY - minGridY) + 1;
    const centerX = Math.floor(minGridX) + Math.floor(width / 2);
    const centerY = Math.floor(minGridY) + Math.floor(height / 2);

    // Convert each point to offset from center (integer arithmetic only)
    const xOffsets = [];
    const yOffsets = [];
    const zValues = [];

    for (let i = 0; i < points.length; i += 3) {
        const gridX = Math.floor(points[i]);      // Grid index (ensure integer)
        const gridY = Math.floor(points[i + 1]);  // Grid index (ensure integer)
        const z = points[i + 2];

        // Calculate offset from tool center (pure integer arithmetic)
        const xOffset = gridX - centerX;
        const yOffset = gridY - centerY;
        // Z relative to tool tip: tip=0, points above tip are positive
        // minZ is the lowest Z (tip), so z - minZ gives positive offsets upward
        const zValue = z;// - minZ;

        xOffsets.push(xOffset);
        yOffsets.push(yOffset);
        zValues.push(zValue);
    }

    return {
        count: xOffsets.length,
        xOffsets: new Int32Array(xOffsets),
        yOffsets: new Int32Array(yOffsets),
        zValues: new Float32Array(zValues),
        referenceZ: minZ
    };
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

    debug.log(`Output: ${pointsPerLine}x${numScanlines} = ${outputSize} points`);

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
    debug.log(`Toolpath complete in ${(endTime - startTime).toFixed(1)}ms`);
    debug.log(`Output: ${result.length} values (${numScanlines} scanlines × ${pointsPerLine} points)`);

    // Log sample of output for debugging
    const sampleSize = Math.min(10, result.length);
    debug.log(`First ${sampleSize} values:`, Array.from(result.slice(0, sampleSize)));

    return {
        pathData: result,
        numScanlines,
        pointsPerLine,
        generationTime: endTime - startTime
    };
}

// Create reusable GPU buffers for tiled toolpath generation
function createReusableToolpathBuffers(terrainWidth, terrainHeight, sparseToolData, xStep, yStep) {
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
        size: 32,
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
function destroyReusableToolpathBuffers(buffers) {
    buffers.terrainBuffer.destroy();
    buffers.toolBuffer.destroy();
    buffers.outputBuffer.destroy();
    buffers.uniformBuffer.destroy();
    buffers.stagingBuffer.destroy();
}

// Run toolpath compute using pre-created reusable buffers
async function runToolpathComputeWithBuffers(terrainData, terrainWidth, terrainHeight, xStep, yStep, oobZ, buffers, startTime) {
    // Update terrain buffer with new tile data
    device.queue.writeBuffer(buffers.terrainBuffer, 0, terrainData);

    // Calculate output dimensions
    const pointsPerLine = Math.ceil(terrainWidth / xStep);
    const numScanlines = Math.ceil(terrainHeight / yStep);
    const outputSize = pointsPerLine * numScanlines;

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

    const outputData = new Float32Array(buffers.stagingBuffer.getMappedRange().slice(0, outputSize * 4));
    const result = new Float32Array(outputData);
    buffers.stagingBuffer.unmap();

    const endTime = performance.now();

    // Debug: Log first few Z values to detect non-determinism
    if (result.length > 0) {
        const samples = [];
        for (let i = 0; i < Math.min(10, result.length); i++) {
            samples.push(result[i].toFixed(3));
        }
        debug.log(`[Toolpath] Output samples (${result.length} total): ${samples.join(', ')}`);
    }

    return {
        pathData: result,
        numScanlines,
        pointsPerLine,
        generationTime: endTime - startTime
    };
}

// Generate toolpath with tiling support (public API)
async function generateToolpath(terrainPoints, toolPoints, xStep, yStep, oobZ, gridStep, terrainBounds = null, singleScanline = false) {
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

    // If singleScanline mode, override bounds to force centerline only
    if (singleScanline) {
        // For radial mode: Y=0 is the X-axis itself (radius=0)
        // The tool centerline should scan at Y=0, not at the midpoint
        const centerY = terrainBounds.min.y;  // Use min.y which should be 0
        terrainBounds = {
            min: { ...terrainBounds.min, y: centerY },
            max: { ...terrainBounds.max, y: centerY }
        };
        debug.log('[Toolpath] Single scanline mode: forcing centerline at Y =', centerY.toFixed(3));
    }

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

// Radial rasterization - two-pass tiled with bit-attention
// Workload budget: triangles × scanlines should stay under this to avoid timeouts
const MAX_WORKLOAD_PER_TILE = 10_000_000; // triangles × gridHeight budget per tile

let cachedRadialCullPipeline = null;
let cachedRadialRasterizePipeline = null;

async function radialRasterize(triangles, stepSize, rotationStepDegrees, zFloor = 0, boundsOverride = null, params = {}) {
    const startTime = performance.now();

    if (!isInitialized) {
        await initWebGPU();
    }

    // Check if SharedArrayBuffer is available and triangle data is large (>1M triangles = 9M floats = 36MB)
    const numTriangles = triangles.length / 9;
    const useSharedBuffer = typeof SharedArrayBuffer !== 'undefined' &&
                           numTriangles > 1000000 &&
                           !(triangles.buffer instanceof SharedArrayBuffer);

    if (useSharedBuffer) {
        debug.log(`Large dataset (${numTriangles.toLocaleString()} triangles), converting to SharedArrayBuffer`);
        const sab = new SharedArrayBuffer(triangles.byteLength);
        const sharedTriangles = new Float32Array(sab);
        sharedTriangles.set(triangles);
        triangles = sharedTriangles;
    }

    const bounds = boundsOverride || calculateBounds(triangles);

    // Calculate max radius (distance from X-axis)
    let maxRadius = 0;
    for (let i = 0; i < triangles.length; i += 3) {
        const y = triangles[i + 1];
        const z = triangles[i + 2];
        const radius = Math.sqrt(y * y + z * z);
        maxRadius = Math.max(maxRadius, radius);
    }

    // Add margin for ray origins to start outside mesh
    maxRadius *= 1.2;

    const circumference = 2 * Math.PI * maxRadius;
    const xRange = bounds.max.x - bounds.min.x;
    const rotationStepRadians = rotationStepDegrees * (Math.PI / 180);

    // Calculate grid height (number of angular scanlines)
    const gridHeight = Math.ceil(360 / rotationStepDegrees) + 1;

    // Calculate number of tiles based on user config or auto-calculation
    let numTiles, trianglesPerTileTarget;

    if (params.trianglesPerTile) {
        // User specified explicit triangles per tile
        trianglesPerTileTarget = params.trianglesPerTile;
        numTiles = Math.max(1, Math.ceil(numTriangles / trianglesPerTileTarget));
        debug.log(`Radial: ${numTriangles} triangles, ${gridHeight} scanlines, ${numTiles} X-tiles (${trianglesPerTileTarget} target tri/tile)`);
    } else {
        // Auto-calculate based on workload budget
        // Total work = numTriangles × gridHeight × culling_efficiency
        // More tiles = smaller X-slices = better culling = less work per tile
        const totalWorkload = numTriangles * gridHeight * 0.5; // 0.5 = expected culling efficiency
        numTiles = Math.max(1, Math.ceil(totalWorkload / MAX_WORKLOAD_PER_TILE));
        trianglesPerTileTarget = Math.ceil(numTriangles / numTiles);
        debug.log(`Radial: ${numTriangles} triangles, ${gridHeight} scanlines, ${numTiles} X-tiles (${trianglesPerTileTarget} avg tri/tile, auto-calculated)`);
    }

    const tileWidth = xRange / numTiles;

    // Create pipelines on first use
    if (!cachedRadialCullPipeline) {
        const cullShaderModule = device.createShaderModule({ code: radialCullShaderCode });
        cachedRadialCullPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: cullShaderModule, entryPoint: 'main' }
        });
        debug.log('Created radial cull pipeline');
    }

    if (!cachedRadialRasterizePipeline) {
        const rasterShaderModule = device.createShaderModule({ code: radialRasterizeShaderCode });
        cachedRadialRasterizePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: rasterShaderModule, entryPoint: 'main' }
        });
        debug.log('Created radial rasterize pipeline');
    }

    // Create shared triangle buffer
    const triangleBuffer = device.createBuffer({
        size: triangles.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(triangleBuffer, 0, triangles);

    // Calculate attention bit array size
    const numWords = Math.ceil(numTriangles / 32);
    debug.log(`Attention array: ${numWords} words (${numWords * 4} bytes)`);

    // Helper function to process a single tile
    async function processTile(tileIdx) {
        const prefix = `Tile ${tileIdx + 1}/${numTiles}:`;
        try {
            const tile_min_x = bounds.min.x + tileIdx * tileWidth;
            const tile_max_x = bounds.min.x + (tileIdx + 1) * tileWidth;

            debug.log(`${prefix} X=[${tile_min_x.toFixed(2)}, ${tile_max_x.toFixed(2)}]`);

        // Pass 1: Cull triangles for this X-tile
        const tileStartTime = performance.now();

        const attentionBuffer = device.createBuffer({
            size: numWords * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Clear attention buffer
        const zeros = new Uint32Array(numWords);
        device.queue.writeBuffer(attentionBuffer, 0, zeros);

        const cullUniformData = new Float32Array(4);
        cullUniformData[0] = tile_min_x;
        cullUniformData[1] = tile_max_x;
        const cullUniformU32 = new Uint32Array(cullUniformData.buffer);
        cullUniformU32[2] = numTriangles;

        const cullUniformBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(cullUniformBuffer, 0, cullUniformData);

        // CRITICAL: Wait for writeBuffer operations before compute dispatch
        await device.queue.onSubmittedWorkDone();

        const cullBindGroup = device.createBindGroup({
            layout: cachedRadialCullPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: triangleBuffer } },
                { binding: 1, resource: { buffer: attentionBuffer } },
                { binding: 2, resource: { buffer: cullUniformBuffer } },
            ],
        });

        const cullEncoder = device.createCommandEncoder();
        const cullPass = cullEncoder.beginComputePass();
        cullPass.setPipeline(cachedRadialCullPipeline);
        cullPass.setBindGroup(0, cullBindGroup);
        cullPass.dispatchWorkgroups(Math.ceil(numTriangles / 256));
        cullPass.end();
        device.queue.submit([cullEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        const cullTime = performance.now() - tileStartTime;
        // debug.log(`  Culling: ${cullTime.toFixed(1)}ms`);

        // Pass 1.5: Read back attention bits and compact to triangle index list
        const compactStartTime = performance.now();

        // Create staging buffer to read attention bits
        const attentionStagingBuffer = device.createBuffer({
            size: numWords * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(attentionBuffer, 0, attentionStagingBuffer, 0, numWords * 4);
        device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await attentionStagingBuffer.mapAsync(GPUMapMode.READ);
        const attentionBits = new Uint32Array(attentionStagingBuffer.getMappedRange());

        // CPU compaction: build list of marked triangle indices
        const compactIndices = [];
        for (let triIdx = 0; triIdx < numTriangles; triIdx++) {
            const wordIdx = Math.floor(triIdx / 32);
            const bitIdx = triIdx % 32;
            const isMarked = (attentionBits[wordIdx] & (1 << bitIdx)) !== 0;
            if (isMarked) {
                compactIndices.push(triIdx);
            }
        }

        attentionStagingBuffer.unmap();
        attentionStagingBuffer.destroy();

        const compactTime = performance.now() - compactStartTime;
        const cullingEfficiency = ((numTriangles - compactIndices.length) / numTriangles * 100).toFixed(1);
        debug.log(`${prefix} Compacted ${numTriangles} → ${compactIndices.length} triangles (${cullingEfficiency}% culled) in ${compactTime.toFixed(1)}ms`);

        // Create compact triangle index buffer
        const compactIndexBuffer = device.createBuffer({
            size: Math.max(4, compactIndices.length * 4), // At least 4 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        if (compactIndices.length > 0) {
            device.queue.writeBuffer(compactIndexBuffer, 0, new Uint32Array(compactIndices));
        }

        // Pass 2: Rasterize this X-tile
        const rasterStartTime = performance.now();

        const gridWidth = Math.ceil(tileWidth / stepSize) + 1;
        const gridHeight = Math.ceil(360 / rotationStepDegrees) + 1; // Number of angular samples
        const totalCells = gridWidth * gridHeight;

        debug.log(`${prefix} Grid: ${gridWidth}×${gridHeight} = ${totalCells} cells (rotationStep=${rotationStepDegrees}°)`);

        const outputBuffer = device.createBuffer({
            size: totalCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Initialize with EMPTY_CELL
        const initData = new Float32Array(totalCells);
        initData.fill(EMPTY_CELL);
        device.queue.writeBuffer(outputBuffer, 0, initData);

        const rotationOffsetRadians = (params.radialRotationOffset ?? 0) * (Math.PI / 180);

        const rasterUniformData = new Float32Array(16);
        rasterUniformData[0] = tile_min_x;
        rasterUniformData[1] = tile_max_x;
        rasterUniformData[2] = maxRadius;
        rasterUniformData[3] = rotationStepRadians;
        rasterUniformData[4] = stepSize;
        rasterUniformData[5] = zFloor;
        rasterUniformData[6] = rotationOffsetRadians;
        rasterUniformData[7] = 0; // padding
        const rasterUniformU32 = new Uint32Array(rasterUniformData.buffer);
        rasterUniformU32[8] = gridWidth;
        rasterUniformU32[9] = gridHeight;
        rasterUniformU32[10] = compactIndices.length; // Use compact count instead of numTriangles
        rasterUniformU32[11] = 0; // No longer using attention words

        const rasterUniformBuffer = device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(rasterUniformBuffer, 0, rasterUniformData);

        // CRITICAL: Wait for writeBuffer operations before compute dispatch
        await device.queue.onSubmittedWorkDone();

        const rasterBindGroup = device.createBindGroup({
            layout: cachedRadialRasterizePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: triangleBuffer } },
                { binding: 1, resource: { buffer: compactIndexBuffer } }, // Use compact indices instead of attention bits
                { binding: 2, resource: { buffer: outputBuffer } },
                { binding: 3, resource: { buffer: rasterUniformBuffer } },
            ],
        });

        const rasterEncoder = device.createCommandEncoder();
        const rasterPass = rasterEncoder.beginComputePass();
        rasterPass.setPipeline(cachedRadialRasterizePipeline);
        rasterPass.setBindGroup(0, rasterBindGroup);
        const workgroupsX = Math.ceil(gridWidth / 16);
        const workgroupsY = Math.ceil(gridHeight / 16);
        rasterPass.dispatchWorkgroups(workgroupsX, workgroupsY);
        rasterPass.end();

        const stagingBuffer = device.createBuffer({
            size: totalCells * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        rasterEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, totalCells * 4);
        device.queue.submit([rasterEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const outputData = new Float32Array(stagingBuffer.getMappedRange());
        const tileResult = new Float32Array(outputData);
        stagingBuffer.unmap();

        const rasterTime = performance.now() - rasterStartTime;
        debug.log(`${prefix} Rasterization: ${rasterTime.toFixed(1)}ms`);

        // Cleanup tile buffers
        attentionBuffer.destroy();
        cullUniformBuffer.destroy();
        compactIndexBuffer.destroy();
        outputBuffer.destroy();
        rasterUniformBuffer.destroy();
        stagingBuffer.destroy();

            return {
                tileIdx,
                data: tileResult,
                gridWidth,
                gridHeight,
                minX: tile_min_x,
                maxX: tile_max_x
            };
        } catch (error) {
            debug.error(`Error processing tile ${tileIdx + 1}/${numTiles}:`, error);
            throw new Error(`Tile ${tileIdx + 1} failed: ${error.message}`);
        }
    }

    // Process tiles with rolling window to maintain constant concurrency
    // This keeps GPU busy while preventing browser from allocating too many buffers at once
    const maxConcurrentTiles = params.maxConcurrentTiles ?? 50;
    debug.log(`Processing ${numTiles} tiles (max ${maxConcurrentTiles} concurrent)...`);

    let completedTiles = 0;
    let nextTileIdx = 0;
    const activeTiles = new Map(); // promise -> tileIdx
    const tileResults = new Array(numTiles);

    // Helper to start a tile and track it
    const startTile = (tileIdx) => {
        const promise = processTile(tileIdx).then(result => {
            completedTiles++;
            const percent = Math.round((completedTiles / numTiles) * 100);

            // Report progress
            self.postMessage({
                type: 'rasterize-progress',
                data: {
                    percent,
                    current: completedTiles,
                    total: numTiles
                }
            });

            tileResults[tileIdx] = result;
            activeTiles.delete(promise);
            return result;
        });
        activeTiles.set(promise, tileIdx);
        return promise;
    };

    // Start initial window of tiles
    while (nextTileIdx < numTiles && activeTiles.size < maxConcurrentTiles) {
        startTile(nextTileIdx++);
    }

    // As tiles complete, start new ones to maintain window size
    while (activeTiles.size > 0) {
        // Wait for at least one tile to complete
        await Promise.race(activeTiles.keys());

        // Start as many new tiles as needed to fill window
        while (nextTileIdx < numTiles && activeTiles.size < maxConcurrentTiles) {
            startTile(nextTileIdx++);
        }
    }

    triangleBuffer.destroy();

    const totalTime = performance.now() - startTime;
    debug.log(`Radial complete in ${totalTime.toFixed(1)}ms`);

    // Stitch tiles together into a single dense array
    const fullGridHeight = Math.ceil(360 / rotationStepDegrees) + 1; // Number of angular samples
    const fullGridWidth = Math.ceil(xRange / stepSize) + 1;
    const stitchedData = new Float32Array(fullGridWidth * fullGridHeight);
    stitchedData.fill(EMPTY_CELL);

    for (const tile of tileResults) {
        const tileXOffset = Math.round((tile.minX - bounds.min.x) / stepSize);

        for (let ty = 0; ty < tile.gridHeight; ty++) {
            for (let tx = 0; tx < tile.gridWidth; tx++) {
                const tileIdx = ty * tile.gridWidth + tx;
                const fullX = tileXOffset + tx;
                const fullY = ty;

                if (fullX >= 0 && fullX < fullGridWidth && fullY >= 0 && fullY < fullGridHeight) {
                    const fullIdx = fullY * fullGridWidth + fullX;
                    stitchedData[fullIdx] = tile.data[tileIdx];
                }
            }
        }
    }

    // For toolpath generation, bounds.max.y must match the actual grid dimensions
    // so that createHeightMapFromPoints calculates the correct height:
    // height = ceil((max.y - min.y) / stepSize) + 1 = gridHeight
    // Therefore: max.y = (gridHeight - 1) * stepSize
    const boundsMaxY = (fullGridHeight - 1) * stepSize;

    return {
        positions: stitchedData,
        pointCount: stitchedData.length,
        bounds: {
            min: { x: bounds.min.x, y: 0, z: 0 },
            max: { x: bounds.max.x, y: boundsMaxY, z: maxRadius }
        },
        conversionTime: totalTime,
        gridWidth: fullGridWidth,
        gridHeight: fullGridHeight,
        isDense: true,
        maxRadius,
        circumference,
        rotationStepDegrees  // NEW: needed for wrapping and toolpath generation
    };
}

// Radial V2: Rasterize model with rotating ray planes and X-bucketing
async function radialRasterizeV2(triangles, bucketData, resolution, angleStep, numAngles, maxRadius, toolWidth, zFloor, bounds) {
    if (!device) {
        throw new Error('WebGPU not initialized');
    }

    const startTime = performance.now();

    // Calculate grid dimensions
    const gridWidth = Math.ceil((bounds.max.x - bounds.min.x) / resolution);
    const gridYHeight = Math.ceil(toolWidth / resolution);
    const bucketGridWidth = Math.ceil((bucketData.buckets[0].maxX - bucketData.buckets[0].minX) / resolution);

    debug.log('[Worker] Radial V2 rasterization:', {
        gridWidth,
        gridYHeight,
        numAngles,
        numBuckets: bucketData.numBuckets,
        bucketGridWidth,
        maxRadius,
        toolWidth,
        resolution,
        angleStepDeg: angleStep,
        boundsMinX: bounds.min.x,
        boundsMaxX: bounds.max.x,
        zFloor
    });

    // Debug first bucket
    if (bucketData.buckets.length > 0) {
        debug.log('[Worker] First bucket:', {
            minX: bucketData.buckets[0].minX,
            maxX: bucketData.buckets[0].maxX,
            startIndex: bucketData.buckets[0].startIndex,
            count: bucketData.buckets[0].count
        });
    }

    // Create GPU buffers
    const triangleBuffer = device.createBuffer({
        size: triangles.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(triangleBuffer.getMappedRange()).set(triangles);
    triangleBuffer.unmap();

    // Create bucket info buffer (f32, f32, u32, u32 per bucket)
    const bucketInfoSize = bucketData.buckets.length * 16; // 4 fields * 4 bytes
    const bucketInfoBuffer = device.createBuffer({
        size: bucketInfoSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });

    const bucketView = new ArrayBuffer(bucketInfoSize);
    const bucketFloatView = new Float32Array(bucketView);
    const bucketUintView = new Uint32Array(bucketView);

    for (let i = 0; i < bucketData.buckets.length; i++) {
        const bucket = bucketData.buckets[i];
        const offset = i * 4;
        bucketFloatView[offset] = bucket.minX;           // f32
        bucketFloatView[offset + 1] = bucket.maxX;       // f32
        bucketUintView[offset + 2] = bucket.startIndex;  // u32
        bucketUintView[offset + 3] = bucket.count;       // u32
    }

    new Uint8Array(bucketInfoBuffer.getMappedRange()).set(new Uint8Array(bucketView));
    bucketInfoBuffer.unmap();

    // Create triangle indices buffer
    const triangleIndicesBuffer = device.createBuffer({
        size: bucketData.triangleIndices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Uint32Array(triangleIndicesBuffer.getMappedRange()).set(bucketData.triangleIndices);
    triangleIndicesBuffer.unmap();

    // Create output buffer (all angles, all buckets)
    const outputSize = numAngles * bucketData.numBuckets * bucketGridWidth * gridYHeight * 4;
    const outputBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    // CRITICAL: Initialize output buffer with zFloor to avoid reading garbage data
    // Use mappedAtCreation for deterministic initialization (not writeBuffer!)
    const initEncoder = device.createCommandEncoder();
    const initData = new Float32Array(outputSize / 4);
    initData.fill(zFloor);
    device.queue.writeBuffer(outputBuffer, 0, initData);
    // Force initialization to complete
    device.queue.submit([initEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Create uniforms with proper alignment (f32 and u32 mixed)
    // Struct layout: f32, f32, u32, f32, f32, u32, f32, u32, f32, f32, u32, u32
    const uniformBuffer = device.createBuffer({
        size: 48,  // 12 fields * 4 bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });

    const uniformView = new ArrayBuffer(48);
    const floatView = new Float32Array(uniformView);
    const uintView = new Uint32Array(uniformView);

    floatView[0] = resolution;                                          // f32
    floatView[1] = angleStep * (Math.PI / 180);                        // f32
    uintView[2] = numAngles;                                            // u32
    floatView[3] = maxRadius;                                           // f32
    floatView[4] = toolWidth;                                           // f32
    uintView[5] = gridYHeight;                                          // u32
    floatView[6] = bucketData.buckets[0].maxX - bucketData.buckets[0].minX;  // f32 bucketWidth
    uintView[7] = bucketGridWidth;                                      // u32
    floatView[8] = bounds.min.x;                                        // f32 global_min_x
    floatView[9] = zFloor;                                              // f32
    uintView[10] = 0;                                                   // u32 filterMode
    uintView[11] = bucketData.numBuckets;                               // u32

    new Uint8Array(uniformBuffer.getMappedRange()).set(new Uint8Array(uniformView));
    uniformBuffer.unmap();

    // Create shader and pipeline
    const shaderModule = device.createShaderModule({ code: radialRasterizeV2ShaderCode });
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });

    // Create bind group
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

    console.time('RADIAL COMPUTE');
    // Dispatch
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);

    // Dispatch: (numAngles/8, gridYHeight/8, numBuckets)
    const dispatchX = Math.ceil(numAngles / 8);
    const dispatchY = Math.ceil(gridYHeight / 8);
    const dispatchZ = bucketData.numBuckets;
    debug.log({ dispatchX, dispatchY, dispatchZ });

    passEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    passEncoder.end();

    // Read back
    const stagingBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputSize);
    device.queue.submit([commandEncoder.finish()]);

    // CRITICAL: Wait for GPU to finish before reading results
    await device.queue.onSubmittedWorkDone();
    console.timeEnd('RADIAL COMPUTE');

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const outputData = new Float32Array(stagingBuffer.getMappedRange());
    const outputCopy = new Float32Array(outputData);
    stagingBuffer.unmap();

    // Cleanup
    triangleBuffer.destroy();
    bucketInfoBuffer.destroy();
    triangleIndicesBuffer.destroy();
    outputBuffer.destroy();
    uniformBuffer.destroy();
    stagingBuffer.destroy();

    const gpuTime = performance.now() - startTime;
    debug.log('[Worker] GPU rasterization complete:', gpuTime.toFixed(1), 'ms');

    // Debug: Check output buffer for uniqueness
    let minZ = Infinity, maxZ = -Infinity, numNonFloor = 0;
    for (let i = 0; i < outputCopy.length; i++) {
        if (outputCopy[i] !== zFloor) {
            minZ = Math.min(minZ, outputCopy[i]);
            maxZ = Math.max(maxZ, outputCopy[i]);
            numNonFloor++;
        }
    }
    debug.log('[Worker] Output buffer stats:', {
        totalCells: outputCopy.length,
        nonFloorCells: numNonFloor,
        minZ: minZ.toFixed(3),
        maxZ: maxZ.toFixed(3),
        first10: Array.from(outputCopy.slice(0, 10)).map(v => v.toFixed(2))
    });

    // Stitch strips
    const stitchStart = performance.now();
    const strips = [];

    for (let angleIdx = 0; angleIdx < numAngles; angleIdx++) {
        const stripData = new Float32Array(gridWidth * gridYHeight);
        stripData.fill(zFloor);  // Initialize with zFloor, not zeros!

        // Gather from each bucket
        for (let bucketIdx = 0; bucketIdx < bucketData.numBuckets; bucketIdx++) {
            const bucket = bucketData.buckets[bucketIdx];
            const bucketMinGridX = Math.floor((bucket.minX - bounds.min.x) / resolution);

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

        // Convert to sparse format [worldX, worldY, Z, ...]
        // NOTE: Must use world coordinates, not grid indices, for toolpath generator
        const sparsePositions = [];
        let minZ = Infinity, maxZ = -Infinity;
        let validCount = 0;
        for (let y = 0; y < gridYHeight; y++) {
            for (let x = 0; x < gridWidth; x++) {
                const z = stripData[y * gridWidth + x];
                minZ = Math.min(minZ, z);
                maxZ = Math.max(maxZ, z);
                // Include ALL cells - let toolpath generator decide what's valid
                // Convert grid indices to world coordinates
                const worldX = bounds.min.x + x * resolution;
                const worldY = y * resolution;  // Y is 0 to toolWidth
                sparsePositions.push(worldX, worldY, z);
                if (z !== zFloor) validCount++;
            }
        }

        // Debug sample strips to check X coverage and angle symmetry
        if (angleIdx === 0 || angleIdx === 90 || angleIdx === 180 || angleIdx === 270) {
            // Sample X positions from start, middle, end to check coverage
            const samples = [];
            const sampleX = [0, Math.floor(gridWidth/4), Math.floor(gridWidth/2), Math.floor(3*gridWidth/4), gridWidth-1];
            for (const x of sampleX) {
                const z = stripData[x]; // First Y row
                samples.push(`x${x}=${z.toFixed(2)}`);
            }
            debug.log(`[Worker] Strip ${angleIdx} (${(angleIdx * angleStep).toFixed(1)}°): minZ=${minZ.toFixed(3)}, maxZ=${maxZ.toFixed(3)}, valid=${validCount}/${gridWidth * gridYHeight}`);
            debug.log(`[Worker]   X samples: ${samples.join(', ')}`);
        }

        strips.push({
            angle: angleIdx * angleStep,
            positions: new Float32Array(sparsePositions),
            gridWidth,
            gridHeight: gridYHeight,
            pointCount: sparsePositions.length / 3,
            bounds: {
                min: { x: bounds.min.x, y: 0, z: zFloor },
                max: { x: bounds.max.x, y: toolWidth, z: bounds.max.z }
            }
        });
    }

    const stitchTime = performance.now() - stitchStart;
    debug.log('[Worker] Strip stitching complete:', stitchTime.toFixed(1), 'ms');
    debug.log('[Worker] Total time:', (performance.now() - startTime).toFixed(1), 'ms');

    return { strips };
}

// Handle messages from main thread
self.onmessage = async function(e) {
    const { type, data } = e.data;

    try {
        switch (type) {
            case 'init':
                // Store config
                config = data?.config || {
                    maxGPUMemoryMB: 256,
                    gpuMemorySafetyMargin: 0.8,
                    tileOverlapMM: 10,
                    autoTiling: true,
                    minTileSize: 50
                };
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
                config = data.config;
                debug.log('Config updated:', config);
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

            case 'radial-rasterize':
                const { triangles: radialTriangles, stepSize: radialStep, rotationStep: radialRotationStep, zFloor: radialZFloor = 0, boundsOverride: radialBounds, maxConcurrentTiles, trianglesPerTile, radialRotationOffset } = data;
                const radialResult = await radialRasterize(radialTriangles, radialStep, radialRotationStep, radialZFloor, radialBounds, { maxConcurrentTiles, trianglesPerTile, radialRotationOffset });
                self.postMessage({
                    type: 'radial-rasterize-complete',
                    data: radialResult
                }, [radialResult.positions.buffer]);
                break;

            case 'radial-rasterize-v2':
                const {
                    triangles: v2Triangles,
                    bucketData,
                    resolution,
                    angleStep,
                    numAngles,
                    maxRadius,
                    toolWidth,
                    zFloor: v2ZFloor,
                    bounds: v2Bounds
                } = data;
                const v2Result = await radialRasterizeV2(
                    v2Triangles,
                    bucketData,
                    resolution,
                    angleStep,
                    numAngles,
                    maxRadius,
                    toolWidth,
                    v2ZFloor,
                    v2Bounds
                );
                // Transfer strip buffers
                const transferBuffers = v2Result.strips.map(strip => strip.positions.buffer);
                self.postMessage({
                    type: 'radial-rasterize-v2-complete',
                    data: v2Result
                }, transferBuffers);
                break;

            case 'radial-generate-toolpaths':
                // Complete radial pipeline: rasterize model + generate toolpaths for all strips
                const {
                    triangles: radialModelTriangles,
                    bucketData: radialBucketData,
                    toolData: radialToolData,
                    resolution: radialResolution,
                    angleStep: radialAngleStep,
                    numAngles: radialNumAngles,
                    maxRadius: radialMaxRadius,
                    toolWidth: radialToolWidth,
                    zFloor: radialToolpathZFloor,
                    bounds: radialToolpathBounds,
                    xStep: radialXStep,
                    yStep: radialYStep,
                    gridStep: radialGridStep
                } = data;

                debug.log('[Worker] Starting complete radial toolpath pipeline...');

                // Step 1: Rasterize model (all strips)
                const radialModelResult = await radialRasterizeV2(
                    radialModelTriangles,
                    radialBucketData,
                    radialResolution,
                    radialAngleStep,
                    radialNumAngles,
                    radialMaxRadius,
                    radialToolWidth,
                    radialToolpathZFloor,
                    radialToolpathBounds
                );

                debug.log(`[Worker] Rasterized ${radialModelResult.strips.length} strips`);

                // Step 2: Generate toolpath for each strip
                const stripToolpaths = [];
                let totalToolpathPoints = 0;

                for (let i = 0; i < radialModelResult.strips.length; i++) {
                    const strip = radialModelResult.strips[i];

                    // Report progress
                    self.postMessage({
                        type: 'toolpath-progress',
                        data: {
                            percent: Math.round(((i + 1) / radialModelResult.strips.length) * 100),
                            current: i + 1,
                            total: radialModelResult.strips.length,
                            layer: i + 1
                        }
                    });

                    // Skip empty strips (no terrain points)
                    if (!strip.positions || strip.positions.length === 0) {
                        // Don't log every empty strip - too verbose
                        continue;
                    }

                    // Generate toolpath for this strip (single scanline)
                    const stripToolpathResult = await generateToolpath(
                        strip.positions,
                        radialToolData.positions,
                        radialXStep,
                        radialYStep,
                        radialToolpathZFloor,
                        radialGridStep,
                        strip.bounds,
                        true  // singleScanline = true
                    );

                    stripToolpaths.push({
                        angle: strip.angle,
                        pathData: stripToolpathResult.pathData,
                        numScanlines: stripToolpathResult.numScanlines,
                        pointsPerLine: stripToolpathResult.pointsPerLine
                    });

                    totalToolpathPoints += stripToolpathResult.pathData.length;

                    // Only log first, middle, last strips to reduce spam
                    if (i < 3 || i === Math.floor(radialModelResult.strips.length / 2) || i >= radialModelResult.strips.length - 3) {
                        debug.log(`[Worker] Strip ${i} (${strip.angle.toFixed(1)}°): ${stripToolpathResult.pathData.length} toolpath points`);
                    }
                }

                debug.log(`[Worker] Complete radial toolpath: ${stripToolpaths.length} strips, ${totalToolpathPoints} total points`);

                // Transfer all strip toolpath buffers
                const toolpathTransferBuffers = stripToolpaths.map(strip => strip.pathData.buffer);
                self.postMessage({
                    type: 'radial-toolpaths-complete',
                    data: {
                        strips: stripToolpaths,
                        totalPoints: totalToolpathPoints,
                        numStrips: stripToolpaths.length
                    }
                }, toolpathTransferBuffers);
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
