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

// Initialize WebGPU device in worker context
async function initWebGPU() {
    if (isInitialized) return true;

    if (!navigator.gpu) {
        console.warn('[WebGPU Worker] WebGPU not supported');
        return false;
    }

    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            console.warn('[WebGPU Worker] WebGPU adapter not available');
            return false;
        }

        // Request device with higher limits for large meshes
        const adapterLimits = adapter.limits;
        console.log('[WebGPU Worker] Adapter limits:', {
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
        console.log('[WebGPU Worker] ✅ Initialized (pipelines cached)');
        return true;
    } catch (error) {
        console.error('[WebGPU Worker] Failed to initialize:', error);
        return false;
    }
}

const rasterizeShaderCode = `
// Sentinel value for empty cells (far below any real geometry)
const EMPTY_CELL: f32 = -1e10;

struct Uniforms {
    bounds_min_x: f32,
    bounds_min_y: f32,
    bounds_min_z: f32,
    bounds_max_x: f32,
    bounds_max_y: f32,
    bounds_max_z: f32,
    step_size: f32,
    grid_width: u32,
    grid_height: u32,
    triangle_count: u32,
    filter_mode: u32,  // 0 = UPWARD (terrain, keep highest), 1 = DOWNWARD (tool, keep lowest)
    spatial_grid_width: u32,
    spatial_grid_height: u32,
    spatial_cell_size: f32,
    rotation_cos: f32,
    rotation_sin: f32,
}

@group(0) @binding(0) var<storage, read> triangles: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_points: array<f32>;
@group(0) @binding(2) var<storage, read_write> valid_mask: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read> spatial_cell_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> spatial_triangle_indices: array<u32>;

// Rotate a point around the X-axis
// X stays the same, Y and Z are rotated
// Y' = Y*cos(θ) - Z*sin(θ)
// Z' = Y*sin(θ) + Z*cos(θ)
fn rotateAroundX(pos: vec3<f32>, cos_angle: f32, sin_angle: f32) -> vec3<f32> {
    return vec3<f32>(
        pos.x,
        pos.y * cos_angle - pos.z * sin_angle,
        pos.y * sin_angle + pos.z * cos_angle
    );
}

// Fast 2D bounding box check for XY plane
fn ray_hits_triangle_bbox_2d(ray_x: f32, ray_y: f32, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> bool {
    let min_x = min(min(v0.x, v1.x), v2.x);
    let max_x = max(max(v0.x, v1.x), v2.x);
    let min_y = min(min(v0.y, v1.y), v2.y);
    let max_y = max(max(v0.y, v1.y), v2.y);

    return ray_x >= min_x && ray_x <= max_x && ray_y >= min_y && ray_y <= max_y;
}

// Ray-triangle intersection using Möller-Trumbore algorithm
fn ray_triangle_intersect(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>
) -> vec2<f32> {  // Returns (hit: 0.0 or 1.0, z: intersection_z)
    let EPSILON = 0.0000001;

    // Early rejection using 2D bounding box (very cheap!)
    if (!ray_hits_triangle_bbox_2d(ray_origin.x, ray_origin.y, v0, v1, v2)) {
        return vec2<f32>(0.0, 0.0);
    }

    // Calculate edges
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;

    // Cross product: ray_dir × edge2
    let h = cross(ray_dir, edge2);

    // Dot product: edge1 · h
    let a = dot(edge1, h);

    if (a > -EPSILON && a < EPSILON) {
        return vec2<f32>(0.0, 0.0); // Ray parallel to triangle
    }

    let f = 1.0 / a;

    // s = ray_origin - v0
    let s = ray_origin - v0;

    // u = f * (s · h)
    let u = f * dot(s, h);

    // Allow small tolerance for edges/vertices to ensure watertight coverage
    if (u < -EPSILON || u > 1.0 + EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }

    // Cross product: s × edge1
    let q = cross(s, edge1);

    // v = f * (ray_dir · q)
    let v = f * dot(ray_dir, q);

    // Allow small tolerance for edges/vertices to ensure watertight coverage
    if (v < -EPSILON || u + v > 1.0 + EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }

    // t = f * (edge2 · q)
    let t = f * dot(edge2, q);

    if (t > EPSILON) {
        // Intersection found - calculate Z coordinate
        let intersection_z = ray_origin.z + ray_dir.z * t;
        return vec2<f32>(1.0, intersection_z);
    }

    return vec2<f32>(0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_x = global_id.x;
    let grid_y = global_id.y;

    if (grid_x >= uniforms.grid_width || grid_y >= uniforms.grid_height) {
        return;
    }

    // Calculate world position for this grid point (center of cell)
    let world_x = uniforms.bounds_min_x + f32(grid_x) * uniforms.step_size;
    let world_y = uniforms.bounds_min_y + f32(grid_y) * uniforms.step_size;

    // Initialize best_z based on filter mode
    var best_z: f32;
    if (uniforms.filter_mode == 0u) {
        best_z = -1e10;  // Terrain: keep highest Z
    } else {
        best_z = 1e10;   // Tool: keep lowest Z
    }

    var found = false;

    // Ray from below mesh pointing up (+Z direction)
    let ray_origin = vec3<f32>(world_x, world_y, uniforms.bounds_min_z - 1.0);
    let ray_dir = vec3<f32>(0.0, 0.0, 1.0);

    // Find which spatial grid cell this ray belongs to
    let spatial_cell_x = u32((world_x - uniforms.bounds_min_x) / uniforms.spatial_cell_size);
    let spatial_cell_y = u32((world_y - uniforms.bounds_min_y) / uniforms.spatial_cell_size);

    // Clamp to spatial grid bounds
    let clamped_cx = min(spatial_cell_x, uniforms.spatial_grid_width - 1u);
    let clamped_cy = min(spatial_cell_y, uniforms.spatial_grid_height - 1u);

    let spatial_cell_idx = clamped_cy * uniforms.spatial_grid_width + clamped_cx;

    // Get triangle range for this cell
    let start_idx = spatial_cell_offsets[spatial_cell_idx];
    let end_idx = spatial_cell_offsets[spatial_cell_idx + 1u];

    // Test only triangles in this spatial cell
    for (var idx = start_idx; idx < end_idx; idx++) {
        let tri_idx = spatial_triangle_indices[idx];
        let tri_base = tri_idx * 9u;

        // Read triangle vertices
        var v0 = vec3<f32>(
            triangles[tri_base],
            triangles[tri_base + 1u],
            triangles[tri_base + 2u]
        );
        var v1 = vec3<f32>(
            triangles[tri_base + 3u],
            triangles[tri_base + 4u],
            triangles[tri_base + 5u]
        );
        var v2 = vec3<f32>(
            triangles[tri_base + 6u],
            triangles[tri_base + 7u],
            triangles[tri_base + 8u]
        );

        // Apply rotation around X-axis if rotation is active (not identity)
        if (uniforms.rotation_cos != 1.0 || uniforms.rotation_sin != 0.0) {
            v0 = rotateAroundX(v0, uniforms.rotation_cos, uniforms.rotation_sin);
            v1 = rotateAroundX(v1, uniforms.rotation_cos, uniforms.rotation_sin);
            v2 = rotateAroundX(v2, uniforms.rotation_cos, uniforms.rotation_sin);
        }

        let result = ray_triangle_intersect(ray_origin, ray_dir, v0, v1, v2);
        let hit = result.x;
        let intersection_z = result.y;

        if (hit > 0.5) {
            if (uniforms.filter_mode == 0u) {
                // Terrain: keep highest
                if (intersection_z > best_z) {
                    best_z = intersection_z;
                    found = true;
                }
            } else {
                // Tool: keep lowest
                if (intersection_z < best_z) {
                    best_z = intersection_z;
                    found = true;
                }
            }
        }
    }

    // Write output based on filter mode
    let output_idx = grid_y * uniforms.grid_width + grid_x;

    if (uniforms.filter_mode == 0u) {
        // Terrain: Dense output (Z-only, sentinel value for empty cells)
        if (found) {
            output_points[output_idx] = best_z;
        } else {
            output_points[output_idx] = EMPTY_CELL;
        }
    } else {
        // Tool: Sparse output (X, Y, Z triplets with valid mask)
        output_points[output_idx * 3u] = f32(grid_x);
        output_points[output_idx * 3u + 1u] = f32(grid_y);
        output_points[output_idx * 3u + 2u] = best_z;

        if (found) {
            valid_mask[output_idx] = 1u;
        } else {
            valid_mask[output_idx] = 0u;
        }
    }
}
`;

const toolpathShaderCode = `
// Sentinel value for empty terrain cells (must match rasterize shader)
const EMPTY_CELL: f32 = -1e10;

struct SparseToolPoint {
    x_offset: i32,
    y_offset: i32,
    z_value: f32,
    padding: f32,
}

struct Uniforms {
    terrain_width: u32,
    terrain_height: u32,
    tool_count: u32,
    x_step: u32,
    y_step: u32,
    oob_z: f32,
    points_per_line: u32,
    num_scanlines: u32,
}

@group(0) @binding(0) var<storage, read> terrain_map: array<f32>;
@group(0) @binding(1) var<storage, read> sparse_tool: array<SparseToolPoint>;
@group(0) @binding(2) var<storage, read_write> output_path: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let scanline = global_id.y;
    let point_idx = global_id.x;

    if (scanline >= uniforms.num_scanlines || point_idx >= uniforms.points_per_line) {
        return;
    }

    let tool_center_x = i32(point_idx * uniforms.x_step);
    let tool_center_y = i32(scanline * uniforms.y_step);

    var min_delta = 3.402823466e+38;

    for (var i = 0u; i < uniforms.tool_count; i++) {
        let tool_point = sparse_tool[i];
        let terrain_x = tool_center_x + tool_point.x_offset;
        let terrain_y = tool_center_y + tool_point.y_offset;

        if (terrain_x < 0 || terrain_y < 0 ||
            terrain_x >= i32(uniforms.terrain_width) ||
            terrain_y >= i32(uniforms.terrain_height)) {
            continue;
        }

        let terrain_idx = u32(terrain_y) * uniforms.terrain_width + u32(terrain_x);
        let terrain_z = terrain_map[terrain_idx];

        // Check if terrain cell has geometry (not empty sentinel value)
        if (terrain_z > EMPTY_CELL + 1.0) {
            let delta = tool_point.z_value - terrain_z;
            min_delta = min(min_delta, delta);
        }
    }

    var output_z = uniforms.oob_z;
    if (min_delta < 3.402823466e+38) {
        output_z = -min_delta;
    }

    let output_idx = scanline * uniforms.points_per_line + point_idx;
    output_path[output_idx] = output_z;
}
`;

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

    // console.log(`[WebGPU Worker] Building spatial grid ${gridWidth}x${gridHeight} (${cellSize}mm cells)`);

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

        const minX = Math.min(v0x, v1x, v2x);
        const maxX = Math.max(v0x, v1x, v2x);
        const minY = Math.min(v0y, v1y, v2y);
        const maxY = Math.max(v0y, v1y, v2y);

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
    // console.log(`[WebGPU Worker] Spatial grid: ${totalTriangleRefs} refs (avg ${avgPerCell.toFixed(1)} per cell)`);

    return {
        gridWidth,
        gridHeight,
        cellSize,
        cellOffsets,
        triangleIndices,
        avgTrianglesPerCell: avgPerCell
    };
}

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
        console.log(`[WebGPU Worker] First-time init: ${(initEnd - initStart).toFixed(1)}ms`);
    }

    // console.log(`[WebGPU Worker] Rasterizing ${triangles.length / 9} triangles (step ${stepSize}mm, mode ${filterMode})...`);

    // Extract options
    const boundsOverride = options.bounds || options.min ? options : null;  // Support old and new format

    // Use bounds override if provided, otherwise calculate from triangles
    const bounds = boundsOverride || calculateBounds(triangles);

    if (boundsOverride) {
        // console.log(`[WebGPU Worker] Using bounds override: min(${bounds.min.x.toFixed(2)}, ${bounds.min.y.toFixed(2)}, ${bounds.min.z.toFixed(2)}) max(${bounds.max.x.toFixed(2)}, ${bounds.max.y.toFixed(2)}, ${bounds.max.z.toFixed(2)})`);

        // Validate bounds
        if (bounds.min.x >= bounds.max.x || bounds.min.y >= bounds.max.y || bounds.min.z >= bounds.max.z) {
            throw new Error(`Invalid bounds: min must be less than max. Got min(${bounds.min.x}, ${bounds.min.y}, ${bounds.min.z}) max(${bounds.max.x}, ${bounds.max.y}, ${bounds.max.z})`);
        }
    }

    const gridWidth = Math.ceil((bounds.max.x - bounds.min.x) / stepSize) + 1;
    const gridHeight = Math.ceil((bounds.max.y - bounds.min.y) / stepSize) + 1;
    const totalGridPoints = gridWidth * gridHeight;

    // console.log(`[WebGPU Worker] Grid: ${gridWidth}x${gridHeight} = ${totalGridPoints.toLocaleString()} points`);

    // Calculate buffer size based on filter mode
    // filterMode 0 (terrain): Dense Z-only output (1 float per grid cell)
    // filterMode 1 (tool): Sparse X,Y,Z output (3 floats per grid cell)
    const floatsPerPoint = filterMode === 0 ? 1 : 3;
    const outputSize = totalGridPoints * floatsPerPoint * 4;
    const maxBufferSize = device.limits.maxBufferSize || 268435456; // 256MB default
    const modeStr = filterMode === 0 ? 'terrain (dense Z-only)' : 'tool (sparse XYZ)';
    // console.log(`[WebGPU Worker] Output buffer size: ${(outputSize / 1024 / 1024).toFixed(2)} MB for ${modeStr} (max: ${(maxBufferSize / 1024 / 1024).toFixed(2)} MB)`);

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
    const outputBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

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
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0  // Extended for rotation_cos and rotation_sin
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

    // Set rotation (identity by default: cos=1, sin=0)
    const rotationAngleDeg = options.rotationAngleDeg ?? 0;
    const rotationAngleRad = rotationAngleDeg * Math.PI / 180;
    uniformDataF32[14] = Math.cos(rotationAngleRad);
    uniformDataF32[15] = Math.sin(rotationAngleRad);

    // Check for u32 overflow
    const maxU32 = 4294967295;
    if (gridWidth > maxU32 || gridHeight > maxU32) {
        throw new Error(`Grid dimensions exceed u32 max: ${gridWidth}x${gridHeight}`);
    }

    // console.log(`[WebGPU Worker] Uniforms: gridWidth=${gridWidth}, gridHeight=${gridHeight}, triangles=${triangles.length / 9}`);

    const uniformBuffer = device.createBuffer({
        size: uniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

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
    // console.log(`[WebGPU Worker] Dispatching ${workgroupsX}x${workgroupsY} workgroups (max per dim: ${maxWorkgroupsPerDim})`);

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

        // Count actual valid points for logging (sentinel value = -1e10)
        const EMPTY_CELL = -1e10;
        let validCount = 0;
        for (let i = 0; i < totalGridPoints; i++) {
            if (result[i] > EMPTY_CELL + 1) validCount++;  // Any value significantly above sentinel
        }
        // console.log(`[WebGPU Worker] Dense terrain: ${totalGridPoints} grid cells, ${validCount} with geometry (${(validCount/totalGridPoints*100).toFixed(1)}% coverage)`);
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
    // console.log(`[WebGPU Worker] ✅ Rasterize complete: ${pointCount} points in ${conversionTime.toFixed(1)}ms`);
    // console.log(`[WebGPU Worker] Bounds: min(${bounds.min.x.toFixed(2)}, ${bounds.min.y.toFixed(2)}, ${bounds.min.z.toFixed(2)}) max(${bounds.max.x.toFixed(2)}, ${bounds.max.y.toFixed(2)}, ${bounds.max.z.toFixed(2)})`);

    // Verify result data integrity
    if (filterMode === 0) {
        // Terrain: Dense Z-only format
        if (result.length > 0) {
            const EMPTY_CELL = -1e10;
            const firstZ = result[0] <= EMPTY_CELL + 1 ? 'EMPTY' : result[0].toFixed(3);
            const lastZ = result[result.length-1] <= EMPTY_CELL + 1 ? 'EMPTY' : result[result.length-1].toFixed(3);
            // console.log(`[WebGPU Worker] First Z: ${firstZ}, Last Z: ${lastZ}`);
        }
    } else {
        // Tool: Sparse X,Y,Z format
        if (result.length > 0) {
            const firstPoint = `(${result[0].toFixed(3)}, ${result[1].toFixed(3)}, ${result[2].toFixed(3)})`;
            const lastIdx = result.length - 3;
            const lastPoint = `(${result[lastIdx].toFixed(3)}, ${result[lastIdx+1].toFixed(3)}, ${result[lastIdx+2].toFixed(3)})`;
            // console.log(`[WebGPU Worker] First point: ${firstPoint}, Last point: ${lastPoint}`);
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
    console.log(`[WebGPU Worker] Dense terrain format: ${bytesPerPoint} bytes/point (was 16), can fit ${(maxPointsPerTile/1e6).toFixed(1)}M points per tile`);

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

    console.log(`[WebGPU Worker] Creating ${tilesX}x${tilesY} = ${tilesX * tilesY} tiles (${actualTileWidth.toFixed(2)}mm × ${actualTileHeight.toFixed(2)}mm each)`);
    console.log(`[WebGPU Worker] Tile grid: ${Math.ceil(actualTileWidth / stepSize)}x${Math.ceil(actualTileHeight / stepSize)} points per tile`);

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
        console.log(`[WebGPU Worker] Stitching ${tileResults.length} dense terrain tiles...`);

        // Calculate global grid dimensions
        const globalWidth = Math.ceil((fullBounds.max.x - fullBounds.min.x) / stepSize) + 1;
        const globalHeight = Math.ceil((fullBounds.max.y - fullBounds.min.y) / stepSize) + 1;
        const totalGridCells = globalWidth * globalHeight;

        // Allocate global dense grid (Z-only), initialize to sentinel value
        const EMPTY_CELL = -1e10;
        const globalGrid = new Float32Array(totalGridCells);
        globalGrid.fill(EMPTY_CELL);

        console.log(`[WebGPU Worker] Global grid: ${globalWidth}x${globalHeight} = ${totalGridCells.toLocaleString()} cells`);

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

        console.log(`[WebGPU Worker] ✅ Stitched: ${totalGridCells} total cells, ${validCount} with geometry (${(validCount/totalGridCells*100).toFixed(1)}% coverage)`);

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
        console.log(`[WebGPU Worker] Stitching ${tileResults.length} sparse tool tiles...`);

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

        console.log(`[WebGPU Worker] ✅ Stitched: ${finalPointCount} unique sparse points`);

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
        console.log('[WebGPU Worker] Tiling required - switching to tiled rasterization');

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
            console.log(`[WebGPU Worker] Processing tile ${i + 1}/${tiles.length}: ${tiles[i].id}`);
            console.log(`[WebGPU Worker]   Tile bounds: min(${tiles[i].bounds.min.x.toFixed(2)}, ${tiles[i].bounds.min.y.toFixed(2)}) max(${tiles[i].bounds.max.x.toFixed(2)}, ${tiles[i].bounds.max.y.toFixed(2)})`);

            const tileResult = await rasterizeMeshSingle(triangles, stepSize, filterMode, {
                ...tiles[i].bounds,
                rotationAngleDeg: options.rotationAngleDeg
            });

            const tileTime = performance.now() - tileStart;
            console.log(`[WebGPU Worker]   Tile ${i + 1} complete: ${tileResult.pointCount} points in ${tileTime.toFixed(1)}ms`);

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
    // console.log(`[WebGPU Worker] Terrain dense format: ${width}x${height} = ${points.length} cells`);

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
function createSparseToolFromPoints(points, gridStep) {
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
        const zValue = z - minZ; // Z relative to tool tip

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
    console.log('[WebGPU Worker] Generating toolpath...');
    console.log(`[WebGPU Worker] Input: terrain ${terrainPoints.length/3} points, tool ${toolPoints.length/3} points, steps (${xStep}, ${yStep}), oobZ ${oobZ}, gridStep ${gridStep}`);

    if (terrainBounds) {
        console.log(`[WebGPU Worker] Using terrain bounds: min(${terrainBounds.min.x.toFixed(2)}, ${terrainBounds.min.y.toFixed(2)}, ${terrainBounds.min.z.toFixed(2)}) max(${terrainBounds.max.x.toFixed(2)}, ${terrainBounds.max.y.toFixed(2)}, ${terrainBounds.max.z.toFixed(2)})`);
    }

    try {
        // Create height map from terrain points (use terrain bounds if provided)
        const terrainMapData = createHeightMapFromPoints(terrainPoints, gridStep, terrainBounds);
        console.log(`[WebGPU Worker] Created terrain map: ${terrainMapData.width}x${terrainMapData.height}`);

        // Create sparse tool representation
        const sparseToolData = createSparseToolFromPoints(toolPoints, gridStep);
        console.log(`[WebGPU Worker] Created sparse tool: ${sparseToolData.count} points`);

        // Run WebGPU compute
        const result = await runToolpathCompute(
            terrainMapData, sparseToolData, xStep, yStep, oobZ, startTime
        );

        return result;
    } catch (error) {
        console.error('[WebGPU Worker] Error generating toolpath:', error);
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

    console.log(`[WebGPU Worker] Output: ${pointsPerLine}x${numScanlines} = ${outputSize} points`);

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
    console.log(`[WebGPU Worker] ✅ Toolpath complete in ${(endTime - startTime).toFixed(1)}ms`);
    console.log(`[WebGPU Worker] Output: ${result.length} values (${numScanlines} scanlines × ${pointsPerLine} points)`);

    // Log sample of output for debugging
    const sampleSize = Math.min(10, result.length);
    console.log(`[WebGPU Worker] First ${sampleSize} values:`, Array.from(result.slice(0, sampleSize)));

    return {
        pathData: result,
        numScanlines,
        pointsPerLine,
        generationTime: endTime - startTime
    };
}

// Generate toolpath with tiling support (public API)
async function generateToolpath(terrainPoints, toolPoints, xStep, yStep, oobZ, gridStep, terrainBounds = null) {
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
    console.log('[WebGPU Worker] 🔲 Using tiled toolpath generation');
    console.log(`[WebGPU Worker] Terrain: DENSE (${terrainPoints.length} cells = ${outputWidth}x${outputHeight})`);
    console.log(`[WebGPU Worker] Tool dimensions: ${toolWidthMm.toFixed(2)}mm × ${toolHeightMm.toFixed(2)}mm (${toolWidthCells}×${toolHeightCells} cells)`);

    // Create tiles with tool-size overlap (pass dimensions in grid cells)
    const tiles = createToolpathTiles(terrainBounds, gridStep, xStep, yStep, toolWidthCells, toolHeightCells, maxSafeSize);
    console.log(`[WebGPU Worker] Created ${tiles.length} tiles`);

    // Process each tile
    const tileResults = [];
    let totalTileTime = 0;
    for (let i = 0; i < tiles.length; i++) {
        const tile = tiles[i];
        const tileStartTime = performance.now();
        console.log(`[WebGPU Worker] Processing tile ${i + 1}/${tiles.length}...`);

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

        // Extract terrain sub-grid for this tile (terrain is ALWAYS dense)
        const tileMinGridX = Math.floor((tile.bounds.min.x - terrainBounds.min.x) / gridStep);
        const tileMaxGridX = Math.ceil((tile.bounds.max.x - terrainBounds.min.x) / gridStep);
        const tileMinGridY = Math.floor((tile.bounds.min.y - terrainBounds.min.y) / gridStep);
        const tileMaxGridY = Math.ceil((tile.bounds.max.y - terrainBounds.min.y) / gridStep);

        const tileWidth = tileMaxGridX - tileMinGridX + 1;
        const tileHeight = tileMaxGridY - tileMinGridY + 1;
        const tileTerrainPoints = new Float32Array(tileWidth * tileHeight);

        console.log(`[WebGPU Worker] Tile ${i+1} dense extraction: ${tileWidth}x${tileHeight} from global ${outputWidth}x${outputHeight}`);

        // Copy relevant sub-grid from full terrain
        const EMPTY_CELL = -1e10;
        tileTerrainPoints.fill(EMPTY_CELL);

        for (let ty = 0; ty < tileHeight; ty++) {
            const globalY = tileMinGridY + ty;
            if (globalY < 0 || globalY >= outputHeight) continue;

            for (let tx = 0; tx < tileWidth; tx++) {
                const globalX = tileMinGridX + tx;
                if (globalX < 0 || globalX >= outputWidth) continue;

                const globalIdx = globalY * outputWidth + globalX;
                const tileIdx = ty * tileWidth + tx;
                tileTerrainPoints[tileIdx] = terrainPoints[globalIdx];
            }
        }

        // Generate toolpath for this tile
        const tileToolpathResult = await generateToolpathSingle(
            tileTerrainPoints,
            toolPoints,
            xStep,
            yStep,
            oobZ,
            gridStep,
            tile.bounds
        );

        const tileTime = performance.now() - tileStartTime;
        totalTileTime += tileTime;

        tileResults.push({
            pathData: tileToolpathResult.pathData,
            numScanlines: tileToolpathResult.numScanlines,
            pointsPerLine: tileToolpathResult.pointsPerLine,
            tile: tile
        });

        console.log(`[WebGPU Worker] Tile ${i + 1}/${tiles.length} complete: ${tileToolpathResult.numScanlines}×${tileToolpathResult.pointsPerLine} in ${tileTime.toFixed(1)}ms`);
    }

    console.log(`[WebGPU Worker] All tiles processed in ${totalTileTime.toFixed(1)}ms (avg ${(totalTileTime/tiles.length).toFixed(1)}ms per tile)`);

    // Stitch tiles together, dropping overlap regions
    const stitchStartTime = performance.now();
    const stitchedResult = stitchToolpathTiles(tileResults, terrainBounds, gridStep, xStep, yStep);
    const stitchTime = performance.now() - stitchStartTime;

    const totalTime = performance.now() - tilingStartTime;
    console.log(`[WebGPU Worker] Stitching took ${stitchTime.toFixed(1)}ms`);
    console.log(`[WebGPU Worker] ✅ Tiled toolpath complete: ${stitchedResult.numScanlines}×${stitchedResult.pointsPerLine} in ${totalTime.toFixed(1)}ms total`);

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

    console.log(`[WebGPU Worker] Creating ${tilesX}×${tilesY} tiles (${coreGridWidth}×${coreGridHeight} cells core + ${toolOverlapX}×${toolOverlapY} cells overlap)`);

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

    return tiles;
}

// Stitch toolpath tiles together, dropping overlap regions (using integer grid coordinates)
function stitchToolpathTiles(tileResults, globalBounds, gridStep, xStep, yStep) {
    // Calculate global output dimensions
    const globalWidth = Math.ceil((globalBounds.max.x - globalBounds.min.x) / gridStep) + 1;
    const globalHeight = Math.ceil((globalBounds.max.y - globalBounds.min.y) / gridStep) + 1;
    const globalPointsPerLine = Math.ceil(globalWidth / xStep);
    const globalNumScanlines = Math.ceil(globalHeight / yStep);

    console.log(`[WebGPU Worker] Stitching toolpath: global grid ${globalWidth}x${globalHeight}, output ${globalPointsPerLine}x${globalNumScanlines}`);

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

        console.log(`[WebGPU Worker]   Tile ${tile.id}: copied ${copiedCount} values`);
    }

    // Count how many output values are still NaN (gaps)
    let nanCount = 0;
    for (let i = 0; i < result.length; i++) {
        if (isNaN(result[i])) nanCount++;
    }
    console.log(`[WebGPU Worker] Stitching complete: ${result.length} total values, ${nanCount} still NaN`);

    return {
        pathData: result,
        numScanlines: globalNumScanlines,
        pointsPerLine: globalPointsPerLine,
        generationTime: 0 // Sum from tiles if needed
    };
}

function generateRadialScanline(data) {
    const { stripPositions, stripBounds, toolPositions, xStep, zFloor, gridStep } = data;
    const EMPTY_CELL = -1e10;

    // console.log('[WebGPU Worker] Generating radial scanline...');
    // console.log(`[WebGPU Worker] Strip: ${stripPositions.length} cells, Tool: ${toolPositions.length/3} points, xStep: ${xStep}`);

    // Create height map from strip (dense Z-only format)
    const stripMap = createHeightMapFromPoints(stripPositions, gridStep, stripBounds);
    // console.log(`[WebGPU Worker] Strip map: ${stripMap.width}x${stripMap.height}`);

    // Create sparse tool representation
    const sparseTool = createSparseToolFromPoints(toolPositions, gridStep);
    // console.log(`[WebGPU Worker] Sparse tool: ${sparseTool.count} points`);

    // Find center row (y=0 in world space)
    const centerY = Math.round((0 - stripMap.minY) / gridStep);

    if (centerY < 0 || centerY >= stripMap.height) {
        console.error(`[WebGPU Worker] Center row ${centerY} out of bounds [0, ${stripMap.height})`);
        throw new Error('Center row out of strip bounds');
    }

    // console.log(`[WebGPU Worker] Center row: ${centerY} (y=0)`);

    // Calculate output dimensions
    const outputWidth = Math.ceil(stripMap.width / xStep);
    const scanline = new Float32Array(outputWidth);

    // Process each output position along X-axis
    for (let outX = 0; outX < outputWidth; outX++) {
        const gridX = outX * xStep;

        if (gridX >= stripMap.width) break;

        // Find maximum tool-terrain collision
        // Note: Don't skip if center is empty - tool edges might still touch terrain
        let maxZ = zFloor;

        for (let i = 0; i < sparseTool.count; i++) {
            const toolOffsetX = sparseTool.xOffsets[i];
            const toolOffsetY = sparseTool.yOffsets[i];
            const toolZ = sparseTool.zValues[i];

            // Calculate terrain lookup position
            const terrainGridX = gridX + toolOffsetX;
            const terrainGridY = centerY + toolOffsetY;

            // Check bounds
            if (terrainGridX < 0 || terrainGridX >= stripMap.width ||
                terrainGridY < 0 || terrainGridY >= stripMap.height) {
                continue;
            }

            // Get terrain height at this position
            const idx = terrainGridY * stripMap.width + terrainGridX;
            const terrainHeight = stripMap.grid[idx];

            if (terrainHeight === EMPTY_CELL) continue;

            // Calculate tool center Z needed to just touch terrain
            const toolCenterZ = terrainHeight - toolZ;
            maxZ = Math.max(maxZ, toolCenterZ);
        }

        scanline[outX] = maxZ;
    }

    // console.log(`[WebGPU Worker] ✅ Radial scanline complete: ${outputWidth} points`);
    // console.log(`[WebGPU Worker] First 10 values: ${Array.from(scanline.slice(0, 10)).map(v => v.toFixed(2)).join(',')}`);

    return {
        scanline
    };
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
                console.log('[WebGPU Worker] Config updated:', config);
                break;

            case 'rasterize':
                const { triangles, stepSize, filterMode, isForTool, boundsOverride, rotationAngleDeg } = data;
                const rasterOptions = boundsOverride ? { ...boundsOverride, rotationAngleDeg } : { rotationAngleDeg };
                const rasterResult = await rasterizeMesh(triangles, stepSize, filterMode, rasterOptions);
                self.postMessage({
                    type: 'rasterize-complete',
                    data: rasterResult,
                    isForTool: isForTool || false // Pass through the flag
                }, [rasterResult.positions.buffer]);
                break;

            case 'generate-toolpath':
                const { terrainPositions, toolPositions, xStep, yStep, zFloor, gridStep, terrainBounds } = data;
                const toolpathResult = await generateToolpath(
                    terrainPositions, toolPositions, xStep, yStep, zFloor, gridStep, terrainBounds
                );
                self.postMessage({
                    type: 'toolpath-complete',
                    data: toolpathResult
                }, [toolpathResult.pathData.buffer]);
                break;

            case 'generate-radial-scanline':
                const scanlineResult = generateRadialScanline(data);
                self.postMessage({
                    type: 'radial-scanline-complete',
                    data: scanlineResult
                }, [scanlineResult.scanline.buffer]);
                break;

            default:
                self.postMessage({
                    type: 'error',
                    message: 'Unknown message type: ' + type
                });
        }
    } catch (error) {
        console.error('[WebGPU Worker] Error:', error);
        self.postMessage({
            type: 'error',
            message: error.message,
            stack: error.stack
        });
    }
};

// Initialize on load
initWebGPU();
