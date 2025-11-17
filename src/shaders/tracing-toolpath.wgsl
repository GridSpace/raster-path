// Tracing toolpath generation
// Follows input polylines and calculates Z-depth at each sampled point
// Sentinel value for empty terrain cells (must match rasterize shader)
const EMPTY_CELL: f32 = -1e10;
const MAX_F32: f32 = 3.402823466e+38;

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
    point_count: u32,        // Number of sampled points to process
    path_index: u32,         // Index of current path being processed
    terrain_min_x: f32,      // Terrain bounding box (world coordinates)
    terrain_min_y: f32,
    grid_step: f32,          // Resolution of terrain rasterization
    oob_z: f32,              // Z value for out-of-bounds points (zFloor)
}

@group(0) @binding(0) var<storage, read> terrain_map: array<f32>;
@group(0) @binding(1) var<storage, read> sparse_tool: array<SparseToolPoint>;
@group(0) @binding(2) var<storage, read> input_points: array<f32>;  // XY pairs
@group(0) @binding(3) var<storage, read_write> output_depths: array<f32>;  // Z values
@group(0) @binding(4) var<storage, read_write> max_z_buffer: array<atomic<i32>>;  // Max Z per path (as bits)
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point_idx = global_id.x;

    if (point_idx >= uniforms.point_count) {
        return;
    }

    // Read input X,Y world coordinates
    let world_x = input_points[point_idx * 2u + 0u];
    let world_y = input_points[point_idx * 2u + 1u];

    // Convert world coordinates to grid coordinates
    let grid_x_f32 = (world_x - uniforms.terrain_min_x) / uniforms.grid_step;
    let grid_y_f32 = (world_y - uniforms.terrain_min_y) / uniforms.grid_step;
    let tool_center_x = i32(grid_x_f32);
    let tool_center_y = i32(grid_y_f32);

    // Check if tool center is outside terrain bounds
    let center_oob = tool_center_x < 0 || tool_center_x >= i32(uniforms.terrain_width) ||
                     tool_center_y < 0 || tool_center_y >= i32(uniforms.terrain_height);

    var max_collision_z = uniforms.oob_z;
    var found_collision = false;

    // Test each tool point for collision with terrain
    for (var i = 0u; i < uniforms.tool_count; i++) {
        let tool_point = sparse_tool[i];
        let terrain_x = tool_center_x + tool_point.x_offset;
        let terrain_y = tool_center_y + tool_point.y_offset;

        // Bounds check: terrain sample must be within terrain grid
        if (terrain_x < 0 || terrain_x >= i32(uniforms.terrain_width) ||
            terrain_y < 0 || terrain_y >= i32(uniforms.terrain_height)) {
            continue;
        }

        let terrain_idx = u32(terrain_y) * uniforms.terrain_width + u32(terrain_x);
        let terrain_z = terrain_map[terrain_idx];

        // Check if terrain cell has geometry (not empty sentinel value)
        if (terrain_z > EMPTY_CELL + 1.0) {
            // Tool z_value is positive offset from tip (tip=0, shaft=+50)
            // Add to terrain height to find where tool center needs to be
            let collision_z = terrain_z + tool_point.z_value;
            max_collision_z = max(max_collision_z, collision_z);
            found_collision = true;
        }
    }

    // If no collision found and center was in-bounds, use oob_z
    var output_z = uniforms.oob_z;
    if (found_collision) {
        output_z = max_collision_z;
    }

    output_depths[point_idx] = output_z;

    // Update max Z for this path using atomic operation
    // Convert float to int bits for atomic comparison
    let z_bits = bitcast<i32>(output_z);
    atomicMax(&max_z_buffer[uniforms.path_index], z_bits);
}
