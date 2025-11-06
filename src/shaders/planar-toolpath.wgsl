// Planar toolpath generation
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
    x_step: u32,
    y_step: u32,
    oob_z: f32,
    points_per_line: u32,
    num_scanlines: u32,
    y_offset: u32,  // Offset to center Y position (for single-scanline radial mode)
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
    let tool_center_y = i32(scanline * uniforms.y_step) + i32(uniforms.y_offset);

    // var max_collision_z = -MAX_F32;  // Track maximum collision height
    var max_collision_z = uniforms.oob_z;  // Track maximum collision height
    var found_collision = false;

    for (var i = 0u; i < uniforms.tool_count; i++) {
        let tool_point = sparse_tool[i];
        let terrain_x = tool_center_x + tool_point.x_offset;
        let terrain_y = tool_center_y + tool_point.y_offset;

        // Bounds check: X must always be in bounds
        if (terrain_x < 0 || terrain_x >= i32(uniforms.terrain_width)) {
            continue;
        }

        // Bounds check: Y must be within terrain strip bounds
        // For single-scanline OUTPUT mode, the tool center is at Y=0 but the terrain
        // strip contains the full Y range (tool width), so tool offsets access different Y rows
        if (terrain_y < 0 || terrain_y >= i32(uniforms.terrain_height)) {
            continue;
        }

        let terrain_idx = u32(terrain_y) * uniforms.terrain_width + u32(terrain_x);
        let terrain_z = terrain_map[terrain_idx];

        // Check if terrain cell has geometry (not empty sentinel value)
        if (terrain_z > EMPTY_CELL + 1.0) {
            // Tool z_value is positive offset from tip (tip=0, shaft=+50)
            // Subtract from terrain to find where tool center needs to be
            let collision_z = terrain_z + tool_point.z_value;
            max_collision_z = max(max_collision_z, collision_z);
            found_collision = true;
        }
    }

    var output_z = uniforms.oob_z;
    if (found_collision) {
        output_z = max_collision_z;
    }

    let output_idx = scanline * uniforms.points_per_line + point_idx;
    output_path[output_idx] = output_z;
}
