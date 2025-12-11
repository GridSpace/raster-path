// Triangle rotation shader for radial rasterization V3
// Rotates all triangles in a bucket by a single angle and computes Y-bounds

struct Uniforms {
    angle: f32,              // Rotation angle in radians
    num_triangles: u32,      // Number of triangles to rotate
}

struct RotatedTriangle {
    v0: vec3<f32>,          // Rotated vertex 0
    v1: vec3<f32>,          // Rotated vertex 1
    v2: vec3<f32>,          // Rotated vertex 2
    y_min: f32,             // Minimum Y coordinate (for filtering)
    y_max: f32,             // Maximum Y coordinate (for filtering)
}

@group(0) @binding(0) var<storage, read> triangles: array<f32>;           // Input: original triangles (flat array)
@group(0) @binding(1) var<storage, read_write> rotated: array<f32>;       // Output: rotated triangles + bounds
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

// Rotate a point around X-axis
fn rotate_around_x(p: vec3<f32>, angle: f32) -> vec3<f32> {
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    return vec3<f32>(
        p.x,
        p.y * cos_a - p.z * sin_a,
        p.y * sin_a + p.z * cos_a
    );
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tri_idx = global_id.x;

    if (tri_idx >= uniforms.num_triangles) {
        return;
    }

    // Read original triangle vertices
    let base = tri_idx * 9u;
    let v0 = vec3<f32>(triangles[base], triangles[base + 1u], triangles[base + 2u]);
    let v1 = vec3<f32>(triangles[base + 3u], triangles[base + 4u], triangles[base + 5u]);
    let v2 = vec3<f32>(triangles[base + 6u], triangles[base + 7u], triangles[base + 8u]);

    // Rotate vertices around X-axis
    let v0_rot = rotate_around_x(v0, uniforms.angle);
    let v1_rot = rotate_around_x(v1, uniforms.angle);
    let v2_rot = rotate_around_x(v2, uniforms.angle);

    // Compute Y bounds for fast filtering during rasterization
    let y_min = min(v0_rot.y, min(v1_rot.y, v2_rot.y));
    let y_max = max(v0_rot.y, max(v1_rot.y, v2_rot.y));

    // Write rotated triangle + bounds
    // Layout: 11 floats per triangle: v0(3), v1(3), v2(3), y_min(1), y_max(1)
    let out_base = tri_idx * 11u;
    rotated[out_base] = v0_rot.x;
    rotated[out_base + 1u] = v0_rot.y;
    rotated[out_base + 2u] = v0_rot.z;
    rotated[out_base + 3u] = v1_rot.x;
    rotated[out_base + 4u] = v1_rot.y;
    rotated[out_base + 5u] = v1_rot.z;
    rotated[out_base + 6u] = v2_rot.x;
    rotated[out_base + 7u] = v2_rot.y;
    rotated[out_base + 8u] = v2_rot.z;
    rotated[out_base + 9u] = y_min;
    rotated[out_base + 10u] = y_max;
}
