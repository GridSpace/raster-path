// Workload Calibration Shader
// Tests GPU watchdog limits by doing configurable amount of work per thread

struct Uniforms {
    workgroup_size_x: u32,
    workgroup_size_y: u32,
    workgroup_size_z: u32,
    triangle_tests: u32,  // How many intersection tests to run
}

@group(0) @binding(0) var<storage, read_write> completion_flags: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

// Ray-triangle intersection using Möller-Trumbore algorithm
// This is the actual production code - same ALU/cache characteristics
fn ray_triangle_intersect(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>
) -> vec2<f32> {  // Returns (hit: 0.0 or 1.0, z: intersection_z)
    let EPSILON = 0.0001;

    // Calculate edges
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;

    // Cross product: ray_dir × edge2
    let h = cross(ray_dir, edge2);

    // Dot product: edge1 · h
    let a = dot(edge1, h);

    // Check if ray is parallel to triangle
    if (abs(a) < EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }

    let f = 1.0 / a;
    let s = ray_origin - v0;
    let u = f * dot(s, h);

    // Check if intersection is outside triangle (u parameter)
    if (u < 0.0 || u > 1.0) {
        return vec2<f32>(0.0, 0.0);
    }

    let q = cross(s, edge1);
    let v = f * dot(ray_dir, q);

    // Check if intersection is outside triangle (v parameter)
    if (v < 0.0 || u + v > 1.0) {
        return vec2<f32>(0.0, 0.0);
    }

    // Calculate intersection point along ray
    let t = f * dot(edge2, q);

    if (t > EPSILON) {
        // Ray hit triangle
        let intersection_z = ray_origin.z + t * ray_dir.z;
        return vec2<f32>(1.0, intersection_z);
    }

    return vec2<f32>(0.0, 0.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_index = global_id.z * (uniforms.workgroup_size_x * uniforms.workgroup_size_y) +
                       global_id.y * uniforms.workgroup_size_x +
                       global_id.x;

    // Synthetic triangle vertices (deterministic, no memory reads needed)
    let v0 = vec3<f32>(0.0, 0.0, 0.0);
    let v1 = vec3<f32>(1.0, 0.0, 0.0);
    let v2 = vec3<f32>(0.5, 1.0, 0.0);

    // Ray parameters based on thread ID (deterministic)
    let ray_origin = vec3<f32>(
        f32(global_id.x) * 0.1,
        f32(global_id.y) * 0.1,
        10.0
    );
    let ray_dir = vec3<f32>(0.0, 0.0, -1.0);

    // Perform N intersection tests (configurable workload)
    var hit_count = 0u;
    for (var i = 0u; i < uniforms.triangle_tests; i++) {
        // Slightly vary triangle vertices to prevent compiler optimization
        let offset = f32(i) * 0.001;
        let v0_offset = v0 + vec3<f32>(offset, 0.0, 0.0);
        let v1_offset = v1 + vec3<f32>(0.0, offset, 0.0);
        let v2_offset = v2 + vec3<f32>(offset, offset, 0.0);

        let result = ray_triangle_intersect(ray_origin, ray_dir, v0_offset, v1_offset, v2_offset);
        if (result.x > 0.5) {
            hit_count += 1u;
        }
    }

    // Write completion flag (1 = thread completed all work)
    // If this thread was killed by watchdog, this write never happens (stays 0)
    completion_flags[thread_index] = 1u;
}
