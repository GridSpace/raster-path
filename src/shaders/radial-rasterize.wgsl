// Radial Pass 2: Rasterization with bit-attention
const EMPTY_CELL: f32 = -1e10;
const PI: f32 = 3.14159265359;

struct Uniforms {
    bounds_min_x: f32,
    bounds_max_x: f32,
    max_radius: f32,
    rotation_step_radians: f32,
    step_size: f32,
    z_floor: f32,
    grid_width: u32,
    grid_height: u32,
    num_triangles: u32,
    num_attention_words: u32,
}

@group(0) @binding(0) var<storage, read> triangles: array<f32>;
@group(0) @binding(1) var<storage, read> attentionBits: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// Ray-triangle intersection (Möller–Trumbore algorithm)
fn rayTriangleIntersect(
    rayOrigin: vec3f,
    rayDir: vec3f,
    v0: vec3f,
    v1: vec3f,
    v2: vec3f
) -> f32 {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(rayDir, edge2);
    let a = dot(edge1, h);

    // Ray parallel to triangle
    if (abs(a) < 1e-8) {
        return -1.0;
    }

    let f = 1.0 / a;
    let s = rayOrigin - v0;
    let u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }

    let q = cross(s, edge1);
    let v = f * dot(rayDir, q);

    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }

    let t = f * dot(edge2, q);

    if (t > 1e-8) {
        return t;
    }

    return -1.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gx = global_id.x;
    let gy = global_id.y;

    if (gx >= uniforms.grid_width || gy >= uniforms.grid_height) {
        return;
    }

    let idx = gy * uniforms.grid_width + gx;

    // Calculate world X position
    let wx = uniforms.bounds_min_x + f32(gx) * uniforms.step_size;

    // Calculate angle from Y grid position
    let theta = f32(gy) * uniforms.rotation_step_radians;

    // Ray origin: start at X-axis (origin in YZ plane)
    let cosTheta = cos(theta);
    let sinTheta = sin(theta);
    let rayOrigin = vec3f(wx, 0.0, 0.0);

    // Ray direction: point outward from X-axis (normalized)
    let rayDir = vec3f(0.0, cosTheta, sinTheta);

    // Find FARTHEST intersection (only check marked triangles)
    var farthestT = 0.0;
    var foundHit = false;

    for (var triIdx = 0u; triIdx < uniforms.num_triangles; triIdx++) {
        // Check attention bit
        let wordIdx = triIdx / 32u;
        let bitIdx = triIdx % 32u;
        let word = attentionBits[wordIdx];
        let isMarked = (word & (1u << bitIdx)) != 0u;

        if (!isMarked) {
            continue; // Skip unmarked triangles
        }

        // Test triangle
        let base = triIdx * 9u;
        let v0 = vec3f(triangles[base + 0u], triangles[base + 1u], triangles[base + 2u]);
        let v1 = vec3f(triangles[base + 3u], triangles[base + 4u], triangles[base + 5u]);
        let v2 = vec3f(triangles[base + 6u], triangles[base + 7u], triangles[base + 8u]);

        let t_hit = rayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2);

        if (t_hit > 0.0 && t_hit > farthestT) {
            farthestT = t_hit;
            foundHit = true;
        }
    }

    // Calculate output
    if (foundHit) {
        let hitY = rayOrigin.y + farthestT * rayDir.y;
        let hitZ = rayOrigin.z + farthestT * rayDir.z;
        let radius = sqrt(hitY * hitY + hitZ * hitZ);

        // Apply floor
        output[idx] = max(radius, uniforms.z_floor);
    } else {
        output[idx] = EMPTY_CELL;
    }
}
