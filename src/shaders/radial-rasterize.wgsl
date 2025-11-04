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
    rotation_offset_radians: f32,
    padding: f32,
    grid_width: u32,
    grid_height: u32,
    num_triangles: u32,
    num_attention_words: u32,
}

@group(0) @binding(0) var<storage, read> triangles: array<f32>;
@group(0) @binding(1) var<storage, read> compactTriangleIndices: array<u32>;
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

    // Allow tolerance for edges/vertices to ensure watertight coverage
    // Larger epsilon needed because near-parallel triangles (small 'a') amplify errors via f=1/a
    let EPSILON = 0.0001;
    if (u < -EPSILON || u > 1.0 + EPSILON) {
        return -1.0;
    }

    let q = cross(s, edge1);
    let v = f * dot(rayDir, q);

    // Allow tolerance for edges/vertices to ensure watertight coverage
    if (v < -EPSILON || u + v > 1.0 + EPSILON) {
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

    // Calculate angle from Y grid position with optional rotation offset
    let theta = f32(gy) * uniforms.rotation_step_radians + uniforms.rotation_offset_radians;

    // Ray origin: start along X-axis (in YZ plane at origin)
    let cosTheta = cos(theta);
    let sinTheta = sin(theta);
    let rayOrigin = vec3f(wx, 0.0, 0.0);

    // Ray direction: point outward radially in YZ plane (negate sinTheta for clockwise from +Y)
    let rayDirOut = vec3f(0.0, cosTheta, -sinTheta);

    // Find FARTHEST intersection in outward direction (positive distances)
    var bestRadius = -1e10;  // Start with very negative (worst case)
    var foundHit = false;

    // Iterate over compact list of marked triangles (no bit checking needed!)
    for (var compactIdx = 0u; compactIdx < uniforms.num_triangles; compactIdx++) {
        // Get actual triangle index from compact list
        let triIdx = compactTriangleIndices[compactIdx];

        // Test triangle
        let base = triIdx * 9u;
        let v0 = vec3f(triangles[base + 0u], triangles[base + 1u], triangles[base + 2u]);
        let v1 = vec3f(triangles[base + 3u], triangles[base + 4u], triangles[base + 5u]);
        let v2 = vec3f(triangles[base + 6u], triangles[base + 7u], triangles[base + 8u]);

        let t_hit = rayTriangleIntersect(rayOrigin, rayDirOut, v0, v1, v2);

        if (t_hit > 0.0) {
            // Positive distance - geometry is outward from axis
            let hitY = rayOrigin.y + t_hit * rayDirOut.y;
            let hitZ = rayOrigin.z + t_hit * rayDirOut.z;
            let radius = sqrt(hitY * hitY + hitZ * hitZ);

            if (radius > bestRadius) {
                bestRadius = radius;
                foundHit = true;
            }
        }
    }

    // Also check inward direction (for lathe ops through center)
    // These will have negative conceptual distances (inside the axis)
    let rayDirIn = vec3f(0.0, -cosTheta, sinTheta);

    if (foundHit == false) {
    for (var compactIdx = 0u; compactIdx < uniforms.num_triangles; compactIdx++) {
        // Get actual triangle index from compact list
        let triIdx = compactTriangleIndices[compactIdx];

        // Test triangle
        let base = triIdx * 9u;
        let v0 = vec3f(triangles[base + 0u], triangles[base + 1u], triangles[base + 2u]);
        let v1 = vec3f(triangles[base + 3u], triangles[base + 4u], triangles[base + 5u]);
        let v2 = vec3f(triangles[base + 6u], triangles[base + 7u], triangles[base + 8u]);

        let t_hit = rayTriangleIntersect(rayOrigin, rayDirIn, v0, v1, v2);

        if (t_hit > 0.0) {
            // Ray found geometry in inward direction
            // Calculate position (will be on opposite side of axis)
            let hitY = rayOrigin.y + t_hit * rayDirIn.y;
            let hitZ = rayOrigin.z + t_hit * rayDirIn.z;
            let radius = -sqrt(hitY * hitY + hitZ * hitZ);  // Negative for inward

            if (radius > bestRadius) {
                bestRadius = radius;
                foundHit = true;
            }
        }
    }
    }

    // Calculate output
    if (foundHit) {
        // Apply floor
        output[idx] = max(bestRadius, uniforms.z_floor);
    } else {
        output[idx] = EMPTY_CELL;
    }
}
