// Radial Pass 1: Triangle culling by X-range (bit-attention)
struct Uniforms {
    tile_min_x: f32,
    tile_max_x: f32,
    num_triangles: u32,
    padding: u32,
}

@group(0) @binding(0) var<storage, read> triangles: array<f32>;
@group(0) @binding(1) var<storage, read_write> attentionBits: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let triIdx = global_id.x;
    if (triIdx >= uniforms.num_triangles) {
        return;
    }

    // Get triangle X coordinates
    let base = triIdx * 9u;
    let v0x = triangles[base + 0u];
    let v1x = triangles[base + 3u];
    let v2x = triangles[base + 6u];

    // Check if triangle overlaps X range
    let allLeft = v0x < uniforms.tile_min_x && v1x < uniforms.tile_min_x && v2x < uniforms.tile_min_x;
    let allRight = v0x > uniforms.tile_max_x && v1x > uniforms.tile_max_x && v2x > uniforms.tile_max_x;

    if (!allLeft && !allRight) {
        // Mark this triangle
        let wordIdx = triIdx / 32u;
        let bitIdx = triIdx % 32u;
        atomicOr(&attentionBits[wordIdx], 1u << bitIdx);
    }
}
