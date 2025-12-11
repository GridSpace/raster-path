// Radial V3 rasterization with Y-filtering
// Uses pre-rotated triangles with Y-bounds for fast spatial culling

const EPSILON: f32 = 0.0001;

struct Uniforms {
    resolution: f32,         // Grid step size (mm)
    tool_radius: f32,        // Tool radius for Y-filtering
    grid_width: u32,         // Number of X cells
    grid_height: u32,        // Number of Y cells
    bucket_min_x: f32,       // Bucket X-axis start
    bucket_min_y: f32,       // Y-axis start (typically 0 or -tool_width/2)
    z_floor: f32,            // Z value for empty cells
    num_triangles: u32,      // Number of triangles in this bucket
    padding: u32,            // Alignment padding
}

@group(0) @binding(0) var<storage, read> rotated_triangles: array<f32>;   // Pre-rotated triangles + bounds (ALL triangles)
@group(0) @binding(1) var<storage, read_write> output: array<f32>;        // Output: dense terrain grid
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read> triangle_indices: array<u32>;    // Indices of triangles to test (bucket subset)

// Simplified ray-triangle intersection for downward rays
// Ray direction is always (0, 0, -1), so we can optimize
fn ray_triangle_intersect_downward(
    ray_origin: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>
) -> vec2<f32> {  // Returns (hit: 0.0 or 1.0, t: distance along ray)
    // Ray direction is (0, 0, -1)
    let ray_dir = vec3<f32>(0.0, 0.0, -1.0);

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

    if (u < -EPSILON || u > 1.0 + EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }

    // Cross product: s × edge1
    let q = cross(s, edge1);

    // v = f * (ray_dir · q)
    let v = f * dot(ray_dir, q);

    if (v < -EPSILON || u + v > 1.0 + EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }

    // t = f * (edge2 · q)
    let t = f * dot(edge2, q);

    if (t > EPSILON) {
        // Intersection found - return distance along ray
        return vec2<f32>(1.0, t);
    }

    return vec2<f32>(0.0, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_x = global_id.x;
    let grid_y = global_id.y;

    // Bounds check
    if (grid_x >= uniforms.grid_width || grid_y >= uniforms.grid_height) {
        return;
    }

    // Calculate world position
    let world_x = uniforms.bucket_min_x + f32(grid_x) * uniforms.resolution;
    let world_y = uniforms.bucket_min_y + f32(grid_y) * uniforms.resolution;

    // Fixed downward ray from high above
    let ray_origin = vec3<f32>(world_x, world_y, 1000.0);  // Start at Z=1000

    // Track best (closest) hit
    var best_z = uniforms.z_floor;
    var found = false;

    // Test triangles in this bucket with Y-bounds filtering
    for (var i = 0u; i < uniforms.num_triangles; i++) {
        // Get triangle index from bucket's index array
        let tri_idx = triangle_indices[i];

        // Read Y-bounds first (cheaper than reading all vertices)
        let base = tri_idx * 11u;
        let y_min = rotated_triangles[base + 9u];
        let y_max = rotated_triangles[base + 10u];

        // Y-bounds check: skip triangles that don't overlap this ray's Y position
        if (y_max < world_y - uniforms.tool_radius ||
            y_min > world_y + uniforms.tool_radius) {
            continue;
        }

        // Read rotated vertices
        let v0 = vec3<f32>(
            rotated_triangles[base],
            rotated_triangles[base + 1u],
            rotated_triangles[base + 2u]
        );
        let v1 = vec3<f32>(
            rotated_triangles[base + 3u],
            rotated_triangles[base + 4u],
            rotated_triangles[base + 5u]
        );
        let v2 = vec3<f32>(
            rotated_triangles[base + 6u],
            rotated_triangles[base + 7u],
            rotated_triangles[base + 8u]
        );

        let result = ray_triangle_intersect_downward(ray_origin, v0, v1, v2);
        let hit = result.x;
        let t = result.y;

        if (hit > 0.5) {
            // Calculate Z position of intersection
            let hit_z = ray_origin.z - t;

            // Keep highest (max Z) hit
            if (hit_z > best_z) {
                best_z = hit_z;
                found = true;
            }
        }
    }

    // Write output
    let output_idx = grid_y * uniforms.grid_width + grid_x;
    output[output_idx] = best_z;
}
