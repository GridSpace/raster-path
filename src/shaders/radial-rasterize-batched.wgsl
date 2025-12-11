// Radial V3 batched bucket rasterization
// Processes ALL buckets in one dispatch - GPU threads find their bucket

const EPSILON: f32 = 0.0001;

struct Uniforms {
    resolution: f32,         // Grid step size (mm)
    tool_radius: f32,        // Tool radius for Y-filtering
    full_grid_width: u32,    // Full grid width (all buckets)
    grid_height: u32,        // Number of Y cells
    global_min_x: f32,       // Global minimum X coordinate
    bucket_min_y: f32,       // Y-axis start (typically -tool_width/2)
    z_floor: f32,            // Z value for empty cells
    num_buckets: u32,        // Number of buckets
}

struct BucketInfo {
    min_x: f32,              // Bucket X range start
    max_x: f32,              // Bucket X range end
    start_index: u32,        // Index into triangle_indices array
    count: u32,              // Number of triangles in this bucket
}

@group(0) @binding(0) var<storage, read> rotated_triangles: array<f32>;   // ALL rotated triangles + bounds
@group(0) @binding(1) var<storage, read_write> output: array<f32>;        // Full-width output grid
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read> all_buckets: array<BucketInfo>;  // All bucket descriptors
@group(0) @binding(4) var<storage, read> triangle_indices: array<u32>;    // All triangle indices

// Simplified ray-triangle intersection for downward rays
fn ray_triangle_intersect_downward(
    ray_origin: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>
) -> vec2<f32> {  // Returns (hit: 0.0 or 1.0, t: distance along ray)
    let ray_dir = vec3<f32>(0.0, 0.0, -1.0);

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(ray_dir, edge2);
    let a = dot(edge1, h);

    if (a > -EPSILON && a < EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }

    let f = 1.0 / a;
    let s = ray_origin - v0;
    let u = f * dot(s, h);

    if (u < -EPSILON || u > 1.0 + EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }

    let q = cross(s, edge1);
    let v = f * dot(ray_dir, q);

    if (v < -EPSILON || u + v > 1.0 + EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }

    let t = f * dot(edge2, q);

    if (t > EPSILON) {
        return vec2<f32>(1.0, t);
    }

    return vec2<f32>(0.0, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_x = global_id.x;
    let grid_y = global_id.y;

    // Bounds check
    if (grid_x >= uniforms.full_grid_width || grid_y >= uniforms.grid_height) {
        return;
    }

    // Calculate world position
    let world_x = uniforms.global_min_x + f32(grid_x) * uniforms.resolution;
    let world_y = uniforms.bucket_min_y + f32(grid_y) * uniforms.resolution;

    // FIND WHICH BUCKET THIS X POSITION BELONGS TO
    // Simple linear search (could be binary search for many buckets)
    var bucket_idx = 0u;
    var found_bucket = false;
    for (var i = 0u; i < uniforms.num_buckets; i++) {
        if (world_x >= all_buckets[i].min_x && world_x < all_buckets[i].max_x) {
            bucket_idx = i;
            found_bucket = true;
            break;
        }
    }

    // If not in any bucket, write floor and return
    if (!found_bucket) {
        let output_idx = grid_y * uniforms.full_grid_width + grid_x;
        output[output_idx] = uniforms.z_floor;
        return;
    }

    let bucket = all_buckets[bucket_idx];

    // Fixed downward ray from high above
    let ray_origin = vec3<f32>(world_x, world_y, 1000.0);

    // Track best (closest) hit
    var best_z = uniforms.z_floor;

    // Test triangles in this bucket with Y-bounds filtering
    for (var i = 0u; i < bucket.count; i++) {
        // Get triangle index from bucket's index array
        let tri_idx = triangle_indices[bucket.start_index + i];

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
            }
        }
    }

    // Write to FULL-WIDTH output (no stitching needed!)
    let output_idx = grid_y * uniforms.full_grid_width + grid_x;
    output[output_idx] = best_z;
}
