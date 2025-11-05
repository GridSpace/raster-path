// Radial V2 rasterization with X-bucketing and rotating ray planes
// Sentinel value for empty cells (far below any real geometry)
const EMPTY_CELL: f32 = -1e10;
const PI: f32 = 3.14159265359;

struct Uniforms {
    resolution: f32,           // Grid step size (mm)
    angle_step: f32,           // Radians between angles
    num_angles: u32,           // Total number of angular strips
    max_radius: f32,           // Ray origin distance from X-axis (maxHypot * 1.01)
    tool_width: f32,           // Tool width in mm
    grid_y_height: u32,        // Tool width in pixels (toolWidth / resolution)
    bucket_width: f32,         // Width of each X-bucket (mm)
    bucket_grid_width: u32,    // Bucket width in pixels
    global_min_x: f32,         // Global minimum X coordinate
    z_floor: f32,              // Z value for empty cells
    filter_mode: u32,          // 0 = max Z (terrain), 1 = min Z (tool)
    num_buckets: u32,          // Total number of X-buckets
}

struct BucketInfo {
    min_x: f32,                // Bucket X range start (mm)
    max_x: f32,                // Bucket X range end (mm)
    start_index: u32,          // Index into triangle_indices array
    count: u32                 // Number of triangles in this bucket
}

@group(0) @binding(0) var<storage, read> triangles: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read> bucket_info: array<BucketInfo>;
@group(0) @binding(4) var<storage, read> triangle_indices: array<u32>;

// Fast 2D bounding box check for ray-triangle intersection
fn ray_hits_triangle_bbox(ray_origin: vec3<f32>, ray_dir: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> bool {
    let epsilon = 0.001;  // 1 micron tolerance
    let min_x = min(min(v0.x, v1.x), v2.x) - epsilon;
    let max_x = max(max(v0.x, v1.x), v2.x) + epsilon;
    let min_y = min(min(v0.y, v1.y), v2.y) - epsilon;
    let max_y = max(max(v0.y, v1.y), v2.y) + epsilon;
    let min_z = min(min(v0.z, v1.z), v2.z) - epsilon;
    let max_z = max(max(v0.z, v1.z), v2.z) + epsilon;

    // Simple AABB check
    return ray_origin.x >= min_x && ray_origin.x <= max_x &&
           ray_origin.y >= min_y && ray_origin.y <= max_y &&
           ray_origin.z >= min_z && ray_origin.z <= max_z;
}

// Ray-triangle intersection using Möller-Trumbore algorithm
fn ray_triangle_intersect(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>
) -> vec2<f32> {  // Returns (hit: 0.0 or 1.0, z: intersection_z)
    let EPSILON = 0.0001;

    // Early rejection using bounding box
    if (!ray_hits_triangle_bbox(ray_origin, ray_dir, v0, v1, v2)) {
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
        // Intersection found - calculate Z coordinate
        let intersection_point = ray_origin + ray_dir * t;
        return vec2<f32>(1.0, intersection_point.z);
    }

    return vec2<f32>(0.0, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bucket_idx = global_id.z;
    let grid_y = global_id.y;
    let angle_idx = global_id.x;

    // Bounds check
    if (angle_idx >= uniforms.num_angles ||
        grid_y >= uniforms.grid_y_height ||
        bucket_idx >= uniforms.num_buckets) {
        return;
    }

    let bucket = bucket_info[bucket_idx];
    let angle = f32(angle_idx) * uniforms.angle_step;

    // Calculate bucket min grid X
    let bucket_min_grid_x = u32((bucket.min_x - uniforms.global_min_x) / uniforms.resolution);

    // Loop over X within this bucket
    for (var local_x = 0u; local_x < uniforms.bucket_grid_width; local_x++) {
        let grid_x = bucket_min_grid_x + local_x;
        let world_x = uniforms.global_min_x + f32(grid_x) * uniforms.resolution;

        // Calculate ray origin (on rotated plane, offset from model)
        let radial_dist = uniforms.max_radius - f32(grid_y) * uniforms.resolution;
        let ray_origin_y = radial_dist * cos(angle);
        let ray_origin_z = radial_dist * sin(angle);
        let ray_origin = vec3<f32>(world_x, ray_origin_y, ray_origin_z);

        // Ray direction: radially inward toward X-axis
        let ray_dir = vec3<f32>(0.0, -cos(angle), -sin(angle));

        // Initialize best_z based on filter mode
        var best_z: f32;
        if (uniforms.filter_mode == 0u) {
            best_z = uniforms.z_floor;  // Terrain: keep highest Z
        } else {
            best_z = 1e10;              // Tool: keep lowest Z
        }

        var found = false;

        // Ray-cast against triangles in this bucket
        for (var i = 0u; i < bucket.count; i++) {
            let tri_idx = triangle_indices[bucket.start_index + i];
            let tri_base = tri_idx * 9u;

            // Read triangle vertices
            let v0 = vec3<f32>(
                triangles[tri_base],
                triangles[tri_base + 1u],
                triangles[tri_base + 2u]
            );
            let v1 = vec3<f32>(
                triangles[tri_base + 3u],
                triangles[tri_base + 4u],
                triangles[tri_base + 5u]
            );
            let v2 = vec3<f32>(
                triangles[tri_base + 6u],
                triangles[tri_base + 7u],
                triangles[tri_base + 8u]
            );

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

        // Write output
        // Layout: bucket_idx * numAngles * bucketWidth * gridHeight
        //       + angle_idx * bucketWidth * gridHeight
        //       + grid_y * bucketWidth
        //       + local_x
        let output_idx = bucket_idx * uniforms.num_angles * uniforms.bucket_grid_width * uniforms.grid_y_height
                       + angle_idx * uniforms.bucket_grid_width * uniforms.grid_y_height
                       + grid_y * uniforms.bucket_grid_width
                       + local_x;

        if (found) {
            output[output_idx] = best_z;
        } else {
            output[output_idx] = uniforms.z_floor;
        }
    }
}
