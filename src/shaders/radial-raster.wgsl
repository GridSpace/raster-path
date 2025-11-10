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
    start_angle: f32,          // Starting angle offset in radians (for batching)
    bucket_offset: u32,        // Offset for bucket batching (bucket_idx in batch writes to bucket_offset + bucket_idx in output)
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

// Note: AABB early rejection removed - X-bucketing already provides spatial filtering
// A proper ray-AABB intersection test would be needed if we wanted bounding box culling,
// but checking if ray_origin is inside AABB was incorrect and rejected valid triangles

// Ray-triangle intersection using Möller-Trumbore algorithm
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
        // Intersection found - return distance along ray (t parameter)
        return vec2<f32>(1.0, t);
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
    let angle = uniforms.start_angle + (f32(angle_idx) * uniforms.angle_step);

    // Calculate bucket min grid X
    let bucket_min_grid_x = u32((bucket.min_x - uniforms.global_min_x) / uniforms.resolution);

    // Loop over X within this bucket
    for (var local_x = 0u; local_x < uniforms.bucket_grid_width; local_x++) {
        let grid_x = bucket_min_grid_x + local_x;
        let world_x = uniforms.global_min_x + f32(grid_x) * uniforms.resolution;

        // Rotating top-down scan: normal XY planar scan, rotated around X-axis
        // Step 1: Define scan position in planar frame (X, Y, Z_above)
        let scan_x = world_x;
        let scan_y = f32(grid_y) * uniforms.resolution - uniforms.tool_width / 2.0;
        let scan_z = uniforms.max_radius;  // Start above the model

        // Step 2: Rotate position (scan_x, scan_y, scan_z) around X-axis by 'angle'
        // X stays the same, rotate YZ plane: y' = y*cos - z*sin, z' = y*sin + z*cos
        // NOTE: This uses right-handed rotation (positive angle rotates +Y towards +Z)
        // To reverse rotation direction (left-handed or opposite), flip signs:
        //   y' = y*cos + z*sin  (flip sign on z term)
        //   z' = -y*sin + z*cos (flip sign on y term)
        let ray_origin_x = scan_x;
        let ray_origin_y = scan_y * cos(angle) - scan_z * sin(angle);
        let ray_origin_z = scan_y * sin(angle) + scan_z * cos(angle);
        let ray_origin = vec3<f32>(ray_origin_x, ray_origin_y, ray_origin_z);

        // Step 3: Rotate ray direction (0, 0, -1) around X-axis by 'angle'
        // X component stays 0, rotate YZ: dy = 0*cos - (-1)*sin = sin, dz = 0*sin + (-1)*cos = -cos
        // NOTE: For reversed rotation, use: vec3<f32>(0.0, -sin(angle), -cos(angle))
        let ray_dir = vec3<f32>(0.0, sin(angle), -cos(angle));

        // Initialize best distance (closest hit)
        var best_t: f32 = 1e10;  // Start with very large distance
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
            let t = result.y;  // Distance along ray

            if (hit > 0.5) {
                // Keep closest hit (minimum t)
                if (t < best_t) {
                    best_t = t;
                    found = true;
                }
            }
        }

        // Write output
        // Layout: (bucket_offset + bucket_idx) * numAngles * bucketWidth * gridHeight
        //       + angle_idx * bucketWidth * gridHeight
        //       + grid_y * bucketWidth
        //       + local_x
        let output_idx = (uniforms.bucket_offset + bucket_idx) * uniforms.num_angles * uniforms.bucket_grid_width * uniforms.grid_y_height
                       + angle_idx * uniforms.bucket_grid_width * uniforms.grid_y_height
                       + grid_y * uniforms.bucket_grid_width
                       + local_x;

        if (found) {
            // Terrain height = distance from scan origin minus ray travel distance
            // Ray started at max_radius from X-axis, traveled best_t distance to hit
            let terrain_height = uniforms.max_radius - best_t;
            output[output_idx] = terrain_height;
        } else {
            output[output_idx] = uniforms.z_floor;
        }
    }
}
