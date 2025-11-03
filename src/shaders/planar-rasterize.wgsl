// Planar rasterization with spatial partitioning
// Sentinel value for empty cells (far below any real geometry)
const EMPTY_CELL: f32 = -1e10;

struct Uniforms {
    bounds_min_x: f32,
    bounds_min_y: f32,
    bounds_min_z: f32,
    bounds_max_x: f32,
    bounds_max_y: f32,
    bounds_max_z: f32,
    step_size: f32,
    grid_width: u32,
    grid_height: u32,
    triangle_count: u32,
    filter_mode: u32,  // 0 = UPWARD (terrain, keep highest), 1 = DOWNWARD (tool, keep lowest)
    spatial_grid_width: u32,
    spatial_grid_height: u32,
    spatial_cell_size: f32,
}

@group(0) @binding(0) var<storage, read> triangles: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_points: array<f32>;
@group(0) @binding(2) var<storage, read_write> valid_mask: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read> spatial_cell_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> spatial_triangle_indices: array<u32>;

// Fast 2D bounding box check for XY plane
fn ray_hits_triangle_bbox_2d(ray_x: f32, ray_y: f32, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> bool {
    // Add small epsilon to catch near-misses (mesh vertex gaps, FP rounding)
    let epsilon = 0.001;  // 1 micron tolerance
    let min_x = min(min(v0.x, v1.x), v2.x) - epsilon;
    let max_x = max(max(v0.x, v1.x), v2.x) + epsilon;
    let min_y = min(min(v0.y, v1.y), v2.y) - epsilon;
    let max_y = max(max(v0.y, v1.y), v2.y) + epsilon;

    return ray_x >= min_x && ray_x <= max_x && ray_y >= min_y && ray_y <= max_y;
}

// Ray-triangle intersection using Möller-Trumbore algorithm
fn ray_triangle_intersect(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>
) -> vec2<f32> {  // Returns (hit: 0.0 or 1.0, z: intersection_z)
    // Larger epsilon needed because near-parallel triangles (small 'a') amplify errors via f=1/a
    let EPSILON = 0.0001;

    // Early rejection using 2D bounding box (very cheap!)
    if (!ray_hits_triangle_bbox_2d(ray_origin.x, ray_origin.y, v0, v1, v2)) {
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

    // Allow tolerance for edges/vertices to ensure watertight coverage
    if (u < -EPSILON || u > 1.0 + EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }

    // Cross product: s × edge1
    let q = cross(s, edge1);

    // v = f * (ray_dir · q)
    let v = f * dot(ray_dir, q);

    // Allow tolerance for edges/vertices to ensure watertight coverage
    if (v < -EPSILON || u + v > 1.0 + EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }

    // t = f * (edge2 · q)
    let t = f * dot(edge2, q);

    if (t > EPSILON) {
        // Intersection found - calculate Z coordinate
        let intersection_z = ray_origin.z + ray_dir.z * t;
        return vec2<f32>(1.0, intersection_z);
    }

    return vec2<f32>(0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_x = global_id.x;
    let grid_y = global_id.y;

    if (grid_x >= uniforms.grid_width || grid_y >= uniforms.grid_height) {
        return;
    }

    // Calculate world position for this grid point (center of cell)
    let world_x = uniforms.bounds_min_x + f32(grid_x) * uniforms.step_size;
    let world_y = uniforms.bounds_min_y + f32(grid_y) * uniforms.step_size;

    // Initialize best_z based on filter mode
    var best_z: f32;
    if (uniforms.filter_mode == 0u) {
        best_z = -1e10;  // Terrain: keep highest Z
    } else {
        best_z = 1e10;   // Tool: keep lowest Z
    }

    var found = false;

    // Ray from below mesh pointing up (+Z direction)
    let ray_origin = vec3<f32>(world_x, world_y, uniforms.bounds_min_z - 1.0);
    let ray_dir = vec3<f32>(0.0, 0.0, 1.0);

    // Find which spatial grid cell this ray belongs to
    let spatial_cell_x = u32((world_x - uniforms.bounds_min_x) / uniforms.spatial_cell_size);
    let spatial_cell_y = u32((world_y - uniforms.bounds_min_y) / uniforms.spatial_cell_size);

    // Clamp to spatial grid bounds
    let clamped_cx = min(spatial_cell_x, uniforms.spatial_grid_width - 1u);
    let clamped_cy = min(spatial_cell_y, uniforms.spatial_grid_height - 1u);

    let spatial_cell_idx = clamped_cy * uniforms.spatial_grid_width + clamped_cx;

    // Get triangle range for this cell
    let start_idx = spatial_cell_offsets[spatial_cell_idx];
    let end_idx = spatial_cell_offsets[spatial_cell_idx + 1u];

    // Test only triangles in this spatial cell
    for (var idx = start_idx; idx < end_idx; idx++) {
        let tri_idx = spatial_triangle_indices[idx];
        let tri_base = tri_idx * 9u;

        // Read triangle vertices (already in local space and rotated if needed)
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

    // Write output based on filter mode
    let output_idx = grid_y * uniforms.grid_width + grid_x;

    if (uniforms.filter_mode == 0u) {
        // Terrain: Dense output (Z-only, sentinel value for empty cells)
        if (found) {
            output_points[output_idx] = best_z;
        } else {
            output_points[output_idx] = EMPTY_CELL;
        }
    } else {
        // Tool: Sparse output (X, Y, Z triplets with valid mask)
        output_points[output_idx * 3u] = f32(grid_x);
        output_points[output_idx * 3u + 1u] = f32(grid_y);
        output_points[output_idx * 3u + 2u] = best_z;

        if (found) {
            valid_mask[output_idx] = 1u;
        } else {
            valid_mask[output_idx] = 0u;
        }
    }
}
