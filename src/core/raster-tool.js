/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Raster Tool - Sparse Tool Representation
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Utilities for creating sparse tool representations from rasterized tool data.
 * Converts dense grid data into compact offset-based format for GPU toolpath
 * generation.
 *
 * EXPORTS:
 * ────────
 * Functions:
 *   - createSparseToolFromPoints(points) - Convert tool points to sparse format
 *
 * DATA FORMATS:
 * ─────────────
 * Input (Dense Grid):
 *   - Float32Array of [gridX, gridY, Z, gridX, gridY, Z, ...]
 *   - gridX, gridY are integer grid indices (not world coordinates)
 *   - Z is height in world units (mm)
 *
 * Output (Sparse Tool):
 *   {
 *     count: number,           // Number of tool points
 *     xOffsets: Int32Array,    // X offset from tool center (grid cells)
 *     yOffsets: Int32Array,    // Y offset from tool center (grid cells)
 *     zValues: Float32Array,   // Z height (mm)
 *     referenceZ: number       // Tool tip Z (lowest point)
 *   }
 *
 * ALGORITHM:
 * ──────────
 * 1. Find bounding box in grid space (integer coordinates)
 * 2. Calculate tool center (grid coordinates)
 * 3. Convert each point to offset from center
 * 4. Store as parallel arrays for GPU consumption
 *
 * This sparse representation reduces memory usage and improves GPU cache
 * coherency during toolpath generation, as the tool can be "stamped" at
 * each position using simple offset arithmetic.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

// Create sparse tool representation from rasterized points
// Points come from GPU as [gridX, gridY, Z] - pure integer grid coordinates for X/Y
export function createSparseToolFromPoints(points) {
    if (!points || points.length === 0) {
        throw new Error('No tool points provided');
    }

    // Points are [gridX, gridY, Z] where gridX/gridY are grid indices (floats but integer values)
    // Find bounds in grid space and tool tip Z
    let minGridX = Infinity, minGridY = Infinity, minZ = Infinity;
    let maxGridX = -Infinity, maxGridY = -Infinity;

    for (let i = 0; i < points.length; i += 3) {
        const gridX = points[i];      // Already a grid index
        const gridY = points[i + 1];  // Already a grid index
        const z = points[i + 2];

        minGridX = Math.min(minGridX, gridX);
        maxGridX = Math.max(maxGridX, gridX);
        minGridY = Math.min(minGridY, gridY);
        maxGridY = Math.max(maxGridY, gridY);
        minZ = Math.min(minZ, z);
    }

    // Calculate tool center in grid coordinates (pure integer)
    const width = Math.floor(maxGridX - minGridX) + 1;
    const height = Math.floor(maxGridY - minGridY) + 1;
    const centerX = Math.floor(minGridX) + Math.floor(width / 2);
    const centerY = Math.floor(minGridY) + Math.floor(height / 2);

    // Convert each point to offset from center (integer arithmetic only)
    const xOffsets = [];
    const yOffsets = [];
    const zValues = [];

    for (let i = 0; i < points.length; i += 3) {
        const gridX = Math.floor(points[i]);      // Grid index (ensure integer)
        const gridY = Math.floor(points[i + 1]);  // Grid index (ensure integer)
        const z = points[i + 2];

        // Calculate offset from tool center (pure integer arithmetic)
        const xOffset = gridX - centerX;
        const yOffset = gridY - centerY;
        // Z relative to tool tip: tip=0, points above tip are positive
        // minZ is the lowest Z (tip), so z - minZ gives positive offsets upward
        const zValue = z;// - minZ;

        xOffsets.push(xOffset);
        yOffsets.push(yOffset);
        zValues.push(zValue);
    }

    return {
        count: xOffsets.length,
        xOffsets: new Int32Array(xOffsets),
        yOffsets: new Int32Array(yOffsets),
        zValues: new Float32Array(zValues),
        referenceZ: minZ
    };
}