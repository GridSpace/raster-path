# Raster Path - WebGPU Toolpath Generator

Fast browser-based terrain + tool path generator using WebGPU compute shaders.

## Features

- **Multiple Operational Modes**: Planar (XY grid), Radial (cylindrical), and Tracing (path-following)
- **CNC Toolpath Generation**: Generate toolpaths by simulating tool movement over terrain
- **GPU Accelerated**: 20-100× faster than CPU-based solutions
- **Optimized Radial Variants**: V2 (default), V3 (memory-optimized), and V4 (slice-based lathe)
- **Unified API**: Clean three-method interface that works uniformly across all modes
- **ESM Module**: Importable package for browser applications

## Quick Start

### As a Module

```javascript
import { RasterPath } from '@gridspace/raster-path';

// Initialize for planar mode
const raster = new RasterPath({
    mode: 'planar',
    resolution: 0.1  // 0.1mm grid resolution
});
await raster.init();

// 1. Load tool (from STL triangles)
const toolTriangles = parseSTL(toolSTLBuffer);
const toolData = await raster.loadTool({
    triangles: toolTriangles
});

// 2. Load terrain (rasterizes immediately in planar mode)
const terrainTriangles = parseSTL(terrainSTLBuffer);
const terrainData = await raster.loadTerrain({
    triangles: terrainTriangles,
    zFloor: -100
});

// 3. Generate toolpaths
const toolpathData = await raster.generateToolpaths({
    xStep: 5,    // Sample every 5th point in X
    yStep: 5,    // Sample every 5th point in Y
    zFloor: -100
});

console.log(`Generated ${toolpathData.pathData.length} toolpath points`);

// Cleanup
raster.terminate();
```

### Radial Mode (for cylindrical parts)

```javascript
// Initialize for radial mode (V2 default)
const raster = new RasterPath({
    mode: 'radial',
    resolution: 0.1,      // Radial resolution (mm)
    rotationStep: 1.0     // 1 degree between rays
});
await raster.init();

// Load tool and terrain (same API!)
await raster.loadTool({ triangles: toolTriangles });
await raster.loadTerrain({
    triangles: terrainTriangles,
    zFloor: 0
});

// Generate toolpaths
const toolpathData = await raster.generateToolpaths({
    xStep: 5,
    yStep: 5,
    zFloor: 0
});

// Output is array of strips (one per rotation angle)
console.log(`Generated ${toolpathData.numStrips} strips, ${toolpathData.totalPoints} points`);
```

**Radial Variants:**
```javascript
// Use V3 (memory-optimized) for large models
const rasterV3 = new RasterPath({
    mode: 'radial',
    resolution: 0.1,
    rotationStep: 1.0,
    radialV3: true
});

// Use V4 (slice-based lathe, experimental) with pre-sliced data
const rasterV4 = new RasterPath({
    mode: 'radial',
    resolution: 0.5,
    rotationStep: 1.0,
    radialV4: true
});
```

### Tracing Mode (for path-following)

```javascript
// Initialize for tracing mode
const raster = new RasterPath({
    mode: 'tracing',
    resolution: 0.1  // Terrain rasterization resolution
});
await raster.init();

// Load tool and terrain
await raster.loadTool({ triangles: toolTriangles });
await raster.loadTerrain({
    triangles: terrainTriangles,
    zFloor: -100
});

// Define input paths as arrays of XY coordinate pairs
const paths = [
    new Float32Array([x1, y1, x2, y2, x3, y3, ...]),  // Path 1
    new Float32Array([x1, y1, x2, y2, ...])           // Path 2
];

// Generate toolpaths by tracing along paths
const toolpathData = await raster.generateToolpaths({
    paths: paths,
    step: 0.5,    // Sample every 0.5mm along each path
    zFloor: -100
});

// Output is array of XYZ coordinate arrays (one per path)
console.log(`Generated ${toolpathData.pathResults.length} traced paths`);
toolpathData.pathResults.forEach((path, i) => {
    console.log(`  Path ${i}: ${path.length / 3} points`);
});
```

### Demo UI

```bash
npm install
npm run dev
```

Open http://localhost:3000 and drag STL files onto the interface.

## Algorithm

### Planar Mode (XY Grid Rasterization)

1. **Tool Rasterization**: Create XY grid at specified resolution and rasterize tool geometry (keeps min Z per grid cell)
2. **Terrain Rasterization**: Rasterize terrain geometry on matching XY grid (keeps max Z per grid cell)
3. **Toolpath Generation**:
   - Scan tool over terrain in XY grid with configurable step sizes (xStep, yStep)
   - At each position, calculate minimum Z-offset where tool doesn't collide with terrain
   - Output scanline-based toolpath as array of Z-heights

### Radial Mode (Cylindrical Rasterization)

Three variants are available with different performance characteristics:

#### V2 (Default) - Ray-Based Rasterization
1. **Tool Rasterization**: Rasterize tool in planar mode (same as above)
2. **Terrain Preparation**: Center terrain in YZ plane and store triangles
3. **Toolpath Generation**:
   - Cast rays from origin at specified rotation angles (e.g., every 1°)
   - For each ray, rasterize terrain triangles along that angle
   - Use X-bucketing optimization to partition triangles spatially
   - Calculate tool-terrain collisions along each radial strip
   - Output array of strips (one per angle), each containing Z-heights along X-axis

#### V3 - Bucket-Angle Pipeline (Memory Optimized)
Enable with `radialV3: true` option.

**Algorithm:**
1. **Tool Rasterization**: Same as V2
2. **Terrain Preparation**: Bucket triangles by X-coordinate
3. **Toolpath Generation** (for each rotation angle):
   - Rotate all triangles in bucket by angle (GPU parallel)
   - Filter by Y-bounds (skip triangles outside tool radius)
   - Rasterize all buckets in single dispatch → dense terrain strip
   - Generate toolpath from strip immediately

**Advantages over V2:**
- Lower memory usage (only one angle's data in GPU at a time)
- Y-axis filtering reduces unnecessary triangle processing
- Better cache locality by processing each bucket completely

#### V4 - Slice-Based Lathe (Experimental)
Enable with `radialV4: true` option.

**Algorithm:**
1. **Tool Rasterization**: Same as V2
2. **Terrain Slicing** (CPU): Slice model along X-axis at dense intervals
   - Each slice is a YZ plane intersection → array of line segments
3. **Toolpath Generation** (for each rotation angle):
   - Rotate all slice lines around X-axis (CPU)
   - GPU shader traces tool through rotated slices
   - For each X position, ray-cast through corresponding slice to find max Z collision

**Advantages:**
- No rasterization overhead, works directly with geometry
- CPU/GPU balanced workload
- Based on proven Kiri:Moto lathePath algorithm

**Note:** V4 expects pre-sliced data and is designed for integration with external slicing engines.

### Tracing Mode (Path-Following Toolpath)

1. **Tool Rasterization**: Rasterize tool in planar mode
2. **Terrain Rasterization**: Rasterize terrain on XY grid (same as planar mode)
3. **Path Sampling**: Sample each input polyline at specified step resolution (e.g., every 0.5mm)
4. **Toolpath Generation**:
   - For each sampled point on each path:
     - Convert world coordinates to terrain grid coordinates
     - Test tool collision at that grid position using planar algorithm
     - Calculate maximum collision Z-height
   - Output array of XYZ coordinate arrays (one per input path)

**Use Case:** Generate toolpaths that follow pre-defined paths (e.g., outlines, contours) rather than scanning the entire grid.

## Performance

Example (84×84×28mm model, 6,120 triangles):

| Step Size | Points  | WebGPU Time | CPU Time (WASM) |
|-----------|---------|-------------|-----------------|
| 0.5mm     | 48K     | 0.8s        | 20-80s          |
| 0.1mm     | 1.2M    | 2s          | 280s            |

**Speedup**: 20-100× faster with WebGPU

## Project Structure

```
src/
  index.js                    # Main RasterPath API (ESM export)
  web/
    webgpu-worker.js         # WebGPU worker (GPU compute shaders)
    app.js                   # Demo web application
    index.html               # Demo UI entry point
    style.css                # Demo styles
    parse-stl.js             # STL file parser utility
  test/
    planar-test.cjs          # Planar mode regression test
    planar-tiling-test.cjs   # Planar high-resolution test
    radial-test.cjs          # Radial mode regression test
  benchmark/
    fixtures/                # Test STL files (terrain.stl, tool.stl)
build/                        # Built files (generated by npm run build)
```

## API Reference

### `RasterPath`

Constructor: `new RasterPath(options)`

**Options**:
- `mode` (string): `'planar'`, `'radial'`, or `'tracing'`
- `resolution` (number): Grid resolution in mm (e.g., 0.1)
- `rotationStep` (number, radial only): Degrees between rays (e.g., 1.0)
- `radialV3` (boolean, radial only): Enable V3 memory-optimized pipeline (default: false)
- `radialV4` (boolean, radial only): Enable V4 slice-based lathe pipeline (default: false)

#### `async init()`
Initialize WebGPU worker. Must be called before other methods.

**Returns**: `Promise<void>`

**Example**:
```javascript
const raster = new RasterPath({ mode: 'planar', resolution: 0.1 });
await raster.init();
```

---

#### `async loadTool({ triangles, sparseData })`
Load tool geometry for toolpath generation.

**Parameters** (one required):
- `triangles` (Float32Array, optional): STL triangle data (9 floats per triangle: v0.xyz, v1.xyz, v2.xyz)
- `sparseData` (object, optional): Pre-computed raster data with `{ bounds, positions, pointCount }`

**Returns**: `Promise<object>` - Tool raster data with `{ bounds, positions, pointCount }`

**Example**:
```javascript
// From STL triangles
const toolData = await raster.loadTool({
    triangles: toolTriangles
});

// From pre-computed sparse data (Kiri:Moto integration)
const toolData = await raster.loadTool({
    sparseData: { bounds, positions, pointCount }
});
```

---

#### `async loadTerrain({ triangles, zFloor, boundsOverride, onProgress })`
Load terrain geometry. Behavior depends on mode:
- **Planar mode**: Rasterizes immediately and returns terrain data
- **Radial mode**: Stores triangles for later, returns `null`

**Parameters**:
- `triangles` (Float32Array): STL triangle data
- `zFloor` (number, optional): Z floor value for out-of-bounds areas
- `boundsOverride` (object, optional): Override bounding box `{min: {x, y, z}, max: {x, y, z}}`
- `onProgress` (function, optional): Progress callback `(progress: number) => void`

**Returns**:
- Planar mode: `Promise<object>` - Terrain raster data with `{ bounds, positions, pointCount }`
- Radial mode: `Promise<null>`

**Example**:
```javascript
// Planar mode - returns terrain data immediately
const terrainData = await raster.loadTerrain({
    triangles: terrainTriangles,
    zFloor: -100
});

// Radial mode - stores for later
await raster.loadTerrain({
    triangles: terrainTriangles,
    zFloor: 0
});
```

---

#### `async generateToolpaths(options)`
Generate toolpaths from loaded tool and terrain. Must call `loadTool()` and `loadTerrain()` first.

**Parameters (mode-dependent)**:

**Planar and Radial modes:**
- `xStep` (number): Sample every Nth point in X direction
- `yStep` (number): Sample every Nth point in Y direction
- `zFloor` (number): Z floor value for out-of-bounds areas
- `radiusOffset` (number, radial only): Radial offset in mm
- `onProgress` (function, optional): Progress callback `(progress: number) => void`

**Tracing mode:**
- `paths` (Array<Float32Array>): Array of input polylines (each as XY coordinate pairs)
- `step` (number): Sample resolution along paths in world units (e.g., 0.5mm)
- `zFloor` (number): Z floor value for out-of-bounds areas
- `onProgress` (function, optional): Progress callback `(progress: number) => void`

**Returns**:
- Planar mode: `Promise<object>` with:
  - `pathData` (Float32Array): Z-heights in scanline order
  - `width` (number): Points per scanline
  - `height` (number): Number of scanlines

- Radial mode: `Promise<object>` with:
  - `strips` (Array): Array of strip objects, each containing:
    - `angle` (number): Rotation angle in degrees
    - `pathData` (Float32Array): Z-heights along X-axis
  - `numStrips` (number): Total number of strips
  - `totalPoints` (number): Sum of all points across strips

- Tracing mode: `Promise<object>` with:
  - `pathResults` (Array<Float32Array>): Array of XYZ coordinate arrays (one per input path)
  - `totalPoints` (number): Sum of all points across paths

**Examples**:
```javascript
// Planar and radial modes
const toolpathData = await raster.generateToolpaths({
    xStep: 5,
    yStep: 5,
    zFloor: -100,
    radiusOffset: 20  // radial mode only
});

// Tracing mode
const paths = [
    new Float32Array([x1, y1, x2, y2, ...]),
    new Float32Array([x1, y1, x2, y2, ...])
];
const toolpathData = await raster.generateToolpaths({
    paths: paths,
    step: 0.5,  // Sample every 0.5mm
    zFloor: -100
});
```

---

#### `terminate()`
Terminate WebGPU worker and cleanup resources.

**Example**:
```javascript
raster.terminate();
```

## Requirements

- Modern browser with WebGPU support (Chrome 113+, Edge 113+)
- For testing: Electron (provides headless WebGPU environment)

## Development

```bash
# Install dependencies
npm install

# Build (copies web files to build/)
npm run build

# Run demo
npm run serve

# Test
npm test
```

## License

MIT
