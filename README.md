# Raster Path - WebGPU Toolpath Generator

Fast browser-based terrain + tool path generator using WebGPU compute shaders.

## Features

- **STL to Point Cloud**: Convert STL files to point meshes using GPU-accelerated XY rastering
- **CNC Toolpath Generation**: Generate toolpaths by simulating tool movement over terrain
- **GPU Accelerated**: 20-100× faster than CPU-based solutions
- **ESM Module**: Clean, importable package for browser applications

## Quick Start

### As a Module

```javascript
import { RasterPath } from 'raster-path';

// Initialize WebGPU
const converter = new RasterPath();
await converter.init();

// Load and convert STL
const response = await fetch('model.stl');
const stlBuffer = await response.arrayBuffer();

const result = await converter.rasterizeSTL(
    stlBuffer,
    0.5,  // step size (mm)
    0     // filter mode: 0=max Z (terrain), 1=min Z (tool)
);

console.log(`Converted to ${result.pointCount} points`);

// Generate toolpath
const toolpath = await converter.generateToolpath(
    terrainResult.positions,
    toolResult.positions,
    5,      // xStep
    5,      // yStep
    -100,   // zFloor
    0.5     // gridStep
);

// Cleanup
converter.dispose();
```

### Demo UI

```bash
npm install
npm run dev
```

Open http://localhost:3000 and drag STL files onto the interface.

## Algorithm

### XY Rastering (STL → Point Cloud)

1. Calculate bounding box and filter triangles by surface normal
2. Create XY grid at specified step size (0.05mm - 1.0mm)
3. For each grid point, find triangle intersections along Z axis
4. Keep max Z (terrain) or min Z (tool) intersection per grid point

### Toolpath Generation

1. Convert point clouds to indexed 2D grids
2. Scan tool over terrain in XY grid
3. Calculate collision at each position (minimum delta between tool and terrain)
4. Output tool center Z heights as scanline-based toolpath

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
  index.js                    # Main ESM export
  web/
    webgpu-worker.js         # WebGPU worker (rasterization & toolpath)
    index.html               # Demo UI
    main.js                  # Demo app
    styles.css
  test/
    generate-hemisphere.js   # Test fixture generator
```

## API Reference

### `RasterPath`

#### `async init()`
Initialize WebGPU worker. Must be called before processing.

**Returns**: `Promise<boolean>` - Success status

#### `async rasterizeMesh(triangles, stepSize, filterMode, boundsOverride)`
Rasterize triangle mesh to height map. Accepts geometry from any source (Three.js, procedural, etc.).

**Parameters**:
- `triangles` (Float32Array): Unindexed triangle positions (9 floats per triangle: v0.xyz, v1.xyz, v2.xyz)
- `stepSize` (number): Grid resolution in mm (e.g., 0.5)
- `filterMode` (number): 0 for max Z (terrain), 1 for min Z (tool)
- `boundsOverride` (object, optional): Bounding box {min: {x, y, z}, max: {x, y, z}}

**Returns**: `Promise<{positions: Float32Array, pointCount: number, bounds: object}>`

**Example with Three.js**:
```javascript
// Get unindexed positions from Three.js geometry
const geometry = new THREE.BoxGeometry(10, 10, 10);
const positions = geometry.attributes.position.array;

const result = await converter.rasterizeMesh(
    positions,
    0.5,  // step size
    0     // max Z filter
);
```

#### `async rasterizeSTL(stlBuffer, stepSize, filterMode, boundsOverride)`
Convenience wrapper that parses STL and calls `rasterizeMesh()`.

**Parameters**:
- `stlBuffer` (ArrayBuffer): Binary STL data
- Other parameters: Same as `rasterizeMesh()`

**Returns**: `Promise<{positions: Float32Array, pointCount: number, bounds: object}>`

#### `async generateToolpath(terrainPositions, toolPositions, xStep, yStep, zFloor, gridStep)`
Generate CNC toolpath from terrain and tool point clouds.

**Parameters**:
- `terrainPositions` (Float32Array): Terrain point cloud
- `toolPositions` (Float32Array): Tool point cloud
- `xStep` (number): X-axis step size
- `yStep` (number): Y-axis step size
- `zFloor` (number): Z floor value for out-of-bounds
- `gridStep` (number): Grid resolution in mm

**Returns**: `Promise<{pathData: Float32Array, numScanlines: number, pointsPerLine: number, generationTime: number}>`

#### `dispose()`
Terminate worker and cleanup resources.

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
