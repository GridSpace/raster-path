// raster-path: Terrain and Tool Raster Path Finder using WebGPU
// Main ESM entry point

/**
 * ═══════════════════════════════════════════════════════════════════════════
 * RasterPath API Overview
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Unified three-method API for GPU-accelerated toolpath generation.
 * Works uniformly across both planar (XY grid) and radial (cylindrical) modes.
 *
 * USAGE PATTERN:
 * ──────────────
 * 1. Create instance: new RasterPath({ mode, resolution, rotationStep? })
 * 2. Initialize GPU:  await raster.init()
 * 3. Load tool:       await raster.loadTool({ triangles | sparseData })
 * 4. Load terrain:    await raster.loadTerrain({ triangles, zFloor?, ... })
 * 5. Generate paths:  await raster.generateToolpaths({ xStep, yStep, zFloor, ... })
 * 6. Cleanup:         raster.terminate()
 *
 * MODE DIFFERENCES:
 * ─────────────────
 * PLANAR MODE:
 *   - Traditional XY grid rasterization
 *   - loadTerrain() rasterizes immediately and returns data
 *   - Best for flat or gently curved surfaces
 *   - Output: Single 2D array of Z-heights in scanline order
 *
 * RADIAL MODE:
 *   - Cylindrical unwrap rasterization
 *   - loadTerrain() stores triangles, defers rasterization until generateToolpaths()
 *   - Best for cylindrical/rotational parts
 *   - Requires terrain centered in YZ plane (done automatically)
 *   - Output: Array of radial strips, one per rotation angle
 *
 * COORDINATE SYSTEMS:
 * ───────────────────
 * - Tool geometry: Z-axis is flipped during loadTool() for collision detection
 * - Terrain (radial): Auto-centered in YZ plane before storage
 * - All inputs use standard STL coordinates (right-handed, Z-up)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * Configuration options for RasterPath
 * @typedef {Object} RasterPathConfig
 * @property {'planar'|'radial'} mode - Rasterization mode (default: 'planar')
 * @property {boolean} autoTiling - Automatically tile large datasets (default: true)
 * @property {number} gpuMemorySafetyMargin - Safety margin as percentage (default: 0.8 = 80%)
 * @property {number} maxGPUMemoryMB - Maximum GPU memory per tile (default: 256MB)
 * @property {number} resolution - Grid step size in mm (required)
 * @property {number} rotationStep - Radial mode only: degrees between rays (e.g., 1.0 = 360 rays)
 * @property {number} batchDivisor - Testing parameter to artificially divide batch size (default: 1)
 * @property {boolean} debug - Enable debug logging (default: false)
 * @property {boolean} quiet - Suppress log output (default: false)
 */

const ZMAX = 10e6;
const EMPTY_CELL = -1e10;
const log_pre = '[Raster]';

const debug = {
    error: function() { console.error(log_pre, ...arguments) },
    warn: function() { console.warn(log_pre, ...arguments) },
    log: function() { console.log(log_pre, ...arguments) },
    ok: function() { console.log(log_pre, '✅', ...arguments) },
};

/**
 * Main class for rasterizing geometry and generating toolpaths using WebGPU
 * Supports both planar and radial (cylindrical) rasterization modes
 */
export class RasterPath {
    constructor(config = {}) {
        // Validate required parameters
        if (!config.resolution) {
            throw new Error('RasterPath requires resolution parameter');
        }

        // Validate mode
        const mode = config.mode || 'planar';
        if (mode !== 'planar' && mode !== 'radial') {
            throw new Error(`Invalid mode: ${mode}. Must be 'planar' or 'radial'`);
        }

        // Validate rotationStep for radial mode
        if (mode === 'radial' && !config.rotationStep) {
            throw new Error('Radial mode requires rotationStep parameter (degrees between rays)');
        }

        this.mode = mode;
        this.resolution = config.resolution;
        this.rotationStep = config.rotationStep;

        this.worker = null;
        this.isInitialized = false;
        this.messageHandlers = new Map();
        this.messageId = 0;
        this.deviceCapabilities = null;

        // Configure debug output
        if (config.quiet) {
            debug.log = function() {};
        }

        // Configuration with defaults
        this.config = {
            workerName: config.workerName ?? "raster-worker.js",
            maxGPUMemoryMB: config.maxGPUMemoryMB ?? 256,
            gpuMemorySafetyMargin: config.gpuMemorySafetyMargin ?? 0.8,
            autoTiling: config.autoTiling ?? true,
            batchDivisor: config.batchDivisor ?? 1, // For testing batching overhead
            debug: config.debug,
            quiet: config.quiet
        };

        debug.log('config', this.config);
    }

    /**
     * Initialize WebGPU worker
     * Must be called before any processing operations
     * @returns {Promise<boolean>} Success status
     */
    async init() {
        if (this.isInitialized) {
            return true;
        }

        return new Promise((resolve, reject) => {
            try {
                // Create worker from the raster-worker.js file
                const workerName = this.config.workerName;
                const isBuildVersion = import.meta.url.includes('/build/') || import.meta.url.includes('raster-path.js');
                const workerPath = workerName
                    ? new URL(workerName, import.meta.url)
                : isBuildVersion
                    ? new URL(`./raster-worker.js`, import.meta.url)
                    : new URL(`../core/raster-worker.js`, import.meta.url);
                this.worker = new Worker(workerPath, { type: 'module' });

                // Set up message handler
                this.worker.onmessage = (e) => this.#handleMessage(e);
                this.worker.onerror = (error) => {
                    debug.error('[RasterPath] Worker error:', error);
                    reject(error);
                };

                // Send init message with config
                const handler = (data) => {
                    this.isInitialized = data.success;
                    if (data.success) {
                        this.deviceCapabilities = data.capabilities;
                        resolve(true);
                    } else {
                        reject(new Error('Failed to initialize WebGPU'));
                    }
                };

                this.#sendMessage('init', { config: this.config }, 'webgpu-ready', handler);
            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Load tool - accepts either triangles (from STL) or sparse data (from Kiri:Moto)
     * @param {object} params - Parameters
     * @param {Float32Array} params.triangles - Optional: Unindexed triangle vertices
     * @param {object} params.sparseData - Optional: Pre-computed sparse data {bounds, positions, pointCount}
     * @returns {Promise<object>} Tool data (sparse format: {bounds, positions, pointCount})
     */
    async loadTool({ triangles, sparseData }) {
        if (!this.isInitialized) {
            throw new Error('RasterPath not initialized. Call init() first.');
        }

        // If sparse data provided directly (from Kiri:Moto), use it
        if (sparseData) {
            this.toolData = sparseData;
            return sparseData;
        }

        // Otherwise rasterize from triangles
        if (!triangles) {
            throw new Error('loadTool() requires either triangles or sparseData');
        }

        const toolData = await this.#rasterizePlanar({ triangles, isForTool: true });
        const { bounds, positions } = toolData;

        // Transform tool coordinate system: flip Z-axis for collision detection
        // Tool geometry is inverted so that tool-terrain collision can be computed
        // as a simple subtraction (terrainZ - toolZ) instead of complex geometry tests
        for (let i=0; i<positions.length; i += 3) {
            positions[i+2] = -positions[i+2] - bounds.min.z;
        }
        let swapZ = bounds.min.z;
        bounds.min.z = -bounds.max.z;
        bounds.max.z = -swapZ;

        this.toolData = toolData;
        return toolData;
    }

    /**
     * Load terrain - behavior depends on mode
     * Planar mode: Rasterizes and returns terrain data
     * Radial mode: Stores triangles for later use in generateToolpaths()
     * @param {object} params - Parameters
     * @param {Float32Array} params.triangles - Unindexed triangle vertices
     * @param {number} params.zFloor - Z floor for out-of-bounds (optional)
     * @param {object} params.boundsOverride - Optional bounding box {min: {x, y, z}, max: {x, y, z}}
     * @param {function} params.onProgress - Optional progress callback (percent, info) => {}
     * @returns {Promise<object|null>} Planar: terrain data {bounds, positions, pointCount}, Radial: null
     */
    async loadTerrain({ triangles, zFloor, boundsOverride, onProgress }) {
        if (!this.isInitialized) {
            throw new Error('RasterPath not initialized. Call init() first.');
        }

        if (this.mode === 'planar') {
            // Planar: rasterize and return
            const terrainData = await this.#rasterizePlanar({ triangles, zFloor, boundsOverride, isForTool: false, onProgress });
            this.terrainData = terrainData;
            return terrainData;
        } else {
            // Radial: store triangles and metadata for generateToolpaths()
            const originalBounds = boundsOverride || this.#calculateBounds(triangles);

            // Center model in YZ plane (required for radial rasterization)
            // Radial mode casts rays from max_radius distance inward toward the X-axis,
            // and centering ensures the geometry is symmetric around the rotation axis
            const centerY = (originalBounds.min.y + originalBounds.max.y) / 2;
            const centerZ = (originalBounds.min.z + originalBounds.max.z) / 2;

            let centeredTriangles = triangles;
            let bounds = originalBounds;

            if (Math.abs(centerY) > 0.001 || Math.abs(centerZ) > 0.001) {
                debug.log(`Centering model in YZ: offset Y=${centerY.toFixed(3)}, Z=${centerZ.toFixed(3)}`);
                centeredTriangles = new Float32Array(triangles.length);
                for (let i = 0; i < triangles.length; i += 3) {
                    centeredTriangles[i] = triangles[i];                    // X unchanged
                    centeredTriangles[i + 1] = triangles[i + 1] - centerY;  // Center Y
                    centeredTriangles[i + 2] = triangles[i + 2] - centerZ;  // Center Z
                }
                bounds = this.#calculateBounds(centeredTriangles);
            }

            // Store for generateToolpaths()
            this.terrainTriangles = centeredTriangles;
            this.terrainBounds = bounds;
            this.terrainZFloor = zFloor ?? 0;

            return null;
        }
    }

    /**
     * Generate toolpaths from loaded terrain and tool
     * Must call loadTool() and loadTerrain() first
     * @param {object} params - Parameters
     * @param {number} params.xStep - Sample every Nth point in X direction
     * @param {number} params.yStep - Sample every Nth point in Y direction
     * @param {number} params.zFloor - Z floor value for out-of-bounds areas
     * @param {function} params.onProgress - Optional progress callback (progress: number, info?: string) => void
     * @returns {Promise<object>} Planar: {pathData, width, height} | Radial: {strips[], numStrips, totalPoints}
     */
    async generateToolpaths({ xStep, yStep, zFloor, onProgress }) {
        if (!this.isInitialized) {
            throw new Error('RasterPath not initialized. Call init() first.');
        }

        if (!this.toolData) {
            throw new Error('Tool not loaded. Call loadTool() first.');
        }

        debug.log('gen.paths', { xStep, yStep, zFloor });

        if (this.mode === 'planar') {
            if (!this.terrainData) {
                throw new Error('Terrain not loaded. Call loadTerrain() first.');
            }
            return this.#generateToolpathsPlanar({
                terrainData: this.terrainData,
                toolData: this.toolData,
                xStep,
                yStep,
                zFloor,
                onProgress
            });
        } else {
            // Radial mode: use stored triangles
            if (!this.terrainTriangles) {
                throw new Error('Terrain not loaded. Call loadTerrain() first.');
            }
            return this.#generateToolpathsRadial({
                triangles: this.terrainTriangles,
                bounds: this.terrainBounds,
                toolData: this.toolData,
                xStep,
                yStep,
                zFloor: zFloor ?? this.terrainZFloor,
                onProgress
            });
        }
    }

    /**
     * Terminate worker and cleanup resources
     */
    terminate() {
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
            this.isInitialized = false;
            this.messageHandlers.clear();
            this.deviceCapabilities = null;
            // Clear loaded data
            this.toolData = null;
            this.terrainData = null;
            this.terrainTriangles = null;
            this.terrainBounds = null;
            this.terrainZFloor = null;
        }
    }

    // ============================================================================
    // Internal Methods (Planar)
    // ============================================================================

    async #rasterizePlanar({ triangles, zFloor, boundsOverride, isForTool }) {
        const data = await new Promise((resolve, reject) => {
            const handler = (data) => resolve(data);

            this.#sendMessage(
                'rasterize',
                {
                    triangles,
                    stepSize: this.resolution,
                    filterMode: isForTool ? 1 : 0,  // 0 = max Z (terrain), 1 = min Z (tool)
                    boundsOverride
                },
                'rasterize-complete',
                handler
            );
        });

        return data;
    }

    async #generateToolpathsPlanar({ terrainData, toolData, xStep, yStep, zFloor, onProgress, singleScanline = false }) {
        return new Promise((resolve, reject) => {
            // Set up progress handler if callback provided
            if (onProgress) {
                const progressHandler = (data) => {
                    onProgress(data.percent, { current: data.current, total: data.total, layer: data.layer });
                };
                this.messageHandlers.set('toolpath-progress', progressHandler);
            }

            const handler = (data) => {
                // Clean up progress handler
                if (onProgress) {
                    this.messageHandlers.delete('toolpath-progress');
                }
                resolve(data);
            };

            this.#sendMessage(
                'generate-toolpath',
                {
                    terrainPositions: terrainData.positions,
                    toolPositions: toolData.positions,
                    xStep,
                    yStep,
                    zFloor: zFloor ?? 0,
                    gridStep: this.resolution,
                    terrainBounds: terrainData.bounds,
                    singleScanline
                },
                'toolpath-complete',
                handler
            );
        });
    }

    async #generateToolpathsRadial({ triangles, bounds, toolData, xStep, yStep, zFloor, onProgress }) {
        const maxRadius = this.#calculateMaxRadius(triangles);

        // Calculate maximum tool extent in YZ plane (perpendicular to rotation axis)
        // This determines the radial collision search radius for each ray cast
        const toolWidth = Math.max(
            Math.abs(toolData.bounds.max.y - toolData.bounds.min.y),
            Math.abs(toolData.bounds.max.x - toolData.bounds.min.x)
        );

        // Build X-bucketing data for spatial partitioning along rotation axis
        // Divides terrain into buckets along X-axis to reduce triangle intersection tests
        const numAngles = Math.ceil(360 / this.rotationStep);
        const bucketWidth = 1.0; // Bucket size in mm - smaller = better load balancing, more memory
        const bucketData = this.#bucketTrianglesByX(triangles, bounds, bucketWidth);

        return new Promise((resolve, reject) => {
            // Setup progress handler
            if (onProgress) {
                const progressHandler = (data) => {
                    onProgress(data.current, data.total);
                };
                this.messageHandlers.set('toolpath-progress', progressHandler);
            }

            // Setup completion handler
            const completionHandler = (data) => {
                // Clean up progress handler
                if (onProgress) {
                    this.messageHandlers.delete('toolpath-progress');
                }
                resolve(data);
            };

            // Send entire pipeline to worker
            this.#sendMessage(
                'radial-generate-toolpaths',
                {
                    triangles: triangles,
                    bucketData,
                    toolData,
                    resolution: this.resolution,
                    angleStep: this.rotationStep,
                    numAngles,
                    maxRadius: maxRadius * 1.01,
                    toolWidth,
                    zFloor: zFloor,
                    bounds,
                    xStep,
                    yStep
                },
                'radial-toolpaths-complete',
                completionHandler
            );
        });
    }

    // ============================================================================
    // Internal Utilities
    // ============================================================================

    #handleMessage(e) {
        const { type, success, data } = e.data;

        // Handle progress messages (don't delete handler)
        if (type === 'rasterize-progress' || type === 'toolpath-progress') {
            const handler = this.messageHandlers.get(type);
            if (handler) {
                handler(data);
                return;
            }
        }

        // Find handler for this message type (completion messages)
        for (const [id, handler] of this.messageHandlers.entries()) {
            if (handler.responseType === type) {
                this.messageHandlers.delete(id);
                handler.callback(data);
                break;
            }
        }
    }

    #sendMessage(type, data, responseType, callback) {
        const id = this.messageId++;
        this.messageHandlers.set(id, { responseType, callback });
        this.worker.postMessage({ type, data });
    }

    #calculateBounds(triangles) {
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        for (let i = 0; i < triangles.length; i += 3) {
            const x = triangles[i];
            const y = triangles[i + 1];
            const z = triangles[i + 2];

            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
            minZ = Math.min(minZ, z);
            maxZ = Math.max(maxZ, z);
        }

        return {
            min: { x: minX, y: minY, z: minZ },
            max: { x: maxX, y: maxY, z: maxZ }
        };
    }

    #calculateMaxRadius(triangles) {
        let maxRadius = 0;

        for (let i = 0; i < triangles.length; i += 3) {
            const y = triangles[i + 1];
            const z = triangles[i + 2];
            const hypot = Math.sqrt(y * y + z * z);
            maxRadius = Math.max(maxRadius, hypot);
        }

        return maxRadius;
    }

    /**
     * Partition triangles into spatial buckets along X-axis for radial rasterization
     * This optimization reduces triangle intersection tests by only checking triangles
     * within relevant X-ranges during ray casting
     *
     * @returns {object} Bucket data structure with:
     *   - buckets: Array of {minX, maxX, startIndex, count}
     *   - triangleIndices: Uint32Array of triangle indices sorted by bucket
     *   - numBuckets: Total number of buckets
     */
    #bucketTrianglesByX(triangles, bounds, bucketWidth) {
        const numTriangles = triangles.length / 9;
        const numBuckets = Math.ceil((bounds.max.x - bounds.min.x) / bucketWidth);

        // Initialize buckets
        const buckets = [];
        for (let i = 0; i < numBuckets; i++) {
            buckets.push({
                minX: bounds.min.x + i * bucketWidth,
                maxX: bounds.min.x + (i + 1) * bucketWidth,
                triangleIndices: []
            });
        }

        // Assign triangles to overlapping buckets
        for (let triIdx = 0; triIdx < numTriangles; triIdx++) {
            const baseIdx = triIdx * 9;

            // Find triangle X range
            const x0 = triangles[baseIdx];
            const x1 = triangles[baseIdx + 3];
            const x2 = triangles[baseIdx + 6];

            const triMinX = Math.min(x0, x1, x2);
            const triMaxX = Math.max(x0, x1, x2);

            // Find overlapping buckets
            const startBucket = Math.max(0, Math.floor((triMinX - bounds.min.x) / bucketWidth));
            const endBucket = Math.min(numBuckets - 1, Math.floor((triMaxX - bounds.min.x) / bucketWidth));

            for (let b = startBucket; b <= endBucket; b++) {
                buckets[b].triangleIndices.push(triIdx);
            }
        }

        // Flatten triangle indices for GPU
        const triangleIndices = [];
        const bucketInfo = [];

        for (let i = 0; i < buckets.length; i++) {
            const bucket = buckets[i];
            bucketInfo.push({
                minX: bucket.minX,
                maxX: bucket.maxX,
                startIndex: triangleIndices.length,
                count: bucket.triangleIndices.length
            });
            triangleIndices.push(...bucket.triangleIndices);
        }

        return {
            buckets: bucketInfo,
            triangleIndices: new Uint32Array(triangleIndices),
            numBuckets
        };
    }

    // ============================================================================
    // Public Utilities
    // ============================================================================

    /**
     * Get device capabilities
     * @returns {object|null} Device capabilities or null if not initialized
     */
    getDeviceCapabilities() {
        return this.deviceCapabilities;
    }

    /**
     * Get current configuration
     * @returns {object} Current configuration
     */
    getConfig() {
        return {
            mode: this.mode,
            resolution: this.resolution,
            rotationStep: this.rotationStep,
            ...this.config
        };
    }

    /**
     * Parse STL buffer to triangles
     * @param {ArrayBuffer} buffer - Binary STL data
     * @returns {Float32Array} Triangle vertices
     */
    parseSTL(buffer) {
        const view = new DataView(buffer);
        const isASCII = this.#isASCIISTL(buffer);

        if (isASCII) {
            return this.#parseASCIISTL(buffer);
        } else {
            return this.#parseBinarySTL(view);
        }
    }

    #isASCIISTL(buffer) {
        const text = new TextDecoder().decode(buffer.slice(0, 80));
        return text.toLowerCase().startsWith('solid');
    }

    #parseASCIISTL(buffer) {
        const text = new TextDecoder().decode(buffer);
        const lines = text.split('\n');
        const triangles = [];
        let vertexCount = 0;
        let vertices = [];

        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('vertex')) {
                const parts = trimmed.split(/\s+/);
                vertices.push(
                    parseFloat(parts[1]),
                    parseFloat(parts[2]),
                    parseFloat(parts[3])
                );
                vertexCount++;
                if (vertexCount === 3) {
                    triangles.push(...vertices);
                    vertices = [];
                    vertexCount = 0;
                }
            }
        }

        return new Float32Array(triangles);
    }

    #parseBinarySTL(view) {
        const numTriangles = view.getUint32(80, true);
        const triangles = new Float32Array(numTriangles * 9); // 3 vertices * 3 components

        let offset = 84; // Skip 80-byte header + 4-byte count
        let floatIndex = 0;

        for (let i = 0; i < numTriangles; i++) {
            // Skip normal (12 bytes)
            offset += 12;

            // Read 3 vertices (9 floats)
            for (let j = 0; j < 9; j++) {
                triangles[floatIndex++] = view.getFloat32(offset, true);
                offset += 4;
            }

            // Skip attribute byte count (2 bytes)
            offset += 2;
        }

        return triangles;
    }
}
