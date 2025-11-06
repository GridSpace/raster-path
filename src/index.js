// raster-path: Terrain and Tool Raster Path Finder using WebGPU
// Main ESM entry point

/**
 * Configuration options for RasterPath
 * @typedef {Object} RasterPathConfig
 * @property {'planar'|'radial'} mode - Rasterization mode (default: 'planar')
 * @property {boolean} autoTiling - Automatically tile large datasets (default: true)
 * @property {number} gpuMemorySafetyMargin - Safety margin as percentage (default: 0.8 = 80%)
 * @property {number} maxConcurrentTiles - Max concurrent tiles for radial rasterization (default: 50)
 * @property {number} maxGPUMemoryMB - Maximum GPU memory per tile (default: 256MB)
 * @property {number} minTileSize - Minimum tile dimension (default: 50mm)
 * @property {number} radialRotationOffset - Radial mode: rotation offset in degrees (default: 0, use 90 to start at Z-axis)
 * @property {number} resolution - Grid step size in mm (required)
 * @property {number} rotationStep - Radial mode only: degrees between rays (e.g., 1.0 = 360 rays)
 * @property {number} trianglesPerTile - Target triangles per tile for radial rasterization (default: calculated)
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
        let urlOpt = [];
        if (config.quiet) {
            debug.log = function() {};
            urlOpt.push('quiet');
        }
        if (config.debug) {
            urlOpt.push('debug');
        }

        // Configuration with defaults
        this.config = {
            workerName: (config.workerName ?? "webgpu-worker.js") + (urlOpt.length ? "?"+urlOpt.join('&') : ""),
            maxGPUMemoryMB: config.maxGPUMemoryMB ?? 256,
            gpuMemorySafetyMargin: config.gpuMemorySafetyMargin ?? 0.8,
            autoTiling: config.autoTiling ?? true,
            minTileSize: config.minTileSize ?? 50,
            maxConcurrentTiles: config.maxConcurrentTiles ?? 10,
            trianglesPerTile: config.trianglesPerTile, // undefined = auto-calculate
            radialRotationOffset: config.radialRotationOffset ?? 0, // degrees
        };
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
                // Create worker from the webgpu-worker.js file
                const workerName = this.config.workerName;
                const isBuildVersion = import.meta.url.includes('/build/') || import.meta.url.includes('raster-path.js');
                const workerPath = workerName
                    ? new URL(workerName, import.meta.url)
                : isBuildVersion
                    ? new URL(`./webgpu-worker.js`, import.meta.url)
                    : new URL(`./web/webgpu-worker.js`, import.meta.url);
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
     * Rasterize model mesh to terrain heightmap
     * @param {object} params - Parameters
     * @param {Float32Array} params.triangles - Unindexed triangle vertices
     * @param {number} params.zFloor - Z floor for out-of-bounds (optional)
     * @param {object} params.boundsOverride - Optional bounding box {min: {x, y, z}, max: {x, y, z}}
     * @param {function} params.onProgress - Optional progress callback (percent, info) => {}
     * @returns {Promise<object>} Terrain data (format depends on mode)
     */
    async rasterizeModel({ triangles, zFloor, boundsOverride, onProgress }) {
        if (!this.isInitialized) {
            throw new Error('RasterPath not initialized. Call init() first.');
        }

        if (this.mode === 'planar') {
            return this.#rasterizePlanar({ triangles, zFloor, boundsOverride, isForTool: false, onProgress });
        } else {
            throw new Error('Use rasterizeModelRadial() for radial mode');
        }
    }

    /**
     * Rasterize tool mesh
     * @param {object} params - Parameters
     * @param {Float32Array} params.triangles - Unindexed triangle vertices
     * @param {number} params.zFloor - Z floor for out-of-bounds (optional)
     * @param {object} params.boundsOverride - Optional bounding box {min: {x, y, z}, max: {x, y, z}}
     * @returns {Promise<object>} Tool data (sparse format: [gridX, gridY, Z, ...])
     */
    async rasterizeTool({ triangles, zFloor, boundsOverride }) {
        if (!this.isInitialized) {
            throw new Error('RasterPath not initialized. Call init() first.');
        }
        const toolData = await this.#rasterizePlanar({ triangles, zFloor, boundsOverride, isForTool: true });
        const { bounds, positions } = toolData;
        for (let i=0; i<positions.length; i += 3) {
            positions[i+2] = -positions[i+2] - bounds.min.z;
        }
        let swapZ = bounds.min.z;
        bounds.min.z = -bounds.max.z;
        bounds.max.z = -swapZ;
        return toolData;
    }

    /**
     * Generate toolpaths from terrain and tool data
     * @param {object} params - Parameters
     * @param {object} params.terrainData - Output from rasterizeModel()
     * @param {object} params.toolData - Output from rasterizeTool()
     * @param {number} params.xStep - X-axis step size in grid points
     * @param {number} params.yStep - Y-axis step size in grid points
     * @param {number} params.zFloor - Z floor value for out-of-bounds
     * @param {number} params.radiusOffset - Radial only: tool offset above terrain (mm), default 20
     * @param {function} params.onProgress - Optional progress callback (percent, info) => {}
     * @returns {Promise<object>} Toolpath data
     */
    async generateToolpaths({ terrainData, toolData, xStep, yStep, zFloor, radiusOffset = 20, onProgress }) {
        if (!this.isInitialized) {
            throw new Error('RasterPath not initialized. Call init() first.');
        }

        if (this.mode === 'planar') {
            return this.#generateToolpathsPlanar({ terrainData, toolData, xStep, yStep, zFloor, onProgress });
        } else {
            throw new Error('Use generateToolpathsRadial() for radial mode');
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

    // ============================================================================
    // Radial V2 Methods
    // ============================================================================

    /**
     * Radial mode: Rasterize model into angular strips
     * @param {object} params - Parameters
     * @param {Float32Array} params.triangles - Unindexed triangle vertices
     * @param {object} params.toolData - Tool data (required for maxRadius calculation)
     * @param {number} params.zFloor - Z floor for out-of-bounds (optional)
     * @param {object} params.boundsOverride - Optional bounding box {min: {x, y, z}, max: {x, y, z}}
     * @param {function} params.onProgress - Optional progress callback (angle, totalAngles) => {}
     * @returns {Promise<Array>} Array of strip objects [{angle, positions, gridWidth, gridHeight, bounds}, ...]
     */
    async rasterizeModelRadial({ triangles, toolData, zFloor, boundsOverride, onProgress }) {
        if (!this.isInitialized) {
            throw new Error('RasterPath not initialized. Call init() first.');
        }
        if (this.mode !== 'radial') {
            throw new Error('rasterizeModelRadial() only works in radial mode');
        }
        if (!toolData) {
            throw new Error('toolData is required for radial rasterization (tool must be loaded first)');
        }

        // Calculate bounds
        const originalBounds = boundsOverride || this.#calculateBounds(triangles);

        // Center model in YZ plane (required for radial rasterization)
        const centerY = (originalBounds.min.y + originalBounds.max.y) / 2;
        const centerZ = (originalBounds.min.z + originalBounds.max.z) / 2;

        let centeredTriangles = triangles;
        let bounds = originalBounds;

        if (Math.abs(centerY) > 0.001 || Math.abs(centerZ) > 0.001) {
            debug.log(`[RasterPath] Centering model in YZ: offset Y=${centerY.toFixed(3)}, Z=${centerZ.toFixed(3)}`);
            centeredTriangles = new Float32Array(triangles.length);
            for (let i = 0; i < triangles.length; i += 3) {
                centeredTriangles[i] = triangles[i];                    // X unchanged
                centeredTriangles[i + 1] = triangles[i + 1] - centerY;  // Center Y
                centeredTriangles[i + 2] = triangles[i + 2] - centerZ;  // Center Z
            }
            // Recalculate bounds after centering
            bounds = this.#calculateBounds(centeredTriangles);
        }

        const maxRadius = this.#calculateMaxRadius(centeredTriangles);

        // Calculate tool width from toolData bounds
        const toolWidth = Math.max(
            toolData.bounds.max.y - toolData.bounds.min.y,
            toolData.bounds.max.z - toolData.bounds.min.z
        );

        // Create X-buckets for triangle attention
        const bucketWidth = 1.0; // mm - smaller buckets = better load balancing (was 2.0mm)
        const bucketData = this.#bucketTrianglesByX(centeredTriangles, bounds, bucketWidth);

        // Calculate number of angles - full 360° rotation
        const numAngles = Math.ceil(360 / this.rotationStep);

        // Send to worker for GPU processing
        const result = await new Promise((resolve, reject) => {
            const handler = (data) => resolve(data);

            this.#sendMessage(
                'radial-rasterize-v2',
                {
                    triangles: centeredTriangles,
                    bucketData,
                    resolution: this.resolution,
                    angleStep: this.rotationStep,
                    numAngles,
                    maxRadius: maxRadius * 1.01, // Safety margin
                    toolWidth,
                    zFloor: zFloor ?? 0,
                    bounds
                },
                'radial-rasterize-v2-complete',
                handler
            );
        });

        // Stitch strips (result.strips is array of {angle, positions, gridWidth, gridHeight, bounds})
        return result.strips;
    }

    /**
     * Radial mode: Generate toolpaths from strips (DEPRECATED - use rasterizeAndGenerateToolpathsRadial)
     * @param {object} params - Parameters
     * @param {Array} params.strips - Output from rasterizeModelRadial()
     * @param {object} params.toolData - Output from rasterizeTool()
     * @param {number} params.xStep - X-axis step size in grid points
     * @param {number} params.yStep - Y-axis step size in grid points
     * @param {number} params.zFloor - Z floor value for out-of-bounds
     * @param {function} params.onProgress - Optional progress callback (current, total) => {}
     * @returns {Promise<object>} Combined toolpath data { strips: [{angle, pathData, ...}, ...], totalPoints }
     */
    async generateToolpathsRadial({ strips, toolData, xStep, yStep, zFloor, onProgress }) {
        if (!this.isInitialized) {
            throw new Error('RasterPath not initialized. Call init() first.');
        }
        if (this.mode !== 'radial') {
            throw new Error('generateToolpathsRadial() only works in radial mode');
        }

        // Process each strip through planar toolpath generator
        const stripToolpaths = [];
        let totalPoints = 0;

        for (let i = 0; i < strips.length; i++) {
            const strip = strips[i];

            // Skip empty strips
            if (strip.pointCount === 0 || !strip.positions || strip.positions.length === 0) {
                if (onProgress) {
                    onProgress(i + 1, strips.length);
                }
                continue;
            }

            // For radial, we want a single scanline down the center of the strip
            // Use singleScanline flag to force centerline-only processing
            const toolpathResult = await this.#generateToolpathsPlanar({
                terrainData: strip,
                toolData,
                xStep,
                yStep,  // Ignored when singleScanline=true
                zFloor,
                onProgress: null, // Don't forward progress for individual strips
                singleScanline: true  // Force single centerline scan
            });

            stripToolpaths.push({
                angle: strip.angle,
                ...toolpathResult
            });

            totalPoints += toolpathResult.pathData.length;

            // Report progress
            if (onProgress) {
                onProgress(i + 1, strips.length);
            }
        }

        return {
            strips: stripToolpaths,
            totalPoints,
            numStrips: strips.length
        };
    }

    /**
     * Radial mode: Complete pipeline - rasterize model + generate toolpaths (ALL IN WORKER)
     * This is more efficient than separate calls as it avoids transferring intermediate strip data
     * @param {object} params - Parameters
     * @param {Float32Array} params.triangles - Model triangles
     * @param {object} params.toolData - Output from rasterizeTool()
     * @param {number} params.xStep - X-axis step size in grid points
     * @param {number} params.yStep - Y-axis step size in grid points
     * @param {number} params.zFloor - Z floor value for out-of-bounds
     * @param {function} params.onProgress - Optional progress callback (current, total) => {}
     * @returns {Promise<object>} Toolpath data { strips: [{angle, pathData, ...}, ...], totalPoints, numStrips }
     */
    async rasterizeAndGenerateToolpathsRadial({ triangles, toolData, xStep, yStep, zFloor, onProgress }) {
        if (!this.isInitialized) {
            throw new Error('RasterPath not initialized. Call init() first.');
        }
        if (this.mode !== 'radial') {
            throw new Error('rasterizeAndGenerateToolpathsRadial() only works in radial mode');
        }

        // Calculate bounds
        const originalBounds = this.#calculateBounds(triangles);

        // Center model in YZ plane (required for radial rasterization)
        const centerY = (originalBounds.min.y + originalBounds.max.y) / 2;
        const centerZ = (originalBounds.min.z + originalBounds.max.z) / 2;

        let centeredTriangles = triangles;
        let bounds = originalBounds;

        if (Math.abs(centerY) > 0.001 || Math.abs(centerZ) > 0.001) {
            debug.log(`[RasterPath] Centering model in YZ: offset Y=${centerY.toFixed(3)}, Z=${centerZ.toFixed(3)}`);
            centeredTriangles = new Float32Array(triangles.length);
            for (let i = 0; i < triangles.length; i += 3) {
                centeredTriangles[i] = triangles[i];                    // X unchanged
                centeredTriangles[i + 1] = triangles[i + 1] - centerY;  // Center Y
                centeredTriangles[i + 2] = triangles[i + 2] - centerZ;  // Center Z
            }
            // Recalculate bounds after centering
            bounds = this.#calculateBounds(centeredTriangles);
        }

        const maxRadius = this.#calculateMaxRadius(centeredTriangles);

        // Calculate tool width for radial strips
        const toolBounds = toolData.bounds;
        const toolWidth = Math.max(
            Math.abs(toolBounds.max.y - toolBounds.min.y),
            Math.abs(toolBounds.max.x - toolBounds.min.x)
        ) * this.resolution;

        // Build X-bucketing data - full 360° rotation
        const numAngles = Math.ceil(360 / this.rotationStep);
        const bucketWidth = 1.0; // mm - smaller buckets = better load balancing (was 2.0mm)
        const bucketData = this.#bucketTrianglesByX(centeredTriangles, bounds, bucketWidth);

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
                    triangles: centeredTriangles,
                    bucketData,
                    toolData,
                    resolution: this.resolution,
                    angleStep: this.rotationStep,
                    numAngles,
                    maxRadius: maxRadius * 1.01,
                    toolWidth,
                    zFloor: zFloor ?? 0,
                    bounds,
                    xStep,
                    yStep,
                    gridStep: this.resolution
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
