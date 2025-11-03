// raster-path: Terrain and Tool Raster Path Finder using WebGPU
// Main ESM entry point

/**
 * Configuration options for RasterPath
 * @typedef {Object} RasterPathConfig
 * @property {'planar'|'radial'} mode - Rasterization mode (default: 'planar')
 * @property {number} resolution - Grid step size in mm (required)
 * @property {number} rotationStep - Radial mode only: degrees between rays (e.g., 1.0 = 360 rays)
 * @property {number} maxGPUMemoryMB - Maximum GPU memory per tile (default: 256MB)
 * @property {number} gpuMemorySafetyMargin - Safety margin as percentage (default: 0.8 = 80%)
 * @property {boolean} autoTiling - Automatically tile large datasets (default: true)
 * @property {number} minTileSize - Minimum tile dimension (default: 50mm)
 * @property {boolean} quiet - Suppress log output (default: false)
 * @property {boolean} debug - Enable debug logging (default: false)
 * @property {number} maxConcurrentTiles - Max concurrent tiles for radial rasterization (default: 50)
 * @property {number} trianglesPerTile - Target triangles per tile for radial rasterization (default: calculated)
 * @property {number} radialRotationOffset - Radial mode: rotation offset in degrees (default: 0, use 90 to start at Z-axis)
 */

const ZMAX = 10e6;
const EMPTY_CELL = -1e10;

const debug = {
    error: console.error,
    warn: console.warn,
    log: console.log
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
            return this.#rasterizeRadial({ triangles, zFloor, boundsOverride, onProgress });
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

        if (this.mode === 'planar') {
            return this.#rasterizePlanarTool({ triangles, zFloor, boundsOverride });
        } else {
            return this.#rasterizeRadialTool({ triangles, zFloor, boundsOverride });
        }
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
            return this.#generateToolpathsRadial({ terrainData, toolData, xStep, yStep, zFloor, radiusOffset, onProgress });
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
                    isForTool,
                    boundsOverride
                },
                'rasterize-complete',
                handler
            );
        });

        return data;
    }

    async #rasterizePlanarTool({ triangles, zFloor, boundsOverride }) {
        return this.#rasterizePlanar({ triangles, zFloor, boundsOverride, isForTool: true });
    }

    async #generateToolpathsPlanar({ terrainData, toolData, xStep, yStep, zFloor, onProgress }) {
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
                    terrainBounds: terrainData.bounds
                },
                'toolpath-complete',
                handler
            );
        });
    }

    // ============================================================================
    // Internal Methods (Radial)
    // ============================================================================

    async #rasterizeRadial({ triangles, zFloor, boundsOverride, onProgress }) {
        const data = await new Promise((resolve, reject) => {
            // Set up progress handler if callback provided
            if (onProgress) {
                const progressHandler = (data) => {
                    onProgress(data.percent, { current: data.current, total: data.total });
                };
                this.messageHandlers.set('rasterize-progress', progressHandler);
            }

            const handler = (data) => {
                // Clean up progress handler
                if (onProgress) {
                    this.messageHandlers.delete('rasterize-progress');
                }
                resolve(data);
            };

            this.#sendMessage(
                'radial-rasterize',
                {
                    triangles,
                    stepSize: this.resolution,
                    rotationStep: this.rotationStep,
                    zFloor: zFloor ?? 0,
                    boundsOverride,
                    maxConcurrentTiles: this.config.maxConcurrentTiles,
                    trianglesPerTile: this.config.trianglesPerTile,
                    radialRotationOffset: this.config.radialRotationOffset
                },
                'radial-rasterize-complete',
                handler
            );
        });

        return data;
    }

    async #rasterizeRadialTool({ triangles, zFloor, boundsOverride }) {
        // 1. Rasterize tool as planar (standard tool raster)
        const planarToolData = await this.#rasterizePlanar({
            triangles,
            zFloor,
            boundsOverride,
            isForTool: true
        });

        // 2. Apply hemispherical morphing for cylindrical surface
        // Note: This assumes terrainData has already been rasterized and has circumference/maxRadius
        // We'll defer morphing until generateToolpaths where we have terrainData
        // For now, just return the planar tool data
        return planarToolData;
    }

    async #generateToolpathsRadial({ terrainData, toolData, xStep, yStep, zFloor, radiusOffset, onProgress }) {
        // 1. Morph tool for cylindrical surface
        const morphedTool = this.#morphTool(toolData, terrainData, radiusOffset);

        // 2. Stitch radial tiles into dense array for planar toolpath processing
        const { terrainPositions, bounds } = this.#prepareTerrainForToolpath(terrainData);

        // 3. Generate toolpaths using planar algorithm
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

                // Inject bounds for proper visualization
                data.generationBounds = bounds;
                resolve(data);
            };

            this.#sendMessage(
                'generate-toolpath',
                {
                    terrainPositions,
                    toolPositions: morphedTool.positions,
                    xStep,
                    yStep,
                    zFloor: zFloor ?? 0,
                    gridStep: this.resolution,
                    terrainBounds: bounds,
                    isRadial: true
                },
                'toolpath-complete',
                handler
            );
        });
    }

    /**
     * Morph planar tool to cylindrical surface with hemispherical compensation
     * @private
     */
    #morphTool(planarToolData, terrainData, radiusOffset = 20) {
        const { positions, bounds, pointCount } = planarToolData;
        const { circumference, maxRadius } = terrainData;

        debug.log('[RasterPath] Morphing tool for cylindrical surface:', {
            pointCount,
            circumference,
            maxRadius,
            radiusOffset,
            toolBounds: bounds
        });

        // Find tool center and radius for hemispherical morphing
        const toolCenterX = (bounds.max.x + bounds.min.x) / 2;
        const toolCenterY = (bounds.max.y + bounds.min.y) / 2;
        const toolRadiusX = (bounds.max.x - bounds.min.x) / 2;
        const toolRadiusY = (bounds.max.y - bounds.min.y) / 2;
        const toolRadius = Math.max(toolRadiusX, toolRadiusY);

        // Modify Z values IN PLACE to apply hemispherical compensation
        for (let i = 0; i < positions.length; i += 3) {
            const gridX = positions[i];     // Grid index X (unchanged)
            const gridY = positions[i + 1]; // Grid index Y (unchanged)
            const z = positions[i + 2];     // Tool Z value (will be modified)

            // Convert grid indices to world coordinates to calculate distance from center
            const wx = bounds.min.x + gridX * this.resolution;
            const wy = bounds.min.y + gridY * this.resolution;

            // Calculate normalized distance from tool center
            const dx = wx - toolCenterX;
            const dy = wy - toolCenterY;
            const distFromCenter = Math.sqrt(dx * dx + dy * dy);
            const normalizedDist = Math.min(distFromCenter / toolRadius, 1.0);

            // Apply hemispherical morphing (0 at center, increases to toolRadius at edges)
            const hemisphereZ = toolRadius * (1 - Math.sqrt(1 - normalizedDist * normalizedDist));

            // Add hemisphereZ to create hemisphere shape
            positions[i + 2] = z + hemisphereZ;
        }

        // Return the same data object (positions modified in place)
        return planarToolData;
    }

    /**
     * Prepare radial terrain for toolpath generation
     * Handles both tiled and non-tiled radial data
     * @private
     */
    #prepareTerrainForToolpath(terrainData) {
        // If already a dense array (non-tiled), return as-is
        if (terrainData.isDense && terrainData.positions) {
            debug.log('[RasterPath] Using dense radial terrain:', terrainData.gridWidth, 'x', terrainData.gridHeight);
            return {
                terrainPositions: terrainData.positions,
                bounds: terrainData.bounds
            };
        }

        // Otherwise, stitch tiles
        const tiles = terrainData.tiles;
        const gridHeight = tiles[0].gridHeight; // All tiles have same height (circumference)
        const stepSize = this.resolution;

        // Find global X range from tiles (in original STL coordinates)
        let globalMinX = Infinity;
        let globalMaxX = -Infinity;
        for (const tile of tiles) {
            globalMinX = Math.min(globalMinX, tile.minX);
            globalMaxX = Math.max(globalMaxX, tile.maxX);
        }

        // Calculate total grid width
        const totalGridWidth = Math.ceil((globalMaxX - globalMinX) / stepSize) + 1;

        debug.log('[RasterPath] Stitching', tiles.length, 'tiles into', totalGridWidth, 'x', gridHeight, 'dense array');
        debug.log('[RasterPath] Global X range:', globalMinX, 'to', globalMaxX);

        // Create single dense array (row-major: Y * width + X)
        const terrainPositions = new Float32Array(totalGridWidth * gridHeight);
        terrainPositions.fill(EMPTY_CELL);

        for (let tileIdx = 0; tileIdx < tiles.length; tileIdx++) {
            const tile = tiles[tileIdx];
            const tileGridWidth = tile.gridWidth;
            const tileGridHeight = tile.gridHeight;

            // Calculate this tile's offset in the global grid
            const tileStartGridX = Math.round((tile.minX - globalMinX) / stepSize);

            debug.log(`[RasterPath] Tile ${tileIdx}: minX=${tile.minX}, gridWidth=${tileGridWidth}, startGridX=${tileStartGridX}`);

            for (let gy = 0; gy < tileGridHeight; gy++) {
                for (let gx = 0; gx < tileGridWidth; gx++) {
                    const localTileIdx = gy * tileGridWidth + gx;
                    const globalGx = tileStartGridX + gx;
                    const globalIdx = gy * totalGridWidth + globalGx;

                    if (globalGx < 0 || globalGx >= totalGridWidth) {
                        debug.warn(`[RasterPath] OUT OF BOUNDS! tile ${tileIdx}, gx=${gx}, globalGx=${globalGx}, totalGridWidth=${totalGridWidth}`);
                        continue;
                    }

                    terrainPositions[globalIdx] = tile.data[localTileIdx];
                }
            }
        }

        // Count empty cells for diagnostics
        let emptyCells = 0;
        for (let i = 0; i < terrainPositions.length; i++) {
            if (terrainPositions[i] <= -1e9) emptyCells++;
        }
        debug.log('[RasterPath] Stitched terrain:', terrainPositions.length, 'cells,', emptyCells, 'empty (', (emptyCells / terrainPositions.length * 100).toFixed(1), '%)');

        // Create bounds using original STL X coordinates
        const bounds = {
            min: {
                x: globalMinX,  // Original STL X min
                y: 0,           // Angle starts at 0
                z: terrainData.bounds.min.z
            },
            max: {
                x: globalMaxX,              // Original STL X max
                y: terrainData.circumference,  // Full circumference
                z: terrainData.bounds.max.z
            }
        };

        debug.log('[RasterPath] Bounds:', bounds);

        return {
            terrainPositions,
            bounds
        };
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
