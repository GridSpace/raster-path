import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { parseSTL } from './parse-stl.js?v=1';
import { RasterPath } from './raster-path.js';

// Configuration
let STEP_SIZE = 0.1; // mm (default, can be changed via dropdown)

// DOM elements
const canvas = document.getElementById('canvas');
const terrainFileInput = document.getElementById('terrain-file-input');
const terrainFilenameEl = document.getElementById('terrain-filename');
const stepSizeSelect = document.getElementById('step-size-select');
const recomputeBtn = document.getElementById('recompute-btn');
const infoPanel = document.getElementById('info');
const statusEl = document.getElementById('status');
const pointCountEl = document.getElementById('point-count');
const boundsEl = document.getElementById('bounds');

// Bounds override UI elements
const useBoundsOverride = document.getElementById('use-bounds-override');
const boundsInputsDiv = document.getElementById('bounds-inputs');
const boundsMinX = document.getElementById('bounds-min-x');
const boundsMinY = document.getElementById('bounds-min-y');
const boundsMinZ = document.getElementById('bounds-min-z');
const boundsMaxX = document.getElementById('bounds-max-x');
const boundsMaxY = document.getElementById('bounds-max-y');
const boundsMaxZ = document.getElementById('bounds-max-z');
const boundsFromSTLBtn = document.getElementById('bounds-from-stl-btn');
const boundsAddPaddingBtn = document.getElementById('bounds-add-padding-btn');

// Mode switch elements
const toolpathModeRadios = document.getElementsByName('toolpath-mode');

// Toolpath UI elements
const toolFileInput = document.getElementById('tool-file-input');
const toolFilenameEl = document.getElementById('tool-filename');
const xStepInput = document.getElementById('x-step-input');
const yStepInput = document.getElementById('y-step-input');
const yStepControl = document.getElementById('y-step-control');
const xRotationStepInput = document.getElementById('x-rotation-step-input');
const xRotationStepControl = document.getElementById('x-rotation-step-control');
const zFloorInput = document.getElementById('z-floor-input');
const toolpathControls = document.getElementById('toolpath-controls');
const generateToolpathBtn = document.getElementById('generate-toolpath-btn');
const clearToolpathBtn = document.getElementById('clear-toolpath-btn');
const showTerrainCheckbox = document.getElementById('show-terrain');
const showTerrainLabel = document.getElementById('show-terrain-label');
const terrainControl = document.getElementById('terrain-control');
const showUnwrappedCheckbox = document.getElementById('show-unwrapped');
const unwrapControl = document.getElementById('unwrap-control');
const showMeshCheckbox = document.getElementById('show-mesh');

// Timing panel elements
const timingPanel = document.getElementById('timing-panel');
const timingTerrainEl = document.getElementById('timing-terrain');
const timingToolEl = document.getElementById('timing-tool');
const timingToolpathEl = document.getElementById('timing-toolpath');
const memoryTerrainEl = document.getElementById('memory-terrain');
const memoryToolEl = document.getElementById('memory-tool');
const memoryToolpathEl = document.getElementById('memory-toolpath');
const memoryTotalEl = document.getElementById('memory-total');
const tilingStatusEl = document.getElementById('tiling-status');
const gpuLimitEl = document.getElementById('gpu-limit');

// Store last loaded file for recompute
let lastLoadedFile = null;

// Three.js scene setup
let scene, camera, renderer, controls;
let pointCloud = null;
let terrainMesh = null;  // For radial mode mesh display
let toolMesh = null;     // For tool mesh display
let toolCloud = null;
let toolpathCloud = null;
let terrainWorker = null;
let toolWorker = null;
let toolpathWorker = null;

// Toolpath mode: 'planar' or 'radial'
let toolpathMode = 'planar';

// Store terrain and tool data for toolpath generation
let terrainData = null;
let terrainTriangles = null;  // Store original triangles for radial mode
let radialTileData = null;  // Store radial v2 tile data for unwrapped visualization
let toolData = null;
let toolFile = null;
let toolRasterizingInProgress = false;  // Guard against duplicate tool rasterization
let radialAdjustedBounds = null;  // Store adjusted bounds for radial toolpath visualization
let webgpuWorker = null;
let planarRasterPath = null;  // RasterPath API instance for planar mode
let radialRasterPath = null;  // RasterPath API instance for radial mode

// Store STL bounds for bounds override feature
let stlBounds = null;

// Device capabilities
let deviceCapabilities = null;

// Timing and memory data
let timingData = {
    terrainConversion: null,
    toolConversion: null,
    toolpathGeneration: null
};

let memoryData = {
    terrain: null,
    tool: null,
    toolpath: null
};

// Helper: Get or create RasterPath instance for current mode
async function getRasterPath() {
    if (toolpathMode === 'planar') {
        if (!planarRasterPath || planarRasterPath.resolution !== STEP_SIZE) {
            // Terminate old instance if resolution changed
            if (planarRasterPath) {
                planarRasterPath.terminate();
            }
            planarRasterPath = new RasterPath({
                mode: 'planar',
                resolution: STEP_SIZE
            });
            await planarRasterPath.init();
            console.log(`✓ Planar RasterPath initialized (resolution: ${STEP_SIZE}mm)`);
        }
        return planarRasterPath;
    } else {  // radial mode
        if (!radialRasterPath || radialRasterPath.resolution !== STEP_SIZE) {
            // Terminate old instance if resolution changed
            if (radialRasterPath) {
                radialRasterPath.terminate();
            }
            radialRasterPath = new RasterPath({
                mode: 'radial',
                resolution: STEP_SIZE,
                rotationStep: 1.0  // 1 degree between rays = 360 rays
            });
            await radialRasterPath.init();
            console.log(`✓ Radial RasterPath initialized (resolution: ${STEP_SIZE}mm, rotationStep: 1.0°)`);
        }
        return radialRasterPath;
    }
}

// Helper: Format bytes to human readable
function formatBytes(bytes) {
    if (bytes === 0 || bytes === null) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Helper: Calculate memory usage of data
function calculateMemory(data) {
    if (!data) return 0;
    let bytes = 0;

    // Positions (Float32Array)
    if (data.positions) {
        bytes += data.positions.byteLength;
    }

    // Path data (Float32Array)
    if (data.pathData) {
        bytes += data.pathData.byteLength;
    }

    // Add overhead for object structure (rough estimate)
    bytes += 1024; // ~1KB overhead

    return bytes;
}

// Helper: Get bounds override from UI inputs
function getBoundsOverride() {
    if (!useBoundsOverride.checked) {
        return null;
    }

    const minX = parseFloat(boundsMinX.value);
    const minY = parseFloat(boundsMinY.value);
    const minZ = parseFloat(boundsMinZ.value);
    const maxX = parseFloat(boundsMaxX.value);
    const maxY = parseFloat(boundsMaxY.value);
    const maxZ = parseFloat(boundsMaxZ.value);

    if (isNaN(minX) || isNaN(minY) || isNaN(minZ) ||
        isNaN(maxX) || isNaN(maxY) || isNaN(maxZ)) {
        alert('Please fill in all bounds values');
        return null;
    }

    return {
        min: { x: minX, y: minY, z: minZ },
        max: { x: maxX, y: maxY, z: maxZ }
    };
}

// Settings persistence
const SETTINGS_KEY = 'raster-path-settings';

function saveSettings() {
    const settings = {
        stepSize: stepSizeSelect.value,
        toolpathMode: toolpathMode,
        useBoundsOverride: useBoundsOverride.checked,
        boundsMinX: boundsMinX.value,
        boundsMinY: boundsMinY.value,
        boundsMinZ: boundsMinZ.value,
        boundsMaxX: boundsMaxX.value,
        boundsMaxY: boundsMaxY.value,
        boundsMaxZ: boundsMaxZ.value,
        xStep: xStepInput.value,
        yStep: yStepInput.value,
        xRotationStep: xRotationStepInput.value,
        zFloor: zFloorInput.value
    };

    try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
        console.log('Settings saved');
    } catch (e) {
        console.warn('Failed to save settings:', e);
    }
}

function loadSettings() {
    try {
        const saved = localStorage.getItem(SETTINGS_KEY);
        if (!saved) return;

        const settings = JSON.parse(saved);

        // Restore step size
        if (settings.stepSize) {
            stepSizeSelect.value = settings.stepSize;
            STEP_SIZE = parseFloat(settings.stepSize);
        }

        // Restore toolpath mode
        if (settings.toolpathMode) {
            toolpathMode = settings.toolpathMode;
            toolpathModeRadios.forEach(radio => {
                radio.checked = (radio.value === toolpathMode);
            });
            // Show/hide appropriate controls and update label
            if (toolpathMode === 'radial') {
                showTerrainLabel.textContent = 'Show Model';
                yStepControl.style.display = 'none';
                xRotationStepControl.style.display = 'flex';
            } else {
                showTerrainLabel.textContent = 'Show Terrain';
                yStepControl.style.display = 'flex';
                xRotationStepControl.style.display = 'none';
            }
        }

        // Set Z floor default based on mode if not explicitly saved
        if (!settings.zFloor) {
            if (toolpathMode === 'radial') {
                zFloorInput.value = '0';
            } else {
                zFloorInput.value = '-100';
            }
        }

        // Restore bounds override
        if (settings.useBoundsOverride !== undefined) {
            useBoundsOverride.checked = settings.useBoundsOverride;
            boundsInputsDiv.style.display = settings.useBoundsOverride ? 'block' : 'none';
        }

        // Restore bounds values
        if (settings.boundsMinX) boundsMinX.value = settings.boundsMinX;
        if (settings.boundsMinY) boundsMinY.value = settings.boundsMinY;
        if (settings.boundsMinZ) boundsMinZ.value = settings.boundsMinZ;
        if (settings.boundsMaxX) boundsMaxX.value = settings.boundsMaxX;
        if (settings.boundsMaxY) boundsMaxY.value = settings.boundsMaxY;
        if (settings.boundsMaxZ) boundsMaxZ.value = settings.boundsMaxZ;

        // Restore toolpath settings
        if (settings.xStep) xStepInput.value = settings.xStep;
        if (settings.yStep) yStepInput.value = settings.yStep;
        if (settings.xRotationStep) xRotationStepInput.value = settings.xRotationStep;
        if (settings.zFloor) zFloorInput.value = settings.zFloor;

        console.log('Settings restored');
    } catch (e) {
        console.warn('Failed to load settings:', e);
    }
}

function initScene() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // Camera
    camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        10000
    );
    camera.position.set(50, 50, 50);

    // Renderer
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Grid helper
    const gridHelper = new THREE.GridHelper(100, 20, 0x444444, 0x222222);
    scene.add(gridHelper);

    // Axes helper (X=red, Y=green, Z=blue)
    const axesHelper = new THREE.AxesHelper(20);
    scene.add(axesHelper);

    // Add axis labels
    function createTextSprite(text, color) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 64;
        canvas.height = 64;

        context.clearRect(0, 0, 64, 64);
        context.font = 'Bold 40px Arial';
        context.fillStyle = color;
        context.textAlign = 'center';
        context.fillText(text, 32, 40);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(5, 5, 1);
        return sprite;
    }

    const xLabel = createTextSprite('X', '#ff0000');
    xLabel.position.set(25, 0, 0);
    scene.add(xLabel);

    const yLabel = createTextSprite('Z', '#00ff00');
    yLabel.position.set(0, 25, 0);
    scene.add(yLabel);

    const zLabel = createTextSprite('Y', '#0000ff');
    zLabel.position.set(0, 0, 25);
    scene.add(zLabel);

    // Window resize handler
    window.addEventListener('resize', onWindowResize);

    // Step size dropdown handler
    stepSizeSelect.addEventListener('change', (e) => {
        STEP_SIZE = parseFloat(e.target.value);
        console.log('Step size changed to:', STEP_SIZE, 'mm');
        saveSettings();
    });

    // X/Y step and rotation step input handlers
    xStepInput.addEventListener('change', () => saveSettings());
    yStepInput.addEventListener('change', () => saveSettings());
    xRotationStepInput.addEventListener('change', () => saveSettings());
    zFloorInput.addEventListener('change', () => saveSettings());

    // Bounds override checkbox handler
    useBoundsOverride.addEventListener('change', (e) => {
        boundsInputsDiv.style.display = e.target.checked ? 'block' : 'none';

        // If enabling and bounds are empty, auto-populate from terrain
        if (e.target.checked && !boundsMinX.value && terrainData && terrainData.bounds) {
            populateBoundsFromTerrain();
        }
        saveSettings();
    });

    // Helper to populate bounds from terrain data
    function populateBoundsFromTerrain() {
        if (terrainData && terrainData.bounds) {
            const bounds = terrainData.bounds;
            boundsMinX.value = bounds.min.x.toFixed(2);
            boundsMinY.value = bounds.min.y.toFixed(2);
            boundsMinZ.value = bounds.min.z.toFixed(2);
            boundsMaxX.value = bounds.max.x.toFixed(2);
            boundsMaxY.value = bounds.max.y.toFixed(2);
            boundsMaxZ.value = bounds.max.z.toFixed(2);
            console.log('Auto-populated bounds from terrain');
        }
    }

    // "From STL" button - populate bounds from last STL calculation
    boundsFromSTLBtn.addEventListener('click', () => {
        if (stlBounds) {
            boundsMinX.value = stlBounds.min.x.toFixed(2);
            boundsMinY.value = stlBounds.min.y.toFixed(2);
            boundsMinZ.value = stlBounds.min.z.toFixed(2);
            boundsMaxX.value = stlBounds.max.x.toFixed(2);
            boundsMaxY.value = stlBounds.max.y.toFixed(2);
            boundsMaxZ.value = stlBounds.max.z.toFixed(2);
            saveSettings();

            // In radial mode, changing bounds invalidates toolpath
            if (toolpathMode === 'radial' && toolpathCloud) {
                scene.remove(toolpathCloud);
                toolpathCloud.geometry.dispose();
                toolpathCloud.material.dispose();
                toolpathCloud = null;
                clearToolpathBtn.disabled = true;
                console.log('Cleared toolpath due to bounds change (radial mode)');
            }
        } else {
            alert('Please load an STL file first');
        }
    });

    // "Add Padding" button - add 10% padding to current bounds
    boundsAddPaddingBtn.addEventListener('click', () => {
        // Check if inputs are empty or invalid
        if (!boundsMinX.value || !boundsMinY.value || !boundsMinZ.value ||
            !boundsMaxX.value || !boundsMaxY.value || !boundsMaxZ.value) {
            alert('Please fill in all bounds values first (click "From STL" to populate)');
            return;
        }

        const minX = parseFloat(boundsMinX.value);
        const minY = parseFloat(boundsMinY.value);
        const minZ = parseFloat(boundsMinZ.value);
        const maxX = parseFloat(boundsMaxX.value);
        const maxY = parseFloat(boundsMaxY.value);
        const maxZ = parseFloat(boundsMaxZ.value);

        if (isNaN(minX) || isNaN(minY) || isNaN(minZ) ||
            isNaN(maxX) || isNaN(maxY) || isNaN(maxZ)) {
            alert('Invalid bounds values - please check your inputs');
            return;
        }

        if (minX >= maxX || minY >= maxY || minZ >= maxZ) {
            alert('Invalid bounds: min values must be less than max values');
            return;
        }

        const paddingX = (maxX - minX) * 0.1;
        const paddingY = (maxY - minY) * 0.1;
        const paddingZ = (maxZ - minZ) * 0.1;

        boundsMinX.value = (minX - paddingX).toFixed(2);
        boundsMinY.value = (minY - paddingY).toFixed(2);
        boundsMinZ.value = (minZ - paddingZ).toFixed(2);
        boundsMaxX.value = (maxX + paddingX).toFixed(2);
        boundsMaxY.value = (maxY + paddingY).toFixed(2);
        boundsMaxZ.value = (maxZ + paddingZ).toFixed(2);
        saveSettings();

        // In radial mode, changing bounds invalidates toolpath
        if (toolpathMode === 'radial' && toolpathCloud) {
            scene.remove(toolpathCloud);
            toolpathCloud.geometry.dispose();
            toolpathCloud.material.dispose();
            toolpathCloud = null;
            clearToolpathBtn.disabled = true;
            console.log('Cleared toolpath due to bounds padding (radial mode)');
        }
    });

    // Recompute button handler
    recomputeBtn.addEventListener('click', async () => {
        if (lastLoadedFile) {
            console.log('Recomputing with step size:', STEP_SIZE, 'mm');

            // Clear existing point cloud to prevent NaN errors
            if (pointCloud) {
                scene.remove(pointCloud);
                pointCloud.geometry.dispose();
                pointCloud.material.dispose();
                pointCloud = null;
                console.log('Cleared old point cloud');

                // Hide terrain control until new raster is created
                if (terrainControl) {
                    terrainControl.style.display = 'none';
                }
            }

            // Clear existing toolpath since resolution changed
            if (toolpathCloud) {
                scene.remove(toolpathCloud);
                toolpathCloud.geometry.dispose();
                toolpathCloud.material.dispose();
                toolpathCloud = null;
                clearToolpathBtn.disabled = true;
            }

            // Clear tool visualization
            if (toolCloud) {
                scene.remove(toolCloud);
                toolCloud.geometry.dispose();
                toolCloud.material.dispose();
                toolCloud = null;
            }

            // Reset timing data
            timingData.terrainConversion = null;
            timingData.toolConversion = null;
            timingData.toolpathGeneration = null;
            timingTerrainEl.textContent = '-';
            timingToolEl.textContent = '-';
            timingToolpathEl.textContent = '-';

            // Clear tool data so it needs to be regenerated
            toolData = null;

            // Recompute terrain using WebGPU
            console.log('Recomputing terrain with step size:', STEP_SIZE, 'mm');

            {
                // WebGPU path - pass rasterize flag for radial
                // Tool will be automatically rasterized in handleRadialRasterizeComplete
                processFileWebGPU(lastLoadedFile, { rasterize: true });
            }
        }
    });

    // Save settings when bounds inputs change
    boundsMinX.addEventListener('blur', saveSettings);
    boundsMinY.addEventListener('blur', saveSettings);
    boundsMinZ.addEventListener('blur', saveSettings);
    boundsMaxX.addEventListener('blur', saveSettings);
    boundsMaxY.addEventListener('blur', saveSettings);
    boundsMaxZ.addEventListener('blur', saveSettings);

    // Save settings when toolpath inputs change
    xStepInput.addEventListener('blur', saveSettings);
    yStepInput.addEventListener('blur', saveSettings);
    zFloorInput.addEventListener('blur', saveSettings);

    // Start render loop
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// Configuration
const NUM_PARALLEL_WORKERS = 4; // Number of parallel workers for toolpath generation

// Web Worker setup
async function initWorkers() {
    // Initialize WebGPU worker
    webgpuWorker = new Worker('webgpu-worker.js');

    const webgpuReady = new Promise((resolve) => {
        webgpuWorker.onmessage = function(e) {
            if (e.data.type === 'webgpu-ready') {
                deviceCapabilities = e.data.data.capabilities;
                resolve(e.data.data.success);
            }
        };
    });

    // Send init message
    webgpuWorker.postMessage({ type: 'init' });

    // Wait for WebGPU to initialize
    const webgpuAvailable = await webgpuReady;

    // Display device capabilities
    if (deviceCapabilities) {
        console.log('WebGPU Device Capabilities:', deviceCapabilities);
    }

    if (!webgpuAvailable) {
        console.error('❌ WebGPU not available - application requires WebGPU support');
        updateStatus('Error: WebGPU not available');
        return;
    }

    // Set up WebGPU worker message handler
    webgpuWorker.onmessage = function(e) {
        const { type, data, success, isForTool } = e.data;

        switch (type) {
            case 'webgpu-ready':
                // Already handled above
                break;

            case 'rasterize-complete':
                console.log('WebGPU worker: rasterization complete, points:', data.pointCount, 'isForTool:', isForTool);
                handleRasterizeComplete(data, isForTool);
                break;

            case 'radial-rasterize-complete':
                console.log('WebGPU worker: radial rasterization complete');
                console.log('  Grid:', data.gridWidth, 'x', data.gridHeight, '=', data.pointCount, 'cells');
                console.log('  Max radius:', data.maxRadius, 'Circumference:', data.circumference);
                handleRadialRasterizeComplete(data);
                break;

            case 'toolpath-complete':
                console.log('WebGPU worker: toolpath complete');
                handleToolpathComplete(data);
                // Hide progress bar
                const progressContainer = document.getElementById('progress-container');
                if (progressContainer) {
                    progressContainer.style.display = 'none';
                }
                break;

            case 'toolpath-progress':
                // Update progress bar
                const progressFill = document.getElementById('progress-fill');
                const progressText = document.getElementById('progress-text');
                if (progressFill && progressText) {
                    progressFill.style.width = `${data.percent}%`;
                    progressText.textContent = `${data.percent}% (${data.current}/${data.total})`;
                }
                break;

            case 'error':
                console.error('WebGPU worker error:', e.data.message);
                updateStatus('Error: ' + e.data.message);
                generateToolpathBtn.disabled = false;
                // Hide progress bar on error
                const errorProgressContainer = document.getElementById('progress-container');
                if (errorProgressContainer) {
                    errorProgressContainer.style.display = 'none';
                }
                break;
        }
    };

    webgpuWorker.onerror = function(error) {
        console.error('WebGPU worker error:', error);
        updateStatus('WebGPU worker error');
    };

    console.log('✓ WebGPU worker initialized');

    // Initialize RasterPath API instances
    // Note: Instances will be created on-demand when mode is selected to ensure correct resolution
    console.log('✓ RasterPath API ready (instances created on-demand)')

    // Auto-load last files from localStorage and display meshes (but don't rasterize)
    setTimeout(() => {
        const terrainFile = loadFileFromLocalStorage('lastTerrainFile');
        const toolFile = loadFileFromLocalStorage('lastToolFile');

        if (terrainFile) {
            console.log('💾 Auto-loading terrain:', terrainFile.name);
            lastLoadedFile = terrainFile;
            // Load and display mesh, but don't rasterize
            handleFile(terrainFile);
        }

        if (toolFile) {
            console.log('💾 Auto-loading tool:', toolFile.name);
            // Wait for terrain to load first
            setTimeout(() => {
                toolFileInput.files = createFileList(toolFile);
                toolFileInput.dispatchEvent(new Event('change'));
            }, 500);
        }
    }, 100);
}

// Helper to create FileList from File
function createFileList(file) {
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    return dataTransfer.files;
}

// Update status display
function updateStatus(status) {
    statusEl.textContent = status;
}

// Handle rasterization complete from WebGPU worker
function handleRasterizeComplete(data, isForTool) {
    console.log('handleRasterizeComplete:', {
        isForTool: isForTool,
        pointCount: data.pointCount,
        bounds: data.bounds,
        conversionTime: data.conversionTime,
        positionsLength: data.positions?.length
    });

    // Verify data integrity on main thread
    if (data.positions && data.positions.length > 0) {
        if (data.isDense) {
            // Dense format (terrain): Z-only array with sentinel value for empty cells
            const EMPTY_CELL = -1e10;
            const firstZ = data.positions[0] <= EMPTY_CELL + 1 ? 'EMPTY' : data.positions[0].toFixed(3);
            const lastZ = data.positions[data.positions.length-1] <= EMPTY_CELL + 1 ? 'EMPTY' : data.positions[data.positions.length-1].toFixed(3);
            console.log(`[Main Thread] Dense terrain: first Z=${firstZ}, last Z=${lastZ}, grid=${data.gridWidth}x${data.gridHeight}`);

            // Count empty cells (sentinel value)
            let emptyCount = 0;
            for (let i = 0; i < data.positions.length; i++) {
                if (data.positions[i] <= EMPTY_CELL + 1) emptyCount++;
            }
            const coverage = ((1 - emptyCount/data.positions.length) * 100).toFixed(1);
            console.log(`[Main Thread] Dense terrain: ${coverage}% coverage (${data.positions.length - emptyCount} valid cells)`);
        } else {
            // Sparse format (tool): X,Y,Z triplets
            const firstPoint = `(${data.positions[0].toFixed(3)}, ${data.positions[1].toFixed(3)}, ${data.positions[2].toFixed(3)})`;
            const lastIdx = data.positions.length - 3;
            const lastPoint = `(${data.positions[lastIdx].toFixed(3)}, ${data.positions[lastIdx+1].toFixed(3)}, ${data.positions[lastIdx+2].toFixed(3)})`;
            console.log(`[Main Thread] Sparse tool: first=${firstPoint}, last=${lastPoint}`);

            // Check for NaN or Infinity
            let hasInvalid = false;
            for (let i = 0; i < Math.min(data.positions.length, 100); i++) {
                if (!isFinite(data.positions[i])) {
                    hasInvalid = true;
                    console.error(`[Main Thread] Invalid value at index ${i}: ${data.positions[i]}`);
                    break;
                }
            }
            if (!hasInvalid) {
                console.log(`[Main Thread] Data integrity check passed`);
            }
        }
    }

    // Determine if this is terrain or tool based on parameter
    if (isForTool) {
        // Clear the rasterization in-progress flag
        toolRasterizingInProgress = false;

        // For radial mode, morph the planar tool raster to cylindrical surface
        if (toolpathMode === 'radial' && rasterPathRadial && radialTileData) {
            console.log('Morphing planar tool raster for radial mode');
            const morphedData = rasterPathRadial.morphTool(data, radialTileData, STEP_SIZE, 20);
            toolData = morphedData;
            displayTool(morphedData);
        } else {
            toolData = data;
            displayTool(data);
        }

        updateStatus('Tool complete');

        // Hide tool mesh after tool raster completes (in radial mode)
        if (toolpathMode === 'radial' && toolMesh) {
            toolMesh.visible = false;
            // Uncheck the "Show Original Mesh" checkbox
            if (showMeshCheckbox) {
                showMeshCheckbox.checked = false;
            }
        }

        // Store timing and memory
        if (data.conversionTime !== undefined) {
            timingData.toolConversion = data.conversionTime;
        }
        memoryData.tool = calculateMemory(data);
        updateTimingPanel();

        // Enable generate button
        generateToolpathBtn.disabled = false;
    } else {
        terrainData = data;
        displayPointCloud(data);
        updateStatus('Terrain complete');

        // Store timing and memory
        if (data.conversionTime !== undefined) {
            timingData.terrainConversion = data.conversionTime;
        }
        memoryData.terrain = calculateMemory(data);
        updateTimingPanel();

        // Auto-populate bounds if custom bounds is enabled and inputs are empty
        if (useBoundsOverride.checked && !boundsMinX.value) {
            populateBoundsFromTerrain();
        }

        // Enable generate button if tool is already loaded
        if (toolData) {
            generateToolpathBtn.disabled = false;
        }
    }
}

// Handle radial rasterization complete (unwrapped cylinder)
function handleRadialRasterizeComplete(data) {
    console.log('handleRadialRasterizeComplete:', {
        numTiles: data.numTiles,
        maxRadius: data.maxRadius,
        circumference: data.circumference,
        conversionTime: data.conversionTime,
        bounds: data.bounds
    });
    console.log('First tile:', {
        minX: data.tiles[0].minX,
        maxX: data.tiles[0].maxX,
        gridWidth: data.tiles[0].gridWidth,
        gridHeight: data.tiles[0].gridHeight
    });

    // Store as terrain data (keep tiles separate)
    terrainData = data;
    radialTileData = data;  // Store for unwrapped visualization

    // Show unwrap toggle for radial mode
    if (unwrapControl) {
        unwrapControl.style.display = 'block';
    }

    // Convert tiles to dense positions array for visualization
    let totalPoints = 0;
    for (const tile of data.tiles) {
        totalPoints += tile.data.length;
    }

    const positions = new Float32Array(totalPoints * 3);
    let posIdx = 0;

    const stepSize = data.stepSize;
    console.log('Reconstruction using stepSize:', stepSize, 'STEP_SIZE global:', STEP_SIZE);
    console.log('Circumference:', data.circumference, 'MaxRadius:', data.maxRadius);

    for (const tile of data.tiles) {
        const tileGridWidth = tile.gridWidth;
        const tileGridHeight = tile.gridHeight;

        for (let gy = 0; gy < tileGridHeight; gy++) {
            const theta = (gy * stepSize / data.circumference) * 2 * Math.PI;
            const cosTheta = Math.cos(theta);
            const sinTheta = Math.sin(theta);

            for (let gx = 0; gx < tileGridWidth; gx++) {
                const idx = gy * tileGridWidth + gx;
                const radius = tile.data[idx];

                if (radius > -1e9) { // Not EMPTY_CELL
                    const wx = tile.minX + gx * stepSize;
                    positions[posIdx++] = wx;
                    positions[posIdx++] = radius * cosTheta;
                    positions[posIdx++] = radius * sinTheta;
                }
            }
        }
    }

    // Trim to actual point count
    const actualPositions = positions.subarray(0, posIdx);
    const pointCount = posIdx / 3;

    // Create display data
    const displayData = {
        positions: actualPositions,
        pointCount: pointCount,
        bounds: data.bounds,
        conversionTime: data.conversionTime,
        isWorldCoordinates: true  // Positions are already in world XYZ, no conversion needed
    };

    displayPointCloud(displayData);

    updateStatus(`Radial raster complete: ${data.numTiles} tiles, ${pointCount.toLocaleString()} points (${(data.conversionTime / 1000).toFixed(1)}s)`);

    // Hide original meshes after rasterization completes
    if (terrainMesh) {
        terrainMesh.visible = false;
    }
    if (toolMesh) {
        toolMesh.visible = false;
    }
    // Uncheck the "Show Original Mesh" checkbox
    if (showMeshCheckbox) {
        showMeshCheckbox.checked = false;
    }

    // Store timing and memory
    if (data.conversionTime !== undefined) {
        timingData.terrainConversion = data.conversionTime;
    }
    memoryData.terrain = calculateMemory(displayData);
    updateTimingPanel();

    // If tool was already loaded but not rasterized, trigger tool rasterization now
    // (this happens on page reload or when clicking Rasterize button)
    if (toolFile && !toolData && !toolRasterizingInProgress) {
        console.log('Terrain rasterized, now rasterizing tool for morphing');
        toolRasterizingInProgress = true;
        setTimeout(async () => {
            try {
                const buffer = await toolFile.arrayBuffer();
                const { positions, triangleCount } = parseSTL(buffer);
                console.log('Auto-rasterizing tool after terrain:', triangleCount, 'triangles');

                // Copy positions before transferring
                const positionsCopy = new Float32Array(positions);

                webgpuWorker.postMessage({
                    type: 'rasterize',
                    data: {
                        triangles: positionsCopy,
                        stepSize: STEP_SIZE,
                        filterMode: 1,
                        isForTool: true,
                        boundsOverride: null
                    }
                }, [positionsCopy.buffer]);
            } catch (error) {
                console.error('Error auto-rasterizing tool:', error);
                toolRasterizingInProgress = false;
            }
        }, 100);
    }
}

// Display radial terrain as wrapped cylinder (3D)
function displayWrappedTerrain(data) {
    let totalPoints = 0;
    for (const tile of data.tiles) {
        totalPoints += tile.data.length;
    }

    const positions = new Float32Array(totalPoints * 3);
    let posIdx = 0;
    const stepSize = data.stepSize;

    for (const tile of data.tiles) {
        const tileGridWidth = tile.gridWidth;
        const tileGridHeight = tile.gridHeight;

        for (let gy = 0; gy < tileGridHeight; gy++) {
            const theta = (gy * stepSize / data.circumference) * 2 * Math.PI;
            const cosTheta = Math.cos(theta);
            const sinTheta = Math.sin(theta);

            for (let gx = 0; gx < tileGridWidth; gx++) {
                const idx = gy * tileGridWidth + gx;
                const radius = tile.data[idx];

                if (radius > -1e9) { // Not EMPTY_CELL
                    const wx = tile.minX + gx * stepSize;
                    positions[posIdx++] = wx;
                    positions[posIdx++] = radius * cosTheta;
                    positions[posIdx++] = radius * sinTheta;
                }
            }
        }
    }

    const actualPositions = positions.subarray(0, posIdx);
    const pointCount = posIdx / 3;

    const displayData = {
        positions: actualPositions,
        pointCount: pointCount,
        bounds: data.bounds,
        conversionTime: data.conversionTime,
        isWorldCoordinates: true
    };

    displayPointCloud(displayData);
}

// Display radial terrain as unwrapped flat 2D grid
function displayUnwrappedTerrain(data) {
    let totalPoints = 0;
    for (const tile of data.tiles) {
        totalPoints += tile.data.length;
    }

    const positions = new Float32Array(totalPoints * 3);
    let posIdx = 0;
    const stepSize = data.stepSize;

    for (const tile of data.tiles) {
        const tileGridWidth = tile.gridWidth;
        const tileGridHeight = tile.gridHeight;

        for (let gy = 0; gy < tileGridHeight; gy++) {
            // Y is the unwrapped circumferential distance
            const wy = gy * stepSize;

            for (let gx = 0; gx < tileGridWidth; gx++) {
                const idx = gy * tileGridWidth + gx;
                const radius = tile.data[idx];

                if (radius > -1e9) { // Not EMPTY_CELL
                    // Use original STL X coordinates
                    const wx = tile.minX + gx * stepSize;
                    positions[posIdx++] = wx;
                    positions[posIdx++] = wy;
                    positions[posIdx++] = radius;  // Z is the radius (height)
                }
            }
        }
    }

    const actualPositions = positions.subarray(0, posIdx);
    const pointCount = posIdx / 3;

    // Bounds for flat unwrapped view (using original STL X coordinates)
    const unwrappedBounds = {
        min: { x: data.bounds.min.x, y: 0, z: 0 },
        max: { x: data.bounds.max.x, y: data.circumference, z: data.maxRadius }
    };

    const displayData = {
        positions: actualPositions,
        pointCount: pointCount,
        bounds: unwrappedBounds,
        conversionTime: data.conversionTime,
        isWorldCoordinates: true
    };

    displayPointCloud(displayData);
}

// Handle toolpath complete from WebGPU worker
function handleToolpathComplete(data) {
    // For radial mode, inject the adjusted bounds for correct visualization
    if (toolpathMode === 'radial' && radialAdjustedBounds) {
        data.generationBounds = radialAdjustedBounds;
    }

    displayToolpath(data);
    updateStatus('Toolpath complete (webgpu)');
    generateToolpathBtn.disabled = false;

    // Store timing and memory
    timingData.toolpathGeneration = data.generationTime;
    memoryData.toolpath = calculateMemory(data);
    updateTimingPanel();

    // Enable clear button
    clearToolpathBtn.disabled = false;
}

// Display point cloud in scene
function displayPointCloud(data) {
    const renderStart = performance.now();
    const { positions, pointCount, bounds, isDense, gridWidth, gridHeight, isWorldCoordinates } = data;
    console.log('displayPointCloud: received', pointCount, 'points,', positions.byteLength, 'bytes, isDense:', isDense, 'isWorldCoordinates:', isWorldCoordinates);

    // Validation
    if (!positions || positions.length === 0 || pointCount === 0) {
        console.warn('displayPointCloud: No valid positions to display');
        return;
    }

    // Track if this is the first load (to reset camera)
    const isFirstLoad = pointCloud === null && terrainMesh === null;

    // Remove existing point cloud (but keep terrain mesh)
    if (pointCloud) {
        scene.remove(pointCloud);
        pointCloud.geometry.dispose();
        pointCloud.material.dispose();
    }

    const geomStart = performance.now();

    let worldPositions, colors, visualPointCount;

    if (isDense) {
        // DENSE TERRAIN: Convert 2D grid (Z-only) to 3D points
        // For very large grids (>10M cells), sample to avoid memory issues
        const MAX_POINTS_TO_DISPLAY = 5000000; // 5M point limit
        const totalCells = gridWidth * gridHeight;
        const needsSampling = totalCells > MAX_POINTS_TO_DISPLAY;
        const sampleRate = needsSampling ? Math.ceil(totalCells / MAX_POINTS_TO_DISPLAY) : 1;

        if (needsSampling) {
            console.log(`Large grid detected (${totalCells} cells), sampling every ${sampleRate} points`);
        }

        // Only create points for valid cells (skip NaN)
        const tempPositions = [];
        const tempColors = [];
        const color1 = new THREE.Color(0x00ffff); // Cyan
        const color2 = new THREE.Color(0x00cccc); // Slightly darker cyan

        for (let gridY = 0; gridY < gridHeight; gridY += sampleRate) {
            for (let gridX = 0; gridX < gridWidth; gridX += sampleRate) {
                const idx = gridY * gridWidth + gridX;
                const z = positions[idx];

                // Skip empty cells (sentinel value = -1e10)
                const EMPTY_CELL = -1e10;
                if (z > EMPTY_CELL + 1) {
                    // Convert grid indices to world coordinates (mm)
                    const worldX = bounds.min.x + gridX * STEP_SIZE;
                    const worldY = bounds.min.y + gridY * STEP_SIZE;

                    tempPositions.push(worldX, worldY, z);

                    // Checkerboard pattern
                    const isEven = (gridX + gridY) % 2 === 0;
                    const color = isEven ? color1 : color2;
                    tempColors.push(color.r, color.g, color.b);
                }
            }
        }

        worldPositions = new Float32Array(tempPositions);
        colors = new Float32Array(tempColors);
        visualPointCount = tempPositions.length / 3;

        console.log(`displayPointCloud: Dense terrain converted ${visualPointCount} valid points from ${gridWidth}x${gridHeight} grid`);

    } else if (isWorldCoordinates) {
        // Already in world coordinates (e.g., radial v2 mode), use directly
        worldPositions = positions;
        visualPointCount = pointCount;

        // Create simple color array (cyan)
        colors = new Float32Array(pointCount * 3);
        const color = new THREE.Color(0x00ffff);
        for (let i = 0; i < pointCount; i++) {
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }

    } else {
        // SPARSE (tool): Convert grid indices to world coordinates
        // GPU outputs [gridX, gridY, Z] where gridX/gridY are indices
        worldPositions = new Float32Array(positions.length);
        for (let i = 0; i < positions.length; i += 3) {
            const gridX = positions[i];
            const gridY = positions[i + 1];
            const z = positions[i + 2];

            // Convert grid indices to world coordinates
            worldPositions[i] = bounds.min.x + gridX * STEP_SIZE;
            worldPositions[i + 1] = bounds.min.y + gridY * STEP_SIZE;
            worldPositions[i + 2] = z;
        }

        // Create checkerboard pattern colors
        colors = new Float32Array(pointCount * 3);
        const color1 = new THREE.Color(0x00ffff); // Cyan
        const color2 = new THREE.Color(0x00cccc); // Slightly darker cyan

        for (let i = 0; i < pointCount; i++) {
            // Checkerboard based on grid indices (already integers)
            const gridX = Math.floor(positions[i * 3]);
            const gridY = Math.floor(positions[i * 3 + 1]);
            const isEven = (gridX + gridY) % 2 === 0;
            const color = isEven ? color1 : color2;

            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }

        visualPointCount = pointCount;
    }

    // Validate for NaN values before creating geometry
    let hasNaN = false;
    for (let i = 0; i < worldPositions.length; i++) {
        if (isNaN(worldPositions[i])) {
            hasNaN = true;
            console.error(`displayPointCloud: NaN detected at position ${i}`);
            break;
        }
    }

    if (hasNaN) {
        console.error('displayPointCloud: Found NaN values in positions array, aborting display');
        return;
    }

    // Create point cloud geometry
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(worldPositions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const geomTime = performance.now() - geomStart;
    console.log('displayPointCloud: geometry creation took', geomTime.toFixed(2), 'ms');

    // Calculate point size proportional to step size (raster density)
    // Point size should match or slightly exceed step size for complete coverage
    const pointSize = STEP_SIZE * 1.1; // 10% larger than step for slight overlap
    console.log('displayPointCloud: point size', pointSize.toFixed(3), 'mm (step:', STEP_SIZE, 'mm)');

    // Create point cloud material with vertex colors
    const material = new THREE.PointsMaterial({
        size: pointSize,
        vertexColors: true,
        sizeAttenuation: true
    });

    // Create point cloud mesh
    pointCloud = new THREE.Points(geometry, material);

    // Rotate -π/2 around X axis to correct orientation
    pointCloud.rotation.x = -Math.PI / 2;

    scene.add(pointCloud);

    // Show terrain control now that we have raster data
    if (terrainControl) {
        terrainControl.style.display = 'block';
    }

    // Update info panel (use visualPointCount for display)
    pointCountEl.textContent = visualPointCount.toLocaleString();
    boundsEl.textContent = `${formatBounds(bounds.min)} to ${formatBounds(bounds.max)}`;
    infoPanel.classList.remove('hidden');

    // Only reset camera on first load
    if (isFirstLoad) {
        const center = new THREE.Vector3(
            (bounds.min.x + bounds.max.x) / 2,
            (bounds.min.y + bounds.max.y) / 2,
            (bounds.min.z + bounds.max.z) / 2
        );

        const size = Math.max(
            bounds.max.x - bounds.min.x,
            bounds.max.y - bounds.min.y,
            bounds.max.z - bounds.min.z
        );

        camera.position.set(
            center.x + size,
            center.y + size,
            center.z + size
        );

        controls.target.copy(center);
        controls.update();
    }

    const renderTotal = performance.now() - renderStart;
    console.log('displayPointCloud: TOTAL render creation took', renderTotal.toFixed(2), 'ms');
    console.log(`Point cloud created: ${pointCount} points`);
}

// Display terrain as mesh (for radial mode)
function displayTerrainMesh(triangles, bounds) {
    const renderStart = performance.now();
    console.log('displayTerrainMesh: received', triangles.length / 9, 'triangles,', triangles.byteLength, 'bytes');

    // Track if this is the first load (to reset camera)
    const isFirstLoad = terrainMesh === null && pointCloud === null;

    // Remove existing mesh (but keep point cloud if it exists)
    if (terrainMesh) {
        scene.remove(terrainMesh);
        terrainMesh.geometry.dispose();
        terrainMesh.material.dispose();
    }

    // Create geometry from triangles
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(triangles, 3));
    geometry.computeVertexNormals();

    // Create material
    const material = new THREE.MeshPhongMaterial({
        color: 0xdddddd,
        flatShading: true,
        side: THREE.DoubleSide
    });

    // Create mesh
    terrainMesh = new THREE.Mesh(geometry, material);

    // Rotate -π/2 around X axis to correct orientation
    terrainMesh.rotation.x = -Math.PI / 2;

    scene.add(terrainMesh);

    // Update info panel
    const triangleCount = triangles.length / 9;
    pointCountEl.textContent = `${triangleCount.toLocaleString()} triangles`;
    boundsEl.textContent = `${formatBounds(bounds.min)} to ${formatBounds(bounds.max)}`;
    infoPanel.classList.remove('hidden');

    // Only reset camera on first load
    if (isFirstLoad) {
        const center = new THREE.Vector3(
            (bounds.min.x + bounds.max.x) / 2,
            (bounds.min.y + bounds.max.y) / 2,
            (bounds.min.z + bounds.max.z) / 2
        );

        const size = Math.max(
            bounds.max.x - bounds.min.x,
            bounds.max.y - bounds.min.y,
            bounds.max.z - bounds.min.z
        );

        camera.position.set(
            center.x + size,
            center.y + size,
            center.z + size
        );

        controls.target.copy(center);
        controls.update();
    }

    const renderTotal = performance.now() - renderStart;
    console.log('displayTerrainMesh: TOTAL render creation took', renderTotal.toFixed(2), 'ms');
    console.log(`Mesh created: ${triangleCount} triangles`);
}

// Display tool raster (point cloud) for radial mode
function displayToolRaster(data) {
    console.log('displayToolRaster:', {
        numTiles: data.numTiles,
        maxRadius: data.maxRadius,
        circumference: data.circumference
    });

    // Remove existing tool cloud
    if (toolCloud) {
        scene.remove(toolCloud);
        toolCloud.geometry.dispose();
        toolCloud.material.dispose();
    }

    // Convert tiles to positions array
    let totalPoints = 0;
    for (const tile of data.tiles) {
        for (let i = 0; i < tile.data.length; i++) {
            if (tile.data[i] > -1e9) totalPoints++;
        }
    }

    const positions = new Float32Array(totalPoints * 3);
    let posIdx = 0;
    const stepSize = data.stepSize;

    for (const tile of data.tiles) {
        const tileGridWidth = tile.gridWidth;
        const tileGridHeight = tile.gridHeight;

        for (let gy = 0; gy < tileGridHeight; gy++) {
            const theta = (gy * stepSize / data.circumference) * 2 * Math.PI;
            const cosTheta = Math.cos(theta);
            const sinTheta = Math.sin(theta);

            for (let gx = 0; gx < tileGridWidth; gx++) {
                const idx = gy * tileGridWidth + gx;
                const radius = tile.data[idx];

                if (radius > -1e9) {
                    const wx = tile.minX + gx * stepSize;
                    positions[posIdx++] = wx;
                    positions[posIdx++] = radius * cosTheta;
                    positions[posIdx++] = radius * sinTheta;
                }
            }
        }
    }

    const pointCount = posIdx / 3;

    // Create geometry
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Create material - orange to match tool mesh
    const material = new THREE.PointsMaterial({
        size: STEP_SIZE * 1.1,
        color: 0xff8844,
        sizeAttenuation: true
    });

    // Create point cloud
    toolCloud = new THREE.Points(geometry, material);
    toolCloud.rotation.x = -Math.PI / 2;

    // Position tool raster to match tool mesh position
    if (stlBounds) {
        const centerX = (stlBounds.min.x + stlBounds.max.x) / 2;
        const topY = stlBounds.max.y + 5;  // 5mm above terrain
        toolCloud.position.set(centerX, topY, 0);
    }

    scene.add(toolCloud);

    console.log(`Tool raster created: ${pointCount} points`);
}

// Display tool as mesh (for radial mode)
function displayToolMesh(triangles) {
    const renderStart = performance.now();
    console.log('displayToolMesh: received', triangles.length / 9, 'triangles,', triangles.byteLength, 'bytes');

    // Remove existing tool mesh
    if (toolMesh) {
        scene.remove(toolMesh);
        toolMesh.geometry.dispose();
        toolMesh.material.dispose();
    }

    // Create geometry from triangles
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(triangles, 3));
    geometry.computeVertexNormals();

    // Create material - different color from terrain
    const material = new THREE.MeshPhongMaterial({
        color: 0xff8844,  // Orange for tool
        flatShading: true,
        side: THREE.DoubleSide
    });

    // Create mesh
    toolMesh = new THREE.Mesh(geometry, material);

    // Rotate -π/2 around X axis to correct orientation
    toolMesh.rotation.x = -Math.PI / 2;

    // Position tool at terrain bounds if available
    if (stlBounds) {
        // Position tool at center X, max Y (top of terrain)
        const centerX = (stlBounds.min.x + stlBounds.max.x) / 2;
        const topY = stlBounds.max.y + 5;  // 5mm above terrain
        toolMesh.position.set(centerX, topY, 0);
    }

    scene.add(toolMesh);

    const renderTotal = performance.now() - renderStart;
    console.log('displayToolMesh: TOTAL render creation took', renderTotal.toFixed(2), 'ms');
    console.log(`Tool mesh created: ${triangles.length / 9} triangles`);
}

function formatBounds(vec) {
    return `(${vec.x.toFixed(2)}, ${vec.y.toFixed(2)}, ${vec.z.toFixed(2)})`;
}

// Display tool - just render the positions as-is
function displayTool(data) {
    const { positions, pointCount, bounds } = data;
    console.log('displayTool: received', pointCount, 'points');

    // Remove existing tool cloud
    if (toolCloud) {
        scene.remove(toolCloud);
        toolCloud.geometry.dispose();
        toolCloud.material.dispose();
    }

    // Convert grid indices to world coordinates
    // GPU outputs [gridX, gridY, Z] where gridX/gridY are indices
    const worldPositions = new Float32Array(positions.length);
    for (let i = 0; i < positions.length; i += 3) {
        const gridX = positions[i];
        const gridY = positions[i + 1];
        const z = positions[i + 2];

        // Convert grid indices to world coordinates
        worldPositions[i] = bounds.min.x + gridX * STEP_SIZE;     // X
        worldPositions[i + 1] = bounds.min.y + gridY * STEP_SIZE; // Y
        worldPositions[i + 2] = z;                                 // Z
    }

    // Create geometry
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(worldPositions, 3));

    // Color: semi-transparent yellow/gold
    const colors = new Float32Array(pointCount * 3);
    const color = new THREE.Color(0xffaa00); // Gold/yellow
    for (let i = 0; i < pointCount; i++) {
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Calculate point size
    const pointSize = STEP_SIZE * 1.1;

    // Create material with transparency
    const material = new THREE.PointsMaterial({
        size: pointSize,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.6
    });

    // Create point cloud
    toolCloud = new THREE.Points(geometry, material);

    // Rotate to match terrain orientation
    toolCloud.rotation.x = -Math.PI / 2;

    // Position tool at same location as tool STL mesh
    if (stlBounds) {
        const centerX = (stlBounds.min.x + stlBounds.max.x) / 2;
        const topY = stlBounds.max.y + 5;  // 5mm above terrain
        toolCloud.position.set(centerX, topY, 0);
    }

    scene.add(toolCloud);
    console.log('Tool displayed with', pointCount, 'points');
}

// Update timing panel
function updateTimingPanel() {
    // Update timing values
    if (timingData.terrainConversion !== null) {
        timingTerrainEl.textContent = timingData.terrainConversion.toFixed(2) + ' ms';
    }
    if (timingData.toolConversion !== null) {
        timingToolEl.textContent = timingData.toolConversion.toFixed(2) + ' ms';
    }
    if (timingData.toolpathGeneration !== null) {
        timingToolpathEl.textContent = timingData.toolpathGeneration.toFixed(2) + ' ms';
    }

    // Update memory values
    if (memoryData.terrain !== null) {
        memoryTerrainEl.textContent = formatBytes(memoryData.terrain);
    }
    if (memoryData.tool !== null) {
        memoryToolEl.textContent = formatBytes(memoryData.tool);
    }
    if (memoryData.toolpath !== null) {
        memoryToolpathEl.textContent = formatBytes(memoryData.toolpath);
    }

    // Update total JS heap if available
    if (performance.memory) {
        memoryTotalEl.textContent = formatBytes(performance.memory.usedJSHeapSize);
    } else {
        memoryTotalEl.textContent = 'N/A';
    }

    // Update GPU / Tiling info
    if (deviceCapabilities) {
        gpuLimitEl.textContent = formatBytes(deviceCapabilities.maxStorageBufferBindingSize);
    }

    // Update tiling status from terrain data
    if (terrainData && terrainData.tileCount) {
        tilingStatusEl.textContent = `${terrainData.tileCount} tiles`;
    } else if (terrainData) {
        tilingStatusEl.textContent = 'No tiling';
    }

    // Show panel if any timing data exists
    if (timingData.terrainConversion !== null ||
        timingData.toolConversion !== null ||
        timingData.toolpathGeneration !== null) {
        timingPanel.classList.remove('hidden');
    }
}

// Display toolpath in scene
function displayToolpath(data) {
    console.log('displayToolpath: received data', data);
    const { pathData, numScanlines, pointsPerLine, isRadial, rotationStepDegrees, generationBounds } = data;

    // Remove existing toolpath cloud
    if (toolpathCloud) {
        scene.remove(toolpathCloud);
        toolpathCloud.geometry.dispose();
        toolpathCloud.material.dispose();
    }

    // Keep tool visible (it stays at the start position)

    // Get terrain bounds to map grid back to world space
    if (!terrainData || !terrainData.bounds) {
        console.error('No terrain data available for toolpath visualization');
        return;
    }

    // Use generationBounds if provided (for radial mode with padding), otherwise use terrain bounds
    const bounds = generationBounds || terrainData.bounds;

    // The toolpath is in grid index space, we need to map it back to world coordinates
    // The grid step size used for the point cloud
    const gridStep = STEP_SIZE;

    // Get step sizes used in toolpath generation
    const xStep = parseInt(xStepInput.value);
    const yStep = parseInt(yStepInput.value);
    const zFloor = parseFloat(zFloorInput.value);

    console.log('displayToolpath: bounds', bounds);
    console.log('displayToolpath: mode', isRadial ? 'radial' : 'planar');
    console.log('displayToolpath: gridStep', gridStep, 'xStep', xStep, isRadial ? 'rotationStep' : 'yStep', isRadial ? rotationStepDegrees : yStep);
    console.log('displayToolpath: scanlines', numScanlines, 'pointsPerLine', pointsPerLine);

    // Create position array for toolpath points
    const positions = [];

    let pathIdx = 0;

    if (isRadial) {
        // Radial mode: scanlines are at different rotation angles
        // Each point needs to be rotated back into 3D space
        console.log('displayToolpath: radial mode with', numScanlines, 'rotations');
        console.log('displayToolpath: pathData length', pathData.length, 'first 20 values:', Array.from(pathData.slice(0, 20)).map(v => v.toFixed(3)).join(', '));
        console.log('displayToolpath: zFloor', zFloor);

        for (let rotationIdx = 0; rotationIdx < numScanlines; rotationIdx++) {
            const angleDegrees = rotationIdx * rotationStepDegrees;
            const angleRad = angleDegrees * Math.PI / 180;
            const cosAngle = Math.cos(angleRad);
            const sinAngle = Math.sin(angleRad);

            for (let ix = 0; ix < pointsPerLine; ix++) {
                const z = pathData[pathIdx++];

                // Skip floor values
                if (z <= zFloor + 1) continue;

                // Grid X position (along the rotation axis)
                const gridX = ix * xStep;
                const worldX = bounds.min.x + gridX * gridStep;

                // Transform point (x, 0, z) by rotating back by -angle around X axis
                // In the strip coordinate system, the scanline is at y=0
                // To rotate back by -angle: (x, -y*sin + z*cos(-θ), y*cos + z*sin(-θ))
                // At y=0: (x, z*sin(θ), z*cos(θ))
                const worldY = z * sinAngle;
                const worldZ = z * cosAngle;

                positions.push(worldX, worldY, worldZ);
            }
        }
        console.log('displayToolpath: radial generated', positions.length / 3, 'visible points (filtered', pathData.length - positions.length / 3, 'floor values)');
    } else {
        // Planar mode: grid-based scanlines
        console.log('displayToolpath: planar mode');
        console.log('displayToolpath: terrain grid dimensions from bounds:',
            'width =', (bounds.max.x - bounds.min.x) / gridStep,
            'height =', (bounds.max.y - bounds.min.y) / gridStep);

        for (let iy = 0; iy < numScanlines; iy++) {
            for (let ix = 0; ix < pointsPerLine; ix++) {
                const z = pathData[pathIdx++];

                // Map grid indices to world coordinates
                // The toolpath uses stepped indices (ix*xStep, iy*yStep) in grid space
                const gridX = ix * xStep;
                const gridY = iy * yStep;

                // Convert grid indices to world coordinates
                const worldX = bounds.min.x + gridX * gridStep;
                const worldY = bounds.min.y + gridY * gridStep;

                // Only add points that are not at the floor level (filter out oob points)
                if (z > zFloor + 10) { // Filter out oob points (with 10mm margin)
                    positions.push(worldX, worldY, z);
                }
            }
        }
    }

    const positionsArray = new Float32Array(positions);
    const pointCount = positions.length / 3;

    console.log('displayToolpath: created', pointCount, 'toolpath points');
    // console.log('displayToolpath: first 3 points:', positionsArray.slice(0, 9));
    // console.log('displayToolpath: last 3 points:', positionsArray.slice(-9));

    // Calculate XY range of toolpath
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < positionsArray.length; i += 3) {
        const x = positionsArray[i];
        const y = positionsArray[i + 1];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }
    console.log('displayToolpath: XY range: X[', minX, ',', maxX, '] Y[', minY, ',', maxY, ']');
    console.log('displayToolpath: terrain XY range: X[', bounds.min.x, ',', bounds.max.x, '] Y[', bounds.min.y, ',', bounds.max.y, ']');

    // Create geometry
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positionsArray, 3));

    // Create material - use red/orange color for toolpath
    // Scale point size based on the actual step values used
    const effectiveStep = isRadial ? xStep * gridStep : Math.min(xStep, yStep) * gridStep;
    const pointSize = effectiveStep * 1.2; // 20% larger than effective step for visibility
    console.log('displayToolpath: point size', pointSize.toFixed(3), 'mm (xStep:', xStep, isRadial ? '' : 'yStep: ' + yStep, 'gridStep:', gridStep, ')');

    const material = new THREE.PointsMaterial({
        size: pointSize,
        color: 0xff4400, // Orange-red
        sizeAttenuation: true
    });

    // Create point cloud
    toolpathCloud = new THREE.Points(geometry, material);

    // Apply same rotation as terrain
    toolpathCloud.rotation.x = -Math.PI / 2;

    scene.add(toolpathCloud);

    console.log('Toolpath visualization complete');
}

// File handling
function handleFile(file) {
    console.log('handleFile called with:', file);

    if (!file || !file.name.toLowerCase().endsWith('.stl')) {
        alert('Please select a valid STL file');
        return;
    }

    console.log('File is valid STL:', file.name);

    // Store file for recompute
    lastLoadedFile = file;
    recomputeBtn.disabled = false;

    // Update filename display
    terrainFilenameEl.textContent = file.name;

    // Save to localStorage for auto-reload
    saveFileToLocalStorage('lastTerrainFile', file);

    processFile(file);
}

// Save file to localStorage
async function saveFileToLocalStorage(key, file) {
    try {
        const buffer = await file.arrayBuffer();
        const base64 = arrayBufferToBase64(buffer);
        localStorage.setItem(key, JSON.stringify({
            name: file.name,
            data: base64
        }));
        console.log(`💾 Saved ${file.name} to localStorage`);
    } catch (error) {
        console.error('Error saving file to localStorage:', error);
    }
}

// Load file from localStorage
function loadFileFromLocalStorage(key) {
    try {
        const stored = localStorage.getItem(key);
        if (!stored) return null;

        const { name, data } = JSON.parse(stored);
        const buffer = base64ToArrayBuffer(data);
        const file = new File([buffer], name, { type: 'application/octet-stream' });
        console.log(`💾 Loaded ${name} from localStorage`);
        return file;
    } catch (error) {
        console.error('Error loading file from localStorage:', error);
        return null;
    }
}

// Helper: ArrayBuffer to Base64
function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

// Helper: Base64 to ArrayBuffer
function base64ToArrayBuffer(base64) {
    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
}

async function processFileWebGPU(file, options = {}) {
    updateStatus('Loading file...');

    try {
        const buffer = await file.arrayBuffer();
        console.log('File loaded, buffer size:', buffer.byteLength);

        updateStatus('Parsing STL...');
        const { positions, triangleCount, bounds } = parseSTL(buffer);
        console.log('Parsed STL:', triangleCount, 'triangles');
        console.log('Original bounds:', bounds);

        // Center the model based on mode
        let centeredPositions;
        let centeredBounds;

        if (toolpathMode === 'radial' || toolpathMode === 'radial-raster' || toolpathMode === 'radial') {
            // Radial mode: center XYZ at origin for proper rotation
            const centerX = (bounds.min.x + bounds.max.x) / 2;
            const centerY = (bounds.min.y + bounds.max.y) / 2;
            const centerZ = (bounds.min.z + bounds.max.z) / 2;

            centeredPositions = new Float32Array(positions.length);
            for (let i = 0; i < positions.length; i += 3) {
                centeredPositions[i] = positions[i] - centerX;
                centeredPositions[i + 1] = positions[i + 1] - centerY;
                centeredPositions[i + 2] = positions[i + 2] - centerZ;
            }

            centeredBounds = {
                min: {
                    x: bounds.min.x - centerX,
                    y: bounds.min.y - centerY,
                    z: bounds.min.z - centerZ
                },
                max: {
                    x: bounds.max.x - centerX,
                    y: bounds.max.y - centerY,
                    z: bounds.max.z - centerZ
                }
            };

            console.log('Radial mode: centered model at origin (XYZ)');
        } else {
            // Planar mode: center XY, keep Z min at 0
            const centerX = (bounds.min.x + bounds.max.x) / 2;
            const centerY = (bounds.min.y + bounds.max.y) / 2;
            const minZ = bounds.min.z;

            centeredPositions = new Float32Array(positions.length);
            for (let i = 0; i < positions.length; i += 3) {
                centeredPositions[i] = positions[i] - centerX;
                centeredPositions[i + 1] = positions[i + 1] - centerY;
                centeredPositions[i + 2] = positions[i + 2] - minZ;
            }

            centeredBounds = {
                min: {
                    x: bounds.min.x - centerX,
                    y: bounds.min.y - centerY,
                    z: 0
                },
                max: {
                    x: bounds.max.x - centerX,
                    y: bounds.max.y - centerY,
                    z: bounds.max.z - minZ
                }
            };

            console.log('Planar mode: centered XY, Z min at 0');
        }

        console.log('Centered bounds:', centeredBounds);

        // Store STL bounds for bounds override feature
        stlBounds = centeredBounds;

        // Store triangles for radial mode
        terrainTriangles = new Float32Array(centeredPositions);

        if (toolpathMode === 'radial') {
            // Radial mode: Display as mesh, skip rasterization for now
            updateStatus('Displaying mesh...');
            displayTerrainMesh(terrainTriangles, centeredBounds);
            updateStatus('Terrain loaded (radial mode)');

            // Set terrainData with bounds for button state management
            terrainData = { bounds: centeredBounds };

            // Enable generate button if tool is loaded
            if (toolData) {
                generateToolpathBtn.disabled = false;
            }
        } else if (toolpathMode === 'radial') {
            // Radial v2 mode: Display mesh on first load, rasterize on button press

            // Store triangles for rasterize button
            terrainTriangles = new Float32Array(centeredPositions);

            // Check if we should rasterize (called from Rasterize button)
            const shouldRasterize = options && options.rasterize;

            if (shouldRasterize) {
                // Rasterizing - don't recreate the mesh, just rasterize
                updateStatus('Radial rasterizing v2 (bit-attention)...');

                if (!rasterPathRadial) {
                    throw new Error('RasterPathRadial API not initialized');
                }

                // Get bounds override if enabled
                const boundsOverride = getBoundsOverride();

                // Use new RasterPathRadial API
                const startTime = performance.now();
                const result = await rasterPathRadial.rasterize(centeredPositions, STEP_SIZE, {
                    zFloor: 0,
                    boundsOverride
                });
                const endTime = performance.now();

                // Store timing
                timingData.terrainConversion = endTime - startTime;
                timingTerrainEl.textContent = timingData.terrainConversion.toFixed(2) + ' ms';

                // Handle the result - add point cloud alongside mesh
                handleRadialRasterizeComplete(result);
            } else {
                // First load - display the mesh
                updateStatus('Displaying mesh...');
                displayTerrainMesh(centeredPositions, centeredBounds);
                updateStatus('Ready - click Rasterize button to generate terrain raster');
            }
            return;
        } else {
            // Planar mode: Rasterize with WebGPU
            updateStatus('Rasterizing with WebGPU...');

            // Get bounds override if enabled
            const boundsOverride = getBoundsOverride();

            // Send to WebGPU worker
            webgpuWorker.postMessage({
                type: 'rasterize',
                data: {
                    triangles: centeredPositions,
                    stepSize: STEP_SIZE,
                    filterMode: 0, // 0 = UPWARD_FACING for terrain
                    isForTool: false,
                    boundsOverride
                }
            }, [centeredPositions.buffer]);
        }
    } catch (error) {
        console.error('Error processing file with WebGPU:', error);
        updateStatus('Error: ' + error.message);
        alert('Error: ' + error.message);
    }
}

function processFile(file) {
    // Always use WebGPU
    processFileWebGPU(file);
}

// Terrain file input handler
terrainFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    console.log('Terrain file input changed:', file);
    if (file) {
        handleFile(file);
    }
});

// Toolpath UI event handlers
toolFileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        toolFile = file;
        toolFilenameEl.textContent = file.name;

        // Save to localStorage for auto-reload
        saveFileToLocalStorage('lastToolFile', file);

        // Parse the tool STL
        try {
            const buffer = await file.arrayBuffer();
            const { positions, triangleCount } = parseSTL(buffer);
            console.log('Parsed tool STL:', triangleCount, 'triangles');

            // In radial mode, display the tool mesh AND rasterize with PLANAR method
            if (toolpathMode === 'radial') {
                console.log('Displaying tool mesh for radial');
                displayToolMesh(positions);

                // Also rasterize the tool if we have terrain data
                // Use PLANAR rasterization (same as other modes)
                if (terrainData && !toolRasterizingInProgress) {
                    console.log('Rasterizing tool for radial (using PLANAR WebGPU worker)');
                    updateStatus('Rasterizing tool...');
                    toolRasterizingInProgress = true;

                    // Copy positions before transferring to worker (mesh needs to keep original)
                    const positionsCopy = new Float32Array(positions);

                    // Send to WebGPU worker for PLANAR rasterization
                    webgpuWorker.postMessage({
                        type: 'rasterize',
                        data: {
                            triangles: positionsCopy,
                            stepSize: STEP_SIZE,
                            filterMode: 1, // 1 = DOWNWARD_FACING for tool
                            isForTool: true,
                            boundsOverride: null
                        }
                    }, [positionsCopy.buffer]);
                }
            }
            // In other modes, convert tool if terrain data exists
            else if (terrainData) {
                console.log('Converting tool STL:', file.name);
                updateStatus('Converting tool...');

                // Send to WebGPU worker (tools don't use bounds override)
                webgpuWorker.postMessage({
                    type: 'rasterize',
                    data: {
                        triangles: positions,
                        stepSize: STEP_SIZE,
                        filterMode: 1, // 1 = DOWNWARD_FACING for tool
                        isForTool: true,
                        boundsOverride: null
                    }
                }, [positions.buffer]);
            } else {
                // alert('Please load terrain first');
            }
        } catch (error) {
            console.error('Error loading tool:', error);
            alert('Error loading tool: ' + error.message);
        }
    }
});

// Single-threaded WASM toolpath generation
async function generateToolpathSingle(terrainPoints, toolPoints, xStep, yStep, oobZ, gridStep) {
    const startTime = performance.now();
    console.log('🔨 [WASM Single] Starting toolpath generation...');

    return new Promise((resolve, reject) => {
        const worker = new Worker('worker-parallel.js');

        worker.onmessage = function(e) {
            const { type, data } = e.data;

            if (type === 'wasm-ready') {
                // Worker ready, generate full toolpath (0 to end)
                worker.postMessage({
                    type: 'generate-toolpath-partial',
                    data: {
                        terrainPoints,
                        toolPoints,
                        xStep,
                        yStep,
                        oobZ,
                        gridStep,
                        startScanline: 0,
                        endScanline: 999999 // Large number to get all scanlines
                    }
                });
            } else if (type === 'toolpath-partial-complete') {
                const endTime = performance.now();
                const totalTime = endTime - startTime;

                console.log(`🔨 [WASM Single] ✅ Complete in ${totalTime.toFixed(1)}ms`);

                worker.terminate();

                resolve({
                    pathData: data.pathData,
                    numScanlines: data.numScanlines,
                    pointsPerLine: data.pointsPerLine,
                    generationTime: totalTime
                });
            } else if (type === 'error') {
                console.error('[WASM Single] Error:', data);
                worker.terminate();
                reject(new Error(data.message || 'Worker error'));
            }
        };

        worker.onerror = function(error) {
            console.error('[WASM Single] Worker error:', error);
            worker.terminate();
            reject(error);
        };
    });
}

// Parallel toolpath generation coordinator
async function generateToolpathParallel(terrainPoints, toolPoints, xStep, yStep, oobZ, gridStep) {
    const startTime = performance.now();

    // Create temporary worker to get dimensions
    const tempWorker = new Worker('worker-parallel.js');

    return new Promise((resolve, reject) => {
        let terrainDims = null;
        let totalScanlines = null;
        let pointsPerLine = null;

        tempWorker.onmessage = async function(e) {
            const { type, data } = e.data;

            if (type === 'wasm-ready') {
                // Get dimensions by requesting conversion
                const terrainSize = terrainPoints.length * 4;
                const toolSize = toolPoints.length * 4;

                // Use a quick helper: convert and get dimensions
                // For now, we'll calculate dimensions ourselves
                // pointsPerLine = ceil(terrainWidth / xStep), where terrainWidth comes from bounds

                // Calculate from bounds (approximate)
                let minX = Infinity, maxX = -Infinity;
                let minY = Infinity, maxY = -Infinity;

                for (let i = 0; i < terrainPoints.length; i += 3) {
                    const x = terrainPoints[i];
                    const y = terrainPoints[i + 1];
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }

                const terrainWidth = Math.round((maxX - minX) / gridStep) + 1;
                const terrainHeight = Math.round((maxY - minY) / gridStep) + 1;

                pointsPerLine = Math.ceil(terrainWidth / xStep);
                totalScanlines = Math.ceil(terrainHeight / yStep);

                console.log(`📐 Terrain dimensions: ${terrainWidth}x${terrainHeight}, Toolpath: ${pointsPerLine}x${totalScanlines}`);

                // Close temp worker
                tempWorker.terminate();

                // Now spawn parallel workers
                const scanlinesPerWorker = Math.ceil(totalScanlines / NUM_PARALLEL_WORKERS);
                const workers = [];
                const results = [];
                let completedWorkers = 0;

                console.log(`🚀 Spawning ${NUM_PARALLEL_WORKERS} parallel workers...`);

                for (let i = 0; i < NUM_PARALLEL_WORKERS; i++) {
                    const startScanline = i * scanlinesPerWorker;
                    const endScanline = Math.min(startScanline + scanlinesPerWorker, totalScanlines);

                    if (startScanline >= totalScanlines) break;

                    console.log(`  Worker ${i}: scanlines ${startScanline}-${endScanline}`);

                    const worker = new Worker('worker-parallel.js');
                    const workerIndex = i;

                    worker.onmessage = function(e) {
                        const { type, data: workerData } = e.data;

                        if (type === 'wasm-ready') {
                            // Worker ready, send work
                            worker.postMessage({
                                type: 'generate-toolpath-partial',
                                data: {
                                    terrainPoints,
                                    toolPoints,
                                    xStep,
                                    yStep,
                                    oobZ,
                                    gridStep,
                                    startScanline,
                                    endScanline
                                }
                            });
                        } else if (type === 'toolpath-partial-complete') {
                            console.log(`  ✓ Worker ${workerIndex} completed`);
                            results[workerIndex] = workerData;
                            completedWorkers++;

                            // Terminate worker
                            worker.terminate();

                            // Check if all done
                            if (completedWorkers === workers.length) {
                                // Merge results
                                console.log('🔗 Merging results...');
                                const mergedData = new Float32Array(totalScanlines * pointsPerLine);

                                for (const result of results) {
                                    const startIdx = result.startScanline * result.pointsPerLine;
                                    mergedData.set(result.pathData, startIdx);
                                }

                                const endTime = performance.now();
                                const totalTime = endTime - startTime;

                                console.log(`✅ Parallel generation complete in ${totalTime.toFixed(1)}ms`);

                                resolve({
                                    pathData: mergedData,
                                    numScanlines: totalScanlines,
                                    pointsPerLine: pointsPerLine,
                                    generationTime: totalTime
                                });
                            }
                        } else if (type === 'error') {
                            console.error(`Worker ${workerIndex} error:`, workerData);
                            worker.terminate();
                            reject(new Error(workerData.message || 'Worker error'));
                        }
                    };

                    worker.onerror = function(error) {
                        console.error(`Worker ${workerIndex} error:`, error);
                        worker.terminate();
                        reject(error);
                    };

                    workers.push(worker);
                }
            } else if (type === 'error') {
                tempWorker.terminate();
                reject(new Error(data.message || 'Worker error'));
            }
        };

        tempWorker.onerror = function(error) {
            tempWorker.terminate();
            reject(error);
        };
    });
}

generateToolpathBtn.addEventListener('click', async () => {
    if (toolpathMode === 'radial') {
        // Radial v2 mode: Use existing planar toolpath on unwrapped terrain
        if (!radialTileData || !toolData) {
            alert('Please load both terrain and tool STL files first');
            return;
        }

        console.log('🎯 Starting radial toolpath generation...');
        updateStatus('Generating toolpath (radial)...');

        // Disable button during processing
        generateToolpathBtn.disabled = true;

        // Show progress bar
        const progressContainer = document.getElementById('progress-container');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        progressContainer.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = '0%';

        try {
            const xStep = parseInt(xStepInput.value);
            const yStep = parseInt(yStepInput.value);
            const zFloor = parseFloat(zFloorInput.value);

            console.log('Radial-v2 Parameters: X step:', xStep, 'Y step:', yStep, 'Z floor:', zFloor);

            // Use RasterPathRadial to prepare terrain for toolpath generation
            const { terrainPositions, bounds } = rasterPathRadial.prepareTerrainForToolpath(radialTileData);

            // Store bounds for use in visualization
            radialAdjustedBounds = bounds;

            // Send to WebGPU worker for planar toolpath generation
            // Terrain is dense (Z-only), tool is already morphed (sparse)
            webgpuWorker.postMessage({
                type: 'generate-toolpath',
                data: {
                    terrainPositions: terrainPositions,
                    toolPositions: toolData.positions,
                    xStep,
                    yStep,
                    zFloor: zFloor,
                    gridStep: STEP_SIZE,
                    terrainBounds: bounds,  // Use bounds with original STL X coordinates
                    isRadial: true  // Flag to indicate radial mode for visualization
                }
            });
            // Result will be handled by the worker message handler

        } catch (error) {
            console.error('Error generating radial toolpath:', error);
            updateStatus('Error: ' + error.message);
            alert('Error generating radial toolpath: ' + error.message);
            generateToolpathBtn.disabled = false;
            progressContainer.style.display = 'none';
        }

    } else if (toolpathMode === 'radial') {
        // Radial mode: Check for terrain triangles and tool data
        if (!terrainTriangles || !toolData) {
            alert('Please load both terrain and tool STL files first');
            return;
        }

        if (!rasterPath) {
            alert('RasterPath API not initialized');
            return;
        }

        console.log('🎯 Starting radial toolpath generation...');
        updateStatus('Generating radial toolpath...');

        // Disable button during processing
        generateToolpathBtn.disabled = true;

        // Show progress bar
        const progressContainer = document.getElementById('progress-container');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        progressContainer.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = '0%';

        try {
            const xStep = parseInt(xStepInput.value);
            const xRotationStep = parseFloat(xRotationStepInput.value);
            const zFloor = parseFloat(zFloorInput.value);

            console.log('Radial Parameters: X step:', xStep, 'Rotation step:', xRotationStep, 'Z floor:', zFloor, 'Grid step:', STEP_SIZE);

            // In radial mode, apply bounds override only to X-axis
            // Y is determined by tool radius, Z is the full terrain height
            console.log('Original stlBounds:', stlBounds);

            let radialBounds = stlBounds;
            const boundsOverride = getBoundsOverride();
            console.log('Bounds override:', boundsOverride);

            if (boundsOverride) {
                radialBounds = {
                    min: {
                        x: boundsOverride.min.x,  // Use override X
                        y: stlBounds.min.y,        // Keep original Y (will be recalculated by tool radius)
                        z: stlBounds.min.z         // Keep original Z
                    },
                    max: {
                        x: boundsOverride.max.x,  // Use override X
                        y: stlBounds.max.y,        // Keep original Y (will be recalculated by tool radius)
                        z: stlBounds.max.z         // Keep original Z
                    }
                };
                console.log('Radial mode: applying X bounds override');
                console.log('  stlBounds X:', stlBounds.min.x, 'to', stlBounds.max.x);
                console.log('  override X:', boundsOverride.min.x, 'to', boundsOverride.max.x);
                console.log('  radialBounds X:', radialBounds.min.x, 'to', radialBounds.max.x);
            } else {
                console.log('Radial mode: no bounds override, using stlBounds');
            }

            // Call radial toolpath generation with progress callback
            const result = await rasterPath.generateRadialToolpath(
                terrainTriangles,
                toolData.positions,
                xRotationStep,
                xStep,
                zFloor,
                STEP_SIZE,
                radialBounds,
                {
                    onProgress: (percent, info) => {
                        progressFill.style.width = `${percent}%`;
                        progressText.textContent = `${percent}% (${info.current}/${info.total})`;
                    }
                }
            );

            // Hide progress bar
            progressContainer.style.display = 'none';

            console.log('Radial toolpath complete:', result);

            // Handle the result
            handleToolpathComplete({
                pathData: result.pathData,
                numScanlines: result.numRotations,
                pointsPerLine: result.pointsPerLine,
                generationTime: result.generationTime,
                isRadial: true,
                rotationStepDegrees: result.rotationStepDegrees,
                generationBounds: radialBounds  // Pass bounds used for generation
            });

        } catch (error) {
            console.error('Error generating radial toolpath:', error);
            updateStatus('Error: ' + error.message);
            alert('Error generating radial toolpath: ' + error.message);
            generateToolpathBtn.disabled = false;
            progressContainer.style.display = 'none';
        }

    } else {
        // Planar mode: existing implementation
        if (!terrainData || !toolData) {
            alert('Please load both terrain and tool STL files first');
            return;
        }

        console.log('🎯 Starting toolpath generation (planar)...');
        updateStatus('Generating toolpath (planar)...');

        // Disable button during processing
        generateToolpathBtn.disabled = true;

        // Show progress bar
        const progressContainer = document.getElementById('progress-container');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        progressContainer.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = '0%';

        try {
            const xStep = parseInt(xStepInput.value);
            const yStep = parseInt(yStepInput.value);
            const zFloor = parseFloat(zFloorInput.value);

            console.log('Planar Parameters: X step:', xStep, 'Y step:', yStep, 'Z floor:', zFloor, 'Grid step:', STEP_SIZE);

            // Send to WebGPU worker (COPY data, don't transfer - we need to keep it)
            webgpuWorker.postMessage({
                type: 'generate-toolpath',
                data: {
                    terrainPositions: terrainData.positions,
                    toolPositions: toolData.positions,
                    xStep,
                    yStep,
                    zFloor: zFloor,
                    gridStep: STEP_SIZE,
                    terrainBounds: terrainData.bounds // Pass terrain bounds for correct coordinate system
                }
            });
            // Result will be handled by the worker message handler

        } catch (error) {
            console.error('Error generating planar toolpath:', error);
            updateStatus('Error: ' + error.message);
            alert('Error generating planar toolpath: ' + error.message);
            generateToolpathBtn.disabled = false;
            progressContainer.style.display = 'none';
        }
    }
});

clearToolpathBtn.addEventListener('click', () => {
    if (toolpathCloud) {
        scene.remove(toolpathCloud);
        toolpathCloud.geometry.dispose();
        toolpathCloud.material.dispose();
        toolpathCloud = null;
        updateStatus('Toolpath cleared');
        clearToolpathBtn.disabled = true;
        // Tool remains visible at its start position
    }
});

// Mode switch handler
let isModeSwitching = false;  // Guard against recursive calls
toolpathModeRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
        if (isModeSwitching) {
            console.warn('Mode switch already in progress, skipping');
            return;
        }
        isModeSwitching = true;

        toolpathMode = e.target.value;
        console.log('Toolpath mode switched to:', toolpathMode);

        // Update label and controls based on mode
        if (toolpathMode === 'radial') {
            showTerrainLabel.textContent = 'Show Model';
            yStepControl.style.display = 'none';
            xRotationStepControl.style.display = 'flex';
            toolpathControls.style.display = 'block'; // Show toolpath controls

            // Set Z floor to 0 (at X axis) for radial mode
            // User can set negative to reach pockets below centerline
            if (zFloorInput.value === '-100') {
                zFloorInput.value = '0';
            }
        } else if (toolpathMode === 'radial') {
            showTerrainLabel.textContent = 'Show Unwrapped Raster';
            yStepControl.style.display = 'flex'; // Show Y step for unwrapped planar toolpath
            xRotationStepControl.style.display = 'none';
            toolpathControls.style.display = 'block'; // Show toolpath controls
        } else {
            showTerrainLabel.textContent = 'Show Terrain';
            yStepControl.style.display = 'flex';
            xRotationStepControl.style.display = 'none';
            toolpathControls.style.display = 'block'; // Show toolpath controls

            // Set Z floor to -100 (well below part) for planar mode
            if (zFloorInput.value === '0') {
                zFloorInput.value = '-100';
            }
        }

        // Clear existing displays
        if (pointCloud) {
            scene.remove(pointCloud);
            pointCloud.geometry.dispose();
            pointCloud.material.dispose();
            pointCloud = null;
        }
        if (terrainMesh) {
            scene.remove(terrainMesh);
            terrainMesh.geometry.dispose();
            terrainMesh.material.dispose();
            terrainMesh = null;
        }
        if (toolpathCloud) {
            scene.remove(toolpathCloud);
            toolpathCloud.geometry.dispose();
            toolpathCloud.material.dispose();
            toolpathCloud = null;
            clearToolpathBtn.disabled = true;
        }

        // Clear terrain data
        terrainData = null;
        radialTileData = null;
        generateToolpathBtn.disabled = true;

        // Don't auto-reload - user can press recompute button if they want
        isModeSwitching = false;

        saveSettings();
    });
});

// Show/hide terrain toggle (controls rasterized point clouds - both terrain and tool)
showTerrainCheckbox.addEventListener('change', (e) => {
    const showRaster = e.target.checked;
    console.log('Raster visibility:', showRaster);

    if (pointCloud) {
        pointCloud.visible = showRaster;
    }
    if (toolCloud) {
        toolCloud.visible = showRaster;
    }
});

// Toggle between wrapped (cylindrical) and unwrapped (flat) visualization
showUnwrappedCheckbox.addEventListener('change', (e) => {
    if (!radialTileData) return;

    const showUnwrapped = e.target.checked;
    console.log('Showing unwrapped terrain:', showUnwrapped);

    if (showUnwrapped) {
        // Display flat unwrapped 2D grid
        displayUnwrappedTerrain(radialTileData);
    } else {
        // Display wrapped cylindrical 3D
        displayWrappedTerrain(radialTileData);
    }
});

// Toggle mesh visibility (only controls STL meshes)
showMeshCheckbox.addEventListener('change', (e) => {
    const showMesh = e.target.checked;
    console.log('Show original mesh:', showMesh);

    // Show/hide terrain STL mesh
    if (terrainMesh) {
        terrainMesh.visible = showMesh;
    }

    // Show/hide tool STL mesh
    if (toolMesh) {
        toolMesh.visible = showMesh;
    }
});

// Initialize
initScene();
initWorkers();
loadSettings();
