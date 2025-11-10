import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RasterPath } from './raster-path.js';

// ============================================================================
// State
// ============================================================================

let mode = 'planar';
let resolution = 0.1;
let zFloor = -100;
let xStep = 5;
let yStep = 5;
let angleStep = 1.0; // degrees
let toolSize = 2.5; // mm - target tool diameter

let modelSTL = null;  // ArrayBuffer (current, possibly rotated)
let modelOriginalSTL = null;  // ArrayBuffer (original, for reset)
let toolSTL = null;   // ArrayBuffer
let toolOriginalSTL = null;  // ArrayBuffer (original, for reset/scaling)

let modelTriangles = null;  // Float32Array
let toolTriangles = null;   // Float32Array
let toolOriginalTriangles = null;  // Float32Array (original, unscaled)

let modelRasterData = null;
let toolRasterData = null;
let toolpathData = null;

let modelMaxZ = 0;  // Track max Z for tool offset
let rasterPath = null;  // RasterPath instance

// Model rotation state (accumulated 90-degree rotations)
let modelRotation = { x: 0, y: 0, z: 0 };  // In 90-degree increments

// Three.js objects
let scene, camera, renderer, controls;
let rotatedGroup = null;  // Group for 90-degree rotation
let modelMesh = null;
let toolMesh = null;
let modelRasterPoints = null;
let toolRasterPoints = null;
let toolpathPoints = null;

const log_pre = '[App]';
const debug = {
    error: function() { console.error(log_pre, ...arguments) },
    warn: function() { console.warn(log_pre, ...arguments) },
    log: function() { console.log(log_pre, ...arguments) },
    ok: function() { console.log(log_pre, '✅', ...arguments) },
};

// ============================================================================
// Parameter Persistence
// ============================================================================

function saveParameters() {
    localStorage.setItem('raster-mode', mode);
    localStorage.setItem('raster-resolution', resolution);
    localStorage.setItem('raster-zFloor', zFloor);
    localStorage.setItem('raster-xStep', xStep);
    localStorage.setItem('raster-yStep', yStep);
    localStorage.setItem('raster-angleStep', angleStep);
    localStorage.setItem('raster-toolSize', toolSize);
    console.log(`[App] Saved tool size: ${toolSize}mm`);

    // Save view checkboxes
    const showWrappedCheckbox = document.getElementById('show-wrapped');
    if (showWrappedCheckbox) {
        localStorage.setItem('raster-showWrapped', showWrappedCheckbox.checked);
    }
}

function loadParameters() {
    const savedMode = localStorage.getItem('raster-mode');
    if (savedMode !== null) {
        mode = savedMode;
        const modeRadio = document.querySelector(`input[name="mode"][value="${mode}"]`);
        if (modeRadio) {
            modeRadio.checked = true;
            // Trigger mode change to update UI visibility
            updateModeUI();
        }
    }

    const savedResolution = localStorage.getItem('raster-resolution');
    if (savedResolution !== null) {
        resolution = parseFloat(savedResolution);
        const resolutionSelect = document.getElementById('resolution');

        // Check if the saved value exists as an option
        const optionExists = Array.from(resolutionSelect.options).some(opt => parseFloat(opt.value) === resolution);

        if (!optionExists) {
            // Add the custom resolution as an option
            const newOption = document.createElement('option');
            newOption.value = resolution.toFixed(3);
            newOption.textContent = `${resolution.toFixed(3)}mm`;
            resolutionSelect.appendChild(newOption);
        }

        // Set the value as a string to match the option values
        resolutionSelect.value = resolution.toFixed(3);
    }

    const savedZFloor = localStorage.getItem('raster-zFloor');
    if (savedZFloor !== null) {
        zFloor = parseFloat(savedZFloor);
        document.getElementById('z-floor').value = zFloor;
    }

    const savedXStep = localStorage.getItem('raster-xStep');
    if (savedXStep !== null) {
        xStep = parseInt(savedXStep);
        document.getElementById('x-step').value = xStep;
    }

    const savedYStep = localStorage.getItem('raster-yStep');
    if (savedYStep !== null) {
        yStep = parseInt(savedYStep);
        document.getElementById('y-step').value = yStep;
    }

    const savedAngleStep = localStorage.getItem('raster-angleStep');
    if (savedAngleStep !== null) {
        angleStep = parseFloat(savedAngleStep);
        document.getElementById('angle-step').value = angleStep;
    }

    const savedToolSize = localStorage.getItem('raster-toolSize');
    if (savedToolSize !== null) {
        toolSize = parseFloat(savedToolSize);
        // Format to match dropdown option values (e.g., "3.0" not "3")
        document.getElementById('tool-size').value = toolSize.toFixed(1);
        console.log(`[App] Restored tool size: ${toolSize}mm`);
    } else {
        console.log(`[App] No saved tool size, using default: ${toolSize}mm`);
    }

    // Restore view checkboxes
    const savedShowWrapped = localStorage.getItem('raster-showWrapped');
    if (savedShowWrapped !== null) {
        const showWrappedCheckbox = document.getElementById('show-wrapped');
        if (showWrappedCheckbox) {
            showWrappedCheckbox.checked = savedShowWrapped === 'true';
        }
    }
}

// ============================================================================
// IndexedDB for STL Caching
// ============================================================================

const DB_NAME = 'raster-path-cache';
const DB_VERSION = 1;
const STORE_NAME = 'stl-files';

async function openDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);

        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME);
            }
        };
    });
}

async function cacheSTL(key, arrayBuffer, name) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const transaction = db.transaction(STORE_NAME, 'readwrite');
        const store = transaction.objectStore(STORE_NAME);
        const data = { arrayBuffer, name };
        const request = store.put(data, key);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
    });
}

async function getCachedSTL(key) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const transaction = db.transaction(STORE_NAME, 'readonly');
        const store = transaction.objectStore(STORE_NAME);
        const request = store.get(key);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

// ============================================================================
// STL Parsing
// ============================================================================

function calculateTriangleBounds(triangles) {
    const bounds = {
        min: { x: Infinity, y: Infinity, z: Infinity },
        max: { x: -Infinity, y: -Infinity, z: -Infinity }
    };
    for (let i = 0; i < triangles.length; i += 3) {
        bounds.min.x = Math.min(bounds.min.x, triangles[i]);
        bounds.max.x = Math.max(bounds.max.x, triangles[i]);
        bounds.min.y = Math.min(bounds.min.y, triangles[i + 1]);
        bounds.max.y = Math.max(bounds.max.y, triangles[i + 1]);
        bounds.min.z = Math.min(bounds.min.z, triangles[i + 2]);
        bounds.max.z = Math.max(bounds.max.z, triangles[i + 2]);
    }
    return bounds;
}

// Calculate current tool diameter from XY dimensions
function calculateToolDiameter(triangles) {
    if (!triangles || triangles.length === 0) return 0;

    const bounds = calculateTriangleBounds(triangles);
    const xSize = bounds.max.x - bounds.min.x;
    const ySize = bounds.max.y - bounds.min.y;

    // Return the maximum of X and Y dimensions as the diameter
    return Math.max(xSize, ySize);
}

// Scale tool triangles to target diameter
function scaleToolTriangles(triangles, targetDiameter) {
    if (!triangles || triangles.length === 0) return triangles;

    const currentDiameter = calculateToolDiameter(triangles);
    if (currentDiameter === 0) return triangles;

    const scale = targetDiameter / currentDiameter;
    const scaled = new Float32Array(triangles.length);

    for (let i = 0; i < triangles.length; i += 3) {
        scaled[i] = triangles[i] * scale;       // X
        scaled[i + 1] = triangles[i + 1] * scale; // Y
        scaled[i + 2] = triangles[i + 2] * scale; // Z
    }

    return scaled;
}

function parseSTL(arrayBuffer) {
    const view = new DataView(arrayBuffer);

    // Check if ASCII (starts with "solid")
    const text = new TextDecoder().decode(arrayBuffer.slice(0, 80));
    if (text.toLowerCase().startsWith('solid')) {
        return parseASCIISTL(arrayBuffer);
    } else {
        return parseBinarySTL(view);
    }
}

function parseBinarySTL(view) {
    const numTriangles = view.getUint32(80, true);
    const triangles = new Float32Array(numTriangles * 9);

    let offset = 84;
    let floatIndex = 0;

    for (let i = 0; i < numTriangles; i++) {
        offset += 12; // Skip normal

        for (let j = 0; j < 9; j++) {
            triangles[floatIndex++] = view.getFloat32(offset, true);
            offset += 4;
        }

        offset += 2; // Skip attribute byte count
    }

    return triangles;
}

function parseASCIISTL(arrayBuffer) {
    const text = new TextDecoder().decode(arrayBuffer);
    const lines = text.split('\n');
    const triangles = [];
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

            if (vertices.length === 9) {
                triangles.push(...vertices);
                vertices = [];
            }
        }
    }

    return new Float32Array(triangles);
}

// ============================================================================
// STL Creation from Triangles
// ============================================================================

function createSTLFromTriangles(triangles) {
    // Create binary STL from triangle array
    const numTriangles = triangles.length / 9;
    const bufferSize = 80 + 4 + numTriangles * 50; // header + count + triangles
    const buffer = new ArrayBuffer(bufferSize);
    const view = new DataView(buffer);

    // Write header (80 bytes, can be anything)
    const headerText = 'Binary STL - rotated model';
    for (let i = 0; i < Math.min(headerText.length, 80); i++) {
        view.setUint8(i, headerText.charCodeAt(i));
    }

    // Write triangle count
    view.setUint32(80, numTriangles, true);

    // Write triangles
    let offset = 84;
    for (let i = 0; i < triangles.length; i += 9) {
        // Calculate normal (simplified - not computing actual normal)
        view.setFloat32(offset, 0, true); offset += 4; // nx
        view.setFloat32(offset, 0, true); offset += 4; // ny
        view.setFloat32(offset, 1, true); offset += 4; // nz

        // Write vertices
        for (let j = 0; j < 9; j++) {
            view.setFloat32(offset, triangles[i + j], true);
            offset += 4;
        }

        // Attribute byte count
        view.setUint16(offset, 0, true);
        offset += 2;
    }

    return buffer;
}

// ============================================================================
// Model Rotation
// ============================================================================

function rotateTriangles90(triangles, axis, direction) {
    // direction: 1 for +90°, -1 for -90°
    // Rotates triangle vertices around the specified axis
    const result = new Float32Array(triangles.length);
    const angle = direction * Math.PI / 2;  // 90 degrees in radians
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    for (let i = 0; i < triangles.length; i += 3) {
        const x = triangles[i];
        const y = triangles[i + 1];
        const z = triangles[i + 2];

        if (axis === 'x') {
            // Rotate around X-axis: y' = y*cos - z*sin, z' = y*sin + z*cos
            result[i] = x;
            result[i + 1] = y * cos - z * sin;
            result[i + 2] = y * sin + z * cos;
        } else if (axis === 'y') {
            // Rotate around Y-axis: x' = x*cos + z*sin, z' = -x*sin + z*cos
            result[i] = x * cos + z * sin;
            result[i + 1] = y;
            result[i + 2] = -x * sin + z * cos;
        } else if (axis === 'z') {
            // Rotate around Z-axis: x' = x*cos - y*sin, y' = x*sin + y*cos
            result[i] = x * cos - y * sin;
            result[i + 1] = x * sin + y * cos;
            result[i + 2] = z;
        }
    }

    return result;
}

function applyModelRotation(axis, direction) {
    if (!modelMesh) {
        updateInfo('No model loaded');
        return;
    }

    // Update rotation state
    modelRotation[axis] = (modelRotation[axis] + direction) % 4;
    if (modelRotation[axis] < 0) modelRotation[axis] += 4;

    // Rotate the mesh using Three.js
    const angle = direction * Math.PI / 2;
    if (axis === 'x') {
        modelMesh.rotateX(-angle);
    } else if (axis === 'y') {
        modelMesh.rotateY(angle);
    } else if (axis === 'z') {
        modelMesh.rotateZ(-angle);
    }

    let { geometry } = modelMesh;

    modelMesh.updateMatrix();
    geometry.applyMatrix4(modelMesh.matrix);
    modelMesh.position.set(0, 0, 0);
    modelMesh.rotation.set(0, 0, 0);
    modelMesh.scale.set(1, 1, 1);
    modelMesh.updateMatrixWorld(true);
    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();

    // Clear raster and toolpath data (needs recomputation)
    modelRasterData = null;
    toolpathData = null;

    // Update cached STL with rotated triangles
    modelSTL = createSTLFromTriangles(modelTriangles);
    cacheSTL('model-stl', modelSTL, document.getElementById('model-status').textContent || 'model.stl')
        .catch(err => debug.warn('Failed to cache rotated model:', err));

    updateButtonStates();
    updateInfo(`Model rotated ${direction > 0 ? '+' : ''}${direction * 90}° around ${axis.toUpperCase()}`);
}

function resetModelRotation() {
    if (!modelMesh || !modelOriginalSTL) {
        updateInfo('No model loaded');
        return;
    }

    // Re-parse from ORIGINAL STL (not current/rotated)
    modelTriangles = parseSTL(modelOriginalSTL);
    modelSTL = modelOriginalSTL; // Reset current to original
    modelRotation = { x: 0, y: 0, z: 0 };

    // Update the existing mesh geometry
    const positionAttr = modelMesh.geometry.attributes.position;
    for (let i = 0; i < positionAttr.count; i++) {
        positionAttr.setXYZ(i,
            modelTriangles[i * 3],
            modelTriangles[i * 3 + 1],
            modelTriangles[i * 3 + 2]
        );
    }
    positionAttr.needsUpdate = true;
    modelMesh.geometry.computeVertexNormals();
    modelMesh.geometry.computeBoundingBox();

    // Reset mesh rotation
    modelMesh.rotation.set(0, 0, 0);
    modelMesh.updateMatrix();
    modelMesh.updateMatrixWorld(true);

    // Clear raster and toolpath data
    modelRasterData = null;
    toolpathData = null;

    // Update cache with original STL
    cacheSTL('model-stl', modelSTL, document.getElementById('model-status').textContent || 'model.stl')
        .catch(err => debug.warn('Failed to cache reset model:', err));

    updateButtonStates();
    updateInfo('Model rotation reset');
}

// ============================================================================
// File Loading
// ============================================================================

async function loadSTLFile(isModel) {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.stl';

    return new Promise((resolve) => {
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return resolve(null);

            const arrayBuffer = await file.arrayBuffer();
            const triangles = parseSTL(arrayBuffer);

            // Cache in IndexedDB with filename
            const cacheKey = isModel ? 'model-stl' : 'tool-stl';
            await cacheSTL(cacheKey, arrayBuffer, file.name);

            modelRasterData = undefined;
            toolpathData = undefined;
            updateVisualization();

            updateInfo(`Loaded ${file.name}: ${(triangles.length / 9).toLocaleString()} triangles`);
            resolve({ arrayBuffer, triangles, name: file.name });
        };

        input.click();
    });
}

// ============================================================================
// RasterPath Integration
// ============================================================================

async function initRasterPath() {
    if (rasterPath) {
        rasterPath.terminate();
    }

    rasterPath = new RasterPath({
        mode: mode,
        resolution: resolution,
        rotationStep: mode === 'radial' ? angleStep : undefined,
        batchDivisor: 5,
        debug: true
    });

    await rasterPath.init();
    updateInfo(`RasterPath initialized: ${mode} mode, ${resolution}mm resolution`);
}

async function rasterizeAll() {
    if (!modelTriangles && !toolTriangles) {
        updateInfo('No STL files loaded');
        return;
    }

    try {
        // Ensure RasterPath is initialized with current settings
        await initRasterPath();

        // Load tool (works for both modes)
        if (toolTriangles) {
            const t0 = performance.now();
            toolRasterData = await rasterPath.loadTool({
                triangles: toolTriangles
            });
            const t1 = performance.now();
            updateInfo(`Tool loaded in ${(t1 - t0).toFixed(0)}ms`);
        }

        if (mode === 'planar') {
            // Planar mode: rasterize terrain immediately
            if (modelTriangles) {
                updateInfo('Rasterizing terrain...');
                const t0 = performance.now();
                modelRasterData = await rasterPath.loadTerrain({
                    triangles: modelTriangles,
                    zFloor: zFloor
                });
                const t1 = performance.now();
                updateInfo(`Terrain rasterized in ${(t1 - t0).toFixed(0)}ms`);
            }
        } else {
            // Radial mode: MUST load tool FIRST
            if (!toolTriangles) {
                updateInfo('Error: Radial mode requires tool to be loaded first');
                return;
            }

            // Load terrain (stores triangles for later, doesn't rasterize yet)
            if (modelTriangles) {
                const t0 = performance.now();
                await rasterPath.loadTerrain({
                    triangles: modelTriangles,
                    zFloor: zFloor
                });
                const t1 = performance.now();
                updateInfo(`Terrain loaded in ${(t1 - t0).toFixed(0)}ms (rasterized in toolpath generation)`);
            } else {
                updateInfo('Tool loaded. Load model and click "Generate Toolpath" to continue.');
            }

            // Update button states for radial mode
            updateButtonStates();
            return; // Don't rasterize model here - do it in toolpath generation
        }

        updateInfo('Rasterization complete');

        // Auto-enable raster view
        document.getElementById('show-raster').checked = true;

        updateVisualization();
        updateButtonStates();

    } catch (error) {
        debug.error('Rasterization error:', error);
        updateInfo(`Error: ${error.message}`);
    }
}

async function generateToolpath() {
    // Check if tool and terrain are loaded (unified check for both modes)
    if (!toolRasterData) {
        updateInfo('Tool must be loaded first');
        return;
    }

    if (mode === 'planar') {
        if (!modelRasterData) {
            updateInfo('Model must be rasterized first');
            return;
        }
    } else {
        // Radial mode: terrain must be loaded (stored internally)
        if (!modelTriangles) {
            updateInfo('Model STL must be loaded');
            return;
        }
    }

    try {
        const t0 = performance.now();
        updateInfo('Generating toolpath...');

        // Unified API - works for both modes!
        toolpathData = await rasterPath.generateToolpaths({
            xStep: xStep,
            yStep: yStep,
            zFloor: zFloor
        });

        const t1 = performance.now();

        if (mode === 'planar') {
            const numPoints = toolpathData.pathData.length;
            updateInfo(`Toolpath generated: ${numPoints.toLocaleString()} points in ${(t1 - t0).toFixed(0)}ms`);
        } else {
            // debug.log('[Radial] Toolpaths generated:', toolpathData);
            debug.log(`[Radial] Received ${toolpathData.strips.length} strips from worker, numStrips=${toolpathData.numStrips}`);
            updateInfo(`Toolpath generated: ${toolpathData.numStrips} strips, ${toolpathData.totalPoints.toLocaleString()} points in ${(t1 - t0).toFixed(0)}ms`);

            // Store terrain strips for visualization
            if (toolpathData.terrainStrips) {
                modelRasterData = toolpathData.terrainStrips;
                debug.log('[Radial] Terrain strips available:', modelRasterData.length, 'strips');

                // DEBUG: Check X range in strips
                if (modelRasterData.length > 0) {
                    const strip0 = modelRasterData[0];
                    debug.log('[Radial] Strip 0 structure:', {
                        angle: strip0.angle,
                        pointCount: strip0.pointCount,
                        positionsLength: strip0.positions?.length,
                        gridWidth: strip0.gridWidth,
                        gridHeight: strip0.gridHeight,
                        bounds: strip0.bounds
                    });

                    // Find actual X range in positions
                    if (strip0.positions && strip0.positions.length > 0) {
                        let minX = Infinity, maxX = -Infinity;
                        for (let i = 0; i < strip0.positions.length; i += 3) {
                            const x = strip0.positions[i];
                            minX = Math.min(minX, x);
                            maxX = Math.max(maxX, x);
                        }
                        debug.log('[Radial] Strip 0 actual X range in positions:', minX.toFixed(2), 'to', maxX.toFixed(2));
                    }
                }
            } else if (toolpathData.strips && toolpathData.strips[0]?.terrainBounds) {
                // Batched mode: create synthetic terrain strips with bounds only
                modelRasterData = toolpathData.strips.map(strip => ({
                    angle: strip.angle,
                    bounds: strip.terrainBounds
                }));
                debug.log('[Radial] Batched mode: created synthetic terrain strips from bounds:', modelRasterData.length, 'strips');
            }
            // This is fine - we only need toolpathData for visualization
        }

        // Auto-enable toolpath view
        document.getElementById('show-paths').checked = true;

        updateVisualization();

    } catch (error) {
        debug.error('Toolpath generation error:', error);
        updateInfo(`Error: ${error.message}`);
    }
}

// ============================================================================
// Three.js Visualization
// ============================================================================

function initThreeJS() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // Camera
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
    camera.position.set(100, 100, 100);
    camera.lookAt(0, 0, 0);

    // Renderer
    const canvas = document.getElementById('canvas');
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Controls
    controls = new OrbitControls(camera, canvas);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(100, 100, 100);
    scene.add(directionalLight);

    // Grid
    const gridHelper = new THREE.GridHelper(200, 20, 0x444444, 0x222222);
    scene.add(gridHelper);

    // Create rotated group for all visualizations (-90deg around X)
    rotatedGroup = new THREE.Group();
    rotatedGroup.rotation.x = -Math.PI / 2;
    scene.add(rotatedGroup);

    // Axes
    const axesHelper = new THREE.AxesHelper(50);
    scene.add(axesHelper);

    // Axis labels
    function makeTextSprite(message, color) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 128;
        context.font = 'Bold 48px Arial';
        context.fillStyle = color;
        context.textAlign = 'center';
        context.fillText(message, 128, 64);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(10, 5, 1);
        return sprite;
    }

    const xLabel = makeTextSprite('X', '#ff0000');
    xLabel.position.set(60, 0, 0);
    scene.add(xLabel);

    const yLabel = makeTextSprite('Y', '#00ff00');
    yLabel.position.set(0, 0, -60);
    scene.add(yLabel);

    const zLabel = makeTextSprite('Z', '#0000ff');
    zLabel.position.set(0, 60, 6);
    scene.add(zLabel);

    // Window resize
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();
}

function updateVisualization() {
    const showModel = document.getElementById('show-model').checked;
    const showRaster = document.getElementById('show-raster').checked;
    const showPaths = document.getElementById('show-paths').checked;
    const showWrapped = document.getElementById('show-wrapped').checked;

    // Model mesh
    if (modelMesh) {
        modelMesh.visible = showModel;
    } else if (showModel && modelTriangles) {
        displayModelMesh();
    }

    // Tool mesh
    if (toolMesh) {
        toolMesh.visible = showModel;
    } else if (showModel && toolTriangles) {
        displayToolMesh();
    }

    // Model raster points
    if (modelRasterPoints) {
        rotatedGroup.remove(modelRasterPoints);
        modelRasterPoints.geometry.dispose();
        modelRasterPoints.material.dispose();
        modelRasterPoints = null;
    }

    if (showRaster && modelRasterData) {
        displayModelRaster(showWrapped);
    }

    // Tool raster points
    if (toolRasterPoints) {
        rotatedGroup.remove(toolRasterPoints);
        toolRasterPoints.geometry.dispose();
        toolRasterPoints.material.dispose();
        toolRasterPoints = null;
    }

    if (showRaster && toolRasterData) {
        displayToolRaster();
    }

    // Toolpath points
    if (toolpathPoints) {
        rotatedGroup.remove(toolpathPoints);
        toolpathPoints.geometry.dispose();
        toolpathPoints.material.dispose();
        toolpathPoints = null;
    }

    if (showPaths && toolpathData) {
        displayToolpaths(showWrapped);
    }
}

function displayModelMesh() {
    if (!modelTriangles) return;

    // Remove old mesh if it exists
    if (modelMesh) {
        rotatedGroup.remove(modelMesh);
        modelMesh.geometry.dispose();
        modelMesh.material.dispose();
        modelMesh = null;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(modelTriangles, 3));
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();

    const material = new THREE.MeshPhongMaterial({
        color: 0x00ffff,
        shininess: 30,
        transparent: true,
        opacity: 0.6
    });

    modelMesh = new THREE.Mesh(geometry, material);
    rotatedGroup.add(modelMesh);

    // Calculate max Z for tool offset
    modelMaxZ = geometry.boundingBox.max.z + 20;

    // update tool position when terrain (re)loaded
    toolMesh && (toolMesh.position.z = modelMaxZ);  // Offset tool above model
}

function displayToolMesh() {
    if (!toolTriangles) return;

    // Remove old mesh if it exists
    if (toolMesh) {
        rotatedGroup.remove(toolMesh);
        toolMesh.geometry.dispose();
        toolMesh.material.dispose();
        toolMesh = null;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(toolTriangles, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
        color: 0xff6400,  // Orange color for tool
        shininess: 30,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.8
    });

    toolMesh = new THREE.Mesh(geometry, material);
    toolMesh.position.z = modelMaxZ;  // Offset tool above model
    rotatedGroup.add(toolMesh);
}

function displayModelRaster(wrapped) {
    if (!modelRasterData) return;

    const positions = [];
    const colors = [];

    if (mode === 'planar') {
        // Planar: terrain is dense (Z-only array)
        const { positions: rasterPos, bounds, gridWidth, gridHeight } = modelRasterData;
        const stepSize = resolution;

        for (let gy = 0; gy < gridHeight; gy++) {
            for (let gx = 0; gx < gridWidth; gx++) {
                const idx = gy * gridWidth + gx;
                const z = rasterPos[idx];

                if (z > -1e9) {
                    const x = bounds.min.x + gx * stepSize;
                    const y = bounds.min.y + gy * stepSize;

                    positions.push(x, y, z);
                    colors.push(0, 1, 0);  // Green
                }
            }
        }

    } else {
        // Radial: modelRasterData is an array of strips
        // Each strip has: { angle, positions (sparse XYZ), gridWidth, gridHeight, bounds }
        if (!Array.isArray(modelRasterData)) {
            debug.error('[Display] modelRasterData is not an array of strips');
            return;
        }

        if (wrapped) {
            // Wrap each strip around X-axis at its angle
            debug.log('[Display] Wrapping', modelRasterData.length, 'strips');

            // Track overall X range for debug
            let overallMinX = Infinity, overallMaxX = -Infinity;
            let totalPointsRendered = 0;

            for (const strip of modelRasterData) {
                const { angle, positions: stripPositions } = strip;

                if (!stripPositions || stripPositions.length === 0) continue;

                const theta = angle * Math.PI / 180;
                const cosTheta = Math.cos(theta);
                const sinTheta = Math.sin(theta);

                // Strip positions are sparse XYZ triplets
                for (let i = 0; i < stripPositions.length; i += 3) {
                    const x = stripPositions[i];
                    const y_planar = stripPositions[i + 1];  // Y in planar coordinates
                    const terrainHeight = stripPositions[i + 2];  // Distance from X-axis

                    overallMinX = Math.min(overallMinX, x);
                    overallMaxX = Math.max(overallMaxX, x);

                    // Wrap: use terrainHeight as radial distance from X-axis
                    const y = terrainHeight * cosTheta;
                    const z = terrainHeight * sinTheta;

                    positions.push(x, y, z);
                    colors.push(0, 1, 0);  // Green
                    totalPointsRendered++;
                }
            }

            debug.log('[Display] Rendered', totalPointsRendered, 'points from', modelRasterData.length, 'strips');
            debug.log('[Display] X range: [' + overallMinX.toFixed(2) + ', ' + overallMaxX.toFixed(2) + ']');

            // DEBUG: Check angle coverage
            const angles = modelRasterData.map(s => s.angle).sort((a, b) => a - b);
            debug.log('[Display] Angle range: [' + angles[0].toFixed(1) + '°, ' + angles[angles.length-1].toFixed(1) + '°]');
            debug.log('[Display] First 5 angles:', angles.slice(0, 5).map(a => a.toFixed(1) + '°').join(', '));
            debug.log('[Display] Last 5 angles:', angles.slice(-5).map(a => a.toFixed(1) + '°').join(', '));
        } else {
            // Show unwrapped (planar) - lay out strips side by side
            for (let stripIdx = 0; stripIdx < modelRasterData.length; stripIdx++) {
                const strip = modelRasterData[stripIdx];
                const { positions: stripPositions } = strip;

                if (!stripPositions || stripPositions.length === 0) continue;

                const stripY = stripIdx * resolution * 10;  // Offset each strip for visibility

                // Strip positions are sparse XYZ triplets
                for (let i = 0; i < stripPositions.length; i += 3) {
                    const x = stripPositions[i];
                    const terrainHeight = stripPositions[i + 2];

                    positions.push(x, stripY, terrainHeight);
                    colors.push(0, 1, 0);  // Green
                }
            }
        }
    }

    if (positions.length > 0) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        // Scale point size with resolution for proper spacing
        const pointSize = resolution;  // Leave some space between points

        const material = new THREE.PointsMaterial({
            size: pointSize,
            vertexColors: true
        });

        modelRasterPoints = new THREE.Points(geometry, material);
        rotatedGroup.add(modelRasterPoints);
    }
}

function displayToolRaster() {
    if (!toolRasterData) return;

    const positions = [];
    const colors = [];

    // Tool raster is always sparse format [gridX, gridY, Z]
    const { positions: rasterPos, bounds } = toolRasterData;
    const stepSize = resolution;

    for (let i = 0; i < rasterPos.length; i += 3) {
        const gridX = rasterPos[i];
        const gridY = rasterPos[i + 1];
        const z = -rasterPos[i + 2];
        const x = bounds.min.x + gridX * stepSize;
        const y = bounds.min.y + gridY * stepSize;

        positions.push(x, y, z);
        colors.push(1, 0.4, 0);
    }

    if (positions.length > 0) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        // Scale point size with resolution for proper spacing
        const pointSize = resolution;
        const material = new THREE.PointsMaterial({
            size: pointSize,
            vertexColors: true
        });

        toolRasterPoints = new THREE.Points(geometry, material);
        toolRasterPoints.position.z = modelMaxZ;  // Offset tool above model
        rotatedGroup.add(toolRasterPoints);
    }
}

function displayToolpaths(wrapped) {
    if (!toolpathData) {
        return;
    }

    if (mode === 'planar') {
        // Planar toolpaths
        const { pathData, numScanlines, pointsPerLine } = toolpathData;
        const bounds = toolpathData.generationBounds || modelRasterData.bounds;
        const stepSize = resolution;

        const totalPoints = numScanlines * pointsPerLine;
        const MAX_DISPLAY_POINTS = 10000000; // 10M points max for display

        // Check if we need to downsample
        if (totalPoints > MAX_DISPLAY_POINTS) {
            debug.warn(`[Toolpath Display] Toolpath too large for visualization: ${(totalPoints/1e6).toFixed(1)}M points`);
            debug.warn(`[Toolpath Display] Skipping display (max: ${(MAX_DISPLAY_POINTS/1e6).toFixed(1)}M points)`);
            debug.warn(`[Toolpath Display] Toolpath was generated successfully - only visualization is skipped`);
            return;
        }

        // Preallocate typed arrays for better performance
        const positions = new Float32Array(totalPoints * 3);
        const colors = new Float32Array(totalPoints * 3);

        let arrayIdx = 0;
        for (let line = 0; line < numScanlines; line++) {
            for (let pt = 0; pt < pointsPerLine; pt++) {
                const idx = line * pointsPerLine + pt;
                const z = pathData[idx];

                const x = bounds.min.x + pt * xStep * stepSize;
                const y = bounds.min.y + line * yStep * stepSize;

                positions[arrayIdx] = x;
                positions[arrayIdx + 1] = y;
                positions[arrayIdx + 2] = z;

                colors[arrayIdx] = 1;      // R
                colors[arrayIdx + 1] = 0.4; // G
                colors[arrayIdx + 2] = 0;   // B

                arrayIdx += 3;
            }
        }

        // Create geometry from preallocated arrays
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: resolution * 0.5,
            vertexColors: true
        });

        toolpathPoints = new THREE.Points(geometry, material);
        rotatedGroup.add(toolpathPoints);

        return; // Exit early for planar mode

    }

    // Radial V2 toolpaths - each strip is independent
    const { strips } = toolpathData;
    const stepSize = resolution;

    // Calculate total points across all strips
    let totalPoints = 0;
    for (const strip of strips) {
        totalPoints += strip.numScanlines * strip.pointsPerLine;
    }

    const MAX_DISPLAY_POINTS = 10000000; // 10M points max for display

    debug.log('[Toolpath Display] Radial V2 mode:', strips.length, 'strips,', totalPoints, 'total points');

    // Check if we need to skip visualization
    if (totalPoints > MAX_DISPLAY_POINTS) {
        debug.warn(`[Toolpath Display] Toolpath too large for visualization: ${(totalPoints/1e6).toFixed(1)}M points`);
        debug.warn(`[Toolpath Display] Skipping display (max: ${(MAX_DISPLAY_POINTS/1e6).toFixed(1)}M points)`);
        debug.warn(`[Toolpath Display] Toolpath was generated successfully - only visualization is skipped`);
        return;
    }

    // DEBUG: Check angle distribution AND data
    if (strips.length > 0) {
        const angleChecks = [0, 180, 359, 360, 361, 540, 719].filter(i => i < strips.length);
        debug.log('[Toolpath Display] Angle check at indices:', angleChecks.map(i => `${i}=${strips[i].angle.toFixed(1)}°`).join(', '));
        // Check if pathData is actually different between strips
        if (strips.length > 360) {
            const samples0 = strips[0].pathData.slice(0, 5).map(v => v.toFixed(3)).join(',');
            const samples360 = strips[360].pathData.slice(0, 5).map(v => v.toFixed(3)).join(',');
            debug.log('[Toolpath Display] Data check: strip 0 first 5 values:', samples0);
            debug.log('[Toolpath Display] Data check: strip 360 first 5 values:', samples360);
            debug.log('[Toolpath Display] Data is', samples0 === samples360 ? 'SAME (BUG!)' : 'DIFFERENT (OK)');
        }
    }

    if (strips.length > 0) {
        const firstStrip = strips[0];
        // Note: modelRasterData may be null with new pipeline (rasterize + toolpath in one step)
        const terrainStrip = modelRasterData ? modelRasterData.find(s => s.angle === firstStrip.angle) : null;

        // Find min/max without stack overflow
        let minVal = Infinity, maxVal = -Infinity;
        for (let i = 0; i < firstStrip.pathData.length; i++) {
            minVal = Math.min(minVal, firstStrip.pathData[i]);
            maxVal = Math.max(maxVal, firstStrip.pathData[i]);
        }

        debug.log('[Toolpath Display] First strip sample:', {
            angle: firstStrip.angle,
            numScanlines: firstStrip.numScanlines,
            pointsPerLine: firstStrip.pointsPerLine,
            firstValues: Array.from(firstStrip.pathData.slice(0, 10)),
            minValue: minVal,
            maxValue: maxVal,
            terrainBounds: terrainStrip?.bounds
        });
    }

    // Preallocate typed arrays for better performance
    const positions = new Float32Array(totalPoints * 3);
    const colors = new Float32Array(totalPoints * 3);
    let arrayIdx = 0;

    if (wrapped) {
        // Wrap each strip around X-axis at its angle
        // Each strip should have numScanlines=1 (single centerline)
        for (const strip of strips) {
            const { angle, pathData, numScanlines, pointsPerLine } = strip;
            const terrainStrip = modelRasterData.find(s => s.angle === angle);
            const stripBounds = terrainStrip?.bounds || { min: { x: -100, y: 0, z: 0 }, max: { x: 100, y: 10, z: 20 } };

            // Rotate around X-axis at this angle
            const theta = angle * Math.PI / 180;
            const cosTheta = Math.cos(theta);
            const sinTheta = Math.sin(theta);

            // Should only be 1 scanline for radial
            for (let line = 0; line < numScanlines; line++) {
                for (let pt = 0; pt < pointsPerLine; pt++) {
                    const idx = line * pointsPerLine + pt;
                    const radius = pathData[idx];  // Tool tip radius from X-axis

                    const gridX = pt * xStep;
                    const x = stripBounds.min.x + gridX * stepSize;

                    // Wrap around X-axis
                    const yWrapped = radius * cosTheta;
                    const zWrapped = radius * sinTheta;

                    // Use direct indexing instead of push()
                    positions[arrayIdx] = x;
                    positions[arrayIdx + 1] = yWrapped;
                    positions[arrayIdx + 2] = zWrapped;

                    colors[arrayIdx] = 1;      // R
                    colors[arrayIdx + 1] = 0.4; // G
                    colors[arrayIdx + 2] = 0;   // B

                    arrayIdx += 3;
                }
            }
        }
    } else {
        // Show unwrapped (not very useful for radial, but supported)
        for (const strip of strips) {
            const { angle, pathData, numScanlines, pointsPerLine } = strip;
            // Get bounds from the strip's terrain data (or use default if not available)
            const stripBounds = (modelRasterData && modelRasterData.find(s => s.angle === angle)?.bounds) ||
                              { min: { x: -100, y: 0, z: 0 }, max: { x: 100, y: 10, z: 20 } };
            const yOffset = angle; // Use angle as Y offset for visualization

            for (let line = 0; line < numScanlines; line++) {
                const gridY = line * yStep;
                const y = yOffset + gridY * stepSize;

                for (let pt = 0; pt < pointsPerLine; pt++) {
                    const idx = line * pointsPerLine + pt;
                    const radius = pathData[idx];

                    const gridX = pt * xStep;
                    const x = stripBounds.min.x + gridX * stepSize;

                    // Use direct indexing instead of push()
                    positions[arrayIdx] = x;
                    positions[arrayIdx + 1] = y;
                    positions[arrayIdx + 2] = radius;

                    colors[arrayIdx] = 1;      // R
                    colors[arrayIdx + 1] = 0.4; // G
                    colors[arrayIdx + 2] = 0;   // B

                    arrayIdx += 3;
                }
            }
        }
    }

    if (positions.length > 0) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        // Scale point size with resolution
        const pointSize = resolution * 1.5;  // Slightly larger than raster points

        const material = new THREE.PointsMaterial({
            size: pointSize,
            vertexColors: true
        });

        toolpathPoints = new THREE.Points(geometry, material);

        // For wrapped radial mode, rotate 90° around X and center on model mesh
        if (wrapped) {
            toolpathPoints.rotation.x = Math.PI / 2;

            // Center toolpath on model mesh
            if (modelMesh) {
                geometry.computeBoundingBox();
                const toolpathCenter = new THREE.Vector3();
                geometry.boundingBox.getCenter(toolpathCenter);

                const modelCenter = new THREE.Vector3();
                modelMesh.geometry.boundingBox.getCenter(modelCenter);

                // Offset toolpath to match model center
                toolpathPoints.position.copy(modelCenter).sub(toolpathCenter);
            }
        }

        rotatedGroup.add(toolpathPoints);
    }
}

// ============================================================================
// UI Updates
// ============================================================================

function updateInfo(text) {
    debug.log(text);
    document.getElementById('info').textContent = text;
}

function updateButtonStates() {
    const hasModel = modelTriangles !== null;
    const hasTool = toolTriangles !== null;
    const hasAnySTL = hasModel || hasTool;

    // Different requirements for planar vs radial mode
    let canGenerateToolpath;
    if (mode === 'planar') {
        // Planar: need both model and tool rasterized
        canGenerateToolpath = modelRasterData !== null && toolRasterData !== null;
    } else {
        // Radial: only need tool rasterized (model is rasterized during toolpath gen)
        canGenerateToolpath = toolRasterData !== null && hasModel;
    }

    document.getElementById('rasterize').disabled = !hasAnySTL;
    document.getElementById('generate-toolpath').disabled = !canGenerateToolpath;
}

// ============================================================================
// Event Handlers
// ============================================================================

function updateModeUI() {
    // Show/hide wrapped toggle and angle step for radial mode
    const wrappedContainer = document.getElementById('wrapped-container').classList;
    const angleStepContainer = document.getElementById('angle-step-container').classList;
    if (mode === 'radial') {
        wrappedContainer.remove('hide');
        angleStepContainer.remove('hide');
    } else {
        wrappedContainer.add('hide');
        angleStepContainer.add('hide');
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    // Load saved parameters
    loadParameters();

    // Initialize Three.js
    initThreeJS();

    // Load cached STLs from IndexedDB
    const cachedModel = await getCachedSTL('model-stl');
    if (cachedModel) {
        // Handle both old format (raw ArrayBuffer) and new format (object with arrayBuffer and name)
        const isOldFormat = cachedModel instanceof ArrayBuffer;
        modelSTL = isOldFormat ? cachedModel : cachedModel.arrayBuffer;
        modelOriginalSTL = modelSTL; // Cache might have rotated model, but treat as original for now
        modelTriangles = parseSTL(modelSTL);
        document.getElementById('model-status').textContent = isOldFormat ? 'Cached model' : (cachedModel.name || 'Cached model');
        displayModelMesh();
    }

    const cachedTool = await getCachedSTL('tool-stl');
    if (cachedTool) {
        // Handle both old format (raw ArrayBuffer) and new format (object with arrayBuffer and name)
        const isOldFormat = cachedTool instanceof ArrayBuffer;
        toolOriginalSTL = isOldFormat ? cachedTool : cachedTool.arrayBuffer;
        toolOriginalTriangles = parseSTL(toolOriginalSTL);

        // Scale tool to target size
        const originalDiameter = calculateToolDiameter(toolOriginalTriangles);
        toolTriangles = scaleToolTriangles(toolOriginalTriangles, toolSize);
        toolSTL = createSTLFromTriangles(toolTriangles);

        // Update status
        document.getElementById('tool-status').textContent = isOldFormat ? 'Cached tool' : (cachedTool.name || 'Cached tool');
        document.getElementById('tool-size-status').textContent =
            `Original: ${originalDiameter.toFixed(2)}mm → Scaled: ${toolSize}mm`;

        displayToolMesh();
    }

    updateButtonStates();

    // Mode toggle
    document.querySelectorAll('input[name="mode"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            mode = e.target.value;
            updateModeUI();

            // Clear raster data (needs re-rasterization with new mode)
            modelRasterData = null;
            toolRasterData = null;
            toolpathData = null;

            saveParameters();
            updateInfo(`Mode changed to ${mode}`);
            updateButtonStates();
            updateVisualization();
        });
    });

    // Resolution change
    document.getElementById('resolution').addEventListener('change', (e) => {
        resolution = parseFloat(e.target.value);

        // Clear raster data (needs re-rasterization with new resolution)
        modelRasterData = null;
        toolRasterData = null;
        toolpathData = null;

        saveParameters();
        updateInfo(`Resolution changed to ${resolution}mm`);
        updateButtonStates();
    });

    // Z Floor change
    document.getElementById('z-floor').addEventListener('change', (e) => {
        zFloor = parseFloat(e.target.value);

        // Clear raster data (needs re-rasterization with new zFloor)
        modelRasterData = null;
        toolRasterData = null;
        toolpathData = null;

        saveParameters();
        updateInfo(`Z Floor changed to ${zFloor}`);
        updateButtonStates();
    });

    // Stepping controls
    document.getElementById('x-step').addEventListener('change', (e) => {
        xStep = parseInt(e.target.value);
        toolpathData = null;  // Need to regenerate toolpath
        saveParameters();
        updateInfo(`X Step changed to ${xStep}`);
        updateButtonStates();
    });

    document.getElementById('y-step').addEventListener('change', (e) => {
        yStep = parseInt(e.target.value);
        toolpathData = null;  // Need to regenerate toolpath
        saveParameters();
        updateInfo(`Y Step changed to ${yStep}`);
        updateButtonStates();
    });

    document.getElementById('angle-step').addEventListener('change', (e) => {
        angleStep = parseFloat(e.target.value);
        if (mode === 'radial') {
            modelRasterData = null;  // Need to re-rasterize with new angle step
            toolRasterData = null;
            toolpathData = null;
        }
        saveParameters();
        updateInfo(`Angle Step changed to ${angleStep}°`);
        updateButtonStates();
    });

    // Tool size change
    document.getElementById('tool-size').addEventListener('change', async (e) => {
        toolSize = parseFloat(e.target.value);

        // If tool is loaded, rescale it
        if (toolOriginalTriangles) {
            const originalDiameter = calculateToolDiameter(toolOriginalTriangles);
            toolTriangles = scaleToolTriangles(toolOriginalTriangles, toolSize);
            toolSTL = createSTLFromTriangles(toolTriangles);

            // Update status
            document.getElementById('tool-size-status').textContent =
                `Original: ${originalDiameter.toFixed(2)}mm → Scaled: ${toolSize}mm`;

            // Check if tool was already loaded (before clearing)
            const wasToolLoaded = toolRasterData !== null;

            // Clear raster data (tool size changed)
            toolRasterData = null;
            toolpathData = null;

            displayToolMesh();

            // If rasterPath exists and tool was already loaded, reload it with new size
            if (rasterPath && wasToolLoaded) {
                updateInfo(`Reloading tool at ${toolSize}mm...`);
                try {
                    const t0 = performance.now();
                    toolRasterData = await rasterPath.loadTool({
                        triangles: toolTriangles
                    });
                    const t1 = performance.now();
                    updateInfo(`Tool reloaded at ${toolSize}mm in ${(t1 - t0).toFixed(0)}ms`);
                } catch (error) {
                    updateInfo(`Error reloading tool: ${error.message}`);
                }
            } else {
                updateInfo(`Tool size changed to ${toolSize}mm - click Rasterize to apply`);
            }

            updateButtonStates();
        }

        saveParameters();
    });

    // Model rotation buttons
    document.querySelectorAll('.rotate-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const axis = btn.dataset.axis;
            const direction = parseInt(btn.dataset.dir);
            applyModelRotation(axis, direction);
        });
    });

    document.getElementById('reset-rotation').addEventListener('click', resetModelRotation);

    // Load Model button
    document.getElementById('load-model').addEventListener('click', async () => {
        const result = await loadSTLFile(true);
        if (result) {
            modelSTL = result.arrayBuffer;
            modelOriginalSTL = result.arrayBuffer; // Save original for reset
            modelTriangles = result.triangles;
            modelRotation = { x: 0, y: 0, z: 0 };  // Reset rotation on new model
            document.getElementById('model-status').textContent = result.name;

            // Clear raster data
            modelRasterData = null;
            toolpathData = null;

            displayModelMesh();
            updateButtonStates();
        }
    });

    // Load Tool button
    document.getElementById('load-tool').addEventListener('click', async () => {
        const result = await loadSTLFile(false);
        if (result) {
            toolOriginalSTL = result.arrayBuffer;
            toolOriginalTriangles = result.triangles;

            // Calculate original diameter and scale to target size
            const originalDiameter = calculateToolDiameter(toolOriginalTriangles);
            toolTriangles = scaleToolTriangles(toolOriginalTriangles, toolSize);
            toolSTL = createSTLFromTriangles(toolTriangles);

            // Update status with size info
            document.getElementById('tool-status').textContent = result.name;
            document.getElementById('tool-size-status').textContent =
                `Original: ${originalDiameter.toFixed(2)}mm → Scaled: ${toolSize}mm`;

            // Clear raster data
            toolRasterData = null;
            toolpathData = null;

            displayToolMesh();
            updateButtonStates();
        }
    });

    // Rasterize button
    document.getElementById('rasterize').addEventListener('click', rasterizeAll);

    // Generate Toolpath button
    document.getElementById('generate-toolpath').addEventListener('click', generateToolpath);

    // View toggles
    ['show-model', 'show-raster', 'show-paths', 'show-wrapped'].forEach(id => {
        const checkbox = document.getElementById(id);
        if (checkbox) {
            checkbox.addEventListener('change', () => {
                updateVisualization();
                // Save wrapped checkbox state
                if (id === 'show-wrapped') {
                    saveParameters();
                }
            });
        }
    });

    updateInfo('Ready - Load STL files to begin');
});
