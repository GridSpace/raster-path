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

let modelSTL = null;  // ArrayBuffer
let toolSTL = null;   // ArrayBuffer

let modelTriangles = null;  // Float32Array
let toolTriangles = null;   // Float32Array

let modelRasterData = null;
let toolRasterData = null;
let toolpathData = null;

let modelMaxZ = 0;  // Track max Z for tool offset

let rasterPath = null;  // RasterPath instance

// Three.js objects
let scene, camera, renderer, controls;
let rotatedGroup = null;  // Group for 90-degree rotation
let modelMesh = null;
let toolMesh = null;
let modelRasterPoints = null;
let toolRasterPoints = null;
let toolpathPoints = null;

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
        document.getElementById('resolution').value = resolution;
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
        maxGPUMemoryMB: 128,  // Reduce from default 256MB to avoid workgroup failures
        debug: true  // Enable debug logging
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

        // Rasterize model
        if (modelTriangles) {
            // Center mesh on origin for radial mode
            let trianglesToRaster = modelTriangles;
            if (mode === 'radial') {
                const bounds = calculateTriangleBounds(modelTriangles);
                const yzCenterY = (bounds.max.y + bounds.min.y) / 2;
                const yzCenterZ = (bounds.max.z + bounds.min.z) / 2;

                if (Math.abs(yzCenterY) > 0.01 || Math.abs(yzCenterZ) > 0.01) {
                    console.log(`Centering model for radial: Y offset=${yzCenterY.toFixed(2)}, Z offset=${yzCenterZ.toFixed(2)}`);
                    trianglesToRaster = new Float32Array(modelTriangles.length);
                    for (let i = 0; i < modelTriangles.length; i += 3) {
                        trianglesToRaster[i] = modelTriangles[i]; // X unchanged
                        trianglesToRaster[i + 1] = modelTriangles[i + 1] - yzCenterY; // Center Y
                        trianglesToRaster[i + 2] = modelTriangles[i + 2] - yzCenterZ; // Center Z
                    }
                }
            }

            updateInfo('Rasterizing model...');
            const t0 = performance.now();
            modelRasterData = await rasterPath.rasterizeModel({
                triangles: trianglesToRaster,
                zFloor: zFloor
            });
            const t1 = performance.now();
            updateInfo(`Model rasterized in ${(t1 - t0).toFixed(0)}ms`);

            // Debug radial data
            if (mode === 'radial') {
                const gw = modelRasterData.gridWidth;
                const gh = modelRasterData.gridHeight;
                const circ = modelRasterData.circumference;
                const halfRowIdx = Math.floor(gh / 2);

                // Sample different rows
                const row0 = modelRasterData.positions.slice(0, Math.min(10, gw)); // θ=0°
                const rowHalf = modelRasterData.positions.slice(halfRowIdx * gw, halfRowIdx * gw + Math.min(10, gw)); // θ=180°
                const rowQuarter = modelRasterData.positions.slice(Math.floor(gh / 4) * gw, Math.floor(gh / 4) * gw + Math.min(10, gw)); // θ=90°
                const row3Quarter = modelRasterData.positions.slice(Math.floor(3 * gh / 4) * gw, Math.floor(3 * gh / 4) * gw + Math.min(10, gw)); // θ=270°

                console.log('Radial raster debug:', {
                    gridWidth: gw,
                    gridHeight: gh,
                    circumference: circ,
                    maxRadius: modelRasterData.maxRadius,
                    stepSize: resolution,
                    totalPoints: modelRasterData.positions.length,
                    expectedSize: gw * gh,
                    'row_0_theta0deg': Array.from(row0),
                    'row_quarter_theta90deg': Array.from(rowQuarter),
                    'row_half_theta180deg': Array.from(rowHalf),
                    'row_3quarter_theta270deg': Array.from(row3Quarter)
                });

                // Check mesh bounds
                const meshBounds = {
                    minY: Infinity, maxY: -Infinity,
                    minZ: Infinity, maxZ: -Infinity
                };
                for (let i = 0; i < modelTriangles.length; i += 3) {
                    const y = modelTriangles[i + 1];
                    const z = modelTriangles[i + 2];
                    meshBounds.minY = Math.min(meshBounds.minY, y);
                    meshBounds.maxY = Math.max(meshBounds.maxY, y);
                    meshBounds.minZ = Math.min(meshBounds.minZ, z);
                    meshBounds.maxZ = Math.max(meshBounds.maxZ, z);
                }
                console.log('Mesh YZ bounds:', meshBounds);
            }

            // Calculate max Z for tool offset
            const positions = modelRasterData.positions;
            modelMaxZ = -Infinity;
            for (let i = 0; i < positions.length; i++) {
                const z = positions[i];
                if (z > -1e9 && z > modelMaxZ) {
                    modelMaxZ = z;
                }
            }
        }

        // Rasterize tool
        if (toolTriangles) {
            // Center tool on origin for radial mode
            let toolTrianglesToRaster = toolTriangles;
            if (mode === 'radial') {
                const bounds = calculateTriangleBounds(toolTriangles);
                const yzCenterY = (bounds.max.y + bounds.min.y) / 2;
                const yzCenterZ = (bounds.max.z + bounds.min.z) / 2;

                if (Math.abs(yzCenterY) > 0.01 || Math.abs(yzCenterZ) > 0.01) {
                    console.log(`Centering tool for radial: Y offset=${yzCenterY.toFixed(2)}, Z offset=${yzCenterZ.toFixed(2)}`);
                    toolTrianglesToRaster = new Float32Array(toolTriangles.length);
                    for (let i = 0; i < toolTriangles.length; i += 3) {
                        toolTrianglesToRaster[i] = toolTriangles[i]; // X unchanged
                        toolTrianglesToRaster[i + 1] = toolTriangles[i + 1] - yzCenterY; // Center Y
                        toolTrianglesToRaster[i + 2] = toolTriangles[i + 2] - yzCenterZ; // Center Z
                    }
                }
            }

            updateInfo('Rasterizing tool...');
            const t0 = performance.now();
            toolRasterData = await rasterPath.rasterizeTool({
                triangles: toolTrianglesToRaster,
                zFloor: zFloor
            });
            const t1 = performance.now();
            updateInfo(`Tool rasterized in ${(t1 - t0).toFixed(0)}ms`);
        }

        updateInfo('Rasterization complete');

        // Auto-enable raster view
        document.getElementById('show-raster').checked = true;

        updateVisualization();
        updateButtonStates();

    } catch (error) {
        console.error('Rasterization error:', error);
        updateInfo(`Error: ${error.message}`);
    }
}

async function generateToolpath() {
    if (!modelRasterData || !toolRasterData) {
        updateInfo('Both model and tool must be rasterized first');
        return;
    }

    try {
        updateInfo('Generating toolpath...');
        const t0 = performance.now();

        toolpathData = await rasterPath.generateToolpaths({
            terrainData: modelRasterData,
            toolData: toolRasterData,
            xStep: xStep,
            yStep: yStep,
            zFloor: zFloor
        });

        const t1 = performance.now();
        const numPoints = toolpathData.pathData.length;
        updateInfo(`Toolpath generated: ${numPoints.toLocaleString()} points in ${(t1 - t0).toFixed(0)}ms`);

        // Auto-enable toolpath view
        document.getElementById('show-paths').checked = true;

        updateVisualization();

    } catch (error) {
        console.error('Toolpath generation error:', error);
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

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(modelTriangles, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
        color: 0x00ffff,
        shininess: 30,
        transparent: true,
        opacity: 0.6
    });

    modelMesh = new THREE.Mesh(geometry, material);
    rotatedGroup.add(modelMesh);
}

function displayToolMesh() {
    if (!toolTriangles) return;

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
    toolMesh.position.z = modelMaxZ + 20;  // Offset tool above model
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
        // Radial: returned data is planar (unwrapped)
        const { positions: rasterPos, bounds, circumference, maxRadius, rotationStepDegrees } = modelRasterData;

        if (wrapped) {
            // Wrap around X-axis
            const stepSize = resolution;

            if (modelRasterData.isDense) {
                // Dense array format
                const gridWidth = modelRasterData.gridWidth;
                const gridHeight = modelRasterData.gridHeight;

                for (let gy = 0; gy < gridHeight; gy++) {
                    const theta = gy * (rotationStepDegrees * Math.PI / 180);
                    const cosTheta = Math.cos(theta);
                    const sinTheta = Math.sin(theta);

                    for (let gx = 0; gx < gridWidth; gx++) {
                        const idx = gy * gridWidth + gx;
                        const radius = rasterPos[idx];

                        if (radius > -1e9) {
                            const x = bounds.min.x + gx * stepSize;
                            const y = radius * cosTheta;
                            const z = radius * sinTheta;

                            positions.push(x, y, z);
                            colors.push(0, 1, 0);  // Green
                        }
                    }
                }
            }
        } else {
            // Show unwrapped (planar)
            const stepSize = resolution;

            if (modelRasterData.isDense) {
                const gridWidth = modelRasterData.gridWidth;
                const gridHeight = modelRasterData.gridHeight;

                for (let gy = 0; gy < gridHeight; gy++) {
                    const y = gy * stepSize;

                    for (let gx = 0; gx < gridWidth; gx++) {
                        const idx = gy * gridWidth + gx;
                        const radius = rasterPos[idx];

                        if (radius > -1e9) {
                            const x = bounds.min.x + gx * stepSize;

                            positions.push(x, y, radius);
                            colors.push(0, 1, 0);  // Green
                        }
                    }
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
    const zOffset = modelMaxZ + 20;  // Offset tool above model

    for (let i = 0; i < rasterPos.length; i += 3) {
        const gridX = rasterPos[i];
        const gridY = rasterPos[i + 1];
        const z = rasterPos[i + 2];

        const x = bounds.min.x + gridX * stepSize;
        const y = bounds.min.y + gridY * stepSize;

        positions.push(x, y, z + zOffset);
        colors.push(1, 0.4, 0);  // Orange
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
        rotatedGroup.add(toolRasterPoints);
    }
}

function displayToolpaths(wrapped) {
    if (!toolpathData) return;

    const { pathData, numScanlines, pointsPerLine } = toolpathData;
    const positions = [];
    const colors = [];

    if (mode === 'planar') {
        // Planar toolpaths
        const bounds = toolpathData.generationBounds || modelRasterData.bounds;
        const stepSize = resolution;

        for (let line = 0; line < numScanlines; line++) {
            for (let pt = 0; pt < pointsPerLine; pt++) {
                const idx = line * pointsPerLine + pt;
                const z = pathData[idx];

                const x = bounds.min.x + pt * xStep * stepSize;
                const y = bounds.min.y + line * yStep * stepSize;

                positions.push(x, y, z);
                colors.push(1, 0.4, 0);  // Orange
            }
        }

    } else {
        // Radial toolpaths - returned as planar, wrap if needed
        const bounds = toolpathData.generationBounds || modelRasterData.bounds;
        const rotationStepDegrees = modelRasterData.rotationStepDegrees;
        const stepSize = resolution;

        if (wrapped) {
            // Wrap around X-axis
            for (let line = 0; line < numScanlines; line++) {
                const gridY = line * yStep;
                const theta = gridY * (rotationStepDegrees * Math.PI / 180);
                const cosTheta = Math.cos(theta);
                const sinTheta = Math.sin(theta);

                for (let pt = 0; pt < pointsPerLine; pt++) {
                    const idx = line * pointsPerLine + pt;
                    const radius = pathData[idx];

                    const gridX = pt * xStep;
                    const x = bounds.min.x + gridX * stepSize;
                    const yWrapped = radius * cosTheta;
                    const zWrapped = radius * sinTheta;

                    positions.push(x, yWrapped, zWrapped);
                    colors.push(1, 0.4, 0);  // Orange
                }
            }
        } else {
            // Show unwrapped (planar)
            for (let line = 0; line < numScanlines; line++) {
                const gridY = line * yStep;
                const y = gridY * stepSize;

                for (let pt = 0; pt < pointsPerLine; pt++) {
                    const idx = line * pointsPerLine + pt;
                    const radius = pathData[idx];

                    const gridX = pt * xStep;
                    const x = bounds.min.x + gridX * stepSize;

                    positions.push(x, y, radius);
                    colors.push(1, 0.4, 0);  // Orange
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
        rotatedGroup.add(toolpathPoints);
    }
}

// ============================================================================
// UI Updates
// ============================================================================

function updateInfo(text) {
    document.getElementById('info').textContent = text;
    console.log(text);
}

function updateButtonStates() {
    const hasModel = modelTriangles !== null;
    const hasTool = toolTriangles !== null;
    const hasAnySTL = hasModel || hasTool;
    const hasBothRasters = modelRasterData !== null && toolRasterData !== null;

    document.getElementById('rasterize').disabled = !hasAnySTL;
    document.getElementById('generate-toolpath').disabled = !hasBothRasters;
}

// ============================================================================
// Event Handlers
// ============================================================================

function updateModeUI() {
    // Show/hide wrapped toggle and angle step for radial mode
    const wrappedContainer = document.getElementById('wrapped-container');
    const angleStepContainer = document.getElementById('angle-step-container');
    if (mode === 'radial') {
        wrappedContainer.style.display = 'block';
        angleStepContainer.style.display = 'block';
    } else {
        wrappedContainer.style.display = 'none';
        angleStepContainer.style.display = 'none';
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
        modelTriangles = parseSTL(modelSTL);
        document.getElementById('model-status').textContent = isOldFormat ? 'Cached model' : (cachedModel.name || 'Cached model');
        displayModelMesh();
    }

    const cachedTool = await getCachedSTL('tool-stl');
    if (cachedTool) {
        // Handle both old format (raw ArrayBuffer) and new format (object with arrayBuffer and name)
        const isOldFormat = cachedTool instanceof ArrayBuffer;
        toolSTL = isOldFormat ? cachedTool : cachedTool.arrayBuffer;
        toolTriangles = parseSTL(toolSTL);
        document.getElementById('tool-status').textContent = isOldFormat ? 'Cached tool' : (cachedTool.name || 'Cached tool');
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

    // Load Model button
    document.getElementById('load-model').addEventListener('click', async () => {
        const result = await loadSTLFile(true);
        if (result) {
            modelSTL = result.arrayBuffer;
            modelTriangles = result.triangles;
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
            toolSTL = result.arrayBuffer;
            toolTriangles = result.triangles;
            document.getElementById('tool-status').textContent = result.name;

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
            checkbox.addEventListener('change', updateVisualization);
        }
    });

    updateInfo('Ready - Load STL files to begin');
});
