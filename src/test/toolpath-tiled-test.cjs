// toolpath-tiled-test.cjs
// Test case specifically for tiled toolpath generation
// Uses real STL files with bounds override to force 4 tiles
// Matches exact browser test parameters: 0.05mm, 1x1 step, bounds(-230,230)x(-230,230)x(-50,72)

const { app, BrowserWindow } = require('electron');
const path = require('path');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        show: false,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            enableBlinkFeatures: 'WebGPU',
        }
    });

    const htmlPath = path.join(__dirname, '../../build/index.html');
    mainWindow.loadFile(htmlPath);

    mainWindow.webContents.on('did-finish-load', async () => {
        console.log('✓ Page loaded');

        const testScript = `
            (async function() {
                const testStartTime = performance.now();
                console.log('\\n=== Tiled Toolpath Test (Real STL Files) ===');
                console.log('Parameters:');
                console.log('  Resolution: 0.05mm');
                console.log('  XY Step: 1x1');
                console.log('  Bounds override: X(-230, 230) Y(-230, 230) Z(-50, 72)');
                console.log('  Expected: 4 tiles\\n');

                if (!navigator.gpu) {
                    return { error: 'WebGPU not available' };
                }

                // Initialize WebGPU worker
                const worker = new Worker('webgpu-worker.js');
                const timings = {};

                const workerReady = new Promise((resolve) => {
                    worker.onmessage = function(e) {
                        if (e.data.type === 'webgpu-ready') {
                            resolve(e.data.data.success);
                        }
                    };
                });

                worker.postMessage({
                    type: 'init',
                    data: {
                        config: {
                            maxGPUMemoryMB: 128,  // Force exactly 4 tiles (2x2)
                            gpuMemorySafetyMargin: 0.8,
                            autoTiling: true
                        }
                    }
                });
                const ready = await workerReady;

                if (!ready) {
                    return { error: 'Failed to initialize WebGPU worker' };
                }
                const initTime = performance.now() - testStartTime;
                timings.init = initTime;
                console.log('✓ Worker initialized in ' + initTime.toFixed(1) + 'ms');

                // Load STL files from benchmark/fixtures
                console.log('\\nLoading STL files...');
                const loadStart = performance.now();

                const terrainResponse = await fetch('../benchmark/fixtures/terrain.stl');
                const terrainBuffer = await terrainResponse.arrayBuffer();

                const toolResponse = await fetch('../benchmark/fixtures/tool.stl');
                const toolBuffer = await toolResponse.arrayBuffer();

                const loadTime = performance.now() - loadStart;
                timings.load = loadTime;
                console.log('✓ Loaded terrain.stl: ' + terrainBuffer.byteLength + ' bytes');
                console.log('✓ Loaded tool.stl: ' + toolBuffer.byteLength + ' bytes');
                console.log('  Load time: ' + loadTime.toFixed(1) + 'ms');

                // Parse STL files (inline parser)
                function parseBinarySTL(buffer) {
                    const dataView = new DataView(buffer);
                    const numTriangles = dataView.getUint32(80, true);
                    const positions = new Float32Array(numTriangles * 9);
                    let offset = 84;

                    for (let i = 0; i < numTriangles; i++) {
                        offset += 12; // Skip normal
                        for (let j = 0; j < 9; j++) {
                            positions[i * 9 + j] = dataView.getFloat32(offset, true);
                            offset += 4;
                        }
                        offset += 2; // Skip attribute
                    }
                    return { positions, triangleCount: numTriangles };
                }

                // Parse terrain and tool
                const parseStart = performance.now();
                const terrainData = parseBinarySTL(terrainBuffer);
                const toolData = parseBinarySTL(toolBuffer);
                const parseTime = performance.now() - parseStart;
                timings.parse = parseTime;

                console.log('✓ Parsed terrain: ' + terrainData.triangleCount + ' triangles in ' + parseTime.toFixed(1) + 'ms');
                console.log('✓ Parsed tool: ' + toolData.triangleCount + ' triangles');

                const stepSize = 0.05;
                const xStep = 1;
                const yStep = 1;
                const zFloor = -100;

                // Bounds override to force 4 tiles: (-230,230) x (-230,230) x (-50,72)
                const boundsOverride = {
                    min: { x: -230, y: -230, z: -50 },
                    max: { x: 230, y: 230, z: 72 }
                };

                console.log('\\n--- TERRAIN RASTERIZATION ---');
                const terrainStart = performance.now();

                const terrainResult = await new Promise((resolve, reject) => {
                    const msgStart = performance.now();
                    worker.onmessage = function(e) {
                        if (e.data.type === 'rasterize-complete' && !e.data.isForTool) {
                            const msgTime = performance.now() - msgStart;
                            console.log('[TIMING] Terrain rasterization total: ' + msgTime.toFixed(1) + 'ms');
                            resolve(e.data.data);
                        } else if (e.data.type === 'error') {
                            reject(new Error(e.data.message));
                        }
                    };

                    const terrainCopy = new Float32Array(terrainData.positions);
                    worker.postMessage({
                        type: 'rasterize',
                        data: {
                            triangles: terrainCopy,
                            stepSize,
                            filterMode: 0,
                            isForTool: false,
                            boundsOverride
                        }
                    }, [terrainCopy.buffer]);
                });

                const terrainTime = performance.now() - terrainStart;
                timings.terrain = terrainTime;
                console.log('✓ Terrain rasterization: ' + terrainResult.pointCount + ' points');
                if (terrainResult.tileCount) {
                    console.log('  Used ' + terrainResult.tileCount + ' tiles for terrain rasterization');
                }

                // Tool rasterization
                console.log('\\n--- TOOL RASTERIZATION ---');
                const toolStart = performance.now();

                const toolResult = await new Promise((resolve, reject) => {
                    const msgStart = performance.now();
                    worker.onmessage = function(e) {
                        if (e.data.type === 'rasterize-complete' && e.data.isForTool) {
                            const msgTime = performance.now() - msgStart;
                            console.log('[TIMING] Tool rasterization total: ' + msgTime.toFixed(1) + 'ms');
                            resolve(e.data.data);
                        } else if (e.data.type === 'error') {
                            reject(new Error(e.data.message));
                        }
                    };

                    const toolCopy = new Float32Array(toolData.positions);
                    worker.postMessage({
                        type: 'rasterize',
                        data: {
                            triangles: toolCopy,
                            stepSize,
                            filterMode: 1,
                            isForTool: true,
                            boundsOverride: null
                        }
                    }, [toolCopy.buffer]);
                });

                const toolTime = performance.now() - toolStart;
                timings.tool = toolTime;
                console.log('✓ Tool rasterization: ' + toolResult.pointCount + ' points');

                // Toolpath generation
                console.log('\\n--- TOOLPATH GENERATION ---');
                const toolpathStart = performance.now();

                const toolpathResult = await new Promise((resolve, reject) => {
                    const msgStart = performance.now();
                    worker.onmessage = function(e) {
                        if (e.data.type === 'toolpath-complete') {
                            const msgTime = performance.now() - msgStart;
                            console.log('[TIMING] Toolpath generation total: ' + msgTime.toFixed(1) + 'ms');
                            resolve(e.data.data);
                        } else if (e.data.type === 'error') {
                            reject(new Error(e.data.message));
                        }
                    };

                    worker.postMessage({
                        type: 'generate-toolpath',
                        data: {
                            terrainPositions: terrainResult.positions,
                            toolPositions: toolResult.positions,
                            xStep,
                            yStep,
                            zFloor,
                            gridStep: stepSize,
                            terrainBounds: terrainResult.bounds
                        }
                    });
                });

                const toolpathTime = performance.now() - toolpathStart;
                timings.toolpath = toolpathTime;
                const totalTime = performance.now() - testStartTime;
                timings.total = totalTime;

                console.log('✓ Toolpath generation: ' + toolpathResult.pathData.length + ' Z-values');
                console.log('  Worker reported: ' + toolpathResult.generationTime.toFixed(1) + 'ms');

                console.log('\\n--- TIMING SUMMARY ---');
                console.log('  Init: ' + timings.init.toFixed(1) + 'ms');
                console.log('  Load STLs: ' + timings.load.toFixed(1) + 'ms');
                console.log('  Parse STLs: ' + timings.parse.toFixed(1) + 'ms');
                console.log('  Terrain rasterization: ' + timings.terrain.toFixed(1) + 'ms');
                console.log('  Tool rasterization: ' + timings.tool.toFixed(1) + 'ms');
                console.log('  Toolpath generation: ' + timings.toolpath.toFixed(1) + 'ms');
                console.log('  TOTAL: ' + timings.total.toFixed(1) + 'ms');

                worker.terminate();

                return {
                    success: true,
                    tilingUsed: terrainResult.tileCount > 1,
                    terrainTiles: terrainResult.tileCount || 1,
                    toolpathSize: toolpathResult.pathData.length,
                    timings
                };
            })();
        `;

        try {
            const result = await mainWindow.webContents.executeJavaScript(testScript);

            if (result.error) {
                console.error('❌ Test failed:', result.error);
                app.exit(1);
                return;
            }

            console.log('\n✅ Tiled toolpath test passed!');
            console.log('  Terrain tiles:', result.terrainTiles);
            console.log('  Toolpath size:', result.toolpathSize);
            app.exit(0);

        } catch (error) {
            console.error('Error running test:', error);
            app.exit(1);
        }
    });

    mainWindow.webContents.on('console-message', (event, level, message) => {
        if (level === 2) {
            console.error(message);
        } else {
            console.log(message);
        }
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    app.quit();
});
