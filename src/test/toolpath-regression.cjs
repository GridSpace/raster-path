// toolpath-regression.cjs
// Comprehensive regression test for full toolpath generation
// Tests terrain rasterization + tool rasterization + toolpath generation
// Stores complete Z-height output for comparison

const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');

const OUTPUT_DIR = path.join(__dirname, '../../test-output');
const BASELINE_FILE = path.join(OUTPUT_DIR, 'toolpath-baseline.json');
const CURRENT_FILE = path.join(OUTPUT_DIR, 'toolpath-current.json');

if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

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
                console.log('=== Full Toolpath Regression Test ===');

                if (!navigator.gpu) {
                    return { error: 'WebGPU not available' };
                }
                console.log('✓ WebGPU available');

                // Initialize WebGPU worker
                const worker = new Worker('webgpu-worker.js');

                const workerReady = new Promise((resolve) => {
                    worker.onmessage = function(e) {
                        if (e.data.type === 'webgpu-ready') {
                            resolve(e.data.data.success);
                        }
                    };
                });

                // Use normal GPU limit - don't force tiling for baseline test
                worker.postMessage({
                    type: 'init',
                    data: {
                        config: {
                            maxGPUMemoryMB: 256,  // Normal limit - no tiling needed
                            gpuMemorySafetyMargin: 0.8,
                            tileOverlapMM: 10,
                            autoTiling: true,
                            minTileSize: 50
                        }
                    }
                });
                const ready = await workerReady;

                if (!ready) {
                    return { error: 'Failed to initialize WebGPU worker' };
                }
                console.log('✓ Worker initialized');

                // Load STL files from benchmark/fixtures
                console.log('\\nLoading STL files...');
                const terrainResponse = await fetch('../benchmark/fixtures/terrain.stl');
                const terrainBuffer = await terrainResponse.arrayBuffer();

                const toolResponse = await fetch('../benchmark/fixtures/tool.stl');
                const toolBuffer = await toolResponse.arrayBuffer();

                console.log('✓ Loaded terrain.stl:', terrainBuffer.byteLength, 'bytes');
                console.log('✓ Loaded tool.stl:', toolBuffer.byteLength, 'bytes');

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
                        offset += 2; // Skip attribute byte count
                    }
                    return positions;
                }

                const terrainData = parseBinarySTL(terrainBuffer);
                const toolData = parseBinarySTL(toolBuffer);
                console.log('✓ Parsed terrain:', terrainData.length / 9, 'triangles');
                console.log('✓ Parsed tool:', toolData.length / 9, 'triangles');

                // Test parameters - FINE detail for comprehensive test
                const stepSize = 0.05; // 0.05mm high resolution
                const xStep = 1;
                const yStep = 1;
                const zFloor = -100;

                console.log('\\nTest parameters:');
                console.log('  Step size:', stepSize, 'mm');
                console.log('  XY step:', xStep + 'x' + yStep, 'points');
                console.log('  Z floor:', zFloor, 'mm');

                // Rasterize terrain
                console.log('\\n1. Rasterizing terrain...');
                const terrainResult = await new Promise((resolve, reject) => {
                    worker.onmessage = function(e) {
                        if (e.data.type === 'rasterize-complete' && !e.data.isForTool) {
                            resolve(e.data.data);
                        } else if (e.data.type === 'error') {
                            reject(new Error(e.data.message));
                        }
                    };

                    const terrainCopy = new Float32Array(terrainData);
                    worker.postMessage({
                        type: 'rasterize',
                        data: {
                            triangles: terrainCopy,
                            stepSize,
                            filterMode: 0,
                            isForTool: false,
                            boundsOverride: null
                        }
                    }, [terrainCopy.buffer]);
                });
                console.log('✓ Terrain:', terrainResult.pointCount, 'points in', terrainResult.conversionTime.toFixed(1), 'ms');

                // Rasterize tool
                console.log('\\n2. Rasterizing tool...');
                const toolResult = await new Promise((resolve, reject) => {
                    worker.onmessage = function(e) {
                        if (e.data.type === 'rasterize-complete' && e.data.isForTool) {
                            resolve(e.data.data);
                        } else if (e.data.type === 'error') {
                            reject(new Error(e.data.message));
                        }
                    };

                    const toolCopy = new Float32Array(toolData);
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
                console.log('✓ Tool:', toolResult.pointCount, 'points in', toolResult.conversionTime.toFixed(1), 'ms');

                // Generate toolpath
                console.log('\\n3. Generating toolpath...');
                const toolpathResult = await new Promise((resolve, reject) => {
                    worker.onmessage = function(e) {
                        if (e.data.type === 'toolpath-complete') {
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
                            zFloor: zFloor,
                            gridStep: stepSize,
                            terrainBounds: terrainResult.bounds
                        }
                    });
                });
                console.log('✓ Toolpath:', toolpathResult.numScanlines + 'x' + toolpathResult.pointsPerLine, '=', toolpathResult.pathData.length, 'Z-values');
                console.log('  Generation time:', toolpathResult.generationTime.toFixed(1), 'ms');

                worker.terminate();

                // Calculate checksum for regression detection
                let checksum = 0;
                for (let i = 0; i < toolpathResult.pathData.length; i++) {
                    checksum = (checksum + toolpathResult.pathData[i] * (i + 1)) | 0;
                }

                // Sample first 30 Z-values for debugging
                const sampleSize = Math.min(30, toolpathResult.pathData.length);
                const sampleValues = [];
                for (let i = 0; i < sampleSize; i++) {
                    sampleValues.push(toolpathResult.pathData[i].toFixed(2));
                }

                return {
                    success: true,
                    output: {
                        parameters: {
                            stepSize,
                            xStep,
                            yStep,
                            zFloor,
                            terrainTriangles: terrainData.length / 9,
                            toolTriangles: toolData.length / 9
                        },
                        result: {
                            terrainPoints: terrainResult.pointCount,
                            toolPoints: toolResult.pointCount,
                            toolpathSize: toolpathResult.pathData.length,
                            numScanlines: toolpathResult.numScanlines,
                            pointsPerLine: toolpathResult.pointsPerLine,
                            checksum: checksum,
                            sampleValues: sampleValues
                        }
                    }
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

            // Save current output as JSON
            const currentData = {
                parameters: result.output.parameters,
                result: result.output.result
            };

            fs.writeFileSync(CURRENT_FILE, JSON.stringify(currentData, null, 2));
            console.log('\n✓ Saved current output to', CURRENT_FILE);
            console.log(`  Toolpath size: ${result.output.result.toolpathSize} Z-values`);
            console.log(`  Checksum: ${result.output.result.checksum}`);
            console.log(`  Sample values (first 10): ${result.output.result.sampleValues.slice(0, 10).join(', ')}`);

            // Check if baseline exists
            if (!fs.existsSync(BASELINE_FILE)) {
                console.log('\n📝 No baseline found - saving current as baseline');
                fs.writeFileSync(BASELINE_FILE, JSON.stringify(currentData, null, 2));
                console.log('✅ Baseline created');
                app.exit(0);
                return;
            }

            // Compare with baseline
            const baseline = JSON.parse(fs.readFileSync(BASELINE_FILE, 'utf8'));

            console.log('\n=== Comparison ===');
            console.log('Baseline toolpath size:', baseline.result.toolpathSize);
            console.log('Current toolpath size:', result.output.result.toolpathSize);
            console.log('Baseline checksum:', baseline.result.checksum);
            console.log('Current checksum:', result.output.result.checksum);

            let passed = true;

            // Compare basic parameters
            if (baseline.result.toolpathSize !== result.output.result.toolpathSize) {
                console.error('❌ Toolpath size mismatch!');
                console.error(`  Expected: ${baseline.result.toolpathSize}, Got: ${result.output.result.toolpathSize}`);
                passed = false;
            }

            // Compare checksums
            if (baseline.result.checksum !== result.output.result.checksum) {
                console.error('❌ Checksum mismatch!');
                console.error(`  Expected: ${baseline.result.checksum}, Got: ${result.output.result.checksum}`);
                console.error('\nSample comparison (first 10 values):');
                console.error('  Baseline:', baseline.result.sampleValues.slice(0, 10).join(', '));
                console.error('  Current: ', result.output.result.sampleValues.slice(0, 10).join(', '));
                passed = false;
            } else {
                console.log('✓ Checksum matches');
            }

            // Compare dimensions
            if (baseline.result.numScanlines !== result.output.result.numScanlines ||
                baseline.result.pointsPerLine !== result.output.result.pointsPerLine) {
                console.error('❌ Dimension mismatch!');
                console.error(`  Scanlines: ${baseline.result.numScanlines} → ${result.output.result.numScanlines}`);
                console.error(`  Points per line: ${baseline.result.pointsPerLine} → ${result.output.result.pointsPerLine}`);
                passed = false;
            } else {
                console.log('✓ Dimensions match');
            }

            if (passed) {
                console.log('\n✅ All checks passed - output matches baseline');
                app.exit(0);
            } else {
                console.log('\n❌ Regression detected - output differs from baseline');
                console.log('To compare files: diff', BASELINE_FILE, CURRENT_FILE);
                console.log('To update baseline: cp', CURRENT_FILE, BASELINE_FILE);
                app.exit(1);
            }

        } catch (error) {
            console.error('Error running test:', error);
            app.exit(1);
        }
    });

    mainWindow.webContents.on('console-message', (event, level, message) => {
        // Filter out verbose worker initialization messages to reduce output
        if (message.includes('Adapter limits') ||
            message.includes('Initialized (pipelines cached)') ||
            message.includes('Worker') && message.includes('initialized')) {
            return;
        }
        try {
            if (level === 2) {
                console.error(message);
            } else {
                console.log(message);
            }
        } catch (err) {
            // Ignore EPIPE errors from closed stdout
            if (err.code !== 'EPIPE') throw err;
        }
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    app.quit();
});
