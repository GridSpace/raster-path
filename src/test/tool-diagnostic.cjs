// tool-diagnostic.cjs
// Diagnostic test to inspect tool rasterization for diagonal gaps

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
                console.log('=== Tool Rasterization Diagnostic ===');

                if (!navigator.gpu) {
                    return { error: 'WebGPU not available' };
                }

                // Initialize WebGPU worker
                const worker = new Worker('webgpu-worker.js');

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
                            maxGPUMemoryMB: 512,
                            gpuMemorySafetyMargin: 0.8,
                            autoTiling: true
                        }
                    }
                });
                const ready = await workerReady;

                if (!ready) {
                    return { error: 'Failed to initialize WebGPU worker' };
                }
                console.log('✓ Worker initialized');

                // Load tool STL
                const toolResponse = await fetch('../benchmark/fixtures/tool.stl');
                const toolBuffer = await toolResponse.arrayBuffer();

                // Parse tool STL
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

                const toolData = parseBinarySTL(toolBuffer);
                console.log('✓ Parsed tool:', toolData.triangleCount, 'triangles');

                const stepSize = 0.05;

                // Rasterize tool
                console.log('\\nRasterizing tool...');
                const toolResult = await new Promise((resolve, reject) => {
                    worker.onmessage = function(e) {
                        if (e.data.type === 'rasterize-complete' && e.data.isForTool) {
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

                console.log('✓ Tool rasterized:', toolResult.pointCount, 'points');
                console.log('  Bounds:', toolResult.bounds);

                // Analyze the tool raster data
                const positions = toolResult.positions;
                const gridPoints = new Map();

                let minGridX = Infinity, maxGridX = -Infinity;
                let minGridY = Infinity, maxGridY = -Infinity;

                for (let i = 0; i < positions.length; i += 3) {
                    const gridX = Math.round(positions[i]);
                    const gridY = Math.round(positions[i + 1]);
                    const z = positions[i + 2];

                    const key = gridX + ',' + gridY;
                    gridPoints.set(key, z);

                    minGridX = Math.min(minGridX, gridX);
                    maxGridX = Math.max(maxGridX, gridX);
                    minGridY = Math.min(minGridY, gridY);
                    maxGridY = Math.max(maxGridY, gridY);
                }

                console.log('\\n=== DIAGNOSTIC RESULTS ===');
                console.log('Grid extent: X[' + minGridX + ', ' + maxGridX + '] Y[' + minGridY + ', ' + maxGridY + ']');
                console.log('Grid size: ' + (maxGridX - minGridX + 1) + ' x ' + (maxGridY - minGridY + 1));
                const expectedCells = (maxGridX - minGridX + 1) * (maxGridY - minGridY + 1);
                console.log('Expected cells: ' + expectedCells);
                console.log('Actual points: ' + gridPoints.size);
                console.log('Coverage: ' + (gridPoints.size / expectedCells * 100).toFixed(1) + '%');

                // Check main diagonal
                console.log('\\nChecking main diagonal (top-left to bottom-right):');
                const diagSize = Math.min(maxGridX - minGridX + 1, maxGridY - minGridY + 1);
                let diagMissing = 0;
                const diagMissingList = [];
                const diagMissingRanges = [];
                let inMissingRange = false;
                let rangeStart = -1;

                for (let i = 0; i < diagSize; i++) {
                    const x = minGridX + i;
                    const y = minGridY + i;
                    const key = x + ',' + y;
                    const missing = !gridPoints.has(key);

                    if (missing) {
                        diagMissing++;
                        if (diagMissingList.length < 10) {
                            diagMissingList.push('(' + x + ', ' + y + ')');
                        }
                        if (!inMissingRange) {
                            rangeStart = i;
                            inMissingRange = true;
                        }
                    } else if (inMissingRange) {
                        diagMissingRanges.push('[' + rangeStart + '-' + (i-1) + ']');
                        inMissingRange = false;
                    }
                }
                if (inMissingRange) {
                    diagMissingRanges.push('[' + rangeStart + '-' + (diagSize-1) + ']');
                }

                console.log('Main diagonal: ' + (diagSize - diagMissing) + '/' + diagSize + ' present (' + diagMissing + ' missing)');
                if (diagMissingList.length > 0) {
                    console.log('First missing points: ' + diagMissingList.join(', '));
                }
                if (diagMissingRanges.length > 0) {
                    console.log('Missing ranges: ' + diagMissingRanges.join(', '));
                }

                // Check anti-diagonal
                console.log('\\nChecking anti-diagonal (top-right to bottom-left):');
                let antiDiagMissing = 0;
                const antiDiagMissingList = [];
                const antiDiagMissingRanges = [];
                inMissingRange = false;
                rangeStart = -1;

                for (let i = 0; i < diagSize; i++) {
                    const x = maxGridX - i;
                    const y = minGridY + i;
                    const key = x + ',' + y;
                    const missing = !gridPoints.has(key);

                    if (missing) {
                        antiDiagMissing++;
                        if (antiDiagMissingList.length < 10) {
                            antiDiagMissingList.push('(' + x + ', ' + y + ')');
                        }
                        if (!inMissingRange) {
                            rangeStart = i;
                            inMissingRange = true;
                        }
                    } else if (inMissingRange) {
                        antiDiagMissingRanges.push('[' + rangeStart + '-' + (i-1) + ']');
                        inMissingRange = false;
                    }
                }
                if (inMissingRange) {
                    antiDiagMissingRanges.push('[' + rangeStart + '-' + (diagSize-1) + ']');
                }

                console.log('Anti-diagonal: ' + (diagSize - antiDiagMissing) + '/' + diagSize + ' present (' + antiDiagMissing + ' missing)');
                if (antiDiagMissingList.length > 0) {
                    console.log('First missing points: ' + antiDiagMissingList.join(', '));
                }
                if (antiDiagMissingRanges.length > 0) {
                    console.log('Missing ranges: ' + antiDiagMissingRanges.join(', '));
                }

                worker.terminate();

                return {
                    success: true,
                    totalPoints: gridPoints.size,
                    expectedCells: expectedCells,
                    diagMissing: diagMissing,
                    antiDiagMissing: antiDiagMissing
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

            console.log('\n=== SUMMARY ===');
            console.log('Total points:', result.totalPoints);
            console.log('Expected cells:', result.expectedCells);
            console.log('Main diagonal missing:', result.diagMissing);
            console.log('Anti-diagonal missing:', result.antiDiagMissing);

            if (result.diagMissing > 0 || result.antiDiagMissing > 0) {
                console.log('\n⚠️  DIAGONAL GAPS DETECTED!');
                app.exit(1);
            } else {
                console.log('\n✅ No diagonal gaps detected');
                app.exit(0);
            }

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
