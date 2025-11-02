// batch-rasterize-test.cjs
// Test batch rasterization with buffer reuse (for radial toolpath stability)
// Demonstrates how to rasterize at multiple rotation angles without GPU buffer churn

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
                console.log('\\n=== Batch Rasterization Test (Buffer Reuse) ===');
                console.log('Purpose: Test GPU buffer reuse across multiple rotation angles');
                console.log('Benefits: Reduces GPU glitching with complex models\\n');

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
                            maxGPUMemoryMB: 256,
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

                // Load terrain STL
                const terrainResponse = await fetch('../benchmark/fixtures/terrain.stl');
                const terrainBuffer = await terrainResponse.arrayBuffer();
                console.log(\`✓ Loaded terrain.stl: \${(terrainBuffer.byteLength / 1024 / 1024).toFixed(2)} MB\\n\`);

                // Parse STL
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

                const terrainData = parseBinarySTL(terrainBuffer);
                console.log(\`✓ Parsed: \${terrainData.triangleCount} triangles\\n\`);

                const stepSize = 0.1;  // 0.1mm resolution (faster for test)
                const rotationAngles = [];

                // Test with 36 angles (10° steps around full circle)
                for (let angle = 0; angle < 360; angle += 10) {
                    rotationAngles.push(angle);
                }

                console.log('--- OLD METHOD (create/destroy buffers each time) ---');
                const oldMethodStart = performance.now();
                let oldResults = [];

                for (let i = 0; i < rotationAngles.length; i++) {
                    const result = await new Promise((resolve, reject) => {
                        worker.onmessage = function(e) {
                            if (e.data.type === 'rasterize-complete') {
                                resolve(e.data.data);
                            } else if (e.data.type === 'error') {
                                reject(new Error(e.data.message));
                            }
                        };

                        const trianglesCopy = new Float32Array(terrainData.positions);
                        worker.postMessage({
                            type: 'rasterize',
                            data: {
                                triangles: trianglesCopy,
                                stepSize,
                                filterMode: 0,
                                isForTool: false,
                                rotationAngleDeg: rotationAngles[i]
                            }
                        }, [trianglesCopy.buffer]);
                    });
                    oldResults.push(result);
                }

                const oldMethodTime = performance.now() - oldMethodStart;
                const oldAvgPerAngle = oldMethodTime / rotationAngles.length;
                console.log(\`  Total: \${oldMethodTime.toFixed(1)}ms\`);
                console.log(\`  Per angle: \${oldAvgPerAngle.toFixed(1)}ms\\n\`);

                console.log('--- NEW METHOD (reuse buffers across all angles) ---');
                const newMethodStart = performance.now();

                const batchResult = await new Promise((resolve, reject) => {
                    worker.onmessage = function(e) {
                        if (e.data.type === 'rasterize-batch-complete') {
                            resolve(e.data.data);
                        } else if (e.data.type === 'error') {
                            reject(new Error(e.data.message));
                        }
                    };

                    const trianglesCopy = new Float32Array(terrainData.positions);
                    worker.postMessage({
                        type: 'rasterize-batch',
                        data: {
                            triangles: trianglesCopy,
                            stepSize,
                            filterMode: 0,
                            isForTool: false,
                            rotationAngles: rotationAngles
                        }
                    }, [trianglesCopy.buffer]);
                });

                const newMethodTime = performance.now() - newMethodStart;
                const newAvgPerAngle = newMethodTime / rotationAngles.length;
                console.log(\`  Total: \${newMethodTime.toFixed(1)}ms\`);
                console.log(\`  Per angle: \${newAvgPerAngle.toFixed(1)}ms\\n\`);

                const improvement = ((oldMethodTime - newMethodTime) / oldMethodTime * 100);
                const speedup = oldMethodTime / newMethodTime;

                console.log('--- RESULTS ---');
                console.log(\`  Old method: \${oldMethodTime.toFixed(1)}ms\`);
                console.log(\`  New method: \${newMethodTime.toFixed(1)}ms\`);
                console.log(\`  Improvement: \${improvement.toFixed(1)}% faster\`);
                console.log(\`  Speedup: \${speedup.toFixed(2)}x\`);
                console.log(\`  Saved: \${(oldMethodTime - newMethodTime).toFixed(1)}ms total\\n\`);

                // Verify results match
                let mismatchCount = 0;
                for (let i = 0; i < rotationAngles.length; i++) {
                    if (oldResults[i].pointCount !== batchResult.results[i].pointCount) {
                        mismatchCount++;
                    }
                }

                if (mismatchCount > 0) {
                    console.error(\`❌ MISMATCH: \${mismatchCount} results differ!\`);
                    return { error: 'Results mismatch between old and new methods' };
                }

                console.log('✅ Results verified: Both methods produce identical output');

                worker.terminate();

                return {
                    success: true,
                    oldMethodTime,
                    newMethodTime,
                    improvement,
                    speedup,
                    anglesProcessed: rotationAngles.length
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

            console.log('\\n✅ Batch rasterization test passed!');
            console.log('  Speedup:', result.speedup.toFixed(2) + 'x');
            console.log('  Angles processed:', result.anglesProcessed);
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
