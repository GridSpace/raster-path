// radial-batch-comparison.cjs
// Compare radial toolpath generation with and without buffer reuse

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
                console.log('\\n=== Radial Toolpath: Batch Mode Comparison ===\\n');

                if (!navigator.gpu) {
                    return { error: 'WebGPU not available' };
                }

                const { RasterPath } = await import('./raster-path.js');
                const rp = new RasterPath();
                await rp.init({ maxWorkers: 4 });

                // Load STL files
                const terrainResponse = await fetch('../benchmark/fixtures/terrain.stl');
                const terrainBuffer = await terrainResponse.arrayBuffer();
                const toolResponse = await fetch('../benchmark/fixtures/tool.stl');
                const toolBuffer = await toolResponse.arrayBuffer();

                function parseBinarySTL(buffer) {
                    const dataView = new DataView(buffer);
                    const numTriangles = dataView.getUint32(80, true);
                    const positions = new Float32Array(numTriangles * 9);
                    let offset = 84;
                    for (let i = 0; i < numTriangles; i++) {
                        offset += 12;
                        for (let j = 0; j < 9; j++) {
                            positions[i * 9 + j] = dataView.getFloat32(offset, true);
                            offset += 4;
                        }
                        offset += 2;
                    }
                    return positions;
                }

                const terrainTriangles = parseBinarySTL(terrainBuffer);
                const toolTriangles = parseBinarySTL(toolBuffer);

                console.log('✓ Loaded and parsed STL files\\n');

                const stepSize = 0.1;
                const xRotationStep = 5;  // 72 rotations
                const xStep = 1;
                const zFloor = -100;

                // Rasterize terrain and tool
                const terrainBounds = {
                    min: { x: -117.5, y: -117.5, z: -50 },
                    max: { x: 117.5, y: 117.5, z: 72 }
                };

                const toolRaster = await rp.rasterizeMesh(toolTriangles, stepSize, 1);
                console.log('✓ Tool rasterized\\n');

                // Test 1: With buffer reuse (default)
                console.log('--- TEST 1: Buffer Reuse ENABLED (batch: true) ---');
                const test1Start = performance.now();
                const result1 = await rp.generateRadialToolpath(
                    terrainTriangles,
                    toolRaster.positions,
                    xRotationStep,
                    xStep,
                    zFloor,
                    stepSize,
                    terrainBounds,
                    { batch: true }
                );
                const test1Time = performance.now() - test1Start;
                console.log(\`  Time: \${test1Time.toFixed(1)}ms\`);
                console.log(\`  Worker reported: \${result1.generationTime.toFixed(1)}ms\`);
                console.log(\`  Output: \${result1.numRotations} × \${result1.pointsPerLine} points\\n\`);

                // Test 2: Without buffer reuse
                console.log('--- TEST 2: Buffer Reuse DISABLED (batch: false) ---');
                const test2Start = performance.now();
                const result2 = await rp.generateRadialToolpath(
                    terrainTriangles,
                    toolRaster.positions,
                    xRotationStep,
                    xStep,
                    zFloor,
                    stepSize,
                    terrainBounds,
                    { batch: false }
                );
                const test2Time = performance.now() - test2Start;
                console.log(\`  Time: \${test2Time.toFixed(1)}ms\`);
                console.log(\`  Worker reported: \${result2.generationTime.toFixed(1)}ms\`);
                console.log(\`  Output: \${result2.numRotations} × \${result2.pointsPerLine} points\\n\`);

                // Comparison
                const improvement = ((test2Time - test1Time) / test2Time * 100);
                const speedup = test2Time / test1Time;

                console.log('--- RESULTS ---');
                console.log(\`  With buffer reuse:    \${test1Time.toFixed(1)}ms\`);
                console.log(\`  Without buffer reuse: \${test2Time.toFixed(1)}ms\`);
                console.log(\`  Improvement: \${improvement.toFixed(1)}% faster\`);
                console.log(\`  Speedup: \${speedup.toFixed(2)}x\\n\`);

                // Verify outputs match
                let mismatchCount = 0;
                for (let i = 0; i < result1.pathData.length; i++) {
                    if (Math.abs(result1.pathData[i] - result2.pathData[i]) > 0.001) {
                        mismatchCount++;
                    }
                }

                if (mismatchCount > 0) {
                    console.error(\`❌ MISMATCH: \${mismatchCount} values differ!\`);
                    return { error: 'Results differ between modes' };
                }

                console.log('✅ Results verified: Both modes produce identical output');

                return {
                    success: true,
                    withBufferReuse: test1Time,
                    withoutBufferReuse: test2Time,
                    improvement,
                    speedup
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

            console.log('\\n✅ Batch mode comparison test passed!');
            console.log('  Speedup:', result.speedup.toFixed(2) + 'x');
            console.log('  Improvement:', result.improvement.toFixed(1) + '%');
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
