// planar-tiling-test.cjs
// Test for planar tiling with very high resolution (0.01mm)
// This should trigger automatic tiling to avoid GPU memory allocation failures

const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');

const OUTPUT_DIR = path.join(__dirname, '../../test-output');
const OUTPUT_FILE = path.join(OUTPUT_DIR, 'planar-tiling-test.json');

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
                console.log('=== Planar Tiling Test (0.01mm resolution) ===');

                if (!navigator.gpu) {
                    return { error: 'WebGPU not available' };
                }
                console.log('✓ WebGPU available');

                const { RasterPath } = await import('./raster-path.js');

                console.log('\\nLoading STL files...');
                const terrainResponse = await fetch('../benchmark/fixtures/terrain.stl');
                const terrainBuffer = await terrainResponse.arrayBuffer();

                const toolResponse = await fetch('../benchmark/fixtures/tool.stl');
                const toolBuffer = await toolResponse.arrayBuffer();

                console.log('✓ Loaded terrain.stl:', terrainBuffer.byteLength, 'bytes');
                console.log('✓ Loaded tool.stl:', toolBuffer.byteLength, 'bytes');

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
                console.log('✓ Parsed terrain:', terrainTriangles.length / 9, 'triangles');
                console.log('✓ Parsed tool:', toolTriangles.length / 9, 'triangles');

                // HIGH RESOLUTION - should trigger tiling
                const resolution = 0.01;
                const xStep = 1;
                const yStep = 1;
                const zFloor = -100;

                console.log('\\nTest parameters:');
                console.log('  Resolution:', resolution, 'mm (VERY HIGH - should trigger tiling)');
                console.log('  XY step:', xStep + 'x' + yStep, 'points');
                console.log('  Z floor:', zFloor, 'mm');

                // Calculate expected grid size
                // Terrain is roughly 1000x1000mm, so at 0.01mm = 100,000 x 100,000 grid
                // = 10 billion points * 4 bytes = 40GB (way over GPU limit!)
                const terrainSize = 1000; // approximate
                const expectedGridSize = Math.ceil(terrainSize / resolution);
                const expectedPoints = expectedGridSize * expectedGridSize;
                const expectedMemoryMB = (expectedPoints * 4) / (1024 * 1024);
                console.log('\\nExpected memory usage:');
                console.log('  Grid size:', expectedGridSize + 'x' + expectedGridSize);
                console.log('  Total points:', (expectedPoints / 1e6).toFixed(1) + 'M');
                console.log('  Memory needed:', expectedMemoryMB.toFixed(0) + 'MB');
                console.log('  GPU limit: ~512MB (should trigger tiling)');

                console.log('\\nInitializing RasterPath (planar mode)...');
                const raster = new RasterPath({
                    mode: 'planar',
                    resolution: resolution
                });
                await raster.init();
                console.log('✓ RasterPath initialized');

                console.log('\\n1. Rasterizing model (should use tiling)...');
                const t0 = performance.now();
                let terrainData;
                let terrainTime;
                try {
                    terrainData = await raster.rasterizeModel({
                        triangles: terrainTriangles,
                        zFloor: zFloor
                    });
                    terrainTime = performance.now() - t0;
                    console.log('✓ Model:', terrainData.pointCount, 'points in', terrainTime.toFixed(1), 'ms');
                } catch (error) {
                    console.error('❌ Model rasterization FAILED:', error.message);
                    return {
                        error: 'Rasterization failed: ' + error.message,
                        expectedTiling: true,
                        resolution: resolution
                    };
                }

                console.log('\\n2. Rasterizing tool...');
                const t1 = performance.now();
                const toolData = await raster.rasterizeTool({
                    triangles: toolTriangles
                });
                const toolTime = performance.now() - t1;
                console.log('✓ Tool:', toolData.pointCount, 'points in', toolTime.toFixed(1), 'ms');

                console.log('\\n3. Generating toolpaths...');
                const t2 = performance.now();
                const toolpathData = await raster.generateToolpaths({
                    terrainData: terrainData,
                    toolData: toolData,
                    xStep: xStep,
                    yStep: yStep,
                    zFloor: zFloor
                });
                const toolpathTime = performance.now() - t2;
                console.log('✓ Toolpath:', toolpathData.numScanlines + 'x' + toolpathData.pointsPerLine, '=', toolpathData.pathData.length, 'Z-values');
                console.log('  Generation time:', toolpathTime.toFixed(1), 'ms');

                raster.terminate();

                // Calculate checksum
                let checksum = 0;
                for (let i = 0; i < Math.min(1000, toolpathData.pathData.length); i++) {
                    checksum = (checksum + toolpathData.pathData[i] * (i + 1)) | 0;
                }

                return {
                    success: true,
                    tilingWorked: true,
                    output: {
                        parameters: {
                            resolution: resolution,
                            expectedMemoryMB: Math.round(expectedMemoryMB)
                        },
                        result: {
                            terrainPoints: terrainData.pointCount,
                            toolPoints: toolData.pointCount,
                            toolpathSize: toolpathData.pathData.length,
                            checksum: checksum
                        },
                        timing: {
                            terrain: terrainTime,
                            tool: toolTime,
                            toolpath: toolpathTime,
                            total: terrainTime + toolTime + toolpathTime
                        }
                    }
                };
            })();
        `;

        try {
            const result = await mainWindow.webContents.executeJavaScript(testScript);

            if (result.error) {
                console.error('\n❌ TEST FAILED - Tiling did not work!');
                console.error('Error:', result.error);
                console.error('Resolution:', result.resolution);
                console.error('Expected tiling:', result.expectedTiling);

                fs.writeFileSync(OUTPUT_FILE, JSON.stringify({
                    failed: true,
                    error: result.error,
                    resolution: result.resolution
                }, null, 2));

                app.exit(1);
                return;
            }

            console.log('\n=== Test Complete ===');
            console.log('✅ Tiling worked correctly!');
            console.log('Terrain points:', result.output.result.terrainPoints);
            console.log('Tool points:', result.output.result.toolPoints);
            console.log('Toolpath size:', result.output.result.toolpathSize);
            console.log('Total time:', result.output.timing.total.toFixed(1), 'ms');

            fs.writeFileSync(OUTPUT_FILE, JSON.stringify(result.output, null, 2));
            console.log('\n✓ Results written to:', OUTPUT_FILE);

            app.exit(0);
        } catch (error) {
            console.error('❌ Test execution error:', error);
            app.exit(1);
        }
    });

    mainWindow.webContents.on('console-message', (event, level, message) => {
        if (message.includes('[Raster') || message.includes('[Worker') || message.includes('Tiling')) {
            console.log('[Browser]', message);
        }
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    app.quit();
});
