// lathe-cylinder-2-test.cjs
// Test for lathe-cylinder-2.stl (201 buckets) with bucket batching

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
                console.log('=== Lathe Cylinder 2 Test (201 buckets) ===');

                if (!navigator.gpu) {
                    return { error: 'WebGPU not available' };
                }
                console.log('✓ WebGPU available');

                // Import RasterPath
                const { RasterPath } = await import('./raster-path.js');

                // Load STL files
                console.log('\\nLoading lathe-cylinder-2.stl...');
                const terrainResponse = await fetch('../benchmark/fixtures/lathe-cylinder-2.stl');
                const terrainBuffer = await terrainResponse.arrayBuffer();

                const toolResponse = await fetch('../benchmark/fixtures/tool.stl');
                const toolBuffer = await toolResponse.arrayBuffer();

                console.log('✓ Loaded lathe-cylinder-2.stl:', terrainBuffer.byteLength, 'bytes');
                console.log('✓ Loaded tool.stl:', toolBuffer.byteLength, 'bytes');

                // Parse STL files
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

                const terrainTriangles = parseBinarySTL(terrainBuffer);
                const toolTriangles = parseBinarySTL(toolBuffer);
                console.log('✓ Parsed terrain:', terrainTriangles.length / 9, 'triangles');
                console.log('✓ Parsed tool:', toolTriangles.length / 9, 'triangles');

                // Test parameters
                const resolution = 0.5;
                const rotationStep = 5.0;
                const xStep = 1;
                const yStep = 1;
                const zFloor = -50;

                console.log('\\nTest parameters:');
                console.log('  Resolution:', resolution, 'mm');
                console.log('  Rotation step:', rotationStep, '°');
                console.log('  XY step:', xStep + 'x' + yStep, 'points');
                console.log('  Z floor:', zFloor, 'mm');

                // Create RasterPath instance for radial mode
                console.log('\\nInitializing RasterPath (radial mode with diagnostic=true)...');
                const raster = new RasterPath({
                    mode: 'radial',
                    resolution: resolution,
                    rotationStep: rotationStep,
                    diagnostic: true
                });
                await raster.init();
                console.log('✓ RasterPath initialized');

                // Load tool
                console.log('\\nLoading tool...');
                const t0 = performance.now();
                const toolData = await raster.loadTool({ triangles: toolTriangles });
                const toolTime = performance.now() - t0;
                console.log('✓ Tool:', toolData.pointCount, 'points in', toolTime.toFixed(1), 'ms');

                // Load terrain
                console.log('\\nLoading terrain...');
                const t1 = performance.now();
                await raster.loadTerrain({ triangles: terrainTriangles });
                const terrainTime = performance.now() - t1;
                console.log('✓ Terrain loaded in', terrainTime.toFixed(1), 'ms');

                // Generate toolpaths (this should trigger bucket batching)
                console.log('\\n*** Generating toolpaths (watch for bucket batching messages) ***');
                const t2 = performance.now();
                const toolpaths = await raster.generateToolpaths({
                    xStep,
                    yStep,
                    zFloor
                });
                const toolpathTime = performance.now() - t2;

                console.log('\\n✓ Generated', toolpaths.length, 'toolpaths in', (toolpathTime / 1000).toFixed(2), 's');
                if (toolpaths.length > 0) {
                    console.log('  First toolpath:', toolpaths[0].numScanlines, 'scanlines ×', toolpaths[0].pointsPerLine, 'points');
                }

                console.log('\\n✅ TEST PASSED - No GPU timeout!');

                return { success: true };
            })();
        `;

        try {
            const result = await mainWindow.webContents.executeJavaScript(testScript);

            if (result && result.error) {
                console.error('❌ Test failed:', result.error);
                app.exit(1);
            } else {
                console.log('\n✅ Test completed successfully');
                app.exit(0);
            }
        } catch (error) {
            console.error('❌ Test execution failed:', error);
            app.exit(1);
        }
    });

    mainWindow.webContents.on('console-message', (event, level, message) => {
        console.log(message);
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    app.quit();
});
