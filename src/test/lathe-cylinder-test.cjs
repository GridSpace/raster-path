const { app, BrowserWindow } = require('electron');
const fs = require('fs');
const path = require('path');

let testPassed = false;

app.on('ready', async () => {
    const win = new BrowserWindow({
        width: 800,
        height: 600,
        show: false,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true
        }
    });

    // Load test page
    await win.loadFile(path.join(__dirname, '../../build/index.html'));

    // Inject test script
    await win.webContents.executeJavaScript(`
        (async () => {
            console.log('=== Lathe Cylinder Radial Test (0.025mm resolution) ===');

            // Load STL
            const stlPath = '${path.join(__dirname, '../../benchmark/fixtures/lathe-cylinder.stl')}';
            const stlBuffer = await (await fetch('file://' + stlPath)).arrayBuffer();

            console.log('STL Size:', (stlBuffer.byteLength / 1024 / 1024).toFixed(2), 'MB');

            // Parse STL
            function parseSTL(arrayBuffer) {
                const view = new DataView(arrayBuffer);
                const triangleCount = view.getUint32(80, true);
                console.log('Triangle count:', triangleCount);

                const triangles = new Float32Array(triangleCount * 9);
                let offset = 84;

                for (let i = 0; i < triangleCount; i++) {
                    offset += 12; // Skip normal
                    for (let j = 0; j < 9; j++) {
                        triangles[i * 9 + j] = view.getFloat32(offset, true);
                        offset += 4;
                    }
                    offset += 2; // Skip attribute byte count
                }

                return triangles;
            }

            const triangles = parseSTL(stlBuffer);

            // Calculate bounds
            let minX = Infinity, maxX = -Infinity;
            let minY = Infinity, maxY = -Infinity;
            let minZ = Infinity, maxZ = -Infinity;

            for (let i = 0; i < triangles.length; i += 3) {
                const x = triangles[i];
                const y = triangles[i + 1];
                const z = triangles[i + 2];
                minX = Math.min(minX, x); maxX = Math.max(maxX, x);
                minY = Math.min(minY, y); maxY = Math.max(maxY, y);
                minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
            }

            console.log('Bounds:');
            console.log('  X:', minX.toFixed(2), 'to', maxX.toFixed(2), '(size:', (maxX - minX).toFixed(2), 'mm)');
            console.log('  Y:', minY.toFixed(2), 'to', maxY.toFixed(2), '(size:', (maxY - minY).toFixed(2), 'mm)');
            console.log('  Z:', minZ.toFixed(2), 'to', maxZ.toFixed(2), '(size:', (maxZ - minZ).toFixed(2), 'mm)');

            const resolution = 0.025;
            const rotationStep = 1.0;

            console.log('\\nTest parameters:');
            console.log('  Resolution:', resolution, 'mm');
            console.log('  Rotation step:', rotationStep, 'degrees');

            // Calculate expected dimensions
            const xSize = maxX - minX;
            const ySize = maxY - minY;
            const zSize = maxZ - minZ;
            const maxRadius = Math.sqrt(ySize * ySize + zSize * zSize) / 2;

            const gridXSize = Math.ceil(xSize / resolution);
            const gridYHeight = Math.ceil(ySize / resolution);
            const numAngles = Math.ceil(360 / rotationStep);

            console.log('\\nExpected grid dimensions:');
            console.log('  X grid size:', gridXSize, 'cells');
            console.log('  Y grid height:', gridYHeight, 'cells');
            console.log('  Number of angles:', numAngles);
            console.log('  Max radius:', maxRadius.toFixed(2), 'mm');

            // Calculate memory requirements
            const bytesPerCell = 4; // f32
            const totalCells = gridXSize * gridYHeight * numAngles;
            const totalMemoryMB = (totalCells * bytesPerCell / 1024 / 1024).toFixed(2);

            console.log('\\nMemory requirements:');
            console.log('  Total cells:', totalCells.toLocaleString());
            console.log('  Total memory:', totalMemoryMB, 'MB');

            // Test with RasterPath
            console.log('\\n=== Starting RasterPath test ===');

            const { RasterPath } = await import('./raster-path.js');
            const raster = new RasterPath({
                mode: 'radial',
                resolution: resolution,
                rotationStep: rotationStep
            });

            await raster.init();
            console.log('RasterPath initialized');

            try {
                // Load terrain
                console.log('Loading terrain...');
                const startLoad = Date.now();
                await raster.loadTerrain({ triangles, zFloor: 0 });
                const loadTime = Date.now() - startLoad;
                console.log('Terrain loaded in', loadTime, 'ms');

                // Generate toolpaths with a dummy tool
                console.log('Generating toolpaths...');
                const startToolpath = Date.now();

                // Create a simple cylindrical tool (10mm diameter, 10mm height)
                const toolRadius = 5;
                const toolHeight = 10;
                const toolSegments = 16;
                const toolTriangles = new Float32Array(toolSegments * 2 * 9);

                for (let i = 0; i < toolSegments; i++) {
                    const angle1 = (i / toolSegments) * Math.PI * 2;
                    const angle2 = ((i + 1) / toolSegments) * Math.PI * 2;

                    const x1 = Math.cos(angle1) * toolRadius;
                    const y1 = Math.sin(angle1) * toolRadius;
                    const x2 = Math.cos(angle2) * toolRadius;
                    const y2 = Math.sin(angle2) * toolRadius;

                    // Bottom triangle
                    const idx = i * 18;
                    toolTriangles[idx] = 0; toolTriangles[idx + 1] = 0; toolTriangles[idx + 2] = 0;
                    toolTriangles[idx + 3] = x1; toolTriangles[idx + 4] = y1; toolTriangles[idx + 5] = 0;
                    toolTriangles[idx + 6] = x2; toolTriangles[idx + 7] = y2; toolTriangles[idx + 8] = 0;

                    // Side rectangle (as 2 triangles) - just one for simplicity
                    toolTriangles[idx + 9] = x1; toolTriangles[idx + 10] = y1; toolTriangles[idx + 11] = 0;
                    toolTriangles[idx + 12] = x1; toolTriangles[idx + 13] = y1; toolTriangles[idx + 14] = toolHeight;
                    toolTriangles[idx + 15] = x2; toolTriangles[idx + 16] = y2; toolTriangles[idx + 17] = 0;
                }

                await raster.loadTool({ triangles: toolTriangles });

                const toolpathData = await raster.generateToolpaths({
                    xStep: 10,
                    yStep: 10,
                    zFloor: 0,
                    radiusOffset: 20
                });

                const toolpathTime = Date.now() - startToolpath;

                console.log('\\n=== Test Results ===');
                console.log('Toolpath generation time:', toolpathTime, 'ms');
                console.log('Number of strips:', toolpathData.numStrips);
                console.log('Total points:', toolpathData.totalPoints);
                console.log('Average points per strip:', (toolpathData.totalPoints / toolpathData.numStrips).toFixed(0));
                console.log('\\nTEST PASSED');

                return { success: true };

            } catch (error) {
                console.error('\\n=== TEST FAILED ===');
                console.error('Error:', error.message);
                console.error('Stack:', error.stack);
                return { success: false, error: error.message };
            } finally {
                raster.terminate();
            }
        })();
    `).then((result) => {
        testPassed = result.success === true;
        app.quit();
    }).catch((error) => {
        console.error('Test execution error:', error);
        app.quit();
    });
});

app.on('window-all-closed', () => {
    process.exit(testPassed ? 0 : 1);
});
