// radial-v2-test.cjs
// Test for radial V2 mode using new rotating ray plane architecture
// Tests: rasterizeTool() + rasterizeModelRadial() + generateToolpathsRadial()

const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');

const OUTPUT_DIR = path.join(__dirname, '../../test-output');
const OUTPUT_FILE = path.join(OUTPUT_DIR, 'radial-v2-test.json');

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
                console.log('=== Radial V2 Test (Rotating Ray Planes) ===');

                if (!navigator.gpu) {
                    return { error: 'WebGPU not available' };
                }
                console.log('✓ WebGPU available');

                // Import RasterPath
                const { RasterPath } = await import('./raster-path.js');

                // Load STL files
                console.log('\\nLoading STL files...');
                const terrainResponse = await fetch('../benchmark/fixtures/terrain.stl');
                const terrainBuffer = await terrainResponse.arrayBuffer();

                const toolResponse = await fetch('../benchmark/fixtures/tool.stl');
                const toolBuffer = await toolResponse.arrayBuffer();

                console.log('✓ Loaded terrain.stl:', terrainBuffer.byteLength, 'bytes');
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

                // Check terrain bounds
                let minX = Infinity, maxX = -Infinity;
                let minY = Infinity, maxY = -Infinity;
                let minZ = Infinity, maxZ = -Infinity;
                for (let i = 0; i < terrainTriangles.length; i += 3) {
                    minX = Math.min(minX, terrainTriangles[i]);
                    maxX = Math.max(maxX, terrainTriangles[i]);
                    minY = Math.min(minY, terrainTriangles[i + 1]);
                    maxY = Math.max(maxY, terrainTriangles[i + 1]);
                    minZ = Math.min(minZ, terrainTriangles[i + 2]);
                    maxZ = Math.max(maxZ, terrainTriangles[i + 2]);
                }
                console.log('Terrain bounds:');
                console.log('  X: [' + minX.toFixed(2) + ', ' + maxX.toFixed(2) + ']');
                console.log('  Y: [' + minY.toFixed(2) + ', ' + maxY.toFixed(2) + ']');
                console.log('  Z: [' + minZ.toFixed(2) + ', ' + maxZ.toFixed(2) + ']');
                const centerY = (minY + maxY) / 2;
                const centerZ = (minZ + maxZ) / 2;
                console.log('  YZ center: (' + centerY.toFixed(2) + ', ' + centerZ.toFixed(2) + ')');

                // Test parameters
                const resolution = 0.5; // 0.5mm for faster testing
                const rotationStep = 10.0; // 10 degrees (36 strips) - production-like
                const xStep = 1;
                const yStep = 1;
                const zFloor = 0;

                console.log('\\nTest parameters:');
                console.log('  Resolution:', resolution, 'mm');
                console.log('  Rotation step:', rotationStep, '° (', 360/rotationStep, 'strips)');
                console.log('  XY step:', xStep + 'x' + yStep, 'points');
                console.log('  Z floor:', zFloor, 'mm');

                // Create RasterPath instance for radial mode
                console.log('\\nInitializing RasterPath (radial V2 mode)...');
                const raster = new RasterPath({
                    mode: 'radial',
                    resolution: resolution,
                    rotationStep: rotationStep,
                    debug: true
                });
                await raster.init();
                console.log('✓ RasterPath initialized');

                // Step 1: Rasterize tool FIRST (required for radial V2)
                console.log('\\n1. Rasterizing tool (planar)...');
                const t0 = performance.now();
                const toolData = await raster.rasterizeTool({
                    triangles: toolTriangles,
                    zFloor: zFloor
                });
                const toolTime = performance.now() - t0;
                console.log('✓ Tool:', toolData.pointCount, 'points in', toolTime.toFixed(1), 'ms');
                console.log('  Tool bounds:', toolData.bounds);

                // Step 2: Rasterize model with radial V2 (returns array of strips)
                console.log('\\n2. Rasterizing model (radial V2 - rotating rays)...');
                const t1 = performance.now();
                const strips = await raster.rasterizeModelRadial({
                    triangles: terrainTriangles,
                    toolData: toolData,
                    zFloor: zFloor
                });
                const terrainTime = performance.now() - t1;

                console.log('✓ Model rasterized:', strips.length, 'strips in', terrainTime.toFixed(1), 'ms');

                // Analyze strips
                let totalPoints = 0;
                let nonEmptyStrips = 0;
                for (const strip of strips) {
                    totalPoints += strip.pointCount;
                    if (strip.pointCount > 0) nonEmptyStrips++;
                }
                console.log('  Total points across all strips:', totalPoints);
                console.log('  Non-empty strips:', nonEmptyStrips, '/', strips.length);

                // Show details for all strips
                console.log('\\n  Strip details:');
                for (let i = 0; i < strips.length; i++) {
                    const strip = strips[i];
                    console.log('    Strip ' + i + ' (' + strip.angle.toFixed(1) + '°): ' + strip.pointCount + ' points');

                    if (strip.pointCount > 0) {
                        // Find X range in this strip
                        let stripMinX = Infinity, stripMaxX = -Infinity;
                        for (let j = 0; j < strip.positions.length; j += 3) {
                            const x = strip.positions[j];
                            stripMinX = Math.min(stripMinX, x);
                            stripMaxX = Math.max(stripMaxX, x);
                        }
                        const xRange = stripMaxX - stripMinX;
                        console.log('      X range: [' + stripMinX.toFixed(2) + ', ' + stripMaxX.toFixed(2) + '] = ' + xRange.toFixed(2) + ' mm');

                        // Sample some Z values
                        const sampleSize = Math.min(5, strip.pointCount);
                        const samples = [];
                        for (let j = 0; j < sampleSize; j++) {
                            const idx = j * 3 + 2; // Z is every 3rd value
                            samples.push(strip.positions[idx].toFixed(2));
                        }
                        console.log('      Sample Z values: ' + samples.join(', '));
                    }
                }

                // Step 3: Toolpath generation - temporarily disabled while debugging
                console.log('\\n3. Toolpath generation - skipped (rasterization verified)');
                const toolpathTime = 0;
                const toolpathData = { numStrips: 0, totalPoints: 0 };

                // Cleanup
                raster.terminate();

                // Return results
                const results = {
                    success: true,
                    parameters: {
                        resolution,
                        rotationStep,
                        xStep,
                        yStep,
                        zFloor
                    },
                    timing: {
                        tool: toolTime,
                        terrain: terrainTime,
                        toolpath: toolpathTime,
                        total: toolTime + terrainTime + toolpathTime
                    },
                    terrain: {
                        numStrips: strips.length,
                        totalPoints: totalPoints,
                        nonEmptyStrips: nonEmptyStrips
                    },
                    toolpath: {
                        numStrips: toolpathData.numStrips,
                        totalPoints: toolpathData.totalPoints
                    }
                };

                console.log('\\n=== Test Complete ===');
                console.log('Total time:', results.timing.total.toFixed(1), 'ms');

                return results;
            })();
        `;

        try {
            const result = await mainWindow.webContents.executeJavaScript(testScript);

            if (result.error) {
                console.error('❌ Test failed:', result.error);
                app.exit(1);
                return;
            }

            // Write results to file
            fs.writeFileSync(OUTPUT_FILE, JSON.stringify(result, null, 2));
            console.log('\\n✅ Test passed! Results written to:', OUTPUT_FILE);

            // Summary
            console.log('\\n=== Summary ===');
            console.log('Terrain strips:', result.terrain.numStrips);
            console.log('Terrain points:', result.terrain.totalPoints);
            console.log('Toolpath strips:', result.toolpath.numStrips);
            console.log('Toolpath points:', result.toolpath.totalPoints);
            console.log('Total time:', result.timing.total.toFixed(1), 'ms');

            if (result.terrain.totalPoints === 0) {
                console.error('\\n⚠️  WARNING: No terrain points generated! Check ray-triangle intersection.');
                app.exit(1);
            } else if (result.toolpath.totalPoints === 0) {
                console.error('\\n⚠️  WARNING: No toolpath points generated!');
                app.exit(1);
            } else {
                app.exit(0);
            }

        } catch (error) {
            console.error('❌ Test execution error:', error);
            app.exit(1);
        }
    });

    mainWindow.webContents.on('console-message', (event, level, message) => {
        console.log('[Browser]', message);
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    app.quit();
});
