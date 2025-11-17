// tracing-test.cjs
// Test for tracing mode using new RasterPath API
// Tests: loadTool() + loadTerrain() + generateToolpaths() with input paths

const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');

const OUTPUT_DIR = path.join(__dirname, '../../test-output');
const CURRENT_FILE = path.join(OUTPUT_DIR, 'tracing-current.json');

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
                console.log('=== Tracing Mode Test ===');

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

                // Calculate terrain bounds for path generation
                let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
                for (let i = 0; i < terrainTriangles.length; i += 3) {
                    const x = terrainTriangles[i];
                    const y = terrainTriangles[i + 1];
                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                }
                console.log('✓ Terrain bounds:',
                    'X:', minX.toFixed(2), 'to', maxX.toFixed(2),
                    'Y:', minY.toFixed(2), 'to', maxY.toFixed(2));

                // Generate a 45-degree diagonal path across the terrain
                const pathLength = Math.sqrt((maxX - minX) ** 2 + (maxY - minY) ** 2);
                const numSegments = 20; // Intentionally sparse to test densification
                const path1 = new Float32Array(numSegments * 2);

                console.log('\\nGenerating 45° diagonal test path...');
                for (let i = 0; i < numSegments; i++) {
                    const t = i / (numSegments - 1);
                    path1[i * 2] = minX + t * (maxX - minX);      // X
                    path1[i * 2 + 1] = minY + t * (maxY - minY);  // Y
                }
                console.log('✓ Generated path with', numSegments, 'vertices');
                console.log('  Path length:', pathLength.toFixed(2), 'mm');

                // Test parameters
                const resolution = 0.1; // 0.1mm resolution for terrain raster
                const step = 0.5;       // 0.5mm sampling resolution along path
                const zFloor = -100;

                console.log('\\nTest parameters:');
                console.log('  Terrain resolution:', resolution, 'mm');
                console.log('  Path sampling step:', step, 'mm');
                console.log('  Z floor:', zFloor, 'mm');

                // Create RasterPath instance for tracing mode
                console.log('\\nInitializing RasterPath (tracing mode)...');
                const raster = new RasterPath({
                    mode: 'tracing',
                    resolution: resolution
                });
                await raster.init();
                console.log('✓ RasterPath initialized');

                // Load tool
                console.log('\\nLoading tool...');
                const toolStartTime = performance.now();
                await raster.loadTool({ triangles: toolTriangles });
                const toolTime = performance.now() - toolStartTime;
                console.log('✓ Tool loaded in', toolTime.toFixed(1), 'ms');

                // Load terrain
                console.log('\\nLoading terrain...');
                const terrainStartTime = performance.now();
                const terrainData = await raster.loadTerrain({
                    triangles: terrainTriangles,
                    zFloor: zFloor
                });
                const terrainTime = performance.now() - terrainStartTime;
                console.log('✓ Terrain loaded in', terrainTime.toFixed(1), 'ms');
                console.log('  Grid:', terrainData.width, 'x', terrainData.height);

                // Generate traced toolpaths
                console.log('\\nGenerating traced toolpaths...');
                const toolpathStartTime = performance.now();
                const result = await raster.generateToolpaths({
                    paths: [path1],
                    step: step,
                    zFloor: zFloor,
                    onProgress: (percent, info) => {
                        console.log('  Progress:', percent + '%', 'Path', info.current, '/', info.total);
                    }
                });
                const toolpathTime = performance.now() - toolpathStartTime;
                console.log('✓ Toolpaths generated in', toolpathTime.toFixed(1), 'ms');

                // Analyze results
                console.log('\\nResults:');
                console.log('  Number of output paths:', result.paths.length);

                const outputPath = result.paths[0];
                const numOutputPoints = outputPath.length / 3;
                console.log('  Output path points:', numOutputPoints);
                console.log('  Input path points:', numSegments);
                console.log('  Densification factor:', (numOutputPoints / numSegments).toFixed(2) + 'x');

                // Sample Z-values
                const zValues = [];
                for (let i = 0; i < outputPath.length; i += 3) {
                    zValues.push(outputPath[i + 2]);
                }
                const minZ = Math.min(...zValues);
                const maxZ = Math.max(...zValues);
                const avgZ = zValues.reduce((a, b) => a + b, 0) / zValues.length;

                console.log('\\nZ-depth statistics:');
                console.log('  Min Z:', minZ.toFixed(3), 'mm');
                console.log('  Max Z:', maxZ.toFixed(3), 'mm');
                console.log('  Avg Z:', avgZ.toFixed(3), 'mm');
                console.log('  Range:', (maxZ - minZ).toFixed(3), 'mm');

                // Sample and display some output points
                console.log('\\nOutput path samples:');
                const numSamples = Math.min(5, numOutputPoints);
                for (let i = 0; i < numSamples; i++) {
                    const idx = Math.floor(i * numOutputPoints / numSamples);
                    const x = outputPath[idx * 3].toFixed(2);
                    const y = outputPath[idx * 3 + 1].toFixed(2);
                    const z = outputPath[idx * 3 + 2].toFixed(2);
                    console.log(\`  Point \${idx}: (\${x}, \${y}, \${z})\`);
                }

                // Verify that output path lies within terrain XY bounds
                console.log('\\nVerifying path is within terrain bounds...');
                let pathInBounds = true;
                let pointsOutOfBounds = 0;
                for (let i = 0; i < numOutputPoints; i++) {
                    const x = outputPath[i * 3];
                    const y = outputPath[i * 3 + 1];
                    if (x < minX || x > maxX || y < minY || y > maxY) {
                        pathInBounds = false;
                        pointsOutOfBounds++;
                    }
                }
                if (pathInBounds) {
                    console.log('  ✓ All path points within terrain bounds');
                } else {
                    console.warn(\`  ⚠ \${pointsOutOfBounds}/\${numOutputPoints} points outside terrain bounds\`);
                }

                // Check if all Z values are zFloor (indicates no collision detection)
                const allZFloor = zValues.every(z => z === zFloor);
                if (allZFloor) {
                    console.warn('  ⚠ All Z values are zFloor - no terrain collision detected');
                    console.warn('  This may indicate a bug in collision detection or path/terrain mismatch');
                } else {
                    console.log('  ✓ Terrain collision detected');
                }

                // Cleanup
                raster.terminate();
                console.log('\\n✓ Test complete');

                return {
                    success: true,
                    terrainLoad: terrainTime,
                    toolLoad: toolTime,
                    toolpathGeneration: toolpathTime,
                    totalTime: toolTime + terrainTime + toolpathTime,
                    inputPoints: numSegments,
                    outputPoints: numOutputPoints,
                    densificationFactor: numOutputPoints / numSegments,
                    zStats: { minZ, maxZ, avgZ, range: maxZ - minZ },
                    allZFloor: allZFloor
                };
            })();
        `;

        const result = await mainWindow.webContents.executeJavaScript(testScript);

        if (result.error) {
            console.error('❌ Test failed:', result.error);
            app.exit(1);
            return;
        }

        console.log('\n=== Test Summary ===');
        console.log('Terrain load:', result.terrainLoad.toFixed(1), 'ms');
        console.log('Tool load:', result.toolLoad.toFixed(1), 'ms');
        console.log('Toolpath generation:', result.toolpathGeneration.toFixed(1), 'ms');
        console.log('Total time:', result.totalTime.toFixed(1), 'ms');
        console.log('\nDensification:', result.inputPoints, '→', result.outputPoints,
                    '(' + result.densificationFactor.toFixed(2) + 'x)');
        console.log('Z-depth range:', result.zStats.range.toFixed(3), 'mm');
        console.log('Collision detection:', result.allZFloor ? '❌ FAILED (all zFloor)' : '✓ Working');

        // Save current results
        fs.writeFileSync(CURRENT_FILE, JSON.stringify(result, null, 2));
        console.log('\n✓ Results saved to:', CURRENT_FILE);

        console.log('\n✅ Tracing mode test passed');
        app.exit(0);
    });

    mainWindow.webContents.on('console-message', (event, level, message) => {
        // Forward browser console to Node console (optional, for debugging)
        // console.log('[Browser]', message);
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    app.quit();
});
