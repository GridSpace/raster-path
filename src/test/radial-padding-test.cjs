// radial-padding-test.cjs
// Test that X-axis padding actually increases radial toolpath span

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
                try {
                    console.log('=== Radial Padding Test ===');

                    if (!navigator.gpu) {
                        return { error: 'WebGPU not available' };
                    }

                    // Import RasterPath API
                    const { RasterPath } = await import('./raster-path.js');
                    const rasterPath = new RasterPath();
                    await rasterPath.init();
                    console.log('✓ RasterPath initialized');

                    // Load STL files from benchmark/fixtures
                    console.log('\\nLoading STL files...');
                    const terrainResponse = await fetch('../benchmark/fixtures/terrain.stl');
                    const terrainBuffer = await terrainResponse.arrayBuffer();

                    const toolResponse = await fetch('../benchmark/fixtures/tool.stl');
                    const toolBuffer = await toolResponse.arrayBuffer();

                    console.log(\`✓ Loaded terrain.stl: \${terrainBuffer.byteLength} bytes\`);
                    console.log(\`✓ Loaded tool.stl: \${toolBuffer.byteLength} bytes\`);

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

                    const triangles = parseBinarySTL(terrainBuffer);
                    const toolTriangles = parseBinarySTL(toolBuffer);
                    console.log(\`✓ Parsed terrain: \${triangles.length/9} triangles\`);
                    console.log(\`✓ Parsed tool: \${toolTriangles.length/9} triangles\`);

                    // Rasterize tool with standard parameters
                    const stepSize = 0.05; // 0.05mm detail
                    const toolResult = await rasterPath.rasterizeMesh(toolTriangles, stepSize, 1);
                    console.log(\`✓ Tool rasterized: \${toolResult.pointCount} points\`);

                    // Calculate terrain bounds
                    let minX = Infinity, minY = Infinity, minZ = Infinity;
                    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
                    for (let i = 0; i < triangles.length; i += 3) {
                        minX = Math.min(minX, triangles[i]);
                        maxX = Math.max(maxX, triangles[i]);
                        minY = Math.min(minY, triangles[i + 1]);
                        maxY = Math.max(maxY, triangles[i + 1]);
                        minZ = Math.min(minZ, triangles[i + 2]);
                        maxZ = Math.max(maxZ, triangles[i + 2]);
                    }

                    // Original bounds
                    const originalBounds = {
                        min: { x: minX, y: minY, z: minZ },
                        max: { x: maxX, y: maxY, z: maxZ }
                    };

                    // Padded bounds (add 10mm to X axis for testing)
                    const paddedBounds = {
                        min: { x: originalBounds.min.x - 10, y: originalBounds.min.y, z: originalBounds.min.z },
                        max: { x: originalBounds.max.x + 10, y: originalBounds.max.y, z: originalBounds.max.z }
                    };

                    console.log('\\n=== Test 1: Original Bounds ===');
                    console.log('X range:', originalBounds.min.x, 'to', originalBounds.max.x,
                                '(width:', (originalBounds.max.x - originalBounds.min.x).toFixed(2), 'mm)');

                    const result1 = await rasterPath.generateRadialToolpath(
                        triangles,
                        toolResult.positions,
                        1,  // 1 degree rotation step
                        1,  // xStep every grid point
                        -100,
                        stepSize,
                        originalBounds
                    );

                    console.log('Points per line:', result1.pointsPerLine);
                    console.log('First scanline length:', result1.pointsPerLine);
                    const firstLine1 = Array.from(result1.pathData.slice(0, result1.pointsPerLine));
                    const nonFloorCount1 = firstLine1.filter(z => z > -99).length;
                    console.log('Non-floor points in first line:', nonFloorCount1);

                    console.log('\\n=== Test 2: Padded Bounds (+10mm X on each side) ===');
                    console.log('X range:', paddedBounds.min.x, 'to', paddedBounds.max.x,
                                '(width:', (paddedBounds.max.x - paddedBounds.min.x).toFixed(2), 'mm)');

                    const result2 = await rasterPath.generateRadialToolpath(
                        triangles,
                        toolResult.positions,
                        1,  // 1 degree rotation step
                        1,  // xStep every grid point
                        -100,
                        stepSize,
                        paddedBounds
                    );

                    console.log('Points per line:', result2.pointsPerLine);
                    console.log('First scanline length:', result2.pointsPerLine);
                    const firstLine2 = Array.from(result2.pathData.slice(0, result2.pointsPerLine));
                    const nonFloorCount2 = firstLine2.filter(z => z > -99).length;
                    console.log('Non-floor points in first line:', nonFloorCount2);

                    console.log('\\n=== Comparison ===');
                    console.log('Original points per line:', result1.pointsPerLine);
                    console.log('Padded points per line:', result2.pointsPerLine);
                    console.log('Difference:', result2.pointsPerLine - result1.pointsPerLine);
                    const expectedIncrease = Math.floor(20 / stepSize / 1); // 20mm padding (10mm each side) / stepSize / xStep
                    console.log('Expected difference (20mm / 0.05mm / 1 xStep):', expectedIncrease);

                    if (result2.pointsPerLine <= result1.pointsPerLine) {
                        return {
                            error: \`Padding did not increase points per line! Original: \${result1.pointsPerLine}, Padded: \${result2.pointsPerLine}\`
                        };
                    }

                    const actualIncrease = result2.pointsPerLine - result1.pointsPerLine;

                    console.log('\\n✅ Padding test PASSED');
                    console.log('  Points per line increased from', result1.pointsPerLine, 'to', result2.pointsPerLine);
                    console.log('  Increase:', actualIncrease, 'points (expected ~', expectedIncrease, ')');

                    return {
                        success: true,
                        originalPoints: result1.pointsPerLine,
                        paddedPoints: result2.pointsPerLine,
                        increase: actualIncrease,
                        expectedIncrease
                    };

                } catch (error) {
                    console.error('Test error:', error);
                    return { error: error.message, stack: error.stack };
                }
            })();
        `;

        try {
            const result = await mainWindow.webContents.executeJavaScript(testScript);

            if (result.error) {
                console.error('❌ Test failed:', result.error);
                if (result.stack) {
                    console.error(result.stack);
                }
                app.exit(1);
                return;
            }

            console.log('\n=== Test Results ===');
            console.log('✅ Radial padding test successful');
            console.log(`   Original points per line: ${result.originalPoints}`);
            console.log(`   Padded points per line: ${result.paddedPoints}`);
            console.log(`   Increase: ${result.increase} points`);
            console.log(`   Expected: ~${result.expectedIncrease} points`);

            app.exit(0);
        } catch (error) {
            console.error('❌ Test execution failed:', error);
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
            console.log('[Renderer]', message);
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
