// planar-vs-radial-test.cjs
// Compare planar midline vs radial scanline to verify padding consistency

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
                    console.log('=== Planar vs Radial Padding Test ===');

                    if (!navigator.gpu) {
                        return { error: 'WebGPU not available' };
                    }

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

                    const terrainTriangles = parseBinarySTL(terrainBuffer);
                    const toolTriangles = parseBinarySTL(toolBuffer);
                    console.log(\`✓ Parsed terrain: \${terrainTriangles.length/9} triangles\`);
                    console.log(\`✓ Parsed tool: \${toolTriangles.length/9} triangles\`);

                    // Rasterize with standard parameters (detail 0.05mm)
                    const stepSize = 0.05;
                    const terrainResult = await rasterPath.rasterizeMesh(terrainTriangles, stepSize, 0);
                    console.log(\`✓ Terrain rasterized: \${terrainResult.pointCount} points\`);

                    const toolResult = await rasterPath.rasterizeMesh(toolTriangles, stepSize, 1);
                    console.log(\`✓ Tool rasterized: \${toolResult.pointCount} points\`);

                    // Standard parameters: x,y step 1,1
                    const xStep = 1;
                    const yStep = 1;
                    const zFloor = -100;

                    console.log('\\n=== Generating Planar Toolpath ===');
                    const planarResult = await rasterPath.generatePlanarToolpath(
                        terrainResult.positions,
                        toolResult.positions,
                        xStep,
                        yStep,
                        zFloor,
                        stepSize,
                        { terrainBounds: terrainResult.bounds }
                    );

                    console.log(\`Points per line: \${planarResult.pointsPerLine}\`);
                    console.log(\`Number of lines: \${planarResult.numScanlines}\`);

                    // Extract middle line from planar (y=0 is at index numScanlines/2)
                    const midLineIndex = Math.floor(planarResult.numScanlines / 2);
                    const planarMidline = Array.from(
                        planarResult.pathData.slice(
                            midLineIndex * planarResult.pointsPerLine,
                            (midLineIndex + 1) * planarResult.pointsPerLine
                        )
                    );
                    console.log(\`Planar midline (line \${midLineIndex}): \${planarMidline.length} points\`);
                    console.log(\`First 10: \${planarMidline.slice(0, 10).map(v => v.toFixed(2)).join(', ')}\`);

                    console.log('\\n=== Generating Radial Toolpath (0° only) ===');
                    const radialResult = await rasterPath.generateRadialToolpath(
                        terrainTriangles,
                        toolResult.positions,
                        360,  // Only one rotation at 0°
                        xStep,
                        zFloor,
                        stepSize,
                        terrainResult.bounds
                    );

                    console.log(\`Points per line: \${radialResult.pointsPerLine}\`);
                    console.log(\`Number of rotations: \${radialResult.numRotations}\`);

                    const radialScanline = Array.from(radialResult.pathData);
                    console.log(\`Radial scanline: \${radialScanline.length} points\`);
                    console.log(\`First 10: \${radialScanline.slice(0, 10).map(v => v.toFixed(2)).join(', ')}\`);

                    // Compare lengths
                    console.log('\\n=== Comparison ===');
                    console.log(\`Planar midline length: \${planarMidline.length}\`);
                    console.log(\`Radial scanline length: \${radialScanline.length}\`);

                    if (planarMidline.length !== radialScanline.length) {
                        return {
                            error: \`Length mismatch! Planar: \${planarMidline.length}, Radial: \${radialScanline.length}\`
                        };
                    }

                    // Compare values
                    let maxDiff = 0;
                    let diffCount = 0;
                    const threshold = 0.1; // 0.1mm tolerance

                    for (let i = 0; i < planarMidline.length; i++) {
                        const diff = Math.abs(planarMidline[i] - radialScanline[i]);
                        if (diff > threshold) {
                            diffCount++;
                            maxDiff = Math.max(maxDiff, diff);
                        }
                    }

                    console.log(\`Max difference: \${maxDiff.toFixed(3)}mm\`);
                    console.log(\`Points with diff > \${threshold}mm: \${diffCount} / \${planarMidline.length}\`);

                    if (maxDiff > 1.0) {
                        return {
                            error: \`Large difference detected: \${maxDiff.toFixed(3)}mm\`,
                            planarMidline,
                            radialScanline
                        };
                    }

                    console.log('\\n✅ Planar and radial match (within tolerance)');
                    return {
                        success: true,
                        planarLength: planarMidline.length,
                        radialLength: radialScanline.length,
                        maxDiff,
                        diffCount,
                        planarMidline,
                        radialScanline
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
            console.log('✅ Planar vs Radial comparison successful');
            console.log(`   Planar midline: ${result.planarLength} points`);
            console.log(`   Radial scanline: ${result.radialLength} points`);
            console.log(`   Max difference: ${result.maxDiff.toFixed(3)}mm`);
            console.log(`   Points with significant diff: ${result.diffCount}`);

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
