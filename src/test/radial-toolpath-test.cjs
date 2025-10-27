// radial-toolpath-test.cjs
// Test for radial toolpath generation (lathe-like operation)
// Tests terrain rotation + strip rasterization + scanline generation

const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');

const OUTPUT_DIR = path.join(__dirname, '../../test-output');
const RADIAL_OUTPUT = path.join(OUTPUT_DIR, 'radial-toolpath-output.txt');

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
                try {
                console.log('=== Radial Toolpath Test ===');

                if (!navigator.gpu) {
                    return { error: 'WebGPU not available' };
                }
                console.log('✓ WebGPU available');

                // Import RasterPath API
                const { RasterPath } = await import('./raster-path.js');

                // Initialize RasterPath
                const rasterPath = new RasterPath();
                await rasterPath.init();
                console.log('✓ RasterPath initialized');

                // Initialize worker pool for parallel processing
                try {
                    await rasterPath.initWorkerPool();
                    console.log('✓ Worker pool initialized');
                } catch (error) {
                    console.warn('Failed to initialize worker pool:', error);
                }

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
                const terrainBounds = {
                    min: { x: minX, y: minY, z: minZ },
                    max: { x: maxX, y: maxY, z: maxZ }
                };
                console.log('✓ Terrain bounds:', terrainBounds);

                // Rasterize tool with standard parameters
                const stepSize = 0.05; // 0.05mm detail
                const toolResult = await rasterPath.rasterizeMesh(toolTriangles, stepSize, 1); // filterMode=1 for min Z (tool)
                console.log(\`✓ Tool rasterized: \${toolResult.pointCount} points\`);

                // Generate radial toolpath with standard parameters
                const xRotationStep = 1; // 1 degree (360 rotations for full circle)
                const xStep = 1; // sample every grid point along X
                const zFloor = -50;

                console.log('Generating radial toolpath...');
                console.log(\`  Rotation step: \${xRotationStep}°\`);
                console.log(\`  X step: \${xStep} points\`);
                console.log(\`  Z floor: \${zFloor} mm\`);
                console.log(\`  Grid step: \${stepSize} mm\`);

                const startTime = performance.now();
                const result = await rasterPath.generateRadialToolpath(
                    triangles,
                    toolResult.positions,
                    xRotationStep,
                    xStep,
                    zFloor,
                    stepSize,
                    terrainBounds
                );
                const endTime = performance.now();

                console.log(\`✓ Radial toolpath generated in \${(endTime - startTime).toFixed(1)}ms\`);
                console.log(\`  Rotations: \${result.numRotations}\`);
                console.log(\`  Points per line: \${result.pointsPerLine}\`);
                console.log(\`  Total points: \${result.pathData.length}\`);

                // Validate output
                const expectedRotations = Math.floor(360 / xRotationStep);
                if (result.numRotations !== expectedRotations) {
                    return {
                        error: \`Expected \${expectedRotations} rotations, got \${result.numRotations}\`
                    };
                }

                // Check for reasonable Z values
                let outputMinZ = Infinity, outputMaxZ = -Infinity;
                let floorCount = 0;
                for (let i = 0; i < result.pathData.length; i++) {
                    const z = result.pathData[i];
                    if (z === zFloor) floorCount++;
                    outputMinZ = Math.min(outputMinZ, z);
                    outputMaxZ = Math.max(outputMaxZ, z);
                }

                console.log(\`  Z range: [\${outputMinZ.toFixed(2)}, \${outputMaxZ.toFixed(2)}]\`);
                console.log(\`  Floor values: \${floorCount} / \${result.pathData.length}\`);

                // Check first rotation (should be similar values since cone is symmetric)
                const firstRotation = result.pathData.slice(0, result.pointsPerLine);
                console.log('  First rotation (first 10 points):',
                    Array.from(firstRotation.slice(0, 10)).map(v => v.toFixed(2)).join(', '));

                return {
                    success: true,
                    numRotations: result.numRotations,
                    pointsPerLine: result.pointsPerLine,
                    totalPoints: result.pathData.length,
                    generationTime: endTime - startTime,
                    zRange: { min: outputMinZ, max: outputMaxZ },
                    floorCount,
                    pathData: Array.from(result.pathData)
                };
                } catch (error) {
                    console.error('Test script error:', error);
                    return { error: error.message, stack: error.stack };
                }
            })();
        `;

        try {
            const result = await mainWindow.webContents.executeJavaScript(testScript);

            if (result.error) {
                console.error('❌ Test failed:', result.error);
                app.exit(1);
                return;
            }

            console.log('\n=== Test Results ===');
            console.log('✅ Radial toolpath generation successful');
            console.log(`   Rotations: ${result.numRotations}`);
            console.log(`   Points per line: ${result.pointsPerLine}`);
            console.log(`   Total points: ${result.totalPoints}`);
            console.log(`   Generation time: ${result.generationTime.toFixed(1)}ms`);
            console.log(`   Z range: [${result.zRange.min.toFixed(2)}, ${result.zRange.max.toFixed(2)}]`);
            console.log(`   Floor values: ${result.floorCount} / ${result.totalPoints}`);

            // Save output for inspection
            const output = {
                timestamp: new Date().toISOString(),
                ...result
            };
            fs.writeFileSync(RADIAL_OUTPUT, JSON.stringify(output, null, 2));
            console.log(`\n📝 Output saved to: ${RADIAL_OUTPUT}`);

            console.log('\n✅ All tests passed');
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
