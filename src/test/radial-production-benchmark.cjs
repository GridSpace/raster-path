// radial-production-benchmark.cjs
// Benchmark radial toolpath with production settings
// Matches typical user workflow: 0.05mm detail, 5° rotation, X step 1

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
                    console.log('=== Radial Production Benchmark ===');
                    console.log('Settings: Detail 0.05mm, Rotation 1°, X step 1');

                    if (!navigator.gpu) {
                        return { error: 'WebGPU not available' };
                    }

                    const { RasterPath } = await import('./raster-path.js');

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

                    // Production settings
                    const stepSize = 0.05; // 0.05mm detail level
                    const xRotationStep = 1;  // 1° steps = 360 rotations
                    const xStep = 1;          // Sample every grid point
                    const zFloor = 0;

                    // Test 1: Sequential (disable worker pool)
                    console.log('\\n--- Test 1: Sequential Processing ---');
                    const rasterPathSeq = new RasterPath({ parallelWorkers: 0 });
                    await rasterPathSeq.init();
                    console.log('✓ RasterPath initialized (sequential mode)');

                    const toolResult = await rasterPathSeq.rasterizeMesh(toolTriangles, stepSize, 1);
                    console.log(\`✓ Tool rasterized: \${toolResult.pointCount} points\`);

                    console.log('Generating sequential radial toolpath...');
                    const seqStart = performance.now();
                    const seqResult = await rasterPathSeq.generateRadialToolpath(
                        triangles,
                        toolResult.positions,
                        xRotationStep,
                        xStep,
                        zFloor,
                        stepSize,
                        terrainBounds
                    );
                    const seqTime = performance.now() - seqStart;
                    console.log(\`✅ Sequential complete: \${seqTime.toFixed(1)}ms\`);

                    // Clean up sequential instance
                    rasterPathSeq.dispose();

                    // Test 2: Parallel (with worker pool)
                    console.log('\\n--- Test 2: Parallel Processing (4 workers) ---');
                    const rasterPathPar = new RasterPath({ parallelWorkers: 4 });
                    await rasterPathPar.init();
                    await rasterPathPar.initWorkerPool();
                    console.log('✓ RasterPath initialized with worker pool');

                    // Rasterize tool again for parallel instance
                    const toolResultPar = await rasterPathPar.rasterizeMesh(toolTriangles, stepSize, 1);

                    console.log('Generating parallel radial toolpath...');
                    const parStart = performance.now();
                    const parResult = await rasterPathPar.generateRadialToolpath(
                        triangles,
                        toolResultPar.positions,
                        xRotationStep,
                        xStep,
                        zFloor,
                        stepSize,
                        terrainBounds
                    );
                    const parTime = performance.now() - parStart;
                    console.log(\`✅ Parallel complete: \${parTime.toFixed(1)}ms\`);

                    // Calculate speedup
                    const speedup = seqTime / parTime;

                    console.log('\\n=== Benchmark Results ===');
                    console.log(\`Sequential: \${seqTime.toFixed(1)}ms\`);
                    console.log(\`Parallel:   \${parTime.toFixed(1)}ms\`);
                    console.log(\`Speedup:    \${speedup.toFixed(2)}x\`);
                    console.log(\`Rotations:  \${seqResult.numRotations}\`);
                    console.log(\`Points/line: \${seqResult.pointsPerLine}\`);
                    console.log(\`Total points: \${seqResult.pathData.length}\`);

                    // Clean up parallel instance
                    rasterPathPar.dispose();

                    return {
                        success: true,
                        sequentialTime: seqTime,
                        parallelTime: parTime,
                        speedup,
                        numRotations: seqResult.numRotations,
                        pointsPerLine: seqResult.pointsPerLine,
                        totalPoints: seqResult.pathData.length,
                        triangles: triangles.length / 9
                    };

                } catch (error) {
                    console.error('Benchmark error:', error);
                    return { error: error.message, stack: error.stack };
                }
            })();
        `;

        try {
            const result = await mainWindow.webContents.executeJavaScript(testScript);

            if (result.error) {
                console.error('❌ Benchmark failed:', result.error);
                if (result.stack) {
                    console.error(result.stack);
                }
                app.exit(1);
                return;
            }

            console.log('\n=== Final Results ===');
            console.log('✅ Production benchmark complete');
            console.log(`   Terrain: ${result.triangles} triangles`);
            console.log(`   Rotations: ${result.numRotations}`);
            console.log(`   Points per line: ${result.pointsPerLine}`);
            console.log(`   Total points: ${result.totalPoints}`);
            console.log(`   Sequential: ${result.sequentialTime.toFixed(1)}ms`);
            console.log(`   Parallel:   ${result.parallelTime.toFixed(1)}ms`);
            console.log(`   Speedup:    ${result.speedup.toFixed(2)}x`);

            app.exit(0);
        } catch (error) {
            console.error('❌ Benchmark execution failed:', error);
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
