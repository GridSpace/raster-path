// radial-v3-benchmark.cjs
// Benchmark comparison: V2 (current) vs V3 (rotate-filter-toolpath)

const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');

const OUTPUT_DIR = path.join(__dirname, '../../test-output');

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
                console.log('=== Radial V2 vs V3 Benchmark ===\\n');

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

                // Benchmark function
                async function benchmarkRadial(version, useV3) {
                    console.log(\`\\n=== Running \${version} ===\`);

                    const rp = new RasterPath({
                        resolution: 1.0,
                        mode: 'radial',
                        rotationStep: 2.0,  // 180 angles
                        radialV3: useV3,
                        quiet: true
                    });

                    await rp.init();
                    await rp.loadTool({ triangles: toolTriangles });
                    await rp.loadTerrain({ triangles: terrainTriangles, zFloor: 0 });

                    const startTime = performance.now();
                    const result = await rp.generateToolpaths({ xStep: 5, yStep: 5, zFloor: 0 });
                    const endTime = performance.now();

                    const duration = endTime - startTime;
                    console.log(\`\${version} completed in \${duration.toFixed(0)}ms\`);
                    console.log(\`  Strips: \${result.strips.length}\`);
                    console.log(\`  Total points: \${result.totalPoints}\`);

                    return {
                        version,
                        duration,
                        strips: result.strips.length,
                        totalPoints: result.totalPoints
                    };
                }

                // Run benchmarks
                const results = [];

                try {
                    // Run V2 (current implementation)
                    const v2Result = await benchmarkRadial('V2 (current)', false);
                    results.push(v2Result);

                    // Give GPU a moment to settle
                    await new Promise(resolve => setTimeout(resolve, 1000));

                    // Run V3 (rotate-filter-toolpath)
                    const v3Result = await benchmarkRadial('V3 (rotate-filter)', true);
                    results.push(v3Result);

                    // Calculate speedup
                    const speedup = v2Result.duration / v3Result.duration;
                    console.log(\`\\n=== Results ===\`);
                    console.log(\`V2: \${v2Result.duration.toFixed(0)}ms\`);
                    console.log(\`V3: \${v3Result.duration.toFixed(0)}ms\`);
                    console.log(\`Speedup: \${speedup.toFixed(2)}x\`);

                    return {
                        success: true,
                        results,
                        speedup
                    };
                } catch (error) {
                    console.error('Benchmark failed:', error);
                    return {
                        success: false,
                        error: error.message,
                        stack: error.stack
                    };
                }
            })();
        `;

        try {
            const result = await mainWindow.webContents.executeJavaScript(testScript);

            if (result.error) {
                console.error('✗ Test failed:', result.error);
                if (result.stack) console.error(result.stack);
                app.exit(1);
            } else if (!result.success) {
                console.error('✗ Benchmark failed:', result.error);
                if (result.stack) console.error(result.stack);
                app.exit(1);
            } else {
                console.log('\\n✓ Benchmark completed successfully');
                console.log('Results:', JSON.stringify(result.results, null, 2));

                // Save results
                const outputPath = path.join(OUTPUT_DIR, 'radial-v3-benchmark.json');
                fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
                console.log('✓ Results saved to:', outputPath);

                app.exit(0);
            }
        } catch (error) {
            console.error('✗ Script execution failed:', error);
            app.exit(1);
        }
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    app.quit();
});
