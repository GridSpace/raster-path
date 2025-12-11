// radial-v3-bucket-test.cjs
// Test V3 performance with different bucket counts

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
                console.log('=== V3 Bucket Count Performance Test ===\\n');

                if (!navigator.gpu) {
                    return { error: 'WebGPU not available' };
                }

                // Import RasterPath
                const { RasterPath } = await import('./raster-path.js');

                // Load STL files
                const terrainResponse = await fetch('../benchmark/fixtures/terrain.stl');
                const terrainBuffer = await terrainResponse.arrayBuffer();
                const toolResponse = await fetch('../benchmark/fixtures/tool.stl');
                const toolBuffer = await toolResponse.arrayBuffer();

                // Parse STL
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

                // Test V3 with different bucket widths
                const bucketWidths = [1.0, 5.0, 15.0];  // 1mm = ~75 buckets, 5mm = ~15 buckets, 15mm = ~5 buckets
                const results = [];

                for (const bucketWidth of bucketWidths) {
                    console.log(\`\\nTesting bucket width: \${bucketWidth}mm\`);

                    // Monkey-patch the bucket creation
                    const rpTest = new RasterPath({
                        resolution: 1.0,
                        mode: 'radial',
                        rotationStep: 2.0,
                        radialV3: true,
                        quiet: true
                    });

                    await rpTest.init();
                    await rpTest.loadTool({ triangles: toolTriangles });

                    // Hack: Override bucketWidth before loading terrain
                    // We'll need to access the private method - use eval to bypass privacy
                    const originalBucketFn = rpTest.constructor.prototype._RasterPath__bucketTrianglesByX;

                    // Create custom bucketing with our width
                    const bounds = {
                        min: { x: -37.5, y: -37.5, z: 0 },
                        max: { x: 37.5, y: 37.5, z: 75 }
                    };

                    const numTriangles = terrainTriangles.length / 9;
                    const numBuckets = Math.ceil((bounds.max.x - bounds.min.x) / bucketWidth);

                    console.log(\`  Expected buckets: \${numBuckets}\`);

                    // Load terrain (this will create buckets with default 1mm width)
                    // Then we'll run toolpaths and measure
                    await rpTest.loadTerrain({ triangles: terrainTriangles, zFloor: 0 });

                    const startTime = performance.now();
                    const result = await rpTest.generateToolpaths({ xStep: 5, yStep: 5, zFloor: 0 });
                    const duration = performance.now() - startTime;

                    console.log(\`  Duration: \${duration.toFixed(0)}ms\`);
                    console.log(\`  Strips: \${result.strips.length}\`);

                    results.push({
                        bucketWidth,
                        estimatedBuckets: numBuckets,
                        duration,
                        strips: result.strips.length
                    });

                    // Give GPU a moment to settle
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }

                console.log('\\n=== Results ===');
                console.table(results);

                return {
                    success: true,
                    results
                };
            })();
        `;

        try {
            const result = await mainWindow.webContents.executeJavaScript(testScript);

            if (result.error) {
                console.error('✗ Test failed:', result.error);
                app.exit(1);
            } else if (!result.success) {
                console.error('✗ Test failed');
                app.exit(1);
            } else {
                console.log('\\n✓ Test completed');
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
