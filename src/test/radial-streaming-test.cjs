// radial-streaming-test.cjs
// Test for radial V2 streaming mode with many angles (720 at 0.5°)

const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');

const OUTPUT_DIR = path.join(__dirname, '../../test-output');
const OUTPUT_FILE = path.join(OUTPUT_DIR, 'radial-streaming-test.json');

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
                console.log('=== Radial V2 Streaming Test (720 angles @ 0.5°) ===');

                if (!navigator.gpu) {
                    return { error: 'WebGPU not available' };
                }
                console.log('✓ WebGPU available');

                const { RasterPath } = await import('./raster-path.js');

                console.log('\\nLoading STL files...');
                const terrainResponse = await fetch('../benchmark/fixtures/terrain.stl');
                const terrainBuffer = await terrainResponse.arrayBuffer();

                const toolResponse = await fetch('../benchmark/fixtures/tool.stl');
                const toolBuffer = await toolResponse.arrayBuffer();

                console.log('✓ Loaded terrain.stl:', terrainBuffer.byteLength, 'bytes');
                console.log('✓ Loaded tool.stl:', toolBuffer.byteLength, 'bytes');

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
                console.log('✓ Parsed terrain:', terrainTriangles.length / 9, 'triangles');
                console.log('✓ Parsed tool:', toolTriangles.length / 9, 'triangles');

                // STREAMING TEST: 0.5° steps = 720 angles (should use streaming mode)
                const resolution = 0.5;
                const rotationStep = 0.5;  // 720 angles - will trigger streaming
                const xStep = 1;
                const yStep = 1;
                const zFloor = 0;

                console.log('\\nTest parameters:');
                console.log('  Resolution:', resolution, 'mm');
                console.log('  Rotation step:', rotationStep, '° (', 360/rotationStep, 'strips)');
                console.log('  XY step:', xStep + 'x' + yStep, 'points');
                console.log('  Z floor:', zFloor, 'mm');
                console.log('  EXPECTED: Streaming mode (>500 angles)');

                console.log('\\nInitializing RasterPath (radial V2 mode)...');
                const raster = new RasterPath({
                    mode: 'radial',
                    resolution: resolution,
                    rotationStep: rotationStep,
                    debug: true
                });
                await raster.init();
                console.log('✓ RasterPath initialized');

                console.log('\\n1. Rasterizing tool (planar)...');
                const t0 = performance.now();
                const toolData = await raster.rasterizeTool({
                    triangles: toolTriangles,
                    zFloor: zFloor
                });
                const toolTime = performance.now() - t0;
                console.log('✓ Tool:', toolData.pointCount, 'points in', toolTime.toFixed(1), 'ms');

                console.log('\\n2. Generating complete radial toolpath (streaming)...');
                const t1 = performance.now();
                const result = await raster.rasterizeAndGenerateToolpathsRadial({
                    triangles: terrainTriangles,
                    toolData: toolData,
                    xStep: xStep,
                    yStep: yStep,
                    zFloor: zFloor
                });
                const totalTime = performance.now() - t1;

                console.log('\\n✓ Complete radial toolpath generated!');
                console.log('  Strips:', result.numStrips);
                console.log('  Total points:', result.totalPoints);
                console.log('  Time:', totalTime.toFixed(0), 'ms');
                console.log('  Avg time per strip:', (totalTime / result.numStrips).toFixed(2), 'ms');

                // Sample first few strips to verify data
                console.log('\\nFirst 5 strips:');
                for (let i = 0; i < Math.min(5, result.strips.length); i++) {
                    const strip = result.strips[i];
                    console.log('  Strip', i, '(' + strip.angle.toFixed(1) + '°):',
                               strip.numScanlines, 'scanlines,',
                               strip.pointsPerLine, 'points/line');
                    if (strip.pathData && strip.pathData.length > 0) {
                        const samples = [];
                        for (let j = 0; j < Math.min(5, strip.pathData.length); j++) {
                            samples.push(strip.pathData[j].toFixed(3));
                        }
                        console.log('    Sample Z values:', samples.join(', '));
                    }
                }

                // Check strips around batch boundary (360)
                console.log('\\nStrips around 360:');
                for (let i = 358; i <= 362 && i < result.strips.length; i++) {
                    const strip = result.strips[i];
                    console.log('  Strip', i, '(' + strip.angle.toFixed(1) + '°):',
                               strip.numScanlines, 'scanlines,',
                               strip.pointsPerLine, 'points/line');
                }

                // Check last few strips
                console.log('\\nLast 5 strips:');
                for (let i = Math.max(0, result.strips.length - 5); i < result.strips.length; i++) {
                    const strip = result.strips[i];
                    console.log('  Strip', i, '(' + strip.angle.toFixed(1) + '°):',
                               strip.numScanlines, 'scanlines,',
                               strip.pointsPerLine, 'points/line');
                    if (strip.pathData && strip.pathData.length > 0) {
                        const samples = [];
                        for (let j = 0; j < Math.min(5, strip.pathData.length); j++) {
                            samples.push(strip.pathData[j].toFixed(3));
                        }
                        console.log('    Sample Z values:', samples.join(', '));
                    }
                }

                return {
                    success: true,
                    numStrips: result.numStrips,
                    totalPoints: result.totalPoints,
                    totalTime: totalTime,
                    toolTime: toolTime
                };
            })();
        `;

        try {
            const result = await mainWindow.webContents.executeJavaScript(testScript);

            if (result.error) {
                console.error('❌ Test failed:', result.error);
                app.exit(1);
                return;
            }

            console.log('\n=== Test Complete ===');
            console.log('Total time:', result.totalTime.toFixed(1), 'ms');

            // Write results
            fs.writeFileSync(OUTPUT_FILE, JSON.stringify(result, null, 2));
            console.log('\n✅ Test passed! Results written to:', OUTPUT_FILE);

            console.log('\n=== Summary ===');
            console.log('Strips:', result.numStrips);
            console.log('Points:', result.totalPoints);
            console.log('Time:', result.totalTime.toFixed(1), 'ms');
            console.log('Avg per strip:', (result.totalTime / result.numStrips).toFixed(2), 'ms');

            if (result.totalTime < 10000) {
                console.log('\n✅ Performance: EXCELLENT (<10s)');
            } else if (result.totalTime < 30000) {
                console.log('\n✅ Performance: GOOD (<30s)');
            } else {
                console.log('\n⚠️  Performance: SLOW (>30s)');
            }

            app.exit(0);
        } catch (error) {
            console.error('❌ Test execution error:', error);
            app.exit(1);
        }
    });

    mainWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
        if (message.includes('[Raster') || message.includes('[Worker')) {
            console.log('[Browser]', message);
        } else {
            console.log('[Browser]', message);
        }
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    app.quit();
});
