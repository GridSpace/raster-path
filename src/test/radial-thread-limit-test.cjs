// radial-thread-limit-test.cjs
// Test that radial rasterization respects thread limits and produces correct output

const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');

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
                console.log('=== Radial Thread Limit Test ===\\n');

                if (!navigator.gpu) {
                    return { error: 'WebGPU not available' };
                }

                const { RasterPath } = await import('./raster-path.js');

                // Read test STL
                const stlPath = '${path.join(__dirname, '../../benchmark/fixtures/lathe-cylinder.stl')}';
                const response = await fetch('file://' + stlPath);
                if (!response.ok) {
                    return { error: 'Failed to load test STL: ' + stlPath };
                }
                const arrayBuffer = await response.arrayBuffer();

                // Test with different thread limits
                const threadLimits = [
                    { limit: 256, desc: 'Default limit (256 threads)' },
                    { limit: 128, desc: 'Reduced limit (128 threads)' },
                    { limit: 64, desc: 'Very low limit (64 threads)' },
                ];

                const results = [];

                for (const config of threadLimits) {
                    console.log(\`Testing: \${config.desc}\`);

                    const raster = new RasterPath({
                        mode: 'radial',
                        resolution: 0.5,
                        rotationStep: 1.0,
                        toolWidth: 5.0,
                        maxConcurrentThreads: config.limit
                    });

                    await raster.init();

                    const startTime = performance.now();
                    const result = await raster.processSTL(arrayBuffer);
                    const elapsed = performance.now() - startTime;

                    // Calculate checksum
                    let checksum = 0;
                    for (const strip of result.strips) {
                        for (let i = 0; i < strip.pathData.length; i++) {
                            checksum = (checksum * 31 + strip.pathData[i]) | 0;
                        }
                    }

                    console.log(\`  Time: \${elapsed.toFixed(1)}ms\`);
                    console.log(\`  Strips: \${result.strips.length}\`);
                    console.log(\`  Checksum: \${checksum}\`);
                    console.log('');

                    results.push({
                        limit: config.limit,
                        desc: config.desc,
                        time: elapsed,
                        strips: result.strips.length,
                        checksum,
                    });
                }

                // Verify all checksums match
                const referenceChecksum = results[0].checksum;
                const allMatch = results.every(r => r.checksum === referenceChecksum);

                console.log('=== Results ===');
                console.log(\`Reference checksum: \${referenceChecksum}\`);
                console.log(\`All checksums match: \${allMatch ? '✓ YES' : '❌ NO'}\`);

                if (!allMatch) {
                    console.log('\\nChecksum mismatches:');
                    for (const r of results) {
                        if (r.checksum !== referenceChecksum) {
                            console.log(\`  \${r.desc}: \${r.checksum} (expected \${referenceChecksum})\`);
                        }
                    }
                }

                console.log('\\nTiming comparison:');
                const baselineTime = results[0].time;
                for (const r of results) {
                    const ratio = (r.time / baselineTime).toFixed(2);
                    console.log(\`  \${r.desc}: \${r.time.toFixed(1)}ms (\${ratio}x)\`);
                }

                return { success: allMatch, results };
            })();
        `;

        try {
            const result = await mainWindow.webContents.executeJavaScript(testScript);

            if (result.error) {
                console.error('❌ Test failed:', result.error);
                app.exit(1);
            } else if (!result.success) {
                console.error('❌ Thread limit test failed - checksums do not match');
                app.exit(1);
            } else {
                console.log('\\n✅ Thread limit test passed - all configurations produce identical output');
                app.exit(0);
            }
        } catch (error) {
            console.error('❌ Test error:', error);
            app.exit(1);
        }
    });

    mainWindow.webContents.on('console-message', (event, level, message) => {
        console.log(message);
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
