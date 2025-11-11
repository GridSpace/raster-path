// extreme-work-test.cjs
// Test if there's ANY per-thread compute limit by pushing to extreme levels

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
                console.log('=== Extreme Per-Thread Work Test ===');
                console.log('Testing if there is ANY limit to work per thread...\\n');

                if (!navigator.gpu) {
                    return { error: 'WebGPU not available' };
                }

                const { RasterPath } = await import('./raster-path.js');

                const raster = new RasterPath({ mode: 'planar', resolution: 0.1 });
                await raster.init();

                // Test progressively larger workloads on a SINGLE 16x16x1 workgroup (256 threads)
                const testLevels = [
                    { tests: 1_000_000_000, label: '1 billion' },
                    { tests: 2_000_000_000, label: '2 billion' },
                    { tests: 5_000_000_000, label: '5 billion' },
                    { tests: 10_000_000_000, label: '10 billion' },
                ];

                const results = [];

                for (const level of testLevels) {
                    console.log(\`Testing \${level.label} tests/thread (\${level.tests.toLocaleString()})...\`);

                    const calibrationPromise = new Promise((resolve, reject) => {
                        const timeout = setTimeout(() => {
                            reject(new Error('Test timed out after 30s'));
                        }, 30000);

                        const handler = raster.worker.onmessage;
                        raster.worker.onmessage = (e) => {
                            if (e.data.type === 'calibrate-complete') {
                                clearTimeout(timeout);
                                resolve(e.data.data);
                            } else if (e.data.type === 'error') {
                                clearTimeout(timeout);
                                reject(new Error(e.data.message));
                            } else {
                                handler(e);
                            }
                        };
                    });

                    const startTime = performance.now();

                    raster.worker.postMessage({
                        type: 'calibrate',
                        data: {
                            calibrationType: 'workgroup',
                            options: {
                                workgroupSizes: [[16, 16, 1]],
                                minWork: level.tests,
                                maxWork: level.tests,
                                verbose: false,
                            }
                        }
                    });

                    try {
                        const result = await calibrationPromise;
                        const elapsed = performance.now() - startTime;
                        const success = result.safeWorkloadMatrix[0]?.maxWork === level.tests;

                        const status = success ? '✓' : '❌';
                        const totalTests = (level.tests * 256).toLocaleString();
                        console.log(\`  \${status} \${level.label}: \${elapsed.toFixed(0)}ms (\${totalTests} total tests)\\n\`);

                        results.push({
                            tests: level.tests,
                            label: level.label,
                            success,
                            elapsed,
                        });

                        if (!success) {
                            console.log('Found failure point - stopping test.');
                            break;
                        }
                    } catch (error) {
                        console.log(\`  ❌ \${level.label}: FAILED - \${error.message}\\n\`);
                        results.push({
                            tests: level.tests,
                            label: level.label,
                            success: false,
                            error: error.message,
                        });
                        break;
                    }
                }

                console.log('\\n=== Summary ===');
                for (const r of results) {
                    const status = r.success ? '✓' : '❌';
                    const time = r.elapsed ? \` in \${r.elapsed.toFixed(0)}ms\` : '';
                    const err = r.error ? \` - \${r.error}\` : '';
                    console.log(\`\${status} \${r.label}\${time}\${err}\`);
                }

                const maxSuccess = results.filter(r => r.success).pop();
                if (maxSuccess) {
                    console.log(\`\\nMax verified work per thread: \${maxSuccess.label} (\${maxSuccess.tests.toLocaleString()} tests)\`);
                    console.log('Conclusion: No practical per-thread compute limit detected');
                } else {
                    console.log('\\nFound per-thread compute limit');
                }

                return { success: true, results };
            })();
        `;

        try {
            const result = await mainWindow.webContents.executeJavaScript(testScript);

            if (result.error) {
                console.error('❌ Test failed:', result.error);
                app.exit(1);
            } else {
                console.log('\n✅ Extreme work test complete');
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
