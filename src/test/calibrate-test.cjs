// calibrate-test.cjs
// Test GPU workload calibration

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
                console.log('=== GPU Workload Calibration Test ===');

                if (!navigator.gpu) {
                    return { error: 'WebGPU not available' };
                }
                console.log('✓ WebGPU available');

                // Import RasterPath
                const { RasterPath } = await import('./raster-path.js');

                // Create RasterPath instance (initializes worker)
                console.log('\\nInitializing worker...');
                const raster = new RasterPath({ mode: 'planar', resolution: 0.1 });
                await raster.init();
                console.log('✓ Worker initialized');

                // Send calibration request
                console.log('\\nRunning GPU dispatch count calibration...');
                console.log('This will test how many workgroups can be dispatched simultaneously.');

                const startTime = performance.now();

                // Send calibrate message to worker
                const calibrationPromise = new Promise((resolve, reject) => {
                    const handler = raster.worker.onmessage;
                    raster.worker.onmessage = (e) => {
                        if (e.data.type === 'calibrate-complete') {
                            resolve(e.data.data);
                        } else if (e.data.type === 'error') {
                            reject(new Error(e.data.message));
                        } else {
                            handler(e); // Pass through other messages
                        }
                    };
                });

                raster.worker.postMessage({
                    type: 'calibrate',
                    data: {
                        calibrationType: 'dispatch',
                        options: {
                            workgroupSize: [4, 4, 1],    // VERY SMALL workgroup (16 threads)
                            triangleTests: 1000,
                            minDispatch: 1,
                            maxDispatch: 1000,
                            verbose: true,
                        }
                    }
                });

                const results = await calibrationPromise;
                const elapsed = performance.now() - startTime;

                console.log('\\n✓ Calibration complete in', elapsed.toFixed(0) + 'ms');
                console.log('\\n=== Results ===');
                console.log('Max safe dispatch count:', results.maxSafeDispatchCount.toLocaleString());
                console.log('Workgroup size:', results.workgroupSize.join('x'));
                console.log('Triangle tests per thread:', results.triangleTests.toLocaleString());

                const maxThreads = results.maxSafeDispatchCount * results.workgroupSize[0] * results.workgroupSize[1] * results.workgroupSize[2];
                const maxTests = maxThreads * results.triangleTests;
                console.log('\\nMax concurrent threads:', maxThreads.toLocaleString());
                console.log('Max total ray tests:', maxTests.toLocaleString());

                console.log('\\n=== Dispatch Test Results ===');
                for (const entry of results.results) {
                    const status = entry.success ? '✓' : '❌';
                    const threads = entry.totalThreads.toLocaleString();
                    const time = entry.elapsed.toFixed(1);
                    const failed = entry.failedThreads > 0 ? \` (\${entry.failedThreads} failed)\` : '';
                    console.log(\`  \${status} \${entry.dispatchCount.toString().padStart(6)} workgroups: \${threads.padStart(10)} threads in \${time.padStart(7)}ms\${failed}\`);
                }

                return { success: true, results };
            })();
        `;

        try {
            const result = await mainWindow.webContents.executeJavaScript(testScript);

            if (result.error) {
                console.error('❌ Test failed:', result.error);
                app.exit(1);
            } else if (!result.success) {
                console.error('❌ Test returned unsuccessful result');
                app.exit(1);
            } else {
                console.log('\n✅ Calibration test complete');
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
