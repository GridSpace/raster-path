// radial-test.cjs
// Regression test for radial mode using new RasterPath API
// Tests: loadTool() + loadTerrain() + generateToolpaths() with radial projection

const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');

const OUTPUT_DIR = path.join(__dirname, '../../test-output');
const BASELINE_FILE = path.join(OUTPUT_DIR, 'radial-baseline.json');
const CURRENT_FILE = path.join(OUTPUT_DIR, 'radial-current.json');

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
                console.log('=== Radial Mode Regression Test ===');

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

                // Test parameters
                const resolution = 0.1; // 0.1mm for radial (coarser than planar)
                const rotationStep = 1.0; // 1 degree between rays
                const xStep = 5; // Sample every 5th point
                const yStep = 5;
                const zFloor = 0;
                const radiusOffset = 20; // Tool offset above terrain surface

                console.log('\\nTest parameters:');
                console.log('  Resolution:', resolution, 'mm');
                console.log('  Rotation step:', rotationStep, '°');
                console.log('  XY step:', xStep + 'x' + yStep, 'points');
                console.log('  Z floor:', zFloor, 'mm');
                console.log('  Radius offset:', radiusOffset, 'mm');

                // Create RasterPath instance for radial mode
                console.log('\\nInitializing RasterPath (radial mode)...');
                const raster = new RasterPath({
                    mode: 'radial',
                    resolution: resolution,
                    rotationStep: rotationStep
                });
                await raster.init();
                console.log('✓ RasterPath initialized');

                // Load tool (NEW API)
                console.log('\\n1. Loading tool (NEW API)...');
                const t0 = performance.now();
                const toolData = await raster.loadTool({
                    triangles: toolTriangles
                });
                const toolTime = performance.now() - t0;
                console.log('✓ Tool:', toolData.pointCount, 'points in', toolTime.toFixed(1), 'ms');

                // Load terrain (NEW API - stores triangles for later)
                console.log('\\n2. Loading terrain (NEW API - radial mode)...');
                const t1 = performance.now();
                await raster.loadTerrain({
                    triangles: terrainTriangles,
                    zFloor: zFloor
                });
                const terrainTime = performance.now() - t1;
                console.log('✓ Terrain loaded (triangles stored, will rasterize during toolpath generation)');
                console.log('  Time:', terrainTime.toFixed(1), 'ms');

                // Generate toolpaths (NEW API - does rasterization + toolpath generation)
                console.log('\\n3. Generating toolpaths (NEW API - radial)...');
                const t2 = performance.now();
                const toolpathData = await raster.generateToolpaths({
                    xStep: xStep,
                    yStep: yStep,
                    zFloor: zFloor,
                    radiusOffset: radiusOffset
                });
                const toolpathTime = performance.now() - t2;
                console.log('✓ Toolpath generated');
                console.log('  Strips:', toolpathData.numStrips);
                console.log('  Total points:', toolpathData.totalPoints);
                console.log('  Generation time:', toolpathTime.toFixed(1), 'ms');

                // Cleanup
                raster.terminate();

                // Calculate checksum for regression detection (NEW API - radial mode uses strips)
                let checksum = 0;
                let totalValues = 0;
                for (const strip of toolpathData.strips) {
                    for (let i = 0; i < strip.pathData.length; i++) {
                        checksum = (checksum + strip.pathData[i] * (totalValues + i + 1)) | 0;
                    }
                    totalValues += strip.pathData.length;
                }

                // Sample first 30 Z-values for debugging (from first strip)
                const firstStrip = toolpathData.strips[0];
                const sampleSize = Math.min(30, firstStrip ? firstStrip.pathData.length : 0);
                const sampleValues = [];
                if (firstStrip) {
                    for (let i = 0; i < sampleSize; i++) {
                        sampleValues.push(firstStrip.pathData[i].toFixed(2));
                    }
                }

                return {
                    success: true,
                    output: {
                        parameters: {
                            mode: 'radial',
                            resolution: resolution,
                            rotationStep: rotationStep,
                            xStep,
                            yStep,
                            zFloor,
                            radiusOffset,
                            terrainTriangles: terrainTriangles.length / 9,
                            toolTriangles: toolTriangles.length / 9
                        },
                        result: {
                            toolPoints: toolData.pointCount,
                            numStrips: toolpathData.numStrips,
                            totalPoints: toolpathData.totalPoints,
                            checksum: checksum,
                            sampleValues: sampleValues
                        },
                        timing: {
                            terrain: terrainTime,
                            tool: toolTime,
                            toolpath: toolpathTime,
                            total: terrainTime + toolTime + toolpathTime
                        }
                    }
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

            // Save current output
            const currentData = {
                parameters: result.output.parameters,
                result: result.output.result,
                timing: result.output.timing
            };

            fs.writeFileSync(CURRENT_FILE, JSON.stringify(currentData, null, 2));
            console.log('\n✓ Saved current output to', CURRENT_FILE);
            console.log(`  Toolpath size: ${result.output.result.toolpathSize} Z-values`);
            console.log(`  Checksum: ${result.output.result.checksum}`);
            console.log(`  Total time: ${result.output.timing.total.toFixed(1)}ms`);

            // Check if baseline exists
            if (!fs.existsSync(BASELINE_FILE)) {
                console.log('\n📝 No baseline found - saving current as baseline');
                fs.writeFileSync(BASELINE_FILE, JSON.stringify(currentData, null, 2));
                console.log('✅ Baseline created');
                app.exit(0);
                return;
            }

            // Compare with baseline
            const baseline = JSON.parse(fs.readFileSync(BASELINE_FILE, 'utf8'));

            console.log('\n=== Comparison ===');
            console.log('Baseline checksum:', baseline.result.checksum);
            console.log('Current checksum:', result.output.result.checksum);

            let passed = true;

            if (baseline.result.toolpathSize !== result.output.result.toolpathSize) {
                console.error('❌ Toolpath size mismatch!');
                console.error(`  Expected: ${baseline.result.toolpathSize}, Got: ${result.output.result.toolpathSize}`);
                passed = false;
            }

            if (baseline.result.checksum !== result.output.result.checksum) {
                console.error('❌ Checksum mismatch!');
                console.error(`  Expected: ${baseline.result.checksum}, Got: ${result.output.result.checksum}`);
                passed = false;
            } else {
                console.log('✓ Checksum matches');
            }

            if (passed) {
                console.log('\n✅ All checks passed - output matches baseline');
                app.exit(0);
            } else {
                console.log('\n❌ Regression detected - output differs from baseline');
                console.log('To update baseline: cp', CURRENT_FILE, BASELINE_FILE);
                app.exit(1);
            }

        } catch (error) {
            console.error('Error running test:', error);
            app.exit(1);
        }
    });
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => app.quit());
