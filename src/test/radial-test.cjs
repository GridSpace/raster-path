// radial-test.cjs
// Regression test for radial mode using new RasterPath API
// Tests: rasterizeModel() + rasterizeTool() + generateToolpaths() with radial projection

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

                // Rasterize model (terrain) with radial projection
                console.log('\\n1. Rasterizing model (radial)...');
                const t0 = performance.now();
                const terrainData = await raster.rasterizeModel({
                    triangles: terrainTriangles,
                    zFloor: zFloor
                });
                const terrainTime = performance.now() - t0;
                console.log('✓ Model rasterized');
                if (terrainData.tiles) {
                    console.log('  Tiles:', terrainData.tiles.length);
                    console.log('  Grid:', terrainData.tiles[0].gridWidth, 'x', terrainData.tiles[0].gridHeight);
                } else {
                    console.log('  Grid:', terrainData.gridWidth, 'x', terrainData.gridHeight);
                }
                console.log('  Circumference:', terrainData.circumference.toFixed(2), 'mm');
                console.log('  Max radius:', terrainData.maxRadius.toFixed(2), 'mm');
                console.log('  Time:', terrainTime.toFixed(1), 'ms');

                // Rasterize tool (planar)
                console.log('\\n2. Rasterizing tool (planar)...');
                const t1 = performance.now();
                const toolData = await raster.rasterizeTool({
                    triangles: toolTriangles
                });
                const toolTime = performance.now() - t1;
                console.log('✓ Tool:', toolData.pointCount, 'points in', toolTime.toFixed(1), 'ms');

                // Generate toolpaths (morphs tool + stitches terrain + generates paths)
                console.log('\\n3. Generating toolpaths (radial)...');
                const t2 = performance.now();
                const toolpathData = await raster.generateToolpaths({
                    terrainData: terrainData,
                    toolData: toolData,
                    xStep: xStep,
                    yStep: yStep,
                    zFloor: zFloor,
                    radiusOffset: radiusOffset
                });
                const toolpathTime = performance.now() - t2;
                console.log('✓ Toolpath:', toolpathData.numScanlines + 'x' + toolpathData.pointsPerLine, '=', toolpathData.pathData.length, 'Z-values');
                console.log('  Generation time:', toolpathTime.toFixed(1), 'ms');

                // Cleanup
                raster.terminate();

                // Calculate checksum for regression detection
                let checksum = 0;
                for (let i = 0; i < toolpathData.pathData.length; i++) {
                    checksum = (checksum + toolpathData.pathData[i] * (i + 1)) | 0;
                }

                // Sample first 30 Z-values for debugging
                const sampleSize = Math.min(30, toolpathData.pathData.length);
                const sampleValues = [];
                for (let i = 0; i < sampleSize; i++) {
                    sampleValues.push(toolpathData.pathData[i].toFixed(2));
                }

                // Calculate total terrain cells for reporting
                let totalTerrainCells = 0;
                let terrainTiles = 0;
                if (terrainData.tiles) {
                    terrainTiles = terrainData.tiles.length;
                    for (const tile of terrainData.tiles) {
                        totalTerrainCells += tile.gridWidth * tile.gridHeight;
                    }
                } else {
                    // Non-tiled dense array
                    totalTerrainCells = terrainData.pointCount;
                    terrainTiles = 1;
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
                            terrainCells: totalTerrainCells,
                            terrainTiles: terrainTiles,
                            circumference: terrainData.circumference,
                            maxRadius: terrainData.maxRadius,
                            toolPoints: toolData.pointCount,
                            toolpathSize: toolpathData.pathData.length,
                            numScanlines: toolpathData.numScanlines,
                            pointsPerLine: toolpathData.pointsPerLine,
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
