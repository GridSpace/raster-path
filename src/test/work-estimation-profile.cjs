// work-estimation-profile.cjs
// Profile different models to determine work estimation heuristics for optimal batch sizing
// Tests multiple models at different batch divisors to find inflection points

const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');

const OUTPUT_DIR = path.join(__dirname, '../../test-output');
const RESULTS_FILE = path.join(OUTPUT_DIR, 'tool-diameter-scaling.json');

if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Tool scale factors to test (baseline is 5mm radius = 10mm diameter)
// Scale factors: 0.2 → 2mm diameter, 0.4 → 4mm diameter, 0.6 → 6mm diameter, 0.8 → 8mm diameter, 1.0 → 10mm diameter
// Only test tool scaling with divisor=1 to measure tool diameter impact on toolpath generation
const TOOL_SCALES = [0.2, 0.4, 0.6, 0.8, 1.0];

// Test configurations - use divisor=1 only for tool scaling tests
const TEST_CONFIGS = [
    {
        name: 'terrain',
        file: '../benchmark/fixtures/terrain.stl',
        resolution: 0.05,
        rotationStep: 1.0,
        divisors: [1], // Only baseline for tool diameter testing
        expectedTriangles: 75586
    },
    {
        name: 'lathe-cylinder',
        file: '../benchmark/fixtures/lathe-cylinder.stl',
        resolution: 0.05,
        rotationStep: 1.0,
        divisors: [1],
        expectedTriangles: 90620
    },
    {
        name: 'lathe-cylinder-2',
        file: '../benchmark/fixtures/lathe-cylinder-2.stl',
        resolution: 0.05,
        rotationStep: 1.0,
        divisors: [1],
        expectedTriangles: 144890
    },
    {
        name: 'lathe-torture',
        file: '../benchmark/fixtures/lathe-torture.stl',
        resolution: 0.05,
        rotationStep: 1.0,
        divisors: [1],
        expectedTriangles: 1491718
    }
];

console.log('=== Work Estimation Profiling with Tool Diameter Scaling ===');
console.log(`Testing ${TEST_CONFIGS.length} models × ${TOOL_SCALES.length} tool scales`);
console.log(`Tool diameters: ${TOOL_SCALES.map(s => `${(s * 10).toFixed(1)}mm`).join(', ')}`);
console.log('');

let mainWindow;
let currentConfigIndex = 0;
let currentDivisorIndex = 0;
let currentToolScaleIndex = 0;
const results = [];

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
        console.log('✓ Page loaded\n');
        await runNextTest();
    });

    // Capture detailed timing logs
    mainWindow.webContents.on('console-message', (event, level, message) => {
        if (message.includes('Batch') && message.includes('timing:')) {
            console.log('[TIMING]', message);
        }
    });
}

async function runNextTest() {
    if (currentConfigIndex >= TEST_CONFIGS.length) {
        // All tests complete
        await analyzeResults();
        return;
    }

    const config = TEST_CONFIGS[currentConfigIndex];
    const divisors = config.divisors;

    if (currentToolScaleIndex >= TOOL_SCALES.length) {
        // Move to next divisor
        currentDivisorIndex++;
        currentToolScaleIndex = 0;
        await runNextTest();
        return;
    }

    if (currentDivisorIndex >= divisors.length) {
        // Move to next model
        currentConfigIndex++;
        currentDivisorIndex = 0;
        currentToolScaleIndex = 0;
        await runNextTest();
        return;
    }

    const divisor = divisors[currentDivisorIndex];
    const toolScale = TOOL_SCALES[currentToolScaleIndex];
    const toolDiameter = (toolScale * 10).toFixed(1);

    console.log(`${'='.repeat(70)}`);
    console.log(`Model: ${config.name} | Divisor: ${divisor} | Tool: ${toolDiameter}mm`);
    console.log('='.repeat(70));

    const testScript = `
        (async function() {
            const config = ${JSON.stringify(config)};
            const divisor = ${divisor};
            const toolScale = ${toolScale};
            const toolDiameter = ${toolDiameter};

            if (!navigator.gpu) {
                return { error: 'WebGPU not available' };
            }

            const { RasterPath } = await import('./raster-path.js');

            // Load STL files
            const terrainResponse = await fetch('${config.file}');
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
            let toolTriangles = parseBinarySTL(toolBuffer);

            // Scale the tool triangles
            if (toolScale !== 1.0) {
                toolTriangles = toolTriangles.map(v => v * toolScale);
            }

            const actualTriangleCount = terrainTriangles.length / 9;
            const toolTriangleCount = toolTriangles.length / 9;

            console.log('Terrain triangles:', actualTriangleCount);
            console.log('Tool triangles:', toolTriangleCount);
            console.log('Tool diameter:', toolDiameter, 'mm (scale:', toolScale, ')');

            // Create RasterPath instance
            const raster = new RasterPath({
                mode: 'radial',
                resolution: config.resolution,
                rotationStep: config.rotationStep,
                batchDivisor: divisor
            });
            await raster.init();

            // Load tool
            const t0 = performance.now();
            await raster.loadTool({ triangles: toolTriangles });
            const toolTime = performance.now() - t0;

            // Load terrain
            const t1 = performance.now();
            await raster.loadTerrain({
                triangles: terrainTriangles,
                zFloor: 0
            });
            const terrainTime = performance.now() - t1;

            // Generate toolpaths (batching happens here)
            const t2 = performance.now();
            const toolpathData = await raster.generateToolpaths({
                xStep: 1,
                yStep: 1,
                zFloor: 0,
                radiusOffset: 20
            });
            const toolpathTime = performance.now() - t2;

            raster.terminate();

            return {
                success: true,
                model: config.name,
                divisor: divisor,
                toolScale: toolScale,
                toolDiameter: toolDiameter,
                triangleCount: actualTriangleCount,
                toolTriangleCount: toolTriangleCount,
                timing: {
                    tool: toolTime,
                    terrain: terrainTime,
                    toolpath: toolpathTime,
                    total: toolTime + terrainTime + toolpathTime
                },
                result: {
                    numStrips: toolpathData.numStrips,
                    totalPoints: toolpathData.totalPoints
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

        results.push(result);

        console.log(`\nResults:`);
        console.log(`  Terrain triangles:   ${result.triangleCount.toLocaleString()}`);
        console.log(`  Tool triangles:      ${result.toolTriangleCount}`);
        console.log(`  Tool diameter:       ${result.toolDiameter}mm`);
        console.log(`  Toolpath time:       ${result.timing.toolpath.toFixed(1)}ms`);
        console.log(`  Total time:          ${result.timing.total.toFixed(1)}ms`);
        console.log(`  Strips generated:    ${result.result.numStrips}`);

        // Move to next test
        currentToolScaleIndex++;
        setTimeout(() => runNextTest(), 500);

    } catch (error) {
        console.error('Error running test:', error);
        app.exit(1);
    }
}

async function analyzeResults() {
    console.log('\n' + '='.repeat(70));
    console.log('WORK ESTIMATION ANALYSIS');
    console.log('='.repeat(70));

    // Save raw results
    const resultsData = {
        timestamp: new Date().toISOString(),
        configs: TEST_CONFIGS,
        results: results
    };
    fs.writeFileSync(RESULTS_FILE, JSON.stringify(resultsData, null, 2));
    console.log(`\n✓ Raw results saved to: ${RESULTS_FILE}\n`);

    // Group by model
    const modelGroups = {};
    for (const result of results) {
        if (!modelGroups[result.model]) {
            modelGroups[result.model] = [];
        }
        modelGroups[result.model].push(result);
    }

    // Analyze each model - focus on tool diameter impact
    console.log('--- Tool Diameter Impact Analysis ---\n');

    const toolDiameterData = [];

    for (const [modelName, modelResults] of Object.entries(modelGroups)) {
        console.log(`\n${modelName.toUpperCase()}`);
        console.log('─'.repeat(70));

        const triangleCount = modelResults[0].triangleCount;
        const totalAngles = modelResults[0].result.numStrips; // 360 for 1° step

        console.log(`Triangle count: ${triangleCount.toLocaleString()}`);
        console.log(`Total angles: ${totalAngles}`);
        console.log(`Resolution: 0.05mm\n`);

        console.log('Tool Ø | Tool Tris | Toolpath (ms) | Total (ms) | Toolpath/Strip');
        console.log('-------|-----------|---------------|------------|---------------');

        const baseline = modelResults[0]; // Smallest tool

        for (const result of modelResults) {
            const toolpathPerStrip = result.timing.toolpath / totalAngles;
            const speedup = baseline.timing.toolpath / result.timing.toolpath;

            console.log(
                `${String(result.toolDiameter + 'mm').padEnd(6)} | ` +
                `${String(result.toolTriangleCount).padStart(9)} | ` +
                `${result.timing.toolpath.toFixed(1).padStart(13)} | ` +
                `${result.timing.total.toFixed(1).padStart(10)} | ` +
                `${toolpathPerStrip.toFixed(2).padStart(14)}ms`
            );

            // Store data for correlation analysis
            toolDiameterData.push({
                model: modelName,
                triangleCount: triangleCount,
                toolDiameter: result.toolDiameter,
                toolTriangleCount: result.toolTriangleCount,
                toolpathTime: result.timing.toolpath,
                toolpathPerStrip: toolpathPerStrip
            });
        }

        console.log('');
    }

    // Correlation analysis - tool diameter vs toolpath time
    console.log('\n' + '='.repeat(70));
    console.log('TOOL DIAMETER CORRELATION ANALYSIS');
    console.log('='.repeat(70));

    // Group by model to analyze scaling
    const modelToolData = {};
    for (const data of toolDiameterData) {
        if (!modelToolData[data.model]) {
            modelToolData[data.model] = [];
        }
        modelToolData[data.model].push(data);
    }

    console.log('\n--- Scaling Relationship Analysis ---\n');

    for (const [modelName, dataPoints] of Object.entries(modelToolData)) {
        console.log(`${modelName.toUpperCase()}:`);

        // Sort by tool diameter
        dataPoints.sort((a, b) => a.toolDiameter - b.toolDiameter);

        const baseline = dataPoints[0];

        console.log(`  Tool Ø vs Time (normalized to ${baseline.toolDiameter}mm baseline):`);

        for (const point of dataPoints) {
            const diameterRatio = point.toolDiameter / baseline.toolDiameter;
            const timeRatio = point.toolpathTime / baseline.toolpathTime;
            const expectedLinear = diameterRatio;
            const expectedSquare = diameterRatio ** 2;

            console.log(`    ${point.toolDiameter.toFixed(1)}mm: ${timeRatio.toFixed(2)}x slower`);
            console.log(`      Diameter ratio: ${diameterRatio.toFixed(2)}x`);
            console.log(`      If linear (Ø):  ${expectedLinear.toFixed(2)}x (error: ${Math.abs(timeRatio - expectedLinear).toFixed(2)})`);
            console.log(`      If square (Ø²): ${expectedSquare.toFixed(2)}x (error: ${Math.abs(timeRatio - expectedSquare).toFixed(2)})`);
        }

        // Calculate correlation coefficient for linear and square relationships
        const diameterRatios = dataPoints.map(p => p.toolDiameter / baseline.toolDiameter);
        const timeRatios = dataPoints.map(p => p.toolpathTime / baseline.toolpathTime);

        // Linear correlation: timeRatio ~ diameterRatio
        const linearErrors = diameterRatios.map((dr, i) => Math.abs(timeRatios[i] - dr));
        const avgLinearError = linearErrors.reduce((sum, e) => sum + e, 0) / linearErrors.length;

        // Square correlation: timeRatio ~ diameterRatio²
        const squareErrors = diameterRatios.map((dr, i) => Math.abs(timeRatios[i] - dr ** 2));
        const avgSquareError = squareErrors.reduce((sum, e) => sum + e, 0) / squareErrors.length;

        console.log(`  Average error: Linear ${avgLinearError.toFixed(3)}, Square ${avgSquareError.toFixed(3)}`);
        console.log(`  ${avgSquareError < avgLinearError ? '✓ Square relationship fits better' : '✓ Linear relationship fits better'}\n`);
    }

    // Summary
    console.log('\n--- Summary ---\n');
    console.log('This test measures the relationship between tool diameter and toolpath');
    console.log('generation time. The hypothesis is that work scales with tool diameter²');
    console.log('because the sparse tool representation has more points to check.');
    console.log('\nThe analysis compares linear (Ø) vs square (Ø²) scaling to validate');
    console.log('or refute this hypothesis.');

    console.log('\n✅ Profiling complete!');
    app.exit(0);
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => app.quit());
