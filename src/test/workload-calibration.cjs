// workload-calibration.cjs
// Comprehensive test matrix to calibrate workload estimation formulas
// Tests: resolution × tool diameter × angular step × model

const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');

const OUTPUT_DIR = path.join(__dirname, '../../test-output');
const RESULTS_FILE = path.join(OUTPUT_DIR, 'workload-calibration.json');
const CSV_FILE = path.join(OUTPUT_DIR, 'workload-calibration.csv');

if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Test matrix dimensions
const RESOLUTIONS = [0.05, 0.04, 0.03, 0.02, 0.01];
const TOOL_SCALES = [0.1, 0.2, 0.3, 0.4, 0.5]; // 1mm, 2mm, 3mm, 4mm, 5mm (base tool is 10mm)
const ANGULAR_STEPS = [2.0, 1.0, 0.5];

// Find all lathe-*.stl models
const MODELS = [
    { name: 'lathe-cylinder', file: '../benchmark/fixtures/lathe-cylinder.stl' },
    { name: 'lathe-cylinder-2', file: '../benchmark/fixtures/lathe-cylinder-2.stl' },
    { name: 'lathe-puck', file: '../benchmark/fixtures/lathe-puck.stl' },
    { name: 'lathe-torture', file: '../benchmark/fixtures/lathe-torture.stl' }
];

const totalTests = MODELS.length * RESOLUTIONS.length * TOOL_SCALES.length * ANGULAR_STEPS.length;

console.log('=== Workload Calibration Test Matrix ===');
console.log(`Models: ${MODELS.length} (${MODELS.map(m => m.name).join(', ')})`);
console.log(`Resolutions: ${RESOLUTIONS.length} (${RESOLUTIONS.join(', ')})`);
console.log(`Tool sizes: ${TOOL_SCALES.length} (${TOOL_SCALES.map(s => (s * 10).toFixed(1) + 'mm').join(', ')})`);
console.log(`Angular steps: ${ANGULAR_STEPS.length} (${ANGULAR_STEPS.join('°, ')}°)`);
console.log(`Total tests: ${totalTests}`);
console.log('');

let mainWindow;
let currentModelIndex = 0;
let currentResolutionIndex = 0;
let currentToolScaleIndex = 0;
let currentAngularStepIndex = 0;
const results = [];
let testNumber = 0;

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
}

async function runNextTest() {
    // Check if all tests complete
    if (currentModelIndex >= MODELS.length) {
        await analyzeResults();
        return;
    }

    // Advance through dimensions
    if (currentAngularStepIndex >= ANGULAR_STEPS.length) {
        currentToolScaleIndex++;
        currentAngularStepIndex = 0;
    }

    if (currentToolScaleIndex >= TOOL_SCALES.length) {
        currentResolutionIndex++;
        currentToolScaleIndex = 0;
        currentAngularStepIndex = 0;
    }

    if (currentResolutionIndex >= RESOLUTIONS.length) {
        currentModelIndex++;
        currentResolutionIndex = 0;
        currentToolScaleIndex = 0;
        currentAngularStepIndex = 0;
        await runNextTest();
        return;
    }

    const model = MODELS[currentModelIndex];
    const resolution = RESOLUTIONS[currentResolutionIndex];
    const toolScale = TOOL_SCALES[currentToolScaleIndex];
    const angularStep = ANGULAR_STEPS[currentAngularStepIndex];
    const toolDiameter = (toolScale * 10).toFixed(1);

    testNumber++;
    const progress = `[${testNumber}/${totalTests}]`;

    console.log(`${progress} ${model.name} | res=${resolution} | tool=${toolDiameter}mm | step=${angularStep}°`);

    const testScript = `
        (async function() {
            const modelFile = '${model.file}';
            const resolution = ${resolution};
            const toolScale = ${toolScale};
            const angularStep = ${angularStep};

            if (!navigator.gpu) {
                return { error: 'WebGPU not available' };
            }

            const { RasterPath } = await import('./raster-path.js');

            // Load STL files
            const terrainResponse = await fetch(modelFile);
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

            // Scale the tool
            if (toolScale !== 1.0) {
                toolTriangles = toolTriangles.map(v => v * toolScale);
            }

            const triangleCount = terrainTriangles.length / 9;
            const toolTriangleCount = toolTriangles.length / 9;

            // Create RasterPath instance
            const raster = new RasterPath({
                mode: 'radial',
                resolution: resolution,
                rotationStep: angularStep
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

            // Generate toolpaths
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
                triangleCount: triangleCount,
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

        // Store result with test parameters
        results.push({
            model: model.name,
            resolution: resolution,
            toolDiameter: parseFloat(toolDiameter),
            angularStep: angularStep,
            triangleCount: result.triangleCount,
            toolTriangleCount: result.toolTriangleCount,
            numStrips: result.result.numStrips,
            timing: result.timing
        });

        console.log(`  ✓ Toolpath: ${result.timing.toolpath.toFixed(0)}ms | Total: ${result.timing.total.toFixed(0)}ms\n`);

        // Move to next test
        currentAngularStepIndex++;
        setTimeout(() => runNextTest(), 100);

    } catch (error) {
        console.error('Error running test:', error);
        app.exit(1);
    }
}

async function analyzeResults() {
    console.log('\n' + '='.repeat(70));
    console.log('WORKLOAD CALIBRATION ANALYSIS');
    console.log('='.repeat(70));

    // Save raw results as JSON
    const resultsData = {
        timestamp: new Date().toISOString(),
        testMatrix: {
            models: MODELS.map(m => m.name),
            resolutions: RESOLUTIONS,
            toolDiameters: TOOL_SCALES.map(s => s * 10),
            angularSteps: ANGULAR_STEPS
        },
        totalTests: totalTests,
        results: results
    };
    fs.writeFileSync(RESULTS_FILE, JSON.stringify(resultsData, null, 2));
    console.log(`\n✓ Raw JSON saved to: ${RESULTS_FILE}`);

    // Generate CSV for easy analysis
    const csvHeader = 'model,resolution,toolDiameter,angularStep,triangleCount,numStrips,toolTime,terrainTime,toolpathTime,totalTime\n';
    const csvRows = results.map(r =>
        `${r.model},${r.resolution},${r.toolDiameter},${r.angularStep},${r.triangleCount},${r.numStrips},` +
        `${r.timing.tool.toFixed(2)},${r.timing.terrain.toFixed(2)},${r.timing.toolpath.toFixed(2)},${r.timing.total.toFixed(2)}`
    ).join('\n');
    fs.writeFileSync(CSV_FILE, csvHeader + csvRows);
    console.log(`✓ CSV saved to: ${CSV_FILE}\n`);

    // Summary statistics
    console.log('--- Summary Statistics ---\n');

    // Group by model
    const modelGroups = {};
    for (const result of results) {
        if (!modelGroups[result.model]) {
            modelGroups[result.model] = [];
        }
        modelGroups[result.model].push(result);
    }

    for (const [modelName, modelResults] of Object.entries(modelGroups)) {
        const times = modelResults.map(r => r.timing.toolpath);
        const avgTime = times.reduce((sum, t) => sum + t, 0) / times.length;
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);

        console.log(`${modelName}:`);
        console.log(`  Tests: ${modelResults.length}`);
        console.log(`  Toolpath time: min=${minTime.toFixed(0)}ms, max=${maxTime.toFixed(0)}ms, avg=${avgTime.toFixed(0)}ms`);
        console.log(`  Range: ${(maxTime / minTime).toFixed(1)}x variation\n`);
    }

    // Correlation hints
    console.log('--- Key Observations ---\n');
    console.log('The CSV file contains all test data for correlation analysis.');
    console.log('Use it to validate workload formulas by comparing predicted vs actual times.');
    console.log('\nVariables available for formula calibration:');
    console.log('  - resolution (inverse square relationship expected)');
    console.log('  - toolDiameter (linear relationship confirmed)');
    console.log('  - angularStep (inverse relationship expected)');
    console.log('  - triangleCount (linear or density-based relationship)');
    console.log('  - numStrips (360 / angularStep)');

    console.log('\n✅ Calibration complete!');
    app.exit(0);
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => app.quit());
