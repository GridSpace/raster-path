// batch-divisor-benchmark.cjs
// Benchmark test to measure batching overhead with different batch divisors
// Usage: node batch-divisor-benchmark.cjs [divisor1,divisor2,...]
// Example: node batch-divisor-benchmark.cjs 1,2,4,8,16,32

const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');

const OUTPUT_DIR = path.join(__dirname, '../../test-output');
const RESULTS_FILE = path.join(OUTPUT_DIR, 'batch-divisor-results.json');

if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Parse batch divisors from command line args or use defaults
const args = process.argv.slice(2);
const BATCH_DIVISORS = args.length > 0
    ? args[0].split(',').map(n => parseInt(n.trim()))
    : [1, 2, 4, 8, 16, 32];

console.log('=== Batch Divisor Benchmark ===');
console.log('Testing with divisors:', BATCH_DIVISORS.join(', '));
console.log('');

let mainWindow;
let currentDivisorIndex = 0;
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
        console.log('✓ Page loaded');
        await runNextTest();
    });

    // Capture console output from renderer process
    mainWindow.webContents.on('console-message', (event, level, message) => {
        // Filter for our timing logs
        if (message.includes('Batch') && message.includes('timing:')) {
            console.log('[TIMING]', message);
        }
    });
}

async function runNextTest() {
    if (currentDivisorIndex >= BATCH_DIVISORS.length) {
        // All tests complete - analyze and report
        await analyzeResults();
        return;
    }

    const divisor = BATCH_DIVISORS[currentDivisorIndex];
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Testing with BATCH_DIVISOR = ${divisor}`);
    console.log('='.repeat(60));

    const testScript = `
        (async function() {
            const divisor = ${divisor};

            if (!navigator.gpu) {
                return { error: 'WebGPU not available' };
            }

            // Import RasterPath
            const { RasterPath } = await import('./raster-path.js');

            // Load STL files (same as radial-test.cjs - large enough to require batching)
            const terrainResponse = await fetch('../benchmark/fixtures/terrain.stl');
            const terrainBuffer = await terrainResponse.arrayBuffer();

            const toolResponse = await fetch('../benchmark/fixtures/tool.stl');
            const toolBuffer = await toolResponse.arrayBuffer();

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

            // Test parameters - configured to require batching
            const resolution = 0.05; // 0.05mm - finer resolution to force batching
            const rotationStep = 1.0; // 1 degree between rays = 360 angles
            const xStep = 1;
            const yStep = 1;
            const zFloor = 0;
            const radiusOffset = 20;

            console.log('Test parameters:');
            console.log('  Resolution:', resolution, 'mm');
            console.log('  Rotation step:', rotationStep, '°');
            console.log('  Batch divisor:', divisor);

            // Create RasterPath instance with specific batch divisor
            const raster = new RasterPath({
                mode: 'radial',
                resolution: resolution,
                rotationStep: rotationStep,
                batchDivisor: divisor
            });
            await raster.init();

            // Load tool
            const t0 = performance.now();
            await raster.loadTool({ triangles: toolTriangles });
            const toolTime = performance.now() - t0;

            // Load terrain
            const t1 = performance.now();
            await raster.loadTerrain({ triangles: terrainTriangles, zFloor: zFloor });
            const terrainTime = performance.now() - t1;

            // Generate toolpaths (this is where batching happens)
            const t2 = performance.now();
            const toolpathData = await raster.generateToolpaths({
                xStep: xStep,
                yStep: yStep,
                zFloor: zFloor,
                radiusOffset: radiusOffset
            });
            const toolpathTime = performance.now() - t2;

            // Cleanup
            raster.terminate();

            return {
                success: true,
                divisor: divisor,
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

        console.log(`\nResults for BATCH_DIVISOR = ${divisor}:`);
        console.log(`  Tool load time:      ${result.timing.tool.toFixed(1)}ms`);
        console.log(`  Terrain load time:   ${result.timing.terrain.toFixed(1)}ms`);
        console.log(`  Toolpath time:       ${result.timing.toolpath.toFixed(1)}ms`);
        console.log(`  Total time:          ${result.timing.total.toFixed(1)}ms`);
        console.log(`  Strips generated:    ${result.result.numStrips}`);
        console.log(`  Total points:        ${result.result.totalPoints}`);

        // Move to next test
        currentDivisorIndex++;

        // Small delay before next test to ensure clean state
        setTimeout(() => runNextTest(), 1000);

    } catch (error) {
        console.error('Error running test:', error);
        app.exit(1);
    }
}

async function analyzeResults() {
    console.log('\n' + '='.repeat(60));
    console.log('ANALYSIS');
    console.log('='.repeat(60));

    // Save raw results
    const resultsData = {
        timestamp: new Date().toISOString(),
        divisors: BATCH_DIVISORS,
        results: results
    };
    fs.writeFileSync(RESULTS_FILE, JSON.stringify(resultsData, null, 2));
    console.log(`\n✓ Raw results saved to: ${RESULTS_FILE}`);

    // Analyze overhead
    console.log('\n--- Timing Comparison ---');
    const baseline = results[0]; // Divisor = 1
    console.log(`\nBaseline (divisor=1): ${baseline.timing.total.toFixed(1)}ms total`);
    console.log(`  Breakdown: ${baseline.timing.tool.toFixed(1)}ms tool + ${baseline.timing.terrain.toFixed(1)}ms terrain + ${baseline.timing.toolpath.toFixed(1)}ms toolpath`);

    console.log('\nOverhead Analysis:');
    console.log('┌──────────┬───────────┬────────────┬──────────────┬──────────────┐');
    console.log('│ Divisor  │ Total (ms)│ vs Baseline│ Overhead (ms)│ Overhead (%) │');
    console.log('├──────────┼───────────┼────────────┼──────────────┼──────────────┤');

    for (const result of results) {
        const overhead = result.timing.total - baseline.timing.total;
        const overheadPercent = ((overhead / baseline.timing.total) * 100);
        const comparison = result.divisor === 1 ? 'baseline' : `+${overhead.toFixed(0)}ms`;

        console.log(
            `│ ${String(result.divisor).padEnd(8)} │ ` +
            `${result.timing.total.toFixed(1).padStart(9)} │ ` +
            `${comparison.padStart(10)} │ ` +
            `${overhead.toFixed(1).padStart(12)} │ ` +
            `${overheadPercent.toFixed(1).padStart(12)}% │`
        );
    }
    console.log('└──────────┴───────────┴────────────┴──────────────┴──────────────┘');

    // Calculate per-batch overhead
    if (results.length > 1) {
        console.log('\n--- Per-Batch Overhead Estimation ---');
        // Assume divisor creates divisor times more batches
        // So divisor=2 creates 2x batches, divisor=4 creates 4x batches, etc.
        for (let i = 1; i < results.length; i++) {
            const result = results[i];
            const extraBatches = result.divisor - 1; // Assuming baseline has 1 effective batch unit
            const overhead = result.timing.total - baseline.timing.total;
            const perBatchOverhead = overhead / (result.divisor - 1);

            console.log(`Divisor ${result.divisor}: ${overhead.toFixed(1)}ms overhead / ${extraBatches} extra batch(es) ≈ ${perBatchOverhead.toFixed(1)}ms per batch boundary`);
        }
    }

    // Recommendations
    console.log('\n--- Recommendations ---');
    const maxResult = results[results.length - 1];
    const maxOverheadPercent = ((maxResult.timing.total - baseline.timing.total) / baseline.timing.total) * 100;

    if (maxOverheadPercent < 15) {
        console.log('✓ LOW OVERHEAD (<15%): Batching overhead is acceptable.');
        console.log('  Focus on other optimizations (shader efficiency, toolpath generation).');
    } else if (maxOverheadPercent < 30) {
        console.log('⚠ MEDIUM OVERHEAD (15-30%): Consider batch size tuning.');
        console.log('  Investigate buffer creation/destruction costs.');
        console.log('  Consider reusing buffers across batches.');
    } else {
        console.log('⚠ HIGH OVERHEAD (>30%): Priority optimization needed!');
        console.log('  Critical to reduce batch overhead before increasing batch count.');
        console.log('  Primary suspects:');
        console.log('    - createReusableToolpathBuffers() per batch');
        console.log('    - destroyReusableToolpathBuffers() per batch');
        console.log('    - GPU context switching between batches');
        console.log('  Recommendation: Implement buffer pooling or batch-level buffer reuse.');
    }

    console.log('\n✅ Benchmark complete!');
    app.exit(0);
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => app.quit());
