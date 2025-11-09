// lathe-cylinder-2-debug.cjs
// Debug test for lathe-cylinder-2.stl with specific parameters
// Run multiple times to check for non-deterministic behavior

const { app, BrowserWindow } = require('electron');
const path = require('path');

const TEST_ITERATIONS = 5; // Run test 5 times to check consistency

const TEST_CONFIG = {
    model: '../benchmark/fixtures/lathe-cylinder-2.stl',
    tool: '../benchmark/fixtures/tool.stl',
    resolution: 0.05,
    rotationStep: 1.0,
    toolDiameter: 5.0, // Scale tool to 5mm
    xStep: 1,
    yStep: 1
};

console.log('=== Lathe Cylinder 2 Debug Test ===');
console.log('Testing for non-deterministic behavior / missing toolpaths');
console.log(`Model: lathe-cylinder-2.stl`);
console.log(`Resolution: ${TEST_CONFIG.resolution}mm`);
console.log(`Tool size: ${TEST_CONFIG.toolDiameter}mm`);
console.log(`Rotation step: ${TEST_CONFIG.rotationStep}°`);
console.log(`XY steps: ${TEST_CONFIG.xStep}`);
console.log(`Iterations: ${TEST_ITERATIONS}`);
console.log('');

let win;
let testResults = [];
let currentIteration = 0;

app.whenReady().then(async () => {
    win = new BrowserWindow({
        width: 1200,
        height: 800,
        show: false,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true
        }
    });

    // Capture console messages
    win.webContents.on('console-message', (event, level, message) => {
        console.log(message);
    });

    await win.loadFile(path.join(__dirname, '../../build/index.html'));
    console.log('✓ Page loaded\n');

    // Run test multiple times
    await runNextIteration();
});

async function runNextIteration() {
    if (currentIteration >= TEST_ITERATIONS) {
        // All iterations complete - analyze results
        analyzeResults();
        app.quit();
        return;
    }

    currentIteration++;
    console.log('======================================================================');
    console.log(`ITERATION ${currentIteration} of ${TEST_ITERATIONS}`);
    console.log('======================================================================');

    try {
        const result = await runTest();
        testResults.push(result);

        // Wait a bit between iterations
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Run next iteration
        await runNextIteration();
    } catch (error) {
        console.error(`✗ Iteration ${currentIteration} failed:`, error);
        testResults.push({
            iteration: currentIteration,
            success: false,
            error: error.message
        });

        // Continue with next iteration even on failure
        await new Promise(resolve => setTimeout(resolve, 1000));
        await runNextIteration();
    }
}

async function runTest() {
    const result = await win.webContents.executeJavaScript(`
        (async () => {
            const startTime = performance.now();

            // Load model
            const modelResponse = await fetch('${TEST_CONFIG.model}');
            const modelBuffer = await modelResponse.arrayBuffer();

            // Parse model triangles
            function parseSTL(arrayBuffer) {
                const view = new DataView(arrayBuffer);
                const numTriangles = view.getUint32(80, true);
                const triangles = new Float32Array(numTriangles * 9);

                let offset = 84;
                for (let i = 0; i < numTriangles; i++) {
                    offset += 12; // Skip normal
                    for (let v = 0; v < 3; v++) {
                        triangles[i * 9 + v * 3 + 0] = view.getFloat32(offset + 0, true);
                        triangles[i * 9 + v * 3 + 1] = view.getFloat32(offset + 4, true);
                        triangles[i * 9 + v * 3 + 2] = view.getFloat32(offset + 8, true);
                        offset += 12;
                    }
                    offset += 2; // Skip attribute
                }

                return triangles;
            }

            const modelTriangles = parseSTL(modelBuffer);

            // Load and scale tool
            const toolResponse = await fetch('${TEST_CONFIG.tool}');
            const toolBuffer = await toolResponse.arrayBuffer();
            const toolTriangles = parseSTL(toolBuffer);

            // Scale tool to target diameter (${TEST_CONFIG.toolDiameter}mm)
            function calculateToolDiameter(triangles) {
                let minX = Infinity, maxX = -Infinity;
                let minY = Infinity, maxY = -Infinity;
                for (let i = 0; i < triangles.length; i += 3) {
                    minX = Math.min(minX, triangles[i]);
                    maxX = Math.max(maxX, triangles[i]);
                    minY = Math.min(minY, triangles[i + 1]);
                    maxY = Math.max(maxY, triangles[i + 1]);
                }
                return Math.max(maxX - minX, maxY - minY);
            }

            const originalDiameter = calculateToolDiameter(toolTriangles);
            const scale = ${TEST_CONFIG.toolDiameter} / originalDiameter;
            const scaledToolTriangles = new Float32Array(toolTriangles.length);
            for (let i = 0; i < toolTriangles.length; i++) {
                scaledToolTriangles[i] = toolTriangles[i] * scale;
            }

            // Import worker
            const { RasterPath } = await import('./raster-path.js');
            const rasterPath = new RasterPath({
                mode: 'radial',
                resolution: ${TEST_CONFIG.resolution},
                rotationStep: ${TEST_CONFIG.rotationStep}
            });

            // Initialize
            await rasterPath.init();

            const toolStart = performance.now();
            await rasterPath.loadTool({ triangles: scaledToolTriangles });
            const toolTime = performance.now() - toolStart;

            const terrainStart = performance.now();
            await rasterPath.loadTerrain({
                triangles: modelTriangles,
                zFloor: 0
            });
            const terrainTime = performance.now() - terrainStart;

            const toolpathStart = performance.now();
            const toolpathResult = await rasterPath.generateToolpaths({
                xStep: ${TEST_CONFIG.xStep},
                yStep: ${TEST_CONFIG.yStep},
                zFloor: 0,
                radiusOffset: 20
            });
            const toolpathTime = performance.now() - toolpathStart;

            const totalTime = performance.now() - startTime;

            // Count strips and points
            let numStrips = 0;
            let totalPoints = 0;
            let minPoints = Infinity;
            let maxPoints = -Infinity;
            let emptyStrips = 0;

            if (toolpathResult) {
                numStrips = toolpathResult.numStrips || 0;
                totalPoints = toolpathResult.totalPoints || 0;

                // If we have the raw strips data, analyze it
                if (toolpathResult.strips) {
                    for (const strip of toolpathResult.strips) {
                        const pointCount = strip.length / 3;
                        if (pointCount === 0) {
                            emptyStrips++;
                        }
                        minPoints = Math.min(minPoints, pointCount);
                        maxPoints = Math.max(maxPoints, pointCount);
                    }
                } else if (toolpathResult.result && toolpathResult.result.strips) {
                    // Alternative format
                    numStrips = toolpathResult.result.strips.length;
                    for (const strip of toolpathResult.result.strips) {
                        const pointCount = strip.length / 3;
                        totalPoints += pointCount;
                        if (pointCount === 0) {
                            emptyStrips++;
                        }
                        minPoints = Math.min(minPoints, pointCount);
                        maxPoints = Math.max(maxPoints, pointCount);
                    }
                }
            }

            return {
                success: true,
                triangleCount: modelTriangles.length / 9,
                timing: {
                    terrain: terrainTime,
                    tool: toolTime,
                    toolpath: toolpathTime,
                    total: totalTime
                },
                result: {
                    numStrips,
                    totalPoints,
                    minPoints: minPoints === Infinity ? 0 : minPoints,
                    maxPoints: maxPoints === -Infinity ? 0 : maxPoints,
                    emptyStrips,
                    avgPointsPerStrip: numStrips > 0 ? (totalPoints / numStrips).toFixed(1) : 0
                }
            };
        })()
    `);

    console.log(`\nResults (Iteration ${currentIteration}):`);
    console.log(`  Triangle count:      ${result.triangleCount.toLocaleString()}`);
    console.log(`  Terrain time:        ${result.timing.terrain.toFixed(1)}ms`);
    console.log(`  Tool time:           ${result.timing.tool.toFixed(1)}ms`);
    console.log(`  Toolpath time:       ${result.timing.toolpath.toFixed(1)}ms`);
    console.log(`  Total time:          ${result.timing.total.toFixed(1)}ms`);
    console.log('');
    console.log('Toolpath Output:');
    console.log(`  Strips:              ${result.result.numStrips}`);
    console.log(`  Total points:        ${result.result.totalPoints.toLocaleString()}`);
    console.log(`  Empty strips:        ${result.result.emptyStrips}`);
    console.log(`  Min points/strip:    ${result.result.minPoints}`);
    console.log(`  Max points/strip:    ${result.result.maxPoints}`);
    console.log(`  Avg points/strip:    ${result.result.avgPointsPerStrip}`);
    console.log('');

    return {
        iteration: currentIteration,
        ...result
    };
}

function analyzeResults() {
    console.log('');
    console.log('======================================================================');
    console.log('CONSISTENCY ANALYSIS');
    console.log('======================================================================');

    const successful = testResults.filter(r => r.success);

    if (successful.length === 0) {
        console.log('✗ All iterations failed!');
        return;
    }

    if (successful.length < TEST_ITERATIONS) {
        console.log(`⚠️  Only ${successful.length} of ${TEST_ITERATIONS} iterations succeeded`);
    }

    // Check consistency of results
    const totalPointsCounts = successful.map(r => r.result.totalPoints);
    const uniqueTotalPoints = [...new Set(totalPointsCounts)];

    const numStripsCounts = successful.map(r => r.result.numStrips);
    const uniqueNumStrips = [...new Set(numStripsCounts)];

    const emptyStripsCounts = successful.map(r => r.result.emptyStrips);
    const uniqueEmptyStrips = [...new Set(emptyStripsCounts)];

    console.log('\nTotal Points Consistency:');
    if (uniqueTotalPoints.length === 1) {
        console.log(`  ✓ All iterations produced ${totalPointsCounts[0].toLocaleString()} points`);
    } else {
        console.log(`  ✗ INCONSISTENT! Got ${uniqueTotalPoints.length} different values:`);
        uniqueTotalPoints.forEach(val => {
            const count = totalPointsCounts.filter(v => v === val).length;
            console.log(`    ${val.toLocaleString()} points: ${count} times`);
        });
    }

    console.log('\nEmpty Strips Consistency:');
    if (uniqueEmptyStrips.length === 1) {
        console.log(`  ${uniqueEmptyStrips[0] === 0 ? '✓' : '⚠️'}  All iterations had ${emptyStripsCounts[0]} empty strips`);
    } else {
        console.log(`  ✗ INCONSISTENT! Got ${uniqueEmptyStrips.length} different values:`);
        uniqueEmptyStrips.forEach(val => {
            const count = emptyStripsCounts.filter(v => v === val).length;
            console.log(`    ${val} empty strips: ${count} times`);
        });
    }

    // Timing statistics
    const toolpathTimes = successful.map(r => r.timing.toolpath);
    const avgTime = toolpathTimes.reduce((a, b) => a + b, 0) / toolpathTimes.length;
    const minTime = Math.min(...toolpathTimes);
    const maxTime = Math.max(...toolpathTimes);
    const stdDev = Math.sqrt(
        toolpathTimes.reduce((sum, t) => sum + Math.pow(t - avgTime, 2), 0) / toolpathTimes.length
    );

    console.log('\nToolpath Timing Statistics:');
    console.log(`  Average: ${avgTime.toFixed(1)}ms`);
    console.log(`  Min:     ${minTime.toFixed(1)}ms`);
    console.log(`  Max:     ${maxTime.toFixed(1)}ms`);
    console.log(`  StdDev:  ${stdDev.toFixed(1)}ms (${(stdDev / avgTime * 100).toFixed(1)}%)`);

    // Overall verdict
    console.log('');
    if (uniqueTotalPoints.length === 1 && uniqueEmptyStrips[0] === 0) {
        console.log('✅ RESULTS ARE CONSISTENT - No non-deterministic behavior detected');
    } else {
        console.log('🛑 RESULTS ARE INCONSISTENT - Non-deterministic behavior detected!');
        console.log('   This suggests workgroups may be getting killed or timing out.');
    }
}
