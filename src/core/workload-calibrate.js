/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Workload Calibrate - GPU Watchdog Limit Detection
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Calibrates GPU capabilities by testing workgroup configurations until
 * watchdog kills are detected. Uses actual ray-triangle intersection code
 * to simulate real workload characteristics.
 *
 * EXPORTS:
 * ────────
 * Functions:
 *   - calibrateGPU(device, options) - Run calibration and return limits
 *
 * DETECTION STRATEGY:
 * ───────────────────
 * Problem: Watchdog kills are SILENT - threads just stop executing
 * Solution: Initialize output buffer to zeros, each thread writes 1 on completion
 * Detection: Any zeros remaining = that thread was killed by watchdog
 *
 * TEST PARAMETERS:
 * ────────────────
 * 1. Workgroup dimensions (x, y, z)
 *    - Start: 16x16x1 (256 threads)
 *    - Increase: 32x32x1, 64x64x1, etc.
 *
 * 2. Work intensity (triangle_tests)
 *    - Start: 1000 intersection tests per thread
 *    - Increase: 2000, 5000, 10000, etc.
 *
 * BINARY SEARCH:
 * ──────────────
 * For each workgroup size:
 *   1. Start with low work intensity (known to pass)
 *   2. Binary search for max intensity before watchdog kill
 *   3. Record (workgroup_size, max_intensity) pair
 *
 * OUTPUT:
 * ───────
 * {
 *   maxWorkgroupSize: { x: 64, y: 64, z: 1 },    // Largest safe config
 *   maxWorkPerThread: 50000,                      // Max intersection tests
 *   safeWorkloadMatrix: [                         // Safe configs tested
 *     { workgroupSize: [16,16,1], maxWork: 100000, timingMs: 45 },
 *     { workgroupSize: [32,32,1], maxWork: 50000, timingMs: 123 },
 *     ...
 *   ],
 *   deviceInfo: {
 *     maxComputeWorkgroupSizeX: 256,
 *     maxComputeWorkgroupsPerDimension: 65535,
 *     ...
 *   }
 * }
 *
 * USAGE:
 * ──────
 * const limits = await calibrateGPU(device, {
 *   minWorkgroupSize: [8, 8, 1],
 *   maxWorkgroupSize: [64, 64, 1],
 *   minWork: 1000,
 *   maxWork: 100000,
 * });
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

const calibrateShaderCode = 'SHADER:workload-calibrate';

// Test multiple workgroup dispatches (simulates real-world usage)
async function testWorkloadDispatch(device, pipeline, workgroupSize, triangleTests, dispatchCount) {
    const [x, y, z] = workgroupSize;
    const threadsPerWorkgroup = x * y * z;
    const totalThreads = threadsPerWorkgroup * dispatchCount;

    // Create completion flags buffer for ALL workgroups
    const completionBuffer = device.createBuffer({
        size: totalThreads * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Initialize to zeros
    const zeroData = new Uint32Array(totalThreads);
    device.queue.writeBuffer(completionBuffer, 0, zeroData);

    // Create uniforms
    const uniformData = new Uint32Array([x, y, z, triangleTests]);
    const uniformBuffer = device.createBuffer({
        size: uniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    await device.queue.onSubmittedWorkDone();

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: completionBuffer } },
            { binding: 1, resource: { buffer: uniformBuffer } },
        ],
    });

    // Dispatch multiple workgroups
    const startTime = performance.now();
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);

    // Dispatch NxN workgroups (e.g., 10×10 = 100 workgroups)
    const dispatchX = Math.ceil(Math.sqrt(dispatchCount));
    const dispatchY = Math.ceil(dispatchCount / dispatchX);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY, 1);
    passEncoder.end();

    // Readback
    const stagingBuffer = device.createBuffer({
        size: totalThreads * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    commandEncoder.copyBufferToBuffer(completionBuffer, 0, stagingBuffer, 0, totalThreads * 4);
    device.queue.submit([commandEncoder.finish()]);

    await device.queue.onSubmittedWorkDone();
    const elapsed = performance.now() - startTime;

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const completionData = new Uint32Array(stagingBuffer.getMappedRange());
    const completionCopy = new Uint32Array(completionData);
    stagingBuffer.unmap();

    // Check for failures
    let failedThreads = 0;
    for (let i = 0; i < totalThreads; i++) {
        if (completionCopy[i] === 0) {
            failedThreads++;
        }
    }

    // Cleanup
    completionBuffer.destroy();
    uniformBuffer.destroy();
    stagingBuffer.destroy();

    return {
        success: failedThreads === 0,
        failedThreads,
        totalThreads,
        dispatchCount,
        elapsed,
    };
}

// Test a specific workload configuration (single workgroup)
async function testWorkload(device, pipeline, workgroupSize, triangleTests) {
    const [x, y, z] = workgroupSize;
    const totalThreads = x * y * z;

    // Create completion flags buffer (initialized to zeros)
    const completionBuffer = device.createBuffer({
        size: totalThreads * 4, // u32 per thread
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Initialize to zeros (so we can detect threads that never completed)
    const zeroData = new Uint32Array(totalThreads);
    device.queue.writeBuffer(completionBuffer, 0, zeroData);

    // Create uniforms
    const uniformData = new Uint32Array([x, y, z, triangleTests]);
    const uniformBuffer = device.createBuffer({
        size: uniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Wait for writes to complete
    await device.queue.onSubmittedWorkDone();

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: completionBuffer } },
            { binding: 1, resource: { buffer: uniformBuffer } },
        ],
    });

    // Dispatch compute shader
    const startTime = performance.now();
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);

    // Dispatch exactly 1 workgroup (16x16x1 = 256 threads by default)
    // The shader itself is parameterized with the workgroup size to test
    passEncoder.dispatchWorkgroups(1, 1, 1);
    passEncoder.end();

    // Create staging buffer for readback
    const stagingBuffer = device.createBuffer({
        size: totalThreads * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    commandEncoder.copyBufferToBuffer(completionBuffer, 0, stagingBuffer, 0, totalThreads * 4);
    device.queue.submit([commandEncoder.finish()]);

    // Wait for GPU to complete
    await device.queue.onSubmittedWorkDone();
    const elapsed = performance.now() - startTime;

    // Read back completion flags
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const completionData = new Uint32Array(stagingBuffer.getMappedRange());
    const completionCopy = new Uint32Array(completionData);
    stagingBuffer.unmap();

    // Check for failures (any zeros = thread didn't complete)
    let failedThreads = 0;
    for (let i = 0; i < totalThreads; i++) {
        if (completionCopy[i] === 0) {
            failedThreads++;
        }
    }

    // Cleanup
    completionBuffer.destroy();
    uniformBuffer.destroy();
    stagingBuffer.destroy();

    return {
        success: failedThreads === 0,
        failedThreads,
        totalThreads,
        elapsed,
    };
}

// Binary search for max work intensity at a given workgroup size
async function findMaxWork(device, pipeline, workgroupSize, minWork, maxWork) {
    let low = minWork;
    let high = maxWork;
    let lastSuccess = minWork;

    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        const result = await testWorkload(device, pipeline, workgroupSize, mid);

        if (result.success) {
            lastSuccess = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return lastSuccess;
}

// Calibrate dispatch count limits (how many workgroups can be queued)
export async function calibrateDispatchLimits(device, options = {}) {
    const {
        workgroupSize = [16, 16, 1],  // Use known-good size
        triangleTests = 10000,         // Moderate work per thread
        minDispatch = 1,
        maxDispatch = 100000,          // Test up to 100k workgroups
        verbose = true,
    } = options;

    const shaderModule = device.createShaderModule({ code: calibrateShaderCode });
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'main' },
    });

    if (verbose) {
        console.log('[Calibrate] Testing dispatch count limits...');
        console.log(`[Calibrate] Workgroup size: ${workgroupSize.join('x')}, work: ${triangleTests} tests/thread`);
    }

    // Binary search for max dispatch count
    let low = minDispatch;
    let high = maxDispatch;
    let lastSuccess = minDispatch;
    const results = [];

    while (low <= high) {
        const mid = Math.floor((low + high) / 2);

        if (verbose) {
            console.log(`[Calibrate] Testing ${mid} workgroups...`);
        }

        const result = await testWorkloadDispatch(device, pipeline, workgroupSize, triangleTests, mid);
        results.push({ ...result, dispatchCount: mid });

        if (result.success) {
            lastSuccess = mid;
            if (verbose) {
                const totalThreads = result.totalThreads.toLocaleString();
                const totalWork = (result.totalThreads * triangleTests).toLocaleString();
                console.log(`[Calibrate]   ✓ ${mid} workgroups OK (${totalThreads} threads, ${totalWork} total tests) in ${result.elapsed.toFixed(1)}ms`);
            }
            low = mid + 1;
        } else {
            if (verbose) {
                console.log(`[Calibrate]   ❌ ${mid} workgroups FAILED (${result.failedThreads}/${result.totalThreads} threads killed)`);
            }
            high = mid - 1;
        }
    }

    if (verbose) {
        console.log(`[Calibrate] Max safe dispatch count: ${lastSuccess} workgroups`);
    }

    return {
        maxSafeDispatchCount: lastSuccess,
        workgroupSize,
        triangleTests,
        results,
    };
}

// Main calibration function
export async function calibrateGPU(device, options = {}) {
    const {
        workgroupSizes = [
            [8, 8, 1],
            [16, 16, 1],
            [32, 32, 1],
            [64, 64, 1],
        ],
        minWork = 1000,
        maxWork = 100000,
        verbose = true,
    } = options;

    // Compile calibration shader
    const shaderModule = device.createShaderModule({ code: calibrateShaderCode });
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'main' },
    });

    const results = [];

    if (verbose) {
        console.log('[Calibrate] Starting GPU calibration...');
        console.log('[Calibrate] Testing workgroup sizes:', workgroupSizes);
    }

    for (const size of workgroupSizes) {
        const [x, y, z] = size;
        const totalThreads = x * y * z;

        if (verbose) {
            console.log(`[Calibrate] Testing ${x}x${y}x${z} (${totalThreads} threads)...`);
        }

        // First, verify minimal work succeeds
        const minTest = await testWorkload(device, pipeline, size, minWork);
        if (!minTest.success) {
            if (verbose) {
                console.log(`[Calibrate]   ❌ Failed even at minimum work (${minWork} tests)`);
            }
            break; // This workgroup size is too large
        }

        // Binary search for maximum work
        const maxWorkFound = await findMaxWork(device, pipeline, size, minWork, maxWork);
        const finalTest = await testWorkload(device, pipeline, size, maxWorkFound);

        results.push({
            workgroupSize: size,
            totalThreads,
            maxWork: maxWorkFound,
            timingMs: finalTest.elapsed,
            msPerThread: finalTest.elapsed / totalThreads,
            testsPerSecond: (maxWorkFound * totalThreads) / (finalTest.elapsed / 1000),
        });

        if (verbose) {
            console.log(`[Calibrate]   ✓ Max work: ${maxWorkFound} tests (${finalTest.elapsed.toFixed(1)}ms)`);
            console.log(`[Calibrate]     ${(maxWorkFound * totalThreads).toLocaleString()} total ray-triangle tests`);
        }
    }

    // Determine overall limits
    const maxWorkgroupResult = results[results.length - 1];
    const minWorkPerThread = Math.min(...results.map(r => r.maxWork));

    const calibration = {
        maxWorkgroupSize: maxWorkgroupResult.workgroupSize,
        maxWorkPerThread: minWorkPerThread, // Conservative: min across all sizes
        safeWorkloadMatrix: results,
        deviceInfo: {
            maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
            maxComputeWorkgroupSizeY: device.limits.maxComputeWorkgroupSizeY,
            maxComputeWorkgroupSizeZ: device.limits.maxComputeWorkgroupSizeZ,
            maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
        },
    };

    if (verbose) {
        console.log('[Calibrate] Calibration complete:');
        console.log(`[Calibrate]   Max safe workgroup: ${maxWorkgroupResult.workgroupSize.join('x')}`);
        console.log(`[Calibrate]   Max work per thread: ${minWorkPerThread.toLocaleString()}`);
    }

    return calibration;
}
