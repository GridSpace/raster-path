# GPU Workload Calibration Results

**System**: macOS / Electron / WebGPU
**Date**: 2025-11-10
**Test Method**: Binary search with completion flag detection

## Executive Summary

This GPU has severe concurrent execution limits that affect production workloads:

1. **Workgroup Dimension Limit**: Any dimension ≥32 fails (max(X,Y,Z) ≤ 16)
2. **Total Thread Limit**: ~256-384 concurrent threads across all workgroups
3. **Production Impact**: Batched radial rasterization exceeds these limits

## Test 1: Workgroup Dimension Limits

**Method**: Single workgroup dispatch with varying dimensions and work intensity

### Key Findings:

- **16×16×1 (256 threads)**: Can handle 1 billion tests/thread ✓
- **32×32×1 (1024 threads)**: Fails even at minimum work ❌
- **32×8×1 (256 threads)**: Fails despite same thread count as 16×16×1 ❌

**Conclusion**: Limit is **max(X, Y, Z) ≤ 16**, not total thread count.

## Test 2: Dispatch Count Limits

**Method**: Multiple workgroups dispatched simultaneously with moderate work (1000 tests/thread)

### Results:

| Workgroup Size | Threads/WG | Max Workgroups | Total Threads | Status |
|----------------|------------|----------------|---------------|--------|
| 16×16×1        | 256        | 1              | 256           | ✓      |
| 8×8×1          | 64         | 4              | 256           | ✓      |
| 4×4×1          | 16         | 24             | 384           | ✓      |

**Conclusion**: Limit is **~256-384 total concurrent threads** across all workgroups in a single dispatch.

### Detailed Test Output (4×4×1):

```
✓     24 workgroups:        384 threads in     2.8ms
❌     25 workgroups:        400 threads in     2.7ms (4 failed)
❌     27 workgroups:        432 threads in     2.7ms (20 failed)
```

The limit appears to be around 384-400 threads total.

## Production Implications

### Current Radial Rasterization Code:

```javascript
// path-radial.js - Current batching strategy
const anglesPerBatch = 360;  // Process 360 angles at once
passEncoder.dispatchWorkgroups(angleDispatchSize, bucketCount, 1);
```

**Problem**: For a model with 100 X-buckets and 1 batch (360 angles):
- Dispatches: 360 workgroups × 100 = 36,000 workgroups
- Threads: 36,000 × 256 = 9,216,000 concurrent threads
- **This exceeds the ~384 thread limit by 24,000×**

### Why Tests Sometimes Pass:

The observed "flaky" behavior where re-running produces different results is likely due to:
1. GPU thermal throttling
2. OS scheduler preemption
3. Other GPU workloads interfering
4. Watchdog timer variance

## Recommendations

### 1. Serialize Dispatches

Instead of dispatching all workgroups at once, break into chunks:

```javascript
// Process in batches of workgroups that stay under thread limit
const MAX_CONCURRENT_THREADS = 256;  // Conservative
const WORKGROUP_SIZE = 256;
const maxWorkgroupsPerDispatch = Math.floor(MAX_CONCURRENT_THREADS / WORKGROUP_SIZE);

for (let batch = 0; batch < totalWorkgroups; batch += maxWorkgroupsPerDispatch) {
    const count = Math.min(maxWorkgroupsPerDispatch, totalWorkgroups - batch);
    passEncoder.dispatchWorkgroups(count, 1, 1);
}
```

### 2. Use Smaller Workgroups

Reduce from 16×16×1 to 8×8×1 to allow 4× more concurrent dispatches:

```javascript
// Current: 16×16×1 = 256 threads, max 1 workgroup
// Better:  8×8×1 = 64 threads, max 4 workgroups
@compute @workgroup_size(8, 8, 1)
```

### 3. Integrate Calibration into RasterPath

Add auto-calibration as config parameter:

```javascript
const raster = new RasterPath({
    mode: 'radial',
    autoCalibrate: true,  // NEW: Run calibration on init
    resolution: 0.1,
});

await raster.init();
// Calibration runs automatically, stores results in localStorage
```

### 4. Use Calibration Data for Batching

```javascript
// After calibration, use detected limits
const { maxConcurrentThreads, maxWorkgroupSize } = raster.gpuLimits;

const threadsPerWorkgroup = maxWorkgroupSize[0] * maxWorkgroupSize[1];
const maxWorkgroupsPerDispatch = Math.floor(maxConcurrentThreads / threadsPerWorkgroup);

// Batch angle processing to stay under limit
const anglesPerBatch = Math.min(360, maxWorkgroupsPerDispatch);
```

## Detection Method

**Problem**: GPU watchdog kills are silent - threads just stop executing.

**Solution**: Completion flag detection
1. Initialize output buffer to all zeros
2. Each thread writes 1 on completion
3. Any remaining zeros = that thread was killed by watchdog

```wgsl
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // ... do work ...

    // Write completion flag
    completion_flags[thread_index] = 1u;
}
```

```javascript
// CPU side: check for failures
for (let i = 0; i < totalThreads; i++) {
    if (completionData[i] === 0) {
        failedThreads++;  // This thread was killed
    }
}
```

## Next Steps

1. **Integrate calibration into RasterPath API** (src/core/raster-path.js)
2. **Add localStorage caching** to avoid re-running calibration
3. **Update radial batching** to use calibrated limits (path-radial.js)
4. **Add fallback behavior** for systems with extreme limits
5. **Test on multiple GPUs** to gather more data points

## Test Reproducibility

Run calibration tests:

```bash
# Workgroup dimension limits
npm run test:calibrate

# Dispatch count limits
npm run test:calibrate
```

Edit `src/test/calibrate-test.cjs` to configure:
- `calibrationType`: 'workgroup' or 'dispatch'
- `workgroupSize`: [X, Y, Z]
- `triangleTests`: Work intensity per thread
- `maxDispatch`: Maximum workgroups to test

## Device Info

```
maxComputeWorkgroupSizeX: 256
maxComputeWorkgroupSizeY: 256
maxComputeWorkgroupSizeZ: 64
maxComputeWorkgroupsPerDimension: 65535
```

Note: Hardware reports support for much larger workgroups (256×256), but watchdog limits are far more restrictive in practice.
