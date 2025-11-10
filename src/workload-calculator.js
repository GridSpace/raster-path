/**
 * Workload Calculator for Radial Toolpath Generation
 *
 * Estimates GPU workload and optimal batch sizing based on:
 * - Model geometry (triangle count, bounds)
 * - Rasterization parameters (resolution, angular step)
 * - Tool characteristics (diameter)
 *
 * Based on empirical testing:
 * - Tool diameter scales LINEARLY (not quadratically)
 * - Resolution scales INVERSELY SQUARED (finer = more pixels)
 * - Angular step scales INVERSELY LINEAR (finer = more strips)
 * - Triangle count has complex relationship with geometry
 */

export class WorkloadCalculator {
    constructor() {
        // Calibration constants (to be tuned from test data)
        this.BASELINE_TOOL_DIAMETER = 2.5; // mm (most common size)
        this.BASELINE_RESOLUTION = 0.05;    // mm (reference resolution)
        this.BASELINE_ANGULAR_STEP = 1.0;   // degrees

        // Minimum batch parameters
        this.MIN_ANGLES_PER_BATCH = 20;     // Absolute minimum
        this.TARGET_MIN_BATCH_TIME_MS = 400; // Target minimum batch duration

        // GPU overhead constants
        this.GPU_DISPATCH_OVERHEAD_MS = 25; // Per-batch overhead

        // Watchdog timeout thresholds
        this.WATCHDOG_WARNING_MS = 2000;    // Warn if single operation > 2s
        this.WATCHDOG_CRITICAL_MS = 5000;   // Critical if single operation > 5s
        this.WATCHDOG_MAX_SAFE_MS = 1500;   // Target to stay under watchdog
    }

    /**
     * Calculate workload for radial toolpath generation
     *
     * @param {Object} params
     * @param {number} params.triangleCount - Number of terrain triangles
     * @param {Object} params.bounds - Model bounds {minX, maxX, minY, maxY, minZ, maxZ}
     * @param {number} params.resolution - Grid resolution in mm
     * @param {number} params.rotationStep - Angular step in degrees
     * @param {number} params.toolDiameter - Tool diameter in mm
     * @returns {Object} Workload estimate with recommended batch size
     */
    calculateRadialWorkload(params) {
        const {
            triangleCount,
            bounds,
            resolution,
            rotationStep,
            toolDiameter
        } = params;

        // Calculate radial geometry
        const radialRadius = Math.sqrt(
            Math.max(Math.abs(bounds.minY), Math.abs(bounds.maxY)) ** 2 +
            Math.max(Math.abs(bounds.minZ), Math.abs(bounds.maxZ)) ** 2
        );
        const radialHeight = bounds.maxX - bounds.minX;
        const radialVolume = Math.PI * radialRadius * radialRadius * radialHeight;

        // Calculate grid dimensions
        const gridWidth = Math.ceil((2 * radialRadius) / resolution);
        const gridHeight = Math.ceil(radialHeight / resolution);
        const pixelsPerAngle = gridWidth * gridHeight;

        // Calculate number of strips
        const numAngles = Math.ceil(360 / rotationStep);

        // Calculate triangle density
        const triangleDensity = triangleCount / radialVolume;

        // Calculate work factors (normalized to baseline)
        const resolutionFactor = (this.BASELINE_RESOLUTION / resolution) ** 2; // Inverse square
        const angularFactor = this.BASELINE_ANGULAR_STEP / rotationStep;        // Inverse linear
        const toolFactor = toolDiameter / this.BASELINE_TOOL_DIAMETER;          // Linear

        // Formula 6 (User's proposal + tool diameter):
        // work ∝ triangleDensity × (1/resolution²) × (1/angularStep) × toolDiameter
        const workScore = triangleDensity * resolutionFactor * angularFactor * toolFactor;

        // Formula 2 (Simpler):
        // work ∝ triangleCount × pixelsPerAngle × toolDiameter
        const workPerAngle = triangleCount * pixelsPerAngle * toolFactor;

        // Estimate time per angle (calibrated from empirical data)
        // Base calibration: ~3-5ms per angle for baseline parameters
        const baseTimePerAngle = 4.0; // ms (to be tuned from test data)
        const estimatedTimePerAngle = baseTimePerAngle * workScore / 1000;

        // Calculate optimal batch size based on target batch duration
        const optimalAnglesPerBatch = Math.ceil(
            this.TARGET_MIN_BATCH_TIME_MS / estimatedTimePerAngle
        );

        // Apply constraints
        const recommendedAnglesPerBatch = Math.max(
            this.MIN_ANGLES_PER_BATCH,
            Math.min(optimalAnglesPerBatch, numAngles)
        );

        const numBatches = Math.ceil(numAngles / recommendedAnglesPerBatch);

        // Estimate total time
        const rasterTimePerAngle = estimatedTimePerAngle * 0.3; // ~30% raster
        const toolpathTimePerAngle = estimatedTimePerAngle * 0.7; // ~70% toolpath

        const totalRasterTime = rasterTimePerAngle * numAngles;
        const totalToolpathTime = toolpathTimePerAngle * numAngles;
        const totalOverhead = this.GPU_DISPATCH_OVERHEAD_MS * numBatches;
        const estimatedTotalTime = totalRasterTime + totalToolpathTime + totalOverhead;

        // Estimate time per batch (for watchdog detection)
        const estimatedBatchTime = (estimatedTimePerAngle * recommendedAnglesPerBatch) + this.GPU_DISPATCH_OVERHEAD_MS;

        // Watchdog timeout risk assessment
        let watchdogRisk = 'safe';
        let watchdogMessage = 'Batch size is within safe watchdog limits';
        let watchdogSuggestion = null;

        if (estimatedBatchTime >= this.WATCHDOG_CRITICAL_MS) {
            watchdogRisk = 'critical';
            watchdogMessage = `CRITICAL: Estimated batch time (${estimatedBatchTime.toFixed(0)}ms) exceeds watchdog kill threshold (${this.WATCHDOG_CRITICAL_MS}ms)`;
            // Calculate safer batch size
            const safeAnglesPerBatch = Math.floor(this.WATCHDOG_MAX_SAFE_MS / estimatedTimePerAngle);
            watchdogSuggestion = {
                anglesPerBatch: Math.max(this.MIN_ANGLES_PER_BATCH, safeAnglesPerBatch),
                message: `Reduce batch size to ${Math.max(this.MIN_ANGLES_PER_BATCH, safeAnglesPerBatch)} angles or adjust parameters`
            };
        } else if (estimatedBatchTime >= this.WATCHDOG_WARNING_MS) {
            watchdogRisk = 'warning';
            watchdogMessage = `WARNING: Estimated batch time (${estimatedBatchTime.toFixed(0)}ms) approaching watchdog threshold (${this.WATCHDOG_WARNING_MS}ms)`;
            // Calculate safer batch size
            const safeAnglesPerBatch = Math.floor(this.WATCHDOG_MAX_SAFE_MS / estimatedTimePerAngle);
            watchdogSuggestion = {
                anglesPerBatch: Math.max(this.MIN_ANGLES_PER_BATCH, safeAnglesPerBatch),
                message: `Consider reducing batch size to ${Math.max(this.MIN_ANGLES_PER_BATCH, safeAnglesPerBatch)} angles`
            };
        } else if (estimatedBatchTime >= this.WATCHDOG_MAX_SAFE_MS) {
            watchdogRisk = 'caution';
            watchdogMessage = `CAUTION: Estimated batch time (${estimatedBatchTime.toFixed(0)}ms) above safe target (${this.WATCHDOG_MAX_SAFE_MS}ms)`;
        }

        return {
            // Workload metrics
            workScore,
            workPerAngle,
            triangleDensity,
            pixelsPerAngle,

            // Grid info
            gridWidth,
            gridHeight,
            numAngles,

            // Timing estimates
            estimatedTimePerAngle,
            estimatedBatchTime,
            estimatedTotalTime,
            totalRasterTime,
            totalToolpathTime,
            totalOverhead,

            // Batch recommendations
            recommendedAnglesPerBatch,
            numBatches,
            overheadPercent: (totalOverhead / estimatedTotalTime) * 100,

            // Watchdog risk assessment
            watchdog: {
                risk: watchdogRisk,
                message: watchdogMessage,
                estimatedBatchTime,
                suggestion: watchdogSuggestion
            },

            // Work factors
            factors: {
                resolution: resolutionFactor,
                angular: angularFactor,
                tool: toolFactor,
                density: triangleDensity
            }
        };
    }

    /**
     * Calculate memory requirements for radial mode
     *
     * @param {Object} params - Same as calculateRadialWorkload
     * @returns {Object} Memory estimate in MB
     */
    calculateRadialMemory(params) {
        const { bounds, resolution, rotationStep, toolDiameter } = params;

        const radialRadius = Math.sqrt(
            Math.max(Math.abs(bounds.minY), Math.abs(bounds.maxY)) ** 2 +
            Math.max(Math.abs(bounds.minZ), Math.abs(bounds.maxZ)) ** 2
        );
        const radialHeight = bounds.maxX - bounds.minX;

        const gridWidth = Math.ceil((2 * radialRadius) / resolution);
        const gridHeight = Math.ceil(radialHeight / resolution);
        const numAngles = Math.ceil(360 / rotationStep);

        // Each pixel is 4 bytes (float32)
        const bytesPerAngle = gridWidth * gridHeight * 4;
        const totalMemoryBytes = bytesPerAngle * numAngles;
        const totalMemoryMB = totalMemoryBytes / (1024 * 1024);

        return {
            gridWidth,
            gridHeight,
            numAngles,
            bytesPerAngle,
            totalMemoryBytes,
            totalMemoryMB,
            exceedsLimit: totalMemoryMB > 256 // Typical GPU limit
        };
    }

    /**
     * Suggest optimal parameters for memory-constrained scenarios
     *
     * @param {Object} params - Same as calculateRadialWorkload
     * @param {number} maxMemoryMB - Maximum allowed memory in MB
     * @returns {Object} Suggested parameter adjustments
     */
    suggestOptimalParameters(params, maxMemoryMB = 256) {
        const memory = this.calculateRadialMemory(params);

        if (!memory.exceedsLimit && memory.totalMemoryMB < maxMemoryMB) {
            return {
                needsAdjustment: false,
                currentMemory: memory.totalMemoryMB,
                message: 'Current parameters are within memory limits'
            };
        }

        // Calculate required scale factor
        const scaleFactor = Math.sqrt(maxMemoryMB / memory.totalMemoryMB);

        // Suggest coarser resolution
        const suggestedResolution = params.resolution / scaleFactor;
        const roundedResolution = Math.ceil(suggestedResolution * 1000) / 1000;

        // Suggest coarser angular step
        const suggestedAngularStep = params.rotationStep / scaleFactor;
        const roundedAngularStep = Math.ceil(suggestedAngularStep * 10) / 10;

        return {
            needsAdjustment: true,
            currentMemory: memory.totalMemoryMB,
            targetMemory: maxMemoryMB,
            suggestions: {
                resolution: roundedResolution,
                angularStep: roundedAngularStep
            },
            message: `Memory limit exceeded. Suggested: resolution=${roundedResolution}mm or angularStep=${roundedAngularStep}°`
        };
    }

    /**
     * Pretty-print workload analysis
     *
     * @param {Object} params - Workload parameters
     */
    printAnalysis(params) {
        const workload = this.calculateRadialWorkload(params);
        const memory = this.calculateRadialMemory(params);

        console.log('=== Workload Analysis ===');
        console.log(`Model: ${params.triangleCount.toLocaleString()} triangles`);
        console.log(`Resolution: ${params.resolution}mm`);
        console.log(`Angular step: ${params.rotationStep}°`);
        console.log(`Tool diameter: ${params.toolDiameter}mm`);
        console.log('');
        console.log('Grid:');
        console.log(`  ${workload.gridWidth} × ${workload.gridHeight} pixels`);
        console.log(`  ${workload.numAngles} strips`);
        console.log(`  ${workload.pixelsPerAngle.toLocaleString()} pixels per strip`);
        console.log('');
        console.log('Work factors (vs baseline):');
        console.log(`  Resolution: ${workload.factors.resolution.toFixed(2)}x`);
        console.log(`  Angular: ${workload.factors.angular.toFixed(2)}x`);
        console.log(`  Tool: ${workload.factors.tool.toFixed(2)}x`);
        console.log(`  Density: ${workload.factors.density.toFixed(2)} tri/mm³`);
        console.log('');
        console.log('Estimated timing:');
        console.log(`  Per angle: ${workload.estimatedTimePerAngle.toFixed(2)}ms`);
        console.log(`  Total: ${workload.estimatedTotalTime.toFixed(0)}ms`);
        console.log('');
        console.log('Batching:');
        console.log(`  Recommended: ${workload.recommendedAnglesPerBatch} angles/batch`);
        console.log(`  Batches: ${workload.numBatches}`);
        console.log(`  Overhead: ${workload.overheadPercent.toFixed(1)}%`);
        console.log(`  Est. batch time: ${workload.estimatedBatchTime.toFixed(0)}ms`);
        console.log('');
        console.log('Watchdog Risk:');
        const riskSymbol = {
            'safe': '✓',
            'caution': '⚠️',
            'warning': '⚠️⚠️',
            'critical': '🛑'
        }[workload.watchdog.risk];
        console.log(`  ${riskSymbol} ${workload.watchdog.risk.toUpperCase()}`);
        console.log(`  ${workload.watchdog.message}`);
        if (workload.watchdog.suggestion) {
            console.log(`  Suggestion: ${workload.watchdog.suggestion.message}`);
        }
        console.log('');
        console.log('Memory:');
        console.log(`  ${memory.totalMemoryMB.toFixed(1)} MB`);
        console.log(`  ${memory.exceedsLimit ? '⚠️ EXCEEDS LIMIT' : '✓ Within limits'}`);
    }
}
