// workload-calculator-demo.cjs
// Demonstration of the workload calculator

const { WorkloadCalculator } = require('../workload-calculator.js');

const calculator = new WorkloadCalculator();

console.log('=== Workload Calculator Demo ===\n');

// Example 1: Typical lathe part
console.log('Example 1: Lathe cylinder (typical parameters)');
calculator.printAnalysis({
    triangleCount: 90620,
    bounds: {
        minX: 0, maxX: 100,
        minY: -25, maxY: 25,
        minZ: -25, maxZ: 25
    },
    resolution: 0.05,
    rotationStep: 1.0,
    toolDiameter: 2.5
});

console.log('\n' + '='.repeat(70) + '\n');

// Example 2: High resolution
console.log('Example 2: Same part, high resolution (0.01mm)');
calculator.printAnalysis({
    triangleCount: 90620,
    bounds: {
        minX: 0, maxX: 100,
        minY: -25, maxY: 25,
        minZ: -25, maxZ: 25
    },
    resolution: 0.01,
    rotationStep: 1.0,
    toolDiameter: 2.5
});

console.log('\n' + '='.repeat(70) + '\n');

// Example 3: Large tool
console.log('Example 3: Same part, larger tool (5mm)');
calculator.printAnalysis({
    triangleCount: 90620,
    bounds: {
        minX: 0, maxX: 100,
        minY: -25, maxY: 25,
        minZ: -25, maxZ: 25
    },
    resolution: 0.05,
    rotationStep: 1.0,
    toolDiameter: 5.0
});

console.log('\n' + '='.repeat(70) + '\n');

// Example 4: Fine angular step
console.log('Example 4: Same part, fine angular step (0.5°)');
calculator.printAnalysis({
    triangleCount: 90620,
    bounds: {
        minX: 0, maxX: 100,
        minY: -25, maxY: 25,
        minZ: -25, maxZ: 25
    },
    resolution: 0.05,
    rotationStep: 0.5,
    toolDiameter: 2.5
});

console.log('\n' + '='.repeat(70) + '\n');

// Example 5: Complex torture test
console.log('Example 5: Lathe torture (1.5M triangles)');
calculator.printAnalysis({
    triangleCount: 1491718,
    bounds: {
        minX: 0, maxX: 100,
        minY: -30, maxY: 30,
        minZ: -30, maxZ: 30
    },
    resolution: 0.05,
    rotationStep: 1.0,
    toolDiameter: 2.5
});

console.log('\n' + '='.repeat(70) + '\n');

// Example 6: Memory constraint check
console.log('Example 6: Memory limit suggestions');
const params = {
    triangleCount: 90620,
    bounds: {
        minX: 0, maxX: 100,
        minY: -25, maxY: 25,
        minZ: -25, maxZ: 25
    },
    resolution: 0.01, // Very fine - may exceed memory
    rotationStep: 0.5,
    toolDiameter: 2.5
};

const suggestion = calculator.suggestOptimalParameters(params, 256);
console.log(suggestion.message);
if (suggestion.needsAdjustment) {
    console.log(`Current memory: ${suggestion.currentMemory.toFixed(1)} MB`);
    console.log('Suggestions:');
    console.log(`  Resolution: ${suggestion.suggestions.resolution}mm`);
    console.log(`  Angular step: ${suggestion.suggestions.angularStep}°`);
}

console.log('\n=== Demo Complete ===');
