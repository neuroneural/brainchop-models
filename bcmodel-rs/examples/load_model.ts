/**
 * Load a .bcmodel file in Node.js using the TypeScript wrapper.
 *
 * Prerequisites:
 *   cd bcmodel-rs/bcmodel-wasm
 *   wasm-pack build --target web
 *
 * Usage:
 *   npx tsx load_model.ts ../../meshnet/model5_gw_ae/model.bcmodel
 */
import { readFileSync } from "fs";
import { initBcmodelSync, loadFromBuffer, computeTensorStats } from "../bcmodel-ts/src/index.js";

// --- Initialize WASM ---
const wasmBytes = readFileSync(
  new URL("../bcmodel-wasm/pkg/bcmodel_wasm_bg.wasm", import.meta.url)
);
initBcmodelSync(wasmBytes);

// --- Parse CLI args ---
const modelPath = process.argv[2];
if (!modelPath) {
  console.error("Usage: npx tsx load_model.ts <path/to/model.bcmodel>");
  process.exit(1);
}

// --- Load model ---
const data = new Uint8Array(readFileSync(modelPath));
const model = loadFromBuffer(data);

// --- Model info ---
console.log(`Name:         ${model.name}`);
console.log(`Type:         ${model.modelType}`);
console.log(`Input shape:  [${model.inputShape}]`);
console.log(`Classes:      ${model.numClasses}`);
console.log(`Graph nodes:  ${model.graphLength}`);
console.log(`Parameters:   ${model.totalParams.toLocaleString()}`);
console.log(`Weight size:  ${(model.weightSizeBytes / 1024).toFixed(1)} KB`);
console.log();

// --- Architecture graph ---
console.log("Architecture:");
for (const node of model.graph) {
  const inputs =
    node.inputs.length > 0
      ? ` <- ${node.inputs.join(", ")}`
      : " (input)";
  console.log(`  ${node.id.padEnd(20)} ${node.op.padEnd(15)}${inputs}`);
}
console.log();

// --- Tensors ---
console.log("Tensors:");
for (const name of model.tensorNames()) {
  const shape = model.tensorShape(name);
  const data = model.tensorData(name);
  if (data && shape) {
    const stats = computeTensorStats(name, data, shape);
    console.log(
      `  ${name.padEnd(30)} shape=[${shape}]  min=${stats.min.toFixed(4)}  max=${stats.max.toFixed(4)}`
    );
  }
}
console.log();

// --- Labels ---
const labels = model.labels;
if (labels && labels.length > 0) {
  console.log("Labels:");
  for (const label of labels) {
    const [r, g, b] = label.color;
    console.log(
      `  [${String(label.index).padStart(3)}] ${label.name.padEnd(20)} rgb(${r}, ${g}, ${b})`
    );
  }
  console.log();
}

// --- Inference config ---
const config = model.inferenceConfig;
if (config) {
  console.log("Inference config:");
  for (const [key, value] of Object.entries(config)) {
    console.log(`  ${key}: ${value}`);
  }
}

// --- Cleanup ---
model.free();
