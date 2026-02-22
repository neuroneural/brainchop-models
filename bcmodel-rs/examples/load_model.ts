/**
 * Load a .bcmodel file in Node.js using the WASM bindings.
 *
 * Prerequisites:
 *   cd bcmodel-rs/bcmodel-wasm
 *   wasm-pack build --target web
 *
 * Usage:
 *   npx tsx load_model.ts ../../meshnet/model5_gw_ae/model.bcmodel
 */
import { readFileSync } from "fs";
import { initSync, WasmBcmodel } from "../bcmodel-wasm/pkg/bcmodel_wasm.js";

// --- Initialize WASM ---
const wasmBytes = readFileSync(
  new URL("../bcmodel-wasm/pkg/bcmodel_wasm_bg.wasm", import.meta.url)
);
initSync({ module: wasmBytes });

// --- Parse CLI args ---
const modelPath = process.argv[2];
if (!modelPath) {
  console.error("Usage: npx tsx load_model.ts <path/to/model.bcmodel>");
  process.exit(1);
}

// --- Load model ---
const data = new Uint8Array(readFileSync(modelPath));
const model = new WasmBcmodel(data);

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
const graph = model.graph as unknown as Array<{
  id: string;
  op: string;
  inputs: string[];
}>;
for (const node of graph) {
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
  const tensor = model.tensorData(name);
  if (tensor && shape) {
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < tensor.length; i++) {
      if (tensor[i] < min) min = tensor[i];
      if (tensor[i] > max) max = tensor[i];
    }
    console.log(
      `  ${name.padEnd(30)} shape=[${shape}]  min=${min.toFixed(4)}  max=${max.toFixed(4)}`
    );
  }
}
console.log();

// --- Labels ---
const labels = model.labels as unknown as Array<{
  index: number;
  name: string;
  color: number[];
}>;
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
const config = model.inferenceConfig as unknown as Record<string, unknown>;
if (config) {
  console.log("Inference config:");
  for (const [key, value] of Object.entries(config)) {
    console.log(`  ${key}: ${value}`);
  }
}

// --- Cleanup ---
model.free();
