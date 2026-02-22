# bcmodel-rs

Rust parser for [`.bcmodel`](../BCMODEL_SPEC.md) brain MRI segmentation model files, with bindings for **TypeScript/JavaScript** (WebAssembly) and **Python** (native module via PyO3).

Loads model metadata, architecture graphs, weight tensors, labels, and inference configuration from a single `.bcmodel` file. Parsing only — no inference runtime.

## Project Structure

```
bcmodel-rs/
├── bcmodel-core/     # Pure Rust parser library
├── bcmodel-wasm/     # WebAssembly bindings (wasm-bindgen)
├── bcmodel-py/       # Python bindings (PyO3 + maturin)
└── examples/         # Usage examples
```

## Building

### Rust (core library)

```bash
cd bcmodel-rs
cargo build -p bcmodel-core
cargo test -p bcmodel-core
```

### WebAssembly (TypeScript/JavaScript)

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/).

```bash
cd bcmodel-rs/bcmodel-wasm
wasm-pack build --target web
```

This produces `pkg/` with:
- `bcmodel_wasm.js` — ES module
- `bcmodel_wasm.d.ts` — TypeScript type definitions
- `bcmodel_wasm_bg.wasm` — WebAssembly binary

### Python

Requires [maturin](https://www.maturin.rs/) and a Python virtual environment.

```bash
cd bcmodel-rs/bcmodel-py
python -m venv .venv
source .venv/bin/activate
pip install numpy maturin
maturin develop
```

---

## Python API

### Installation

```bash
cd bcmodel-rs/bcmodel-py
pip install maturin numpy
maturin develop
```

### Quick Start

```python
import bcmodel

model = bcmodel.BcmodelFile.load("meshnet/model5_gw_ae/model.bcmodel")
print(model)
# BcmodelFile(name='Tissue Segmentation (Light)', type='tissue-segmentation', classes=3, params=5598, graph_nodes=19)
```

### BcmodelFile Class

```python
import bcmodel
import numpy as np

# Load from file path
model = bcmodel.BcmodelFile.load("/path/to/model.bcmodel")

# Or from bytes
with open("/path/to/model.bcmodel", "rb") as f:
    model = bcmodel.BcmodelFile.from_bytes(f.read())

# -- Properties --
model.name            # "Tissue Segmentation (Light)"
model.model_type      # "tissue-segmentation"
model.description     # "Fast tissue segmentation..."
model.num_classes     # 3
model.input_shape     # [1, 1, 256, 256, 256]
model.graph_length    # 19
model.total_params    # 5598
model.weight_size_bytes  # 22392
model.header          # Full header as a dict

# -- Tensors --
model.tensor_names()           # ["conv0.weight", "conv0.bias", ...]
model.tensor("conv0.weight")   # numpy.ndarray (float32, 1D)
model.tensor_shape("conv0.weight")  # [5, 1, 3, 3, 3]

# Reshape to original dimensions
name = "conv0.weight"
w = model.tensor(name).reshape(model.tensor_shape(name))
print(w.shape)  # (5, 1, 3, 3, 3)

# -- Labels --
model.labels()
# [{"index": 0, "name": "background", "color": [0, 0, 0]},
#  {"index": 1, "name": "White Matter", "color": [255, 255, 255]},
#  {"index": 2, "name": "Grey Matter", "color": [205, 62, 78]}]

# -- Architecture Graph --
model.graph()
# [{"id": "conv0", "op": "conv3d", "params": {...}, "inputs": []}, ...]

# -- Inference Config --
model.inference_config()
# {"enable_seq_conv": False, "crop_padding": 18, "auto_threshold": 0.02, ...}
```

### Compatibility Function

Drop-in replacement for the existing `load_bcmodel.py` API:

```python
import bcmodel

header, tensors = bcmodel.load_bcmodel("/path/to/model.bcmodel")

# header is a dict matching the JSON header
print(header["bcmodel_version"])  # "1.0"
print(header["metadata"]["name"]) # "Tissue Segmentation (Light)"

# tensors is a dict of name -> numpy.ndarray (already reshaped)
print(tensors["conv0.weight"].shape)  # (5, 1, 3, 3, 3)
print(tensors["conv0.weight"].dtype)  # float32
```

---

## TypeScript / JavaScript API

### Setup (Browser)

```html
<script type="module">
import init, { WasmBcmodel } from './pkg/bcmodel_wasm.js';

await init();  // Initialize WASM module

const response = await fetch('model.bcmodel');
const buffer = new Uint8Array(await response.arrayBuffer());
const model = new WasmBcmodel(buffer);

console.log(model.name);       // "Tissue Segmentation (Light)"
console.log(model.numClasses); // 3

model.free();  // Release WASM memory when done
</script>
```

### Setup (Node.js)

```typescript
import { readFileSync } from 'fs';
import { initSync, WasmBcmodel } from './pkg/bcmodel_wasm.js';

// Initialize synchronously in Node
const wasmBytes = readFileSync('./pkg/bcmodel_wasm_bg.wasm');
initSync({ module: wasmBytes });

const data = new Uint8Array(readFileSync('model.bcmodel'));
const model = new WasmBcmodel(data);
```

### WasmBcmodel Class

```typescript
import init, { WasmBcmodel } from './pkg/bcmodel_wasm.js';

await init();

const buffer = new Uint8Array(fileData);
const model = new WasmBcmodel(buffer);

// -- Properties --
model.name              // "Tissue Segmentation (Light)"
model.modelType         // "tissue-segmentation"
model.numClasses        // 3
model.inputShape        // Uint32Array [1, 1, 256, 256, 256]
model.graphLength       // 19
model.totalParams       // 5598
model.weightSizeBytes   // 22392

// -- Structured Data (JS objects) --
model.header            // Full header object
model.metadata          // { name, description, type, source_framework, authors, license }
model.graph             // [{ id, op, params, inputs }, ...]
model.labels            // [{ index, name, color }, ...]
model.inferenceConfig   // { enable_seq_conv, crop_padding, ... }
model.pipelineConfig    // { requires_pre_model, filter_with_pre_mask }
model.performanceHints  // { estimated_time_seconds, memory_requirement_mb }

// -- Tensors --
model.tensorNames()              // ["conv0.weight", "conv0.bias", ...]
model.tensorShape("conv0.weight") // Uint32Array [5, 1, 3, 3, 3]

// Get tensor data as Float32Array (view into WASM memory)
const weights = model.tensorData("conv0.weight");
// IMPORTANT: Copy if you need to retain after other WASM calls
const weightsCopy = new Float32Array(weights);

// -- Cleanup --
model.free();  // Release WASM memory
// Or use `using` syntax (Symbol.dispose):
// using model = new WasmBcmodel(buffer);
```

---

## Rust API

For Rust consumers using `bcmodel-core` directly:

```rust
use bcmodel_core::parser::BcmodelFile;

// From file
let model = BcmodelFile::from_path("model.bcmodel".as_ref())?;

// From bytes
let bytes = std::fs::read("model.bcmodel")?;
let model = BcmodelFile::from_bytes(&bytes)?;

// Metadata
println!("{}", model.metadata().name);        // "Tissue Segmentation (Light)"
println!("{}", model.metadata().model_type);  // "tissue-segmentation"
println!("{}", model.output_spec().num_classes); // 3

// Graph
for node in model.graph() {
    println!("{}: {} (inputs: {:?})", node.id, node.op, node.inputs);
}

// Tensors (zero-copy &[f32] slice)
for name in model.tensor_names() {
    let data = model.get_tensor_data(name).unwrap();
    let shape = model.get_tensor_shape(name).unwrap();
    println!("{name}: shape={shape:?}, len={}", data.len());
}

// Labels
for label in model.labels() {
    println!("[{}] {} color={:?}", label.index, label.name, label.color);
}
```

---

## Examples

| File | Description |
|------|-------------|
| [examples/load_model.py](examples/load_model.py) | Python script — load model and print summary |
| [examples/load_model.ts](examples/load_model.ts) | TypeScript script — Node.js usage with WASM |
| [examples/explore_model.ipynb](examples/explore_model.ipynb) | Jupyter Notebook — interactive model exploration with visualizations |
