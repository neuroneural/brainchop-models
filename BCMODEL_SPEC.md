# BrainChop Model Format (`.bcmodel`) Specification v1.0

A universal single-file format for brain MRI segmentation models. Stores architecture, weights, inference configuration, and labels in one file — readable by tinygrad, TensorFlow.js, and WebGPU.

## File Structure

```
┌──────────────────────────────────────────┐
│ header_size  (8 bytes, uint64 LE)        │
├──────────────────────────────────────────┤
│ JSON header  (header_size bytes, UTF-8)  │
├──────────────────────────────────────────┤
│ Binary data  (remaining bytes)           │
│   ├── tensor 0 bytes                     │
│   ├── tensor 1 bytes                     │
│   └── ...                                │
└──────────────────────────────────────────┘
```

- **Bytes 0–7**: Header size as uint64 little-endian
- **Bytes 8–(8+N)**: JSON header (N = header_size)
- **Bytes (8+N)–EOF**: Binary tensor data, contiguous, no padding

Inspired by [SafeTensors](https://huggingface.co/docs/safetensors). No pickle, no protobuf — just JSON + raw binary.

## JSON Header

```jsonc
{
  "bcmodel_version": "1.0",

  "metadata": {
    "name": "Tissue Segmentation (Light)",
    "description": "Fast tissue segmentation into gray matter and white matter",
    "type": "tissue-segmentation",
    "source_framework": "pytorch"
  },

  "input": {
    "shape": [1, 1, 256, 256, 256],
    "dtype": "float32",
    "data_layout": "channels_first"
  },

  "output": {
    "num_classes": 3,
    "data_layout": "channels_first"
  },

  "graph": [
    {"id": "conv0", "op": "conv3d", "params": {"in_channels": 1, "out_channels": 5, "kernel_size": 3, "padding": 1, "dilation": 1, "bias": true}, "inputs": []},
    {"id": "act0", "op": "relu", "params": {}, "inputs": ["conv0"]},
    {"id": "output", "op": "conv3d", "params": {"in_channels": 5, "out_channels": 3, "kernel_size": 1, "padding": 0, "dilation": 1, "bias": true}, "inputs": ["act8"]}
  ],

  "tensors": {
    "conv0.weight": {"dtype": "float32", "shape": [5, 1, 3, 3, 3], "data_offsets": [0, 540]},
    "conv0.bias":   {"dtype": "float32", "shape": [5],             "data_offsets": [540, 560]}
  },

  "inference": {
    "enable_seq_conv": false,
    "crop_padding": 18,
    "auto_threshold": 0.02,
    "enable_quantile_norm": false,
    "enable_transpose": true
  },

  "performance": {
    "estimated_time_seconds": 3,
    "memory_requirement_mb": 400
  },

  "labels": [
    {"index": 0, "name": "background",  "color": [0, 0, 0]},
    {"index": 1, "name": "White Matter", "color": [255, 255, 255]},
    {"index": 2, "name": "Grey Matter",  "color": [205, 62, 78]}
  ],

  "pipeline": {
    "requires_pre_model": null,
    "filter_with_pre_mask": false
  }
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `bcmodel_version` | string | Format version, currently `"1.0"` |
| `metadata.name` | string | Human-readable model name |
| `metadata.type` | string | One of: `"brain-extraction"`, `"tissue-segmentation"`, `"parcellation"` |
| `input.shape` | int[] | Input tensor shape in NCDHW format |
| `input.dtype` | string | Input data type, typically `"float32"` |
| `input.data_layout` | string | Always `"channels_first"` in v1.0 |
| `output.num_classes` | int | Number of output segmentation classes |
| `graph` | object[] | Architecture as a DAG of operations (see below) |
| `tensors` | object | Weight tensor manifest with offsets into binary data |
| `labels` | object[] | Class names and RGB colors |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `metadata.description` | string | `""` | Detailed description |
| `metadata.source_framework` | string | `""` | Training framework (`"pytorch"`, `"tensorflow"`) |
| `metadata.authors` | string[] | `[]` | Model authors |
| `metadata.license` | string | `""` | License identifier |
| `inference` | object | `{}` | Runtime inference configuration |
| `performance` | object | `{}` | Advisory performance hints |
| `pipeline` | object | `{}` | Model chaining configuration |

### Inference Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_seq_conv` | bool | `false` | Sequential convolution mode (lower memory) |
| `crop_padding` | int | `0` | Voxels of context around the brain crop box |
| `auto_threshold` | float | `0` | Threshold for automatic brain cropping |
| `enable_quantile_norm` | bool | `false` | Apply quantile normalization preprocessing |
| `enable_transpose` | bool | `true` | Whether to transpose the input volume |

## Architecture Graph

The `graph` array represents the model as a directed acyclic graph (DAG). Each node has:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique node identifier |
| `op` | string | yes | Operation type (see catalog below) |
| `params` | object | yes | Operation-specific parameters |
| `inputs` | string[] | yes | IDs of input nodes. Empty `[]` for the entry node. |

### Execution Rules

1. Nodes are listed in **topological order** — a valid execution order.
2. The first node with `"inputs": []` receives the model input tensor.
3. The **last node** in the array produces the model output.
4. Runtimes maintain a `{node_id: tensor}` dictionary and iterate the array sequentially.
5. A tensor must be kept alive until all nodes referencing it have executed.

### Operation Catalog

#### Convolution

| Op | Params | Weight Tensors |
|----|--------|----------------|
| `conv3d` | `in_channels`, `out_channels`, `kernel_size`, `padding`, `dilation`, `stride` (default 1), `bias` (default false), `groups` (default 1) | `{id}.weight` [out, in, D, H, W], `{id}.bias` [out] if bias=true |
| `conv_transpose3d` | Same as `conv3d` + `output_padding` (default 0) | Same as `conv3d` |

#### Normalization

| Op | Params | Weight Tensors |
|----|--------|----------------|
| `batch_norm3d` | `num_features`, `affine` (default true), `eps` (default 1e-5) | If affine: `{id}.weight`, `{id}.bias`, `{id}.running_mean`, `{id}.running_var` |
| `group_norm` | `num_groups`, `num_channels`, `affine` (default true), `eps` (default 1e-5) | If affine: `{id}.weight`, `{id}.bias` |
| `instance_norm3d` | `num_features`, `affine` (default false) | If affine: `{id}.weight`, `{id}.bias` |

#### Activation

| Op | Params |
|----|--------|
| `relu` | *(none)* |
| `elu` | `alpha` (default 1.0) |
| `gelu` | *(none)* |
| `sigmoid` | *(none)* |
| `softmax` | `dim` (default 1) |

#### Tensor Operations

| Op | Params | Description |
|----|--------|-------------|
| `add` | *(none)* | Element-wise addition. `inputs` has 2+ node IDs. |
| `cat` | `dim` | Concatenation along `dim`. `inputs` has 2+ node IDs. |

#### Pooling / Upsampling

| Op | Params |
|----|--------|
| `max_pool3d` | `kernel_size`, `stride` (default = kernel_size), `padding` (default 0) |
| `avg_pool3d` | `kernel_size`, `stride` (default = kernel_size), `padding` (default 0) |
| `upsample` | `scale_factor`, `mode` (`"nearest"` or `"trilinear"`) |

#### Other

| Op | Params |
|----|--------|
| `dropout` | `p` (drop probability). No-op at inference. |
| `linear` | `in_features`, `out_features`, `bias` (default true). Weights: `{id}.weight`, `{id}.bias` |

## Weight Storage

- All weights are stored as **channels-first** (PyTorch convention): conv kernels are `[out_ch, in_ch, D, H, W]`.
- Data is **raw float32** bytes, little-endian.
- Tensors are stored contiguously in the binary section with no padding.
- Each tensor's location is specified by `data_offsets: [begin, end]` in the `tensors` manifest, where offsets are relative to the start of the binary data section.
- Tensor names follow the pattern `{graph_node_id}.{param_name}` (e.g., `conv0.weight`, `bn1.running_mean`).

### TFJS/WebGPU Note

TFJS expects channels-last weights for Conv3D: `[D, H, W, in_ch, out_ch]`. The loader must transpose from the stored `[out_ch, in_ch, D, H, W]` layout using permutation `[2, 3, 4, 1, 0]`.

## Examples

### Skip Connections (U-Net)

The DAG representation handles skip connections by referencing earlier nodes by ID:

```jsonc
"graph": [
  // Encoder
  {"id": "enc1", "op": "conv3d", "params": {"in_channels": 1, "out_channels": 32, ...}, "inputs": []},
  {"id": "enc1_act", "op": "relu", "params": {}, "inputs": ["enc1"]},
  {"id": "pool1", "op": "max_pool3d", "params": {"kernel_size": 2, "stride": 2}, "inputs": ["enc1_act"]},

  // Bottleneck
  {"id": "bot", "op": "conv3d", "params": {"in_channels": 32, "out_channels": 64, ...}, "inputs": ["pool1"]},

  // Decoder with skip connection
  {"id": "up1", "op": "upsample", "params": {"scale_factor": 2, "mode": "nearest"}, "inputs": ["bot"]},
  {"id": "skip1", "op": "cat", "params": {"dim": 1}, "inputs": ["up1", "enc1_act"]},
  {"id": "dec1", "op": "conv3d", "params": {"in_channels": 96, "out_channels": 32, ...}, "inputs": ["skip1"]}
]
```

The `skip1` node concatenates the upsampled bottleneck with the encoder output by referencing both `"up1"` and `"enc1_act"` — two nodes computed at different depths.

### Parallel Branches (Multi-Decoder)

```jsonc
"graph": [
  {"id": "shared", "op": "conv3d", "params": {...}, "inputs": []},

  // Branch 1 (reads from shared)
  {"id": "d1_conv", "op": "conv3d", "params": {"dilation": 16, ...}, "inputs": ["shared"]},

  // Branch 2 (also reads from shared)
  {"id": "d2_conv", "op": "conv3d", "params": {"dilation": 8, ...}, "inputs": ["shared"]},

  // Merge
  {"id": "merge", "op": "cat", "params": {"dim": 1}, "inputs": ["d1_conv", "d2_conv"]},
  {"id": "output", "op": "conv3d", "params": {...}, "inputs": ["merge"]}
]
```

## Extensibility

- **Unknown keys are ignored**: Parsers must skip unrecognized JSON keys without error.
- **New operations**: Adding a new `op` requires updating runtimes. Old runtimes encountering an unknown op should raise: `"Unsupported operation: {op}. Update your runtime."`
- **Version field**: `bcmodel_version` enables schema evolution.

## Parsing Pseudocode

```python
# Python
header_size = struct.unpack('<Q', f.read(8))[0]
header = json.loads(f.read(header_size))
data = f.read()
for name, info in header['tensors'].items():
    begin, end = info['data_offsets']
    tensor = np.frombuffer(data[begin:end], dtype=np.float32).reshape(info['shape'])
```

```javascript
// JavaScript
const view = new DataView(buffer);
const headerSize = view.getUint32(0, true);  // lower 32 bits (sufficient for <4GB headers)
const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerSize)));
const dataOffset = 8 + headerSize;
for (const [name, info] of Object.entries(header.tensors)) {
    const [begin, end] = info.data_offsets;
    const tensor = new Float32Array(buffer, dataOffset + begin, (end - begin) / 4);
}
```
