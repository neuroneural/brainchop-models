#!/usr/bin/env python3
"""Convert existing brainchop model directories to .bcmodel format.

Supports three input formats:
  Format A: Simple model.json with layers array (model5_gw_ae, model18cls, etc.)
  Format B: TF.js Keras layers-model JSON (model11_gw_ae, model20chan3cls, etc.)
  Format C: v2.0 layers.json with forward_pass ops (mindgrab)

Usage:
    python convert_to_bcmodel.py meshnet/model5_gw_ae
    python convert_to_bcmodel.py --all
    python convert_to_bcmodel.py --all --force
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def detect_format(model_dir: Path) -> str:
    """Detect which format a model directory uses."""
    layers_json = model_dir / "layers.json"
    model_json = model_dir / "model.json"

    if layers_json.exists():
        with open(layers_json) as f:
            data = json.load(f)
        if data.get("version") == "2.0" and "forward_pass" in data:
            return "C"

    if model_json.exists():
        with open(model_json) as f:
            data = json.load(f)
        if data.get("format") == "layers-model":
            return "B"
        if "layers" in data:
            return "A"

    raise ValueError(f"Cannot detect format in {model_dir}")


def build_graph_format_a(model_json: dict) -> list[dict]:
    """Build graph from Format A (simple layers array)."""
    layers = model_json["layers"]
    bnorm = model_json.get("bnorm", False)
    gelu = model_json.get("gelu", False)
    elu = model_json.get("elu", False)
    bias_flag = model_json.get("bias", False)
    dropout_p = model_json.get("dropout_p", 0)

    graph = []
    prev_id = None

    for i, layer in enumerate(layers):
        is_last = i == len(layers) - 1
        conv_id = f"conv{i}"

        conv_params = {
            "in_channels": layer["in_channels"],
            "out_channels": layer["out_channels"],
            "kernel_size": layer["kernel_size"],
            "padding": layer["padding"],
            "dilation": layer["dilation"],
            "stride": layer.get("stride", 1),
            "bias": bias_flag,
        }

        graph.append({
            "id": conv_id,
            "op": "conv3d",
            "params": conv_params,
            "inputs": [prev_id] if prev_id else [],
        })
        prev_id = conv_id

        if not is_last:
            if bnorm:
                out_ch = layer["out_channels"]
                norm_id = f"gn{i}"
                graph.append({
                    "id": norm_id,
                    "op": "group_norm",
                    "params": {
                        "num_groups": out_ch,
                        "num_channels": out_ch,
                        "affine": False,
                    },
                    "inputs": [prev_id],
                })
                prev_id = norm_id

            act_id = f"act{i}"
            if gelu:
                act_op = "gelu"
            elif elu:
                act_op = "elu"
            else:
                act_op = "relu"

            graph.append({
                "id": act_id,
                "op": act_op,
                "params": {},
                "inputs": [prev_id],
            })
            prev_id = act_id

            if dropout_p > 0:
                drop_id = f"drop{i}"
                graph.append({
                    "id": drop_id,
                    "op": "dropout",
                    "params": {"p": dropout_p},
                    "inputs": [prev_id],
                })
                prev_id = drop_id

    return graph


def build_graph_format_b(model_json: dict) -> tuple[list[dict], str]:
    """Build graph from Format B (TF.js Keras).

    Returns (graph, activation_type).
    """
    topology = model_json["modelTopology"]["model_config"]["config"]
    keras_layers = topology["layers"]

    graph = []
    prev_id = None
    activation_type = "relu"

    for kl in keras_layers:
        class_name = kl["class_name"]
        config = kl["config"]
        name = kl["name"]

        if class_name == "InputLayer":
            continue

        if class_name == "Conv3D":
            dilation = config["dilation_rate"]
            d = dilation[0] if isinstance(dilation, list) else dilation
            kernel_size = config["kernel_size"]
            k = kernel_size[0] if isinstance(kernel_size, list) else kernel_size
            # "same" padding: padding = dilation * (kernel_size - 1) // 2
            padding = d * (k - 1) // 2

            # Determine in_channels from previous conv or input
            filters = config["filters"]

            conv_params = {
                "out_channels": filters,
                "kernel_size": k,
                "padding": padding,
                "dilation": d,
                "stride": 1,
                "bias": config.get("use_bias", True),
            }

            graph.append({
                "id": name,
                "op": "conv3d",
                "params": conv_params,
                "inputs": [prev_id] if prev_id else [],
            })
            prev_id = name

        elif class_name == "Activation":
            act = config["activation"]
            activation_type = act
            graph.append({
                "id": name,
                "op": act,
                "params": {},
                "inputs": [prev_id] if prev_id else [],
            })
            prev_id = name

    # Fix in_channels by walking the graph
    _fix_in_channels_from_graph(graph)

    return graph, activation_type


def _fix_in_channels_from_graph(graph: list[dict]):
    """Fill in in_channels for conv3d nodes by tracing data flow."""
    # First conv gets in_channels=1 (single-channel brain MRI)
    out_channels = {}
    for node in graph:
        if node["op"] == "conv3d":
            if not node["inputs"]:
                node["params"]["in_channels"] = 1
            else:
                # Find the last conv's out_channels
                for inp_id in reversed(node["inputs"]):
                    if inp_id in out_channels:
                        node["params"]["in_channels"] = out_channels[inp_id]
                        break
            out_channels[node["id"]] = node["params"]["out_channels"]
        elif node["op"] in ("relu", "elu", "gelu", "sigmoid", "dropout"):
            # Pass-through ops preserve channels
            for inp_id in node.get("inputs", []):
                if inp_id in out_channels:
                    out_channels[node["id"]] = out_channels[inp_id]
                    break


def build_graph_format_c(layers_json: dict) -> list[dict]:
    """Build graph from Format C (v2.0 forward_pass)."""
    forward_pass = layers_json["forward_pass"]
    graph = []
    prev_id = None
    counters = {}

    for entry in forward_pass:
        op = entry["op"]
        params = dict(entry["params"])

        # Generate unique ID
        count = counters.get(op, 0)
        counters[op] = count + 1
        node_id = f"{op}{count}"

        graph.append({
            "id": node_id,
            "op": op,
            "params": params,
            "inputs": [prev_id] if prev_id else [],
        })
        prev_id = node_id

    return graph


def load_weights_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Load weights from a SafeTensors file (same binary layout as .bcmodel)."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        manifest = json.loads(f.read(header_size))
        data = f.read()

    weights = {}
    for name, info in manifest.items():
        if name.startswith("__"):  # skip __metadata__
            continue
        shape = info["shape"]
        begin, end = info["data_offsets"]
        n_elements = 1
        for s in shape:
            n_elements *= s
        arr = np.frombuffer(data[begin:end], dtype=np.float32, count=n_elements).copy()
        weights[name] = arr.reshape(shape)

    return weights


def _is_safetensors(path: Path) -> bool:
    """Check if a file is SafeTensors format (uint64 LE header + JSON)."""
    with open(path, "rb") as f:
        header = f.read(16)
    if len(header) < 16:
        return False
    # SafeTensors: first 8 bytes are header size, next bytes start with '{'
    try:
        header_size = struct.unpack("<Q", header[:8])[0]
        return header_size < 10_000_000 and header[8:9] == b"{"
    except Exception:
        return False


def load_weights_pth(pth_path: Path) -> dict[str, np.ndarray]:
    """Load weights from a .pth file (channels-first).

    Handles both real PyTorch pickle/zip files and SafeTensors files
    that happen to have a .pth extension.
    """
    # Check if this is actually a SafeTensors file
    if _is_safetensors(pth_path):
        return load_weights_safetensors(pth_path)

    if torch is None:
        raise RuntimeError("PyTorch required for .pth files: pip install torch")
    try:
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    except Exception:
        # Older .pth files may use pickle ops not supported by weights_only=True
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
    weights = {}
    for key, tensor in state_dict.items():
        weights[key] = tensor.detach().float().numpy()
    return weights


def load_weights_bin_tfjs(bin_path: Path, weights_manifest: list[dict]) -> dict[str, np.ndarray]:
    """Load weights from a TF.js .bin file (channels-last) and transpose to channels-first."""
    data = bin_path.read_bytes()
    weights = {}
    offset = 0

    for entry in weights_manifest:
        for w in entry["weights"]:
            name = w["name"]
            shape = w["shape"]
            n_elements = 1
            for s in shape:
                n_elements *= s
            n_bytes = n_elements * 4  # float32
            arr = np.frombuffer(data, dtype=np.float32, offset=offset, count=n_elements)
            arr = arr.reshape(shape)
            offset += n_bytes

            # Transpose conv kernels from DHWIO → OIDHW
            if "kernel" in name and len(shape) == 5:
                arr = np.transpose(arr, (4, 3, 0, 1, 2))

            weights[name] = arr.copy()

    return weights


def load_weights_bin_native(bin_path: Path, graph: list[dict]) -> dict[str, np.ndarray]:
    """Load weights from a raw .bin file (channels-first, from pth_to_bin.py).

    Reconstructs tensor shapes from the graph architecture.
    """
    data = bin_path.read_bytes()
    offset = 0
    weights = {}

    for node in graph:
        op = node["op"]
        params = node["params"]
        node_id = node["id"]

        if op == "conv3d":
            out_ch = params["out_channels"]
            in_ch = params["in_channels"]
            k = params["kernel_size"]
            # Weight: [out_ch, in_ch, k, k, k]
            shape = [out_ch, in_ch, k, k, k]
            n = 1
            for s in shape:
                n *= s
            arr = np.frombuffer(data, dtype=np.float32, offset=offset, count=n).copy()
            weights[f"{node_id}.weight"] = arr.reshape(shape)
            offset += n * 4

            if params.get("bias", False):
                arr = np.frombuffer(data, dtype=np.float32, offset=offset, count=out_ch).copy()
                weights[f"{node_id}.bias"] = arr.reshape([out_ch])
                offset += out_ch * 4

        elif op == "batch_norm3d" and params.get("affine", True):
            nf = params["num_features"]
            for suffix in ["weight", "bias", "running_mean", "running_var"]:
                arr = np.frombuffer(data, dtype=np.float32, offset=offset, count=nf).copy()
                weights[f"{node_id}.{suffix}"] = arr.reshape([nf])
                offset += nf * 4

        elif op == "group_norm" and params.get("affine", True):
            nc = params["num_channels"]
            for suffix in ["weight", "bias"]:
                arr = np.frombuffer(data, dtype=np.float32, offset=offset, count=nc).copy()
                weights[f"{node_id}.{suffix}"] = arr.reshape([nc])
                offset += nc * 4

    return weights


def map_tfjs_weights_to_graph(tfjs_weights: dict[str, np.ndarray], graph: list[dict]) -> dict[str, np.ndarray]:
    """Map TF.js weight names (e.g. 'conv3d_0/kernel') to graph node names (e.g. 'conv3d_0.weight')."""
    mapped = {}
    for node in graph:
        if node["op"] == "conv3d":
            tfjs_name = node["id"]
            kernel_key = f"{tfjs_name}/kernel"
            bias_key = f"{tfjs_name}/bias"
            if kernel_key in tfjs_weights:
                mapped[f"{node['id']}.weight"] = tfjs_weights[kernel_key]
            if bias_key in tfjs_weights:
                mapped[f"{node['id']}.bias"] = tfjs_weights[bias_key]
    return mapped


def map_pth_weights_to_graph(pth_weights: dict[str, np.ndarray], graph: list[dict]) -> dict[str, np.ndarray]:
    """Map PyTorch state dict keys to graph node names using shape matching.

    Iterates through pth entries in order and graph nodes in order.
    For each graph node, checks if the next pth entry's shape matches
    the expected shape for that node's weight. Skips graph nodes
    whose expected weight shapes don't match (e.g. affine norms with
    no actual parameters).
    """
    pth_keys = list(pth_weights.keys())
    pth_idx = 0
    mapped = {}

    def _try_consume(graph_key: str, expected_shape: tuple | None = None) -> bool:
        """Try to consume the next pth entry if shape matches."""
        nonlocal pth_idx
        if pth_idx >= len(pth_keys):
            return False
        pth_val = pth_weights[pth_keys[pth_idx]]
        if expected_shape is not None and pth_val.shape != expected_shape:
            return False
        mapped[graph_key] = pth_val
        pth_idx += 1
        return True

    for node in graph:
        op = node["op"]
        params = node["params"]
        nid = node["id"]

        if op == "conv3d":
            out_ch = params["out_channels"]
            in_ch = params["in_channels"]
            k = params["kernel_size"]
            w_shape = (out_ch, in_ch, k, k, k)
            _try_consume(f"{nid}.weight", w_shape)
            if params.get("bias", False):
                _try_consume(f"{nid}.bias", (out_ch,))

        elif op == "batch_norm3d":
            nf = params["num_features"]
            nf_shape = (nf,)
            # Try affine params (weight, bias)
            if params.get("affine", True):
                _try_consume(f"{nid}.weight", nf_shape)
                _try_consume(f"{nid}.bias", nf_shape)
            # Running stats (may or may not exist)
            _try_consume(f"{nid}.running_mean", nf_shape)
            _try_consume(f"{nid}.running_var", nf_shape)
            # Skip num_batches_tracked
            if pth_idx < len(pth_keys) and "num_batches" in pth_keys[pth_idx]:
                pth_idx += 1

        elif op == "group_norm":
            nc = params["num_channels"]
            nc_shape = (nc,)
            if params.get("affine", True):
                _try_consume(f"{nid}.weight", nc_shape)
                _try_consume(f"{nid}.bias", nc_shape)

    return mapped


def read_settings(model_dir: Path) -> dict:
    """Read settings.json and return inference/performance config."""
    settings_path = model_dir / "settings.json"
    if not settings_path.exists():
        return {}
    with open(settings_path) as f:
        return json.load(f)


def read_colormap(model_dir: Path, settings: dict) -> list[dict]:
    """Read colormap.json and return labels array."""
    labels_file = "colormap.json"
    if settings and "files" in settings:
        labels_file = settings["files"].get("labels", "colormap.json")

    colormap_path = model_dir / labels_file
    if not colormap_path.exists():
        return []

    with open(colormap_path) as f:
        cmap = json.load(f)

    labels = []
    for i, name in enumerate(cmap.get("labels", [])):
        r = cmap["R"][i] if i < len(cmap.get("R", [])) else 0
        g = cmap["G"][i] if i < len(cmap.get("G", [])) else 0
        b = cmap["B"][i] if i < len(cmap.get("B", [])) else 0
        labels.append({"index": i, "name": name, "color": [r, g, b]})

    return labels


def build_tensors_manifest(weights: dict[str, np.ndarray]) -> tuple[dict, bytes]:
    """Build tensors manifest and concatenated binary data."""
    tensors = {}
    chunks = []
    offset = 0

    for name, arr in weights.items():
        arr = arr.astype(np.float32)
        data = arr.tobytes()
        end = offset + len(data)
        tensors[name] = {
            "dtype": "float32",
            "shape": list(arr.shape),
            "data_offsets": [offset, end],
        }
        chunks.append(data)
        offset = end

    return tensors, b"".join(chunks)


def write_bcmodel(output_path: Path, header: dict, binary_data: bytes):
    """Write a .bcmodel file."""
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_size = len(header_json)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(header_json)
        f.write(binary_data)


def convert_model(model_dir: Path, force: bool = False, verbose: bool = True) -> Path:
    """Convert a single model directory to .bcmodel format."""
    output_path = model_dir / "model.bcmodel"
    if output_path.exists() and not force:
        if verbose:
            print(f"  Skipping {model_dir} (.bcmodel exists, use --force)")
        return output_path

    fmt = detect_format(model_dir)
    if verbose:
        print(f"  Format: {fmt}")

    # Read settings and colormap
    settings = read_settings(model_dir)
    labels = read_colormap(model_dir, settings)

    # Build graph based on format
    model_json_path = model_dir / "model.json"
    layers_json_path = model_dir / "layers.json"

    if fmt == "C":
        with open(layers_json_path) as f:
            layers_data = json.load(f)
        graph = build_graph_format_c(layers_data)
    elif fmt == "B":
        with open(model_json_path) as f:
            model_data = json.load(f)
        graph, _ = build_graph_format_b(model_data)
    else:  # Format A
        with open(model_json_path) as f:
            model_data = json.load(f)
        # Resolve sentinel values (-1)
        _resolve_sentinels(model_data, settings)
        graph = build_graph_format_a(model_data)

    # Load weights
    pth_path = model_dir / "model.pth"
    bin_path = model_dir / "model.bin"

    if pth_path.exists():
        if verbose:
            print(f"  Loading weights from {pth_path}")
        raw_weights = load_weights_pth(pth_path)
        weights = map_pth_weights_to_graph(raw_weights, graph)
    elif bin_path.exists() and fmt == "B":
        if verbose:
            print(f"  Loading TF.js weights from {bin_path}")
        with open(model_json_path) as f:
            model_data = json.load(f)
        raw_weights = load_weights_bin_tfjs(bin_path, model_data["weightsManifest"])
        weights = map_tfjs_weights_to_graph(raw_weights, graph)
    elif bin_path.exists():
        if verbose:
            print(f"  Loading native weights from {bin_path}")
        weights = load_weights_bin_native(bin_path, graph)
    else:
        raise FileNotFoundError(f"No weight file found in {model_dir}")

    # Build tensors manifest
    tensors_manifest, binary_data = build_tensors_manifest(weights)

    # Build header
    num_classes = settings.get("outputClasses", _infer_num_classes(graph))
    input_shape = settings.get("expectedInputShape", [1, 256, 256, 256])
    # Convert to NCDHW: add channel dim
    ncdhw_shape = [input_shape[0], 1] + input_shape[1:]

    inference_settings = settings.get("inference", {})
    perf_settings = settings.get("performance", {})

    header = {
        "bcmodel_version": "1.0",
        "metadata": {
            "name": settings.get("name", model_dir.name),
            "description": settings.get("description", ""),
            "type": settings.get("type", "parcellation"),
            "source_framework": "pytorch",
        },
        "input": {
            "shape": ncdhw_shape,
            "dtype": "float32",
            "data_layout": "channels_first",
        },
        "output": {
            "num_classes": num_classes,
            "data_layout": "channels_first",
        },
        "graph": graph,
        "tensors": tensors_manifest,
        "inference": {
            "enable_seq_conv": inference_settings.get("enableSeqConv", False),
            "crop_padding": inference_settings.get("cropPadding", 0),
            "auto_threshold": inference_settings.get("autoThreshold", 0),
            "enable_quantile_norm": inference_settings.get("enableQuantileNorm", False),
            "enable_transpose": inference_settings.get("enableTranspose", True),
        },
        "performance": {
            "estimated_time_seconds": perf_settings.get("estimatedTimeSeconds", 0),
            "memory_requirement_mb": perf_settings.get("memoryRequirementMB", 0),
        },
        "labels": labels,
    }

    write_bcmodel(output_path, header, binary_data)

    size_kb = output_path.stat().st_size / 1024
    total_params = sum(w.size for w in weights.values())
    if verbose:
        print(f"  Params: {total_params:,}")
        print(f"  Graph nodes: {len(graph)}")
        print(f"  Output: {output_path} ({size_kb:.1f} KB)")

    return output_path


def _resolve_sentinels(model_data: dict, settings: dict):
    """Resolve -1 sentinel values in Format A model.json."""
    layers = model_data["layers"]
    num_classes = settings.get("outputClasses", 2)

    for layer in layers:
        if layer["in_channels"] == -1:
            layer["in_channels"] = 1
        if layer["out_channels"] == -1:
            layer["out_channels"] = num_classes


def _infer_num_classes(graph: list[dict]) -> int:
    """Infer number of output classes from the last conv in the graph."""
    for node in reversed(graph):
        if node["op"] == "conv3d":
            return node["params"]["out_channels"]
    return 2


def find_model_dirs(root: Path) -> list[Path]:
    """Find all model directories under meshnet/."""
    meshnet_dir = root / "meshnet"
    if not meshnet_dir.exists():
        return []
    dirs = []
    for d in sorted(meshnet_dir.iterdir()):
        if d.is_dir() and (d / "model.json").exists():
            dirs.append(d)
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Convert models to .bcmodel format")
    parser.add_argument("input", nargs="?", type=Path, help="Model directory to convert")
    parser.add_argument("--all", action="store_true", help="Convert all meshnet models")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .bcmodel files")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    if args.all:
        root = Path(__file__).parent
        model_dirs = find_model_dirs(root)
        if not model_dirs:
            print("No model directories found under meshnet/")
            return

        for model_dir in model_dirs:
            print(f"Converting {model_dir}...")
            try:
                convert_model(model_dir, force=args.force, verbose=not args.quiet)
            except Exception as e:
                print(f"  ERROR: {e}")
            print()

    elif args.input:
        model_dir = args.input.resolve()
        if not model_dir.exists():
            print(f"Error: {model_dir} not found", file=sys.stderr)
            sys.exit(1)
        print(f"Converting {model_dir}...")
        convert_model(model_dir, force=args.force, verbose=not args.quiet)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
