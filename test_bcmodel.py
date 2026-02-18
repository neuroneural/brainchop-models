#!/usr/bin/env python3
"""Verify .bcmodel files are valid and self-consistent.

Tests:
  1. Every .bcmodel file parses correctly
  2. Header schema is valid
  3. Tensor shapes match graph expectations
  4. Binary data section is fully accounted for
  5. Round-trip: original weights match .bcmodel weights
  6. JS loader compatibility (header parses as valid JSON)
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Import our loader
sys.path.insert(0, str(Path(__file__).parent))
from load_bcmodel import load_bcmodel
from convert_to_bcmodel import (
    detect_format,
    load_weights_pth,
    load_weights_bin_tfjs,
    map_pth_weights_to_graph,
    map_tfjs_weights_to_graph,
    build_graph_format_a,
    build_graph_format_b,
    build_graph_format_c,
    _resolve_sentinels,
    read_settings,
)

MESHNET_DIR = Path(__file__).parent / "meshnet"
PASS = 0
FAIL = 0


def check(condition: bool, msg: str):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {msg}")
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


def test_parse(bcmodel_path: Path):
    """Test that the .bcmodel file parses correctly."""
    header, tensors = load_bcmodel(bcmodel_path)

    # Schema checks
    check(header.get("bcmodel_version") == "1.0", "bcmodel_version is 1.0")
    check("metadata" in header, "has metadata")
    check("graph" in header, "has graph")
    check("tensors" in header, "has tensors manifest")
    check("input" in header, "has input spec")
    check("output" in header, "has output spec")

    meta = header["metadata"]
    check(bool(meta.get("name")), "metadata.name is non-empty")
    check(meta.get("type") in ("brain-extraction", "tissue-segmentation", "parcellation"),
          f"metadata.type is valid: {meta.get('type')}")

    inp = header["input"]
    check(len(inp["shape"]) == 5, f"input.shape is 5D: {inp['shape']}")
    check(inp["dtype"] == "float32", "input.dtype is float32")
    check(inp["data_layout"] == "channels_first", "data_layout is channels_first")

    out = header["output"]
    check(out["num_classes"] > 0, f"num_classes > 0: {out['num_classes']}")

    return header, tensors


def test_graph_structure(header: dict, tensors: dict):
    """Test graph node structure and weight alignment."""
    graph = header["graph"]
    check(len(graph) > 0, f"graph has {len(graph)} nodes")

    # First node should have empty inputs
    check(graph[0]["inputs"] == [], "first node has no inputs (entry point)")

    # All referenced inputs exist
    node_ids = set()
    for node in graph:
        node_ids.add(node["id"])
    for node in graph:
        for inp in node["inputs"]:
            check(inp in node_ids, f"node {node['id']} references valid input {inp}")

    # Last node is the output
    last_node = graph[-1]
    if last_node["op"] == "conv3d":
        expected_classes = header["output"]["num_classes"]
        check(last_node["params"]["out_channels"] == expected_classes,
              f"last conv out_channels ({last_node['params']['out_channels']}) matches num_classes ({expected_classes})")

    # Check weight tensors exist for conv nodes
    for node in graph:
        if node["op"] == "conv3d":
            weight_key = f"{node['id']}.weight"
            has_weight = weight_key in tensors
            check(has_weight, f"weight tensor exists: {weight_key}")
            if has_weight:
                w = tensors[weight_key]
                expected_shape = (
                    node["params"]["out_channels"],
                    node["params"]["in_channels"],
                    node["params"]["kernel_size"],
                    node["params"]["kernel_size"],
                    node["params"]["kernel_size"],
                )
                check(w.shape == expected_shape,
                      f"{weight_key} shape {w.shape} matches expected {expected_shape}")

            if node["params"].get("bias", False):
                bias_key = f"{node['id']}.bias"
                check(bias_key in tensors, f"bias tensor exists: {bias_key}")


def test_binary_coverage(bcmodel_path: Path, header: dict):
    """Test that binary section is fully accounted for by tensor offsets."""
    with open(bcmodel_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + header_size)
        binary_data = f.read()

    binary_len = len(binary_data)
    max_offset = 0
    for name, info in header["tensors"].items():
        begin, end = info["data_offsets"]
        check(begin >= 0, f"{name} begin offset >= 0")
        check(end <= binary_len, f"{name} end offset <= binary length ({end} <= {binary_len})")
        check(end > begin, f"{name} has positive size")
        max_offset = max(max_offset, end)

    check(max_offset == binary_len,
          f"binary section fully covered: max_offset={max_offset}, binary_len={binary_len}")


def test_weight_consistency(model_dir: Path, header: dict, tensors: dict):
    """Compare .bcmodel weights against original source weights."""
    fmt = detect_format(model_dir)
    model_json_path = model_dir / "model.json"
    pth_path = model_dir / "model.pth"
    bin_path = model_dir / "model.bin"

    if fmt == "B" and bin_path.exists():
        # TF.js format: load original weights, transpose, compare
        with open(model_json_path) as f:
            model_data = json.load(f)
        graph, _ = build_graph_format_b(model_data)
        original = load_weights_bin_tfjs(bin_path, model_data["weightsManifest"])
        mapped = map_tfjs_weights_to_graph(original, graph)

        for key in mapped:
            if key in tensors:
                match = np.allclose(mapped[key], tensors[key], atol=1e-6)
                check(match, f"weight consistency: {key}")
            else:
                check(False, f"weight {key} missing from .bcmodel")

    elif pth_path.exists():
        # PyTorch / SafeTensors: load and compare
        original = load_weights_pth(pth_path)

        if fmt == "C":
            layers_json_path = model_dir / "layers.json"
            with open(layers_json_path) as f:
                layers_data = json.load(f)
            graph = build_graph_format_c(layers_data)
        else:
            with open(model_json_path) as f:
                model_data = json.load(f)
            settings = read_settings(model_dir)
            _resolve_sentinels(model_data, settings)
            graph = build_graph_format_a(model_data)

        mapped = map_pth_weights_to_graph(original, graph)

        matched = 0
        for key in mapped:
            if key in tensors:
                match = np.allclose(mapped[key], tensors[key], atol=1e-6)
                if match:
                    matched += 1
                check(match, f"weight consistency: {key}")

        check(matched > 0, f"at least some weights matched ({matched})")
    else:
        print("  SKIP: no original weights to compare")


def test_js_compatibility(bcmodel_path: Path):
    """Verify the header can be parsed by JavaScript-style JSON parsing."""
    with open(bcmodel_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)

    # Verify it's valid UTF-8
    try:
        header_str = header_bytes.decode("utf-8")
        check(True, "header is valid UTF-8")
    except UnicodeDecodeError:
        check(False, "header is valid UTF-8")
        return

    # Verify it's valid JSON
    try:
        parsed = json.loads(header_str)
        check(True, "header is valid JSON")
    except json.JSONDecodeError as e:
        check(False, f"header is valid JSON: {e}")
        return

    # Verify header_size fits in uint32 (JS uses getUint32)
    check(header_size < 2**32, f"header_size fits in uint32: {header_size}")


def main():
    global PASS, FAIL

    model_dirs = sorted(d for d in MESHNET_DIR.iterdir() if d.is_dir())

    for model_dir in model_dirs:
        bcmodel_path = model_dir / "model.bcmodel"
        if not bcmodel_path.exists():
            print(f"\n{model_dir.name}: SKIP (no .bcmodel)")
            continue

        print(f"\n{'='*60}")
        print(f"Testing: {model_dir.name}")
        print(f"{'='*60}")

        try:
            header, tensors = test_parse(bcmodel_path)
            test_graph_structure(header, tensors)
            test_binary_coverage(bcmodel_path, header)
            test_weight_consistency(model_dir, header, tensors)
            test_js_compatibility(bcmodel_path)
        except Exception as e:
            FAIL += 1
            print(f"  FAIL: unexpected error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    print(f"{'='*60}")

    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
