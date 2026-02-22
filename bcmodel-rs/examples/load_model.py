#!/usr/bin/env python3
"""Load a .bcmodel file and print model summary, tensors, and labels.

Prerequisites:
    cd bcmodel-rs/bcmodel-py
    python -m venv .venv && source .venv/bin/activate
    pip install numpy maturin
    maturin develop

Usage:
    python load_model.py ../../meshnet/model5_gw_ae/model.bcmodel
"""
import sys
from pathlib import Path

import bcmodel


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path/to/model.bcmodel>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    # --- Load model ---
    model = bcmodel.BcmodelFile.load(str(path))
    print(model)
    print()

    # --- Model info ---
    print(f"Name:         {model.name}")
    print(f"Type:         {model.model_type}")
    print(f"Description:  {model.description}")
    print(f"Input shape:  {model.input_shape}")
    print(f"Classes:      {model.num_classes}")
    print(f"Graph nodes:  {model.graph_length}")
    print(f"Parameters:   {model.total_params:,}")
    print(f"Weight size:  {model.weight_size_bytes / 1024:.1f} KB")
    print()

    # --- Architecture graph ---
    print("Architecture:")
    for node in model.graph():
        inputs = " <- " + ", ".join(node["inputs"]) if node["inputs"] else " (input)"
        print(f"  {node['id']:20s} {node['op']:15s}{inputs}")
    print()

    # --- Tensors ---
    print("Tensors:")
    for name in model.tensor_names():
        shape = model.tensor_shape(name)
        tensor = model.tensor(name)
        print(f"  {name:30s} shape={shape}  min={tensor.min():.4f}  max={tensor.max():.4f}")
    print()

    # --- Labels ---
    labels = model.labels()
    if labels:
        print("Labels:")
        for label in labels:
            r, g, b = label["color"]
            print(f"  [{label['index']:3d}] {label['name']:20s} rgb({r}, {g}, {b})")
    print()

    # --- Inference config ---
    config = model.inference_config()
    if config:
        print("Inference config:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    # --- Compatibility API (drop-in replacement for load_bcmodel.py) ---
    print("\n--- Compatibility API ---")
    header, tensors = bcmodel.load_bcmodel(str(path))
    print(f"Version:  {header['bcmodel_version']}")
    print(f"Tensors:  {len(tensors)} arrays")
    first = next(iter(tensors))
    print(f"Example:  {first} -> shape={tensors[first].shape}, dtype={tensors[first].dtype}")


if __name__ == "__main__":
    main()
