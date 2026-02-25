#!/usr/bin/env python3
"""Load and run inference with .bcmodel files.

Provides a generic graph executor that works with numpy and optionally tinygrad.

Usage:
    python load_bcmodel.py meshnet/model5_gw_ae/model.bcmodel t1_crop.nii.gz -o output.nii.gz
    python load_bcmodel.py meshnet/mindgrab/model.bcmodel t1_crop.nii.gz -o brain_mask.nii.gz
    python load_bcmodel.py meshnet/model5_gw_ae/model.bcmodel --info
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def load_bcmodel(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Parse a .bcmodel file into header and weight tensors.

    Returns:
        (header, tensors) where tensors maps name -> numpy array.
    """
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data = f.read()

    tensors = {}
    for name, info in header["tensors"].items():
        begin, end = info["data_offsets"]
        arr = np.frombuffer(data[begin:end], dtype=np.float32).copy()
        tensors[name] = arr.reshape(info["shape"])

    return header, tensors


def print_info(header: dict, tensors: dict[str, np.ndarray]):
    """Print model summary."""
    meta = header["metadata"]
    inp = header["input"]
    out = header["output"]

    print(f"Name:        {meta['name']}")
    print(f"Type:        {meta['type']}")
    print(f"Description: {meta.get('description', '')}")
    print(f"Input:       {inp['shape']} ({inp['dtype']})")
    print(f"Classes:     {out['num_classes']}")
    print(f"Graph nodes: {len(header['graph'])}")

    total_params = sum(t.size for t in tensors.values())
    print(f"Parameters:  {total_params:,}")

    size_mb = sum(t.nbytes for t in tensors.values()) / (1024 * 1024)
    print(f"Weight size: {size_mb:.2f} MB")

    if header.get("labels"):
        print(f"Labels:      {', '.join(l['name'] for l in header['labels'])}")

    inf = header.get("inference", {})
    if inf:
        print(f"Inference:   seq_conv={inf.get('enable_seq_conv')}, "
              f"crop_pad={inf.get('crop_padding')}, "
              f"threshold={inf.get('auto_threshold')}, "
              f"quantile_norm={inf.get('enable_quantile_norm')}")


# ── NumPy graph executor ──────────────────────────────────────────────

def _conv3d_numpy(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Simple 3D convolution using numpy. For verification only — very slow."""
    # This is a reference implementation. For real inference, use tinygrad.
    # Shape: x=[N,C,D,H,W], weight=[O,C/g,kD,kH,kW]
    N, C, D, H, W = x.shape
    O, C_g, kD, kH, kW = weight.shape

    # Pad input
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0),
                        (padding, padding),
                        (padding, padding),
                        (padding, padding)))

    D_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
    D_out = (D_in - dilation * (kD - 1) - 1) // stride + 1
    H_out = (H_in - dilation * (kH - 1) - 1) // stride + 1
    W_out = (W_in - dilation * (kW - 1) - 1) // stride + 1

    output = np.zeros((N, O, D_out, H_out, W_out), dtype=np.float32)

    for n in range(N):
        for o in range(O):
            g = o // (O // groups)
            c_start = g * C_g
            for d in range(D_out):
                for h in range(H_out):
                    for w in range(W_out):
                        val = 0.0
                        for c in range(C_g):
                            for kd in range(kD):
                                for kh in range(kH):
                                    for kw in range(kW):
                                        di = d * stride + kd * dilation
                                        hi = h * stride + kh * dilation
                                        wi = w * stride + kw * dilation
                                        val += x[n, c_start + c, di, hi, wi] * weight[o, c, kd, kh, kw]
                        output[n, o, d, h, w] = val

    if bias is not None:
        output += bias.reshape(1, -1, 1, 1, 1)

    return output


def _apply_op_numpy(op, params, inputs, tensors, node_id):
    """Apply a single operation using numpy."""
    x = inputs[0] if inputs else None

    if op == "conv3d":
        w = tensors.get(f"{node_id}.weight")
        b = tensors.get(f"{node_id}.bias")
        return _conv3d_numpy(
            x, w, b,
            stride=params.get("stride", 1),
            padding=params.get("padding", 0),
            dilation=params.get("dilation", 1),
            groups=params.get("groups", 1),
        )

    elif op == "relu":
        return np.maximum(x, 0)

    elif op == "elu":
        alpha = params.get("alpha", 1.0)
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    elif op == "gelu":
        return x * 0.5 * (1.0 + np.vectorize(np.math.erf)(x / np.sqrt(2.0)))

    elif op == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))

    elif op == "group_norm":
        num_groups = params["num_groups"]
        eps = params.get("eps", 1e-5)
        affine = params.get("affine", True)
        N, C = x.shape[:2]
        spatial = x.shape[2:]
        grouped = x.reshape(N, num_groups, C // num_groups, *spatial)
        mean = grouped.mean(axis=tuple(range(2, grouped.ndim)), keepdims=True)
        var = grouped.var(axis=tuple(range(2, grouped.ndim)), keepdims=True)
        grouped = (grouped - mean) / np.sqrt(var + eps)
        result = grouped.reshape(N, C, *spatial)
        if affine:
            w = tensors.get(f"{node_id}.weight")
            b = tensors.get(f"{node_id}.bias")
            if w is not None:
                result = result * w.reshape(1, -1, *([1] * len(spatial)))
            if b is not None:
                result = result + b.reshape(1, -1, *([1] * len(spatial)))
        return result

    elif op == "batch_norm3d":
        eps = params.get("eps", 1e-5)
        affine = params.get("affine", True)
        nf = params["num_features"]
        rm = tensors.get(f"{node_id}.running_mean", np.zeros(nf))
        rv = tensors.get(f"{node_id}.running_var", np.ones(nf))
        spatial_dims = len(x.shape) - 2
        shape = [1, nf] + [1] * spatial_dims
        result = (x - rm.reshape(shape)) / np.sqrt(rv.reshape(shape) + eps)
        if affine:
            w = tensors.get(f"{node_id}.weight")
            b = tensors.get(f"{node_id}.bias")
            if w is not None:
                result = result * w.reshape(shape)
            if b is not None:
                result = result + b.reshape(shape)
        return result

    elif op == "dropout":
        return x  # no-op at inference

    elif op == "add":
        result = inputs[0]
        for inp in inputs[1:]:
            result = result + inp
        return result

    elif op == "cat":
        dim = params.get("dim", 1)
        return np.concatenate(inputs, axis=dim)

    elif op == "max_pool3d":
        raise NotImplementedError("max_pool3d not implemented in numpy executor")

    elif op == "upsample":
        raise NotImplementedError("upsample not implemented in numpy executor")

    elif op == "softmax":
        dim = params.get("dim", 1)
        e = np.exp(x - np.max(x, axis=dim, keepdims=True))
        return e / np.sum(e, axis=dim, keepdims=True)

    else:
        raise ValueError(f"Unsupported operation: {op}. Update your runtime.")


def execute_graph(header: dict, tensors: dict[str, np.ndarray],
                  input_tensor: np.ndarray) -> np.ndarray:
    """Execute the model graph on an input tensor.

    Args:
        header: Parsed .bcmodel header
        tensors: Weight tensors from load_bcmodel()
        input_tensor: Input volume as numpy array, shape matching header input.shape
    """
    activations = {}

    for node in header["graph"]:
        node_id = node["id"]
        op = node["op"]
        params = node["params"]
        input_ids = node["inputs"]

        if not input_ids:
            inputs = [input_tensor]
        else:
            inputs = [activations[iid] for iid in input_ids]

        activations[node_id] = _apply_op_numpy(op, params, inputs, tensors, node_id)

    # Return last node's output
    last_id = header["graph"][-1]["id"]
    return activations[last_id]


# ── Tinygrad graph executor ──────────────────────────────────────────

def execute_graph_tinygrad(header: dict, tensors: dict[str, np.ndarray],
                           input_tensor) -> "Tensor":
    """Execute the model graph using tinygrad for GPU acceleration."""
    from tinygrad import Tensor, dtypes
    from tinygrad import nn

    # Convert numpy tensors to tinygrad
    tg_tensors = {k: Tensor(v) for k, v in tensors.items()}

    if isinstance(input_tensor, np.ndarray):
        input_tensor = Tensor(input_tensor, dtype=dtypes.float)

    activations = {}

    for node in header["graph"]:
        node_id = node["id"]
        op = node["op"]
        params = node["params"]
        input_ids = node["inputs"]

        if not input_ids:
            x = input_tensor
        elif len(input_ids) == 1:
            x = activations[input_ids[0]]
        else:
            x = [activations[iid] for iid in input_ids]

        if op == "conv3d":
            w = tg_tensors[f"{node_id}.weight"]
            b = tg_tensors.get(f"{node_id}.bias")
            result = x.conv2d(w, b,
                              stride=params.get("stride", 1),
                              padding=params.get("padding", 0),
                              dilation=params.get("dilation", 1),
                              groups=params.get("groups", 1))
        elif op == "relu":
            result = x.relu()
        elif op == "elu":
            result = x.elu(alpha=params.get("alpha", 1.0))
        elif op == "gelu":
            result = x.gelu()
        elif op == "sigmoid":
            result = x.sigmoid()
        elif op == "group_norm":
            num_groups = params["num_groups"]
            eps = params.get("eps", 1e-5)
            affine = params.get("affine", True)
            N, C = x.shape[:2]
            spatial = x.shape[2:]
            grouped = x.reshape(N, num_groups, C // num_groups, *spatial)
            axes = tuple(range(2, len(grouped.shape)))
            mean = grouped.mean(axis=axes, keepdim=True)
            var = ((grouped - mean) ** 2).mean(axis=axes, keepdim=True)
            grouped = (grouped - mean) / (var + eps).sqrt()
            result = grouped.reshape(N, C, *spatial)
            if affine:
                w = tg_tensors.get(f"{node_id}.weight")
                b = tg_tensors.get(f"{node_id}.bias")
                shape = [1, C] + [1] * len(spatial)
                if w is not None:
                    result = result * w.reshape(*shape)
                if b is not None:
                    result = result + b.reshape(*shape)
        elif op == "batch_norm3d":
            eps = params.get("eps", 1e-5)
            affine = params.get("affine", True)
            nf = params["num_features"]
            spatial_dims = len(x.shape) - 2
            shape = [1, nf] + [1] * spatial_dims
            rm = tg_tensors.get(f"{node_id}.running_mean", Tensor.zeros(nf))
            rv = tg_tensors.get(f"{node_id}.running_var", Tensor.ones(nf))
            result = (x - rm.reshape(*shape)) / (rv.reshape(*shape) + eps).sqrt()
            if affine:
                w = tg_tensors.get(f"{node_id}.weight")
                b = tg_tensors.get(f"{node_id}.bias")
                if w is not None:
                    result = result * w.reshape(*shape)
                if b is not None:
                    result = result + b.reshape(*shape)
        elif op == "dropout":
            result = x
        elif op == "add":
            result = x[0]
            for inp in x[1:]:
                result = result + inp
        elif op == "cat":
            dim = params.get("dim", 1)
            result = x[0].cat(*x[1:], dim=dim)
        elif op == "softmax":
            dim = params.get("dim", 1)
            result = x.softmax(axis=dim)
        else:
            raise ValueError(f"Unsupported operation: {op}. Update your runtime.")

        activations[node_id] = result

    last_id = header["graph"][-1]["id"]
    return activations[last_id]


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Load and run .bcmodel inference")
    parser.add_argument("model", type=Path, help="Path to .bcmodel file")
    parser.add_argument("input", nargs="?", type=Path, help="Input NIfTI file")
    parser.add_argument("-o", "--output", type=Path, default="output.nii.gz",
                        help="Output NIfTI path")
    parser.add_argument("--info", action="store_true", help="Print model info and exit")
    parser.add_argument("--backend", choices=["numpy", "tinygrad"], default="tinygrad",
                        help="Execution backend")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    header, tensors = load_bcmodel(args.model)

    if args.info:
        print_info(header, tensors)
        return

    if not args.input:
        parser.error("Input NIfTI file required for inference")

    try:
        import nibabel as nib
    except ImportError:
        print("Error: nibabel required for NIfTI I/O: pip install nibabel", file=sys.stderr)
        sys.exit(1)

    # Load input
    print(f"Loading {args.input}...")
    img = nib.load(str(args.input))
    data = img.get_fdata().astype(np.float32)
    affine = img.affine

    # Preprocess
    inference = header.get("inference", {})
    if inference.get("enable_quantile_norm", False):
        qmin, qmax = 0.02, 0.98
        qlow = np.quantile(data, qmin)
        qhigh = np.quantile(data, qmax)
        data = np.clip((data - qlow) / (qhigh - qlow + 1e-8), 0, 1)

    # Reshape to NCDHW
    input_tensor = data.reshape(1, 1, *data.shape)

    # Run inference
    import time
    print("Running inference...")
    start = time.time()

    if args.backend == "tinygrad":
        try:
            output = execute_graph_tinygrad(header, tensors, input_tensor)
            output = output.realize().numpy()
        except ImportError:
            print("tinygrad not available, falling back to numpy")
            output = execute_graph(header, tensors, input_tensor)
    else:
        output = execute_graph(header, tensors, input_tensor)

    elapsed = time.time() - start
    print(f"Inference completed in {elapsed:.2f}s")

    # Post-process: argmax to get segmentation
    seg = output.argmax(axis=1)[0].astype(np.int32)

    # Save
    seg_img = nib.Nifti1Image(seg, affine)
    nib.save(seg_img, str(args.output))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
