#!/usr/bin/env python3
"""Convert the catalyst `gn_hdc_deep` MeshNet (24ch / 104-class, affine GroupNorm,
GELU) into a TensorFlow.js *layers-model* for brainchop-test's WebGL2 path.

Why this exists
---------------
Every other brainchop WebGL2 model is a plain `Conv3D -> relu` stack: their
BatchNorm was *folded* into the conv weights (running stats make BN an affine
map that absorbs into the preceding conv). This model is different:

    block = Conv3d(bias=False) -> GroupNorm(num_groups=C, affine=True) -> GELU

GroupNorm here is per-channel instance norm whose statistics depend on the
*input* at run time, so it cannot be folded. brainchop-test already has a
GroupNorm-aware path: a Conv3D whose layer name ends in `_gn` gets per-channel
instance normalization injected after it (see inference-logic.js /
tensor-utils.js `instanceNorm` / `LayerNormInPlace`). But that injected norm is
*non-affine* (zero-mean/unit-var only) -- it drops the learned scale/shift.

To recover the learned affine (gamma/beta) WITHOUT changing the norm math, we
emit, for every block:

    Conv3D(use_bias=False, activation="linear", name="conv3d_{i}_gn")  # -> instance norm injected by the JS loop
    Conv3D(use_bias=True,  activation="linear", name="affine_{i}",      # -> pure per-channel gamma*x + beta
           kernel_size=1, kernel=diag(gamma), bias=beta)
    Activation("gelu", name="activation_{i}")

Why a 1x1x1 Conv3D and not BatchNormalization? tfjs's BatchNormalization layer
is "not implemented for array of rank 5", so it cannot sit in a volumetric
(D,H,W,C) stack. A 1x1x1 Conv3D whose kernel is diagonal (kernel[0,0,0,i,j] =
gamma_i if i==j else 0) with bias = beta computes exactly  y_c = gamma_c*x_c +
beta_c  per channel -- and Conv3D at rank 5 is the one op every brainchop model
already relies on. It is placed AFTER the `_gn` conv (so the JS loop's injected
instance norm runs first) and BEFORE the GELU activation. The GroupNorm eps
(1e-5, tinygrad default) matches the JS `instanceNorm` eps (1e-5), so the
normalize step is matched too. Net result: numerically faithful affine GN with
ZERO inference-loop changes (it is just another linear Conv3D).

The final classifier is a 1x1x1 Conv3D *with* bias (not `_gn`, no norm).

Reads `model.pth` (a safetensors file written by convert_catalyst_gn_hdc_deep.py
-- tinygrad MeshNet key order) using a tiny built-in safetensors reader, so no
torch / safetensors / tensorflow dependency is needed: numpy only.

Usage
-----
    python convert_gn_hdc_deep_to_tfjs.py \
        --weights meshnet/model24chan104cls/model.pth \
        --config  meshnet/model24chan104cls/model.json \
        --outdir  meshnet/model24chan104cls

Writes  <outdir>/model.json  and  <outdir>/model.bin .
"""
import argparse
import json
import os
import struct

import numpy as np

_ST_DT = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8,
}


def read_safetensors(path):
    """Minimal safetensors reader -> dict[name] = np.ndarray (no deps)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        blob = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = _ST_DT[meta["dtype"]]
        a, b = meta["data_offsets"]
        arr = np.frombuffer(blob[a:b], dtype=dt).reshape(meta["shape"])
        out[name] = np.ascontiguousarray(arr)
    return out


def conv3d_layer(name, kernel_hw_shape, dilation, use_bias, activation, inbound):
    filters = kernel_hw_shape  # filters count
    return {
        "class_name": "Conv3D",
        "config": {
            "name": name, "trainable": False, "dtype": "float32",
            "filters": filters, "kernel_size": [3, 3, 3] if dilation != "1x1" else [1, 1, 1],
            "strides": [1, 1, 1], "padding": "same", "data_format": "channels_last",
            "dilation_rate": dilation if dilation != "1x1" else [1, 1, 1],
            "groups": 1, "activation": activation, "use_bias": use_bias,
        },
        "name": name, "inbound_nodes": [[[inbound, 0, 0, {}]]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="model.pth (safetensors format)")
    ap.add_argument("--config", required=True, help="brainchop model.json (layers/dilations)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    sd = read_safetensors(args.weights)
    cfg = json.load(open(args.config))
    layers_cfg = cfg["layers"]
    n_blocks = len(layers_cfg) - 1            # hidden blocks (last entry is classifier)

    # tinygrad MeshNet module indexing: conv at 3i, GN(weight/bias) at 3i+1,
    # activation (no params) at 3i+2; final conv at 3*n_blocks.
    def conv_w(i):   return sd[f"model.{3*i}.weight"]      # [O, I, 3,3,3]
    def gn_g(i):     return sd[f"model.{3*i+1}.weight"]    # [C]
    def gn_b(i):     return sd[f"model.{3*i+1}.bias"]      # [C]
    final_idx = 3 * n_blocks
    final_w = sd[f"model.{final_idx}.weight"]              # [n_classes, C, 1,1,1]
    final_b = sd[f"model.{final_idx}.bias"]                # [n_classes]

    keras_layers = [{
        "class_name": "InputLayer",
        "config": {"batch_input_shape": [None, 256, 256, 256, 1], "dtype": "float32",
                   "sparse": False, "ragged": False, "name": "input"},
        "name": "input", "inbound_nodes": [],
    }]
    manifest = []          # (weight_name, ndarray) in bin order
    prev = "input"

    for i in range(n_blocks):
        d = int(layers_cfg[i]["dilation"])
        filters = int(layers_cfg[i]["out_channels"])
        C = filters
        cname, bname, aname = f"conv3d_{i}_gn", f"affine_{i}", f"activation_{i}"

        # Conv3D (bias-free, linear); JS injects instance norm because name ends "_gn".
        keras_layers.append(conv3d_layer(cname, filters, [d, d, d], False, "linear", prev))
        # torch [O,I,3,3,3] -> keras [3,3,3,I,O]
        k = np.transpose(conv_w(i), (2, 3, 4, 1, 0)).astype(np.float32)
        manifest.append((f"{cname}/kernel", k))

        # 1x1x1 Conv3D with diagonal kernel = per-channel affine (gamma*x + beta).
        keras_layers.append({
            "class_name": "Conv3D",
            "config": {"name": bname, "trainable": False, "dtype": "float32",
                       "filters": C, "kernel_size": [1, 1, 1], "strides": [1, 1, 1],
                       "padding": "same", "data_format": "channels_last",
                       "dilation_rate": [1, 1, 1], "groups": 1,
                       "activation": "linear", "use_bias": True},
            "name": bname, "inbound_nodes": [[[cname, 0, 0, {}]]],
        })
        # keras kernel [1,1,1,C_in,C_out]; diag so out_c = gamma_c * in_c.
        diag = np.zeros((1, 1, 1, C, C), np.float32)
        gamma = gn_g(i).astype(np.float32)
        diag[0, 0, 0, np.arange(C), np.arange(C)] = gamma
        manifest.append((f"{bname}/kernel", diag))
        manifest.append((f"{bname}/bias", gn_b(i).astype(np.float32)))

        # GELU activation.
        keras_layers.append({
            "class_name": "Activation",
            "config": {"name": aname, "trainable": False, "dtype": "float32", "activation": "gelu"},
            "name": aname, "inbound_nodes": [[[bname, 0, 0, {}]]],
        })
        prev = aname

    # Final 1x1x1 classifier conv WITH bias (no norm, no _gn suffix).
    fname = "conv3d_final"
    n_classes = int(final_w.shape[0])
    keras_layers.append({
        "class_name": "Conv3D",
        "config": {"name": fname, "trainable": False, "dtype": "float32",
                   "filters": n_classes, "kernel_size": [1, 1, 1], "strides": [1, 1, 1],
                   "padding": "same", "data_format": "channels_last",
                   "dilation_rate": [1, 1, 1], "groups": 1, "activation": "linear", "use_bias": True},
        "name": fname, "inbound_nodes": [[[prev, 0, 0, {}]]],
    })
    manifest.append((f"{fname}/kernel", np.transpose(final_w, (2, 3, 4, 1, 0)).astype(np.float32)))
    manifest.append((f"{fname}/bias", final_b.astype(np.float32)))

    model_json = {
        "format": "layers-model",
        "generatedBy": "convert_gn_hdc_deep_to_tfjs.py (affine-GN + GELU)",
        "convertedBy": None,
        "modelTopology": {
            "keras_version": "2.6.0", "backend": "tensorflow",
            "model_config": {"class_name": "Functional",
                             "config": {"name": "model", "layers": keras_layers,
                                        "input_layers": [["input", 0, 0]],
                                        "output_layers": [[fname, 0, 0]]}},
        },
        "weightsManifest": [{
            "paths": ["model.bin"],
            "weights": [{"name": n, "shape": list(a.shape), "dtype": "float32"} for n, a in manifest],
        }],
    }

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "model.json"), "w") as f:
        json.dump(model_json, f)
    with open(os.path.join(args.outdir, "model.bin"), "wb") as f:
        for _, a in manifest:
            f.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())

    nbytes = sum(a.size * 4 for _, a in manifest)
    print(f"[ok] {n_blocks} blocks, {n_classes} classes")
    print(f"[ok] wrote {args.outdir}/model.json  ({len(keras_layers)} layers)")
    print(f"[ok] wrote {args.outdir}/model.bin   ({nbytes} bytes, {len(manifest)} tensors)")


if __name__ == "__main__":
    main()
