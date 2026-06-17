#!/usr/bin/env python3
"""Convert a catalyst `gn_hdc_deep` 104-class checkpoint into brainchop weights.

The catalyst MeshNet (meshnet_gn.py) is a flat conv stack where every hidden
block is  Conv3d(bias=False) -> GroupNorm(affine=True) -> GELU  and the final
1x1 conv carries a bias. brainchop's tinygrad MeshNet builds the same parameter
sequence, so we just rename keys positionally and save as safetensors (brainchop
loads `model.pth` via safe_load first, falling back to torch).

Two ways to run:

  # A) in an env with torch (e.g. ~/venv/brainchop.dev) -- simplest, exact:
  python convert_catalyst_gn_hdc_deep.py \
      --ckpt /path/to/synth104_gn_hdc_deep/model.last.pth \
      --model-json meshnet/model24chan104cls/model.json \
      --out meshnet/model24chan104cls/model.pth

  # B) without torch -- a pure-python torch.pickle reader is used as a fallback.

This produces fp32 weights matching the tinygrad MeshNet state dict, verified to
load strict=True. Hidden convs are bias-free; only the final classifier conv has
a bias; GroupNorm is affine (13 weight/bias pairs).
"""
import argparse, io, json, os, pickle, zipfile
import numpy as np

_DT = {"FloatStorage": np.float32, "HalfStorage": np.float16,
       "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32}


def read_state_dict_torch(path):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for w in ("state_dict", "model_state_dict", "model"):
        if isinstance(sd, dict) and w in sd and isinstance(sd[w], dict):
            sd = sd[w]
            break
    return [(k, v.detach().cpu().float().numpy()) for k, v in sd.items()]


def read_state_dict_pure(path):
    """Read a torch .pth (zip-pickle) without importing torch."""
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    root = pkl[:-len("/data.pkl")] if "/" in pkl else ""

    class Stub:
        def __init__(self, *a, **k): pass

    def rebuild(storage, off, size, stride, *a, **k):
        return ("T", storage, int(off), tuple(size), tuple(stride))

    class TU(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if name == "OrderedDict":
                from collections import OrderedDict
                return OrderedDict
            if name.endswith("Storage"):
                return ("STORAGE", name)
            return Stub

        def persistent_load(self, pid):
            return ("S", pid[2], pid[1][1] if isinstance(pid[1], tuple) else str(pid[1]), pid[4])

    sd = TU(io.BytesIO(z.read(pkl))).load()
    for w in ("state_dict", "model_state_dict", "model"):
        if isinstance(sd, dict) and w in sd and isinstance(sd[w], dict):
            sd = sd[w]
            break

    def materialize(t):
        _, storage, off, shape, stride = t
        _, skey, dtname, _ = storage
        dt = _DT[dtname]
        raw = z.read(f"{root}/data/{skey}" if root else f"data/{skey}")
        flat = np.frombuffer(raw, dtype=dt)
        arr = np.lib.stride_tricks.as_strided(
            flat[off:], shape=shape, strides=tuple(s * dt().itemsize for s in stride))
        return np.ascontiguousarray(arr).astype(np.float32)

    return [(k, materialize(v)) for k, v in sd.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        catalyst = read_state_dict_torch(args.ckpt)
        print("[read] used torch")
    except Exception as e:
        print(f"[read] torch unavailable ({e}); using pure-python reader")
        catalyst = read_state_dict_pure(args.ckpt)

    # Build the brainchop tinygrad MeshNet to learn target key order/shapes.
    from brainchop.tiny_meshnet import MeshNet, load_meshnet
    from tinygrad import Tensor
    from tinygrad.nn.state import get_state_dict, safe_save

    cfg = json.load(open(args.model_json))
    in_c = cfg["layers"][0]["in_channels"]
    chan = cfg["layers"][0]["out_channels"]
    out_c = cfg["layers"][-1]["out_channels"]
    model = MeshNet(in_channels=in_c, n_classes=out_c, channels=chan, config_file=args.model_json)
    tiny = get_state_dict(model)

    assert len(catalyst) == len(tiny), f"param count {len(catalyst)} != {len(tiny)}"
    state = {}
    for (ck, arr), (tk, tt) in zip(catalyst, tiny.items()):
        assert tuple(arr.shape) == tuple(tt.shape), f"shape {ck}{arr.shape} != {tk}{tuple(tt.shape)}"
        state[tk] = Tensor(arr)

    safe_save(state, args.out)
    print(f"[save] {args.out} ({os.path.getsize(args.out)} bytes, {len(state)} tensors)")

    # strict reload sanity check
    load_meshnet(args.model_json, args.out)
    print("[verify] load_meshnet strict=True OK")


if __name__ == "__main__":
    main()
