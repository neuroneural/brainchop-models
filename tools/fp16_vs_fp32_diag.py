#!/usr/bin/env python3
"""fp16-vs-fp32 per-layer divergence diagnostic for the MeshNet stack.

Loads a brainchop `model.pth` (safetensors) + `model.json`, runs the forward
pass twice on the same input -- once in fp32 (reference) and once in "true fp16"
(f16 storage at the inter-layer activation buffers and weights, f32 conv
accumulation, f32 GroupNorm) -- and reports, per layer:

  * max |fp16 - fp32| and relative error of the activation,
  * the f16 storage dynamic range (max |x|, smallest non-zero |x|) so you can see
    overflow (-> inf, >65504) or underflow (-> subnormal/zero, <~6e-5),
  * final per-voxel argmax disagreement between the two paths.

This isolates WHY fp16 misbehaves:
  - error that stays tiny then explodes at one layer with a huge max|x| -> OVERFLOW
    (rescaling / clamping that tensor helps),
  - error that grows smoothly ~2x per layer with in-range values -> RELATIVE
    QUANTIZATION compounding (only fp32 or a less-sensitive retrain helps).

Input: --nii FILE (needs nibabel) or --npy FILE (a preprocessed float volume),
else a synthetic structured volume. Spatial size is capped by --size for speed;
numerics are exercised regardless of anatomical realism.

No torch required (safetensors read in pure numpy).
"""
import argparse, json, struct, sys
import numpy as np

F16MAX = 65504.0
F16_MIN_NORMAL = 6.103515625e-05


# ---------- safetensors (numpy-only) ----------
def load_safetensors(path):
    dtm = {"F16": np.float16, "F32": np.float32, "F64": np.float64,
           "I64": np.int64, "I32": np.int32}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        blob = f.read()
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(blob[s:e], dtype=dtm[v["dtype"]]).reshape(v["shape"]).copy()
    return out


# ---------- ops ----------
def conv3d(x, w, b, dilation, padding, dtype):
    """x: (Cin,D,H,W) f32; w: (Cout,Cin,kd,kh,kw); returns (Cout,D,H,W) f32.
    Accumulation always in f32 (mirrors the GPU's f32 accumulators)."""
    x = x.astype(np.float32)
    w = w.astype(np.float32)
    Cin, D, H, W = x.shape
    Cout = w.shape[0]
    kd, kh, kw = w.shape[2:]
    p = padding
    xp = np.pad(x, ((0, 0), (p, p), (p, p), (p, p)))
    sC, sD, sH, sW = xp.strides
    # strided windows: (Cin, D, H, W, kd, kh, kw), kernel taps stride = dilation
    win = np.lib.stride_tricks.as_strided(
        xp,
        shape=(Cin, D, H, W, kd, kh, kw),
        strides=(sC, sD, sH, sW, sD * dilation, sH * dilation, sW * dilation),
        writeable=False,
    )
    out = np.einsum("cdhwijk,ocijk->odhw", win, w, optimize=True).astype(np.float32)
    if b is not None:
        out += b.astype(np.float32)[:, None, None, None]
    return out


def instance_norm_affine(h, gamma, beta, eps=1e-5):
    """num_groups == num_channels => per-channel norm over spatial dims, f32."""
    mu = h.mean(axis=(1, 2, 3), keepdims=True)
    var = h.var(axis=(1, 2, 3), keepdims=True)
    y = (h - mu) / np.sqrt(var + eps)
    return y * gamma.astype(np.float32)[:, None, None, None] + beta.astype(np.float32)[:, None, None, None]


def gelu(x):  # tinygrad's tanh approximation
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def quantize_f16(x):
    """Storage cast to f16 then back to f32; values >F16MAX become inf."""
    return x.astype(np.float16).astype(np.float32)


def f16_stats(x):
    q = x.astype(np.float16)
    ninf = int(np.isinf(q).sum())
    a = np.abs(x)
    nz = a[a > 0]
    smallest = float(nz.min()) if nz.size else 0.0
    nsub = int(((a > 0) & (a < F16_MIN_NORMAL)).sum())  # underflow to subnormal/zero
    return float(a.max()), smallest, ninf, nsub


# ---------- model ----------
def build_layers(cfg, sd):
    """Return list of dicts describing each conv block in order."""
    convs = [i for i in range(0, 3 * (len(cfg["layers"]) - 1) + 1, 3)]
    layers = []
    for li, lcfg in enumerate(cfg["layers"]):
        cidx = convs[li]
        w = sd[f"model.{cidx}.weight"]
        b = sd.get(f"model.{cidx}.bias")
        gn_w = sd.get(f"model.{cidx + 1}.weight")
        gn_b = sd.get(f"model.{cidx + 1}.bias")
        layers.append(dict(cfg=lcfg, w=w, b=b, gn_w=gn_w, gn_b=gn_b,
                           is_final=(li == len(cfg["layers"]) - 1)))
    return layers


def forward(x0, layers, fp16):
    x = x0.astype(np.float32)
    if fp16:
        x = quantize_f16(x)
    acts = []
    for L in layers:
        c = L["cfg"]
        w = L["w"].astype(np.float16).astype(np.float32) if fp16 else L["w"].astype(np.float32)
        b = L["b"]
        h = conv3d(x, w, b, c["dilation"], c["padding"], None)  # f32 conv buffer
        if not L["is_final"]:
            h = instance_norm_affine(h, L["gn_w"], L["gn_b"])    # f32 GroupNorm
            h = gelu(h)                                          # f32 activation
            if fp16:
                h = quantize_f16(h)                              # <-- f16 storage
        acts.append(h)
        x = h
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--nii")
    ap.add_argument("--npy")
    ap.add_argument("--size", type=int, default=32, help="cube side for synthetic/cropped input")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    sd = load_safetensors(args.weights)
    layers = build_layers(cfg, sd)
    print(f"[model] {len(layers)} conv layers; "
          f"affine={cfg.get('affine')}, gelu={cfg.get('gelu')}")

    # ----- input -----
    S = args.size
    if args.nii:
        import nibabel as nib
        vol = np.asarray(nib.load(args.nii).get_fdata(), np.float32)
        ql, qh = np.quantile(vol, 0.02), np.quantile(vol, 0.98)
        vol = np.clip((vol - ql) / (qh - ql + 1e-3), 0, 1)
        # center crop to SxSxS for tractability
        c = [s // 2 for s in vol.shape]
        h = S // 2
        vol = vol[c[0]-h:c[0]+h, c[1]-h:c[1]+h, c[2]-h:c[2]+h]
        x0 = vol[None]
        print(f"[input] real volume, cropped to {vol.shape}")
    elif args.npy:
        vol = np.load(args.npy).astype(np.float32)
        x0 = vol[None] if vol.ndim == 3 else vol
        print(f"[input] npy {vol.shape}")
    else:
        rng = np.random.default_rng(args.seed)
        zz, yy, xx = np.mgrid[0:S, 0:S, 0:S].astype(np.float32)
        ctr = (S - 1) / 2
        r = np.sqrt((zz-ctr)**2 + (yy-ctr)**2 + (xx-ctr)**2)
        brain = (r < S*0.4).astype(np.float32)
        tex = 0.5 + 0.5 * rng.random((S, S, S)).astype(np.float32)
        x0 = (brain * tex)[None]
        print(f"[input] synthetic structured volume {S}^3 (seed {args.seed})")

    ref = forward(x0, layers, fp16=False)
    half = forward(x0, layers, fp16=True)

    print("\n layer   dilation   max|a|(f32)   max|f16-f32|   relerr     f16:max|x|  min|x>0|  inf  underflow")
    for i, (a, b, L) in enumerate(zip(ref, half, layers)):
        d = L["cfg"]["dilation"]
        adiff = np.abs(a - b)
        denom = np.abs(a).max() + 1e-12
        relerr = adiff.max() / denom
        mx, smallest, ninf, nsub = f16_stats(b)
        tag = "  FINAL(logits)" if L["is_final"] else ""
        flag = ""
        if ninf:
            flag = "  <== OVERFLOW (inf)"
        elif mx > 0.3 * F16MAX:
            flag = "  <== near f16 max"
        print(f"  {i:2d}      {d:3d}      {np.abs(a).max():10.3f}   {adiff.max():11.4f}   "
              f"{relerr:8.2e}   {mx:9.1f}  {smallest:8.1e}  {ninf:4d}  {nsub:9d}{tag}{flag}")

    # final argmax disagreement
    rl = np.argmax(ref[-1], axis=0)
    hl = np.argmax(half[-1], axis=0)
    mask = x0[0] > 0  # foreground only
    dis_all = float((rl != hl).mean())
    dis_fg = float((rl[mask] != hl[mask]).mean()) if mask.any() else 0.0
    print(f"\n[argmax] voxels where fp16 label != fp32 label: "
          f"all={dis_all*100:.3f}%   foreground={dis_fg*100:.3f}%")
    print("[interp] smooth ~2x relerr growth w/ in-range values => relative "
          "quantization (fp32/retrain). A jump at one layer with huge max|x| or "
          "inf => overflow (rescale/clamp that layer).")


if __name__ == "__main__":
    main()
