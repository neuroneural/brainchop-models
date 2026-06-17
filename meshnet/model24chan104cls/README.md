# model24chan104cls — DK-atlas (104 class), deep gridding-free MeshNet

Drop-in replacement for `model21_104class` as the `DKatlas` model in
brainchop-cli. Same task (Desikan-Killiany atlas, 104 labels), more robust
architecture.

- 24 channels, 13 conv layers + final 1×1 classifier.
- Symmetric coprime dilation ramp `1,3,5,7,13,19,31,19,13,7,5,3,1`
  (RF = 255 ≈ the 256³ volume, no gridding holes).
- GroupNorm with learnable affine in every block (`affine: true`), GELU
  activations, no skip/residual connections.
- Hidden convs are bias-free; only the final 1×1 conv carries a bias.
- Peak activation memory ≈ 24 × 256³ (one reused buffer), comparable to the
  21-channel model it replaces, so it runs anywhere brainchop runs (incl.
  WebGPU export).

`affine` GroupNorm and the per-layer `bias` override require the loader change
in brainchop-cli (`brainchop/tiny_meshnet.py`); older brainchop releases load
only models without affine GroupNorm.

## Files
- `model.json` — brainchop MeshNet config (new-style explicit channels).
- `model.pth` — fp32 weights (safetensors; brainchop `safe_load`s it).
- `colormap.json` — 104-class DK-atlas colors (copied from `model21_104class`,
  identical label set).

## Provenance
Converted from the catalyst `gn_hdc_deep_fast_100` checkpoint
(`logs/tmp/synth104_gn_hdc_deep/model.last.pth`, macro-dice ≈ 0.83) via
`tools/convert_catalyst_gn_hdc_deep.py`. Conversion is a positional key rename
(catalyst `model.L.0/1.*` → tinygrad `model.idx.*`); shapes verified 1:1 and the
result loads `strict=True`.
