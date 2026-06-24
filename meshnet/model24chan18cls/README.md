# model24chan18cls — GM/WM + subcortical (18 class), deep gridding-free MeshNet (small)

Smaller (24-channel) sibling of `model32chan18cls`; backs the
`robust_subcortical_mini` model in brainchop-cli. 18-label segmentation
(gray & white matter plus subcortical regions), trained on SynthSeg-style
synthetic data (synth18 / `label18`).

- 24 channels, 13 conv layers + final 1×1 classifier.
- Symmetric coprime dilation ramp `1,3,5,7,13,19,31,19,13,7,5,3,1`
  (RF = 255 ≈ the 256³ volume, no gridding holes).
- GroupNorm with learnable affine in every block (`affine: true`), GELU,
  no skip/residual connections.
- Hidden convs are bias-free; only the final 1×1 conv carries a bias.
- Lower peak memory / faster than `model32chan18cls` (24 vs 32 channels).

Requires brainchop-cli loader support for affine GroupNorm + per-layer `bias`
(`brainchop/tiny_meshnet.py`, ≥ 0.2.4).

## Files
- `model.json` — brainchop MeshNet config (new-style explicit channels).
- `model.pth` — fp32 weights (safetensors; brainchop `safe_load`s it).

## Provenance
Converted from the catalyst `gn_hdc_deep_fast_turbo` checkpoint
(`logs/tmp/synth18_gn_hdc_deep_turbo/model.last.pth`, macro-dice ≈ 0.78) via
`tools/convert_catalyst_gn_hdc_deep.py`. Positional key remap
(catalyst `model.L.0/1.*` → tinygrad `model.idx.*`); shapes verified 1:1 and the
result loads `strict=True`.
