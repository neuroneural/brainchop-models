# model16chan18cls — GM/WM + subcortical (18 class), deep gridding-free MeshNet (smallest)

Lightest sibling of `model24chan18cls` / `model32chan18cls`. 18-label
segmentation (gray & white matter plus subcortical regions), trained on
SynthSeg-style synthetic data (synth18 / `label18`).

- 16 channels, 13 conv layers + final 1×1 classifier.
- Symmetric coprime dilation ramp `1,3,5,7,13,19,31,19,13,7,5,3,1`
  (RF = 255 ≈ the 256³ volume, no gridding holes).
- GroupNorm with learnable affine in every block (`affine: true`), GELU,
  no skip/residual connections.
- Hidden convs are bias-free; only the final 1×1 conv carries a bias.
- Lowest peak memory / fastest of the deep 18-class family (16 vs 24/32 ch) —
  intended as the small/fast browser test variant.

Requires brainchop-cli loader support for affine GroupNorm + per-layer `bias`
(`brainchop/tiny_meshnet.py`, ≥ 0.2.4).

## Files
- `model.json` — brainchop MeshNet config (new-style explicit channels).
- `model.pth` — fp32 weights (safetensors; brainchop `safe_load`s it). Produced
  by the conversion step below.

## Provenance
Converted from the catalyst `gn_hdc_deep` checkpoint
(`logs/tmp/synth18_gn_hdc_deep_turbo16/model.last.pth`) via
`tools/convert_catalyst_gn_hdc_deep.py`. Positional key remap
(catalyst `model.L.0/1.*` → tinygrad `model.idx.*`); shapes verified 1:1 and the
result loads `strict=True`.

## Build steps

```sh
# (run from brainchop-cli/)

# 1) checkpoint -> fp32 brainchop weights (model.pth)
python ../brainchop-models/tools/convert_catalyst_gn_hdc_deep.py \
    --ckpt       ../../meshnet/catalyst_example/logs/tmp/synth18_gn_hdc_deep_turbo16/model.last.pth \
    --model-json ../brainchop-models/meshnet/model16chan18cls/model.json \
    --out        ../brainchop-models/meshnet/model16chan18cls/model.pth

# 2) WebGPU runner + fp16/fp32 safetensors, dropped straight into brainchop-test
IGNORE_BEAM_CACHE=0 python examples/export_meshnet_webgpu.py \
    --model-dir     ../brainchop-models/meshnet/model16chan18cls \
    --bct           ../brainchop-test \
    --runner-name   model16chan18cls \
    --web-model-dir model16chan18cls \
    --beam 3 --which both

# 3) WebGL2 fallback: tfjs layers-model (model.json + model.bin)
python ../brainchop-models/tools/convert_gn_hdc_deep_to_tfjs.py \
    --weights ../brainchop-models/meshnet/model16chan18cls/model.pth \
    --config  ../brainchop-models/meshnet/model16chan18cls/model.json \
    --outdir  ../brainchop-test/public/models/model16chan18cls
```

`colormap.json` (18 labels) is shared with `model32chan18cls` and is already in
`brainchop-test/public/models/model16chan18cls/`.
