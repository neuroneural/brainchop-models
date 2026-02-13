# Brainchop Models

Brain MRI segmentation models for [brainchop-cli](https://github.com/neuroneural/brainchop-cli) and [niivue-desktop](https://github.com/niivue/niivue).

## Meshnet

| Model | Type | Classes | Channels | Description |
|---|---|---|---|---|
| mindgrab | brain-extraction | 2 | 15 | MindGrab brain extraction with 5 decoders |
| model5_gw_ae | tissue-segmentation | 3 | 5 | Tissue segmentation (light) |
| model11_gw_ae | brain-extraction | 3 | 11 | Brain extraction (full) |
| model18cls | parcellation | 18 | 21 | Subcortical segmentation |
| model20chan3cls | tissue-segmentation | 3 | 20 | Tissue segmentation (full) |
| model21_104class | parcellation | 104 | 21 | Desikan-Killiany 104-class atlas |
| model30chan18cls | parcellation | 18 | 30 | Subcortical segmentation (30ch) |
| model30chan50cls | parcellation | 50 | 30 | Aparc+Aseg 50-class cortical parcellation |
| subcortical | parcellation | 18 | 21 | Subcortical segmentation (mini) |

## Others

- multiaxial (ONNX multi-planar models)

## Model Directory Structure

Each meshnet model directory contains:

```
meshnet/<model>/
  model.json       # Architecture definition
  model.pth        # PyTorch weights
  model.bin        # Raw float32 weights (converted from .pth)
  settings.json    # Inference configuration (self-describing)
  colormap.json    # Label names and colors
  preview.png      # Segmentation preview image
```

### settings.json

Each model includes a `settings.json` following the [ModelSettings schema](https://github.com/niivue/niivue/blob/main/packages/niivue-desktop/src/renderer/src/services/brainchop/settingsSchema.ts) for self-describing models:

```json
{
  "name": "Model Name",
  "description": "Model description",
  "type": "tissue-segmentation | brain-extraction | parcellation",
  "outputClasses": 3,
  "expectedInputShape": [1, 256, 256, 256],
  "inference": {
    "enableSeqConv": false,
    "cropPadding": 18,
    "autoThreshold": 0.02,
    "enableQuantileNorm": false,
    "enableTranspose": true
  },
  "performance": {
    "estimatedTimeSeconds": 3,
    "memoryRequirementMB": 400
  },
  "files": {
    "labels": "colormap.json",
    "preview": "preview.png"
  }
}
```

## Converting .pth to .bin

Convert PyTorch weight files to raw float32 binary:

```bash
# Convert a single model
python pth_to_bin.py meshnet/mindgrab/model.pth

# Convert all .pth files missing a .bin
python pth_to_bin.py --all

# Force reconvert all (overwrite existing .bin)
python pth_to_bin.py --all --force
```

Requires: `pip install torch`

## Example Data

Download a sample T1-weighted brain MRI:

```
curl -L https://github.com/neuroneural/brainchop-models/raw/main/t1_crop.nii.gz -o t1_crop.nii.gz
```

## Integration

The models here are tied to [this](https://github.com/neuroneural/brainchop-cli/blob/main/models.json) file.

Brainchop-cli can pull the models if they're available in the same directory in this repository.
