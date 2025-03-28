from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.state import torch_load, load_state_dict
import json
import nibabel as nib
import numpy as np
import time

# Model definitions (same as before)
KERNEL_SIZE = (3,3,3)
LAST_KERNEL_SIZE = (1,1,1)

def normalize(img, qmin=0.02, qmax=0.98):
    """Unit interval preprocessing with clipping"""
    qlow = np.quantile(img, qmin)
    qhigh = np.quantile(img, qmax)
    img = (img - qlow) / (qhigh - qlow)
    img = np.clip(img, 0, 1)  # Clip the values to be between 0 and 1
    return img

def set_channel_num(config, in_channels, n_classes, channels):
  # input layer
  config["layers"][0]["in_channels"] = in_channels
  config["layers"][0]["out_channels"] = channels
  # output layer
  config["layers"][-1]["in_channels"] = channels
  config["layers"][-1]["out_channels"] = n_classes
  # hidden layers
  for layer in config["layers"][1:-1]:
    layer["in_channels"] = layer["out_channels"] = channels
  return config

def construct_layer(dropout_p=0, bnorm=True, gelu=False, *args, **kwargs):
  layers = []
  kwargs["kernel_size"] = KERNEL_SIZE
  layers.append(nn.Conv2d(*args, **kwargs))
  if bnorm:
      layers.append(
          nn.GroupNorm(
              num_groups=1,
              #num_groups=kwargs["out_channels"],
              num_channels=kwargs["out_channels"],
              affine=False,
          )
      )
  relu = lambda x: x.relu()
  gelu = lambda x: x.gelu()
  dropout = lambda x: x.dropout(dropout_p)
  layers.append(gelu if gelu else relu)
  if dropout_p > 0:
    layers.append(dropout)
  return lambda x: x.sequential(layers)

class MeshNet:
  """Configurable MeshNet from https://arxiv.org/pdf/1612.00940.pdf"""
  def __init__(self, in_channels, n_classes, channels, config_file, fat=None):
    """Init"""
    with open(config_file, "r") as f:
      config = set_channel_num(
        json.load(f), in_channels, n_classes, channels
      )
    if fat is not None:
      chn = int(channels * 1.5)
      if fat in {"i", "io"}:
        config["layers"][0]["out_channels"] = chn
        config["layers"][1]["in_channels"] = chn
      if fat == "io":
        config["layers"][-1]["in_channels"] = chn
        config["layers"][-2]["out_channels"] = chn
      if fat == "b":
        config["layers"][3]["out_channels"] = chn
        config["layers"][4]["in_channels"] = chn
        
    # Create a list to store layers with the name 'model' to match state dict keys
    self.model = []
    
    # Populate the model with layers from config
    for block_kwargs in config["layers"][:-1]:  # All but the last layer
      self.model.append(
        construct_layer(
          dropout_p=config["dropout_p"],
          bnorm=config["bnorm"],
          gelu=config["gelu"],
          **{**block_kwargs, "bias": False},
        )
      )
    
    # Handle last layer specially - add it to model list
    last_config = config["layers"][-1]
    self.model.append(
      nn.Conv2d(
        last_config["in_channels"],
        last_config["out_channels"],
        kernel_size=LAST_KERNEL_SIZE,
        padding=last_config["padding"],
        stride=last_config["stride"],
        dilation=last_config["dilation"],
        bias=True # Enable bias in the conv layer
      )
    )
    
  def _forward_impl(self, x):
    # Process all layers except the last one
    for i in range(len(self.model) - 1):
      x = self.model[i](x)
    
    # Apply final layer
    x = self.model[-1](x)
    
    return x
    
  def __call__(self, x):
    """Forward pass"""
    return self._forward_impl(x)

def cast_model_to_fp16(model):
    """Cast model parameters to fp16"""
    from tinygrad.nn.state import get_state_dict
    
    # Get state dict
    state_dict = get_state_dict(model)
    
    # Cast each parameter to fp16 and replace
    for k, v in state_dict.items():
        v.replace(v.cast(dtypes.float16).realize())
    
    print("Model cast to float16")
    return model

def load_nifti(nifti_path):
    """Load a NIfTI file and return its data as a numpy array"""
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.int32)
    # Store affine for later reconstruction
    affine = img.affine
    return data, affine

def save_segmentation(segmentation, affine, output_path):
    """Save the segmentation as a NIfTI file"""
    # Convert to numpy array and get class with highest probability (if multi-class)
    seg_numpy = segmentation.numpy()
    if seg_numpy.shape[1] > 1:  # If multi-class output
        seg_class = np.argmax(seg_numpy, axis=1)[0].astype(np.int32)  # Take argmax along class dimension
    else:
        seg_class = (seg_numpy[0, 0] > 0.5).astype(np.int32)  # Binary threshold
    
    # Create NIfTI image
    seg_img = nib.Nifti1Image(seg_class, affine)
    
    # Save to file
    nib.save(seg_img, output_path)
    print(f"Segmentation saved to {output_path}")

def run_inference(model, nifti_path, output_path):
    """Load NIfTI data, run inference with model, and save the result"""
    print(f"Loading {nifti_path}...")
    volume_data, affine = load_nifti(nifti_path)
    
    print(f"Volume shape: {volume_data.shape}")
    print("Preprocessing volume...")
    input_tensor = Tensor(normalize(volume_data), dtype=dtypes.float).rearrange("... -> 1 1 ...") 
    print("Running inference...")
    start_time = time.time()
    output = model(input_tensor).realize()
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    print("Post-processing and saving result...")
    save_segmentation(output, affine, output_path)

if __name__ == "__main__":
    # Paths
    nifti_path = "t1_crop.nii.gz"  # Input NIfTI file
    model_path = "mindgrab.pth"    # Pretrained model
    config_path = "mindgrab.json"  # Model config
    output_path = "segmentation_output.nii.gz"  # Output segmentation
    
    # Model parameters
    in_chan = 1  # T1 MRI input has 1 channel
    channel = 15  # Number of features in hidden layers
    n_class = 2   # Binary segmentation (background/foreground)
    
    # Initialize model
    print("Initializing model...")
    model = MeshNet(
        in_channels=in_chan, 
        n_classes=n_class,
        channels=channel, 
        config_file=config_path
    )
    
    # Load pretrained weights
    print(f"Loading weights from {model_path}...")
    state_dict = torch_load(model_path)
    load_state_dict(model, state_dict, strict=True)
    print("Model loaded successfully")
    
    # Cast model parameters to fp16
    model = cast_model_to_fp16(model)
    
    # Run inference
    run_inference(model, nifti_path, output_path)
