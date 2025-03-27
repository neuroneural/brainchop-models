from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.state import torch_load, load_state_dict
import json
import nibabel as nib
import numpy as np
import time
import cc3d
import os
from glob import glob
from pathlib import Path
from tinygrad.helpers import tqdm

KERNEL_SIZE = (3,3,3)
LAST_KERNEL_SIZE = (1,1,1)

def set_channel_num(config, in_channels, n_classes, channels):
  config["layers"][0]["in_channels"] = in_channels
  config["layers"][0]["out_channels"] = channels
  config["layers"][-1]["in_channels"] = channels
  config["layers"][-1]["out_channels"] = n_classes
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
        num_groups=kwargs["out_channels"],
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
  def __init__(self, in_channels, n_classes, channels, config_file, fat=None):
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
        
    self.model = []
    
    for block_kwargs in config["layers"][:-1]:
      self.model.append(
        construct_layer(
          dropout_p=config["dropout_p"],
          bnorm=config["bnorm"],
          gelu=config["gelu"],
          **{**block_kwargs, "bias": False},
        )
      )
    
    last_config = config["layers"][-1]
    self.model.append(
      nn.Conv2d(
        last_config["in_channels"],
        last_config["out_channels"],
        kernel_size=LAST_KERNEL_SIZE,
        padding=last_config["padding"],
        stride=last_config["stride"],
        dilation=last_config["dilation"],
        bias=True
      )
    )
    
  def _forward_impl(self, x):
    for i in range(len(self.model) - 1):
      x = self.model[i](x)
    
    x = self.model[-1](x)
    return x
    
  def __call__(self, x):
    return self._forward_impl(x)

def cast_model_to_fp16(model):
  from tinygrad.nn.state import get_state_dict
  state_dict = get_state_dict(model)
  for k, v in state_dict.items():
    v.replace(v.cast(dtypes.float16).realize())
  return model

def normalize(data_array, qmin=0.02, qmax=0.98):
  numpy_data = data_array.numpy() if hasattr(data_array, 'numpy') else data_array
  qlow = np.quantile(numpy_data, qmin)
  qhigh = np.quantile(numpy_data, qmax)
  normalized = (numpy_data - qlow) / (qhigh - qlow)
  normalized = np.clip(normalized, 0, 1)
  
  if hasattr(data_array, 'numpy'):
    return Tensor(normalized, dtype=data_array.dtype)
  return normalized

def load_nifti(nifti_path):
  img = nib.load(nifti_path)
  data = img.get_fdata().astype(np.float32)
  return data, img.affine, img.header

def preprocess_volume(data):
  # First normalize the numpy data
  data = normalize(data)
  
  # Create tensor with shape [1, 1, 256, 256, 256]
  # This matches the expected input format for 3D medical image processing
  data_tensor = Tensor(data[np.newaxis, np.newaxis, :, :, :], dtype=dtypes.float32)
  
  # Cast to fp16 if configured
  if USE_FP16:
    data_tensor = data_tensor.cast(dtypes.float16)
    
  return data_tensor

def postprocess_segmentation(segmentation):
  # Handle tensor with shape [1, 2, 256, 256, 256] or [2, 256, 256, 256]
  seg_numpy = segmentation.numpy()
  print(f"Segmentation tensor shape: {seg_numpy.shape}")
  
  # Check if we have shape [1, 2, 256, 256, 256] (with batch dimension)
  if len(seg_numpy.shape) == 5:
    seg_numpy = seg_numpy[0]  # Remove batch dimension
    
  # Now we should have shape [2, 256, 256, 256] (for binary classification)
  if seg_numpy.shape[0] == 2:
    # Binary classification case (background/foreground)
    seg_class = np.argmax(seg_numpy, axis=0).astype(np.uint8)
  else:
    # Single channel case
    seg_class = (seg_numpy > 0.5).astype(np.uint8)
  
  return seg_class

def save_segmentation(segmentation, affine, header, output_path):
  seg_img = nib.Nifti1Image(segmentation, affine, header)
  nib.save(seg_img, output_path)

def process_file(model, nifti_path, output_path, connectivity=26):
  volume_data, affine, header = load_nifti(nifti_path)
  input_tensor = preprocess_volume(volume_data)
  
  print(f"Input tensor shape: {input_tensor.shape}")
  start_time = time.time()
  output = model(input_tensor)
  inference_time = time.time() - start_time
  print(f"Inference completed in {inference_time:.2f} seconds")
  print(f"Output tensor shape: {output.shape}")
  
  segmentation = postprocess_segmentation(output) # this should do connected components
  save_segmentation(segmentation, affine, header, output_path)

def process_directory(model, input_dir, output_dir, connectivity=26):
  os.makedirs(output_dir, exist_ok=True)
  
  nifti_files = glob(os.path.join(input_dir, "*.nii.gz"))
  if not nifti_files:
    nifti_files = glob(os.path.join(input_dir, "*.nii"))
  
  print(f"Found {len(nifti_files)} files in {input_dir}")
  
  for image_path in tqdm(nifti_files, desc=f"Processing {Path(input_dir).name}"):
    try:
      image_name = os.path.basename(image_path)
      if image_name.endswith('.nii.gz'):
        image_name = image_name[:-7]
      elif image_name.endswith('.nii'):
        image_name = image_name[:-4]
      mask_image_path = os.path.join(output_dir, f"{image_name}_mask.nii.gz")
      
      process_file(model, image_path, mask_image_path, connectivity)
      
    except Exception as e:
      print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
  USE_FP16 = False
  CONNECTIVITY = 26
  
  nifti_path = "t1_crop.nii.gz"
  model_path = "mindgrab.pth"
  config_path = "mindgrabAE.json"
  output_path = "segmentation_output.nii.gz"
  
  in_chan = 1
  channel = 15
  n_class = 2
  
  print("Initializing model...")
  model = MeshNet(
    in_channels=in_chan, 
    n_classes=n_class,
    channels=channel, 
    config_file=config_path
  )
  
  print(f"Loading weights from {model_path}...")
  state_dict = torch_load(model_path)
  load_state_dict(model, state_dict, strict=True)
  
  if USE_FP16:
    model = cast_model_to_fp16(model)
  
  process_file(model, nifti_path, output_path, CONNECTIVITY)
  
  # Base directory processing code (commented)
  # base_input_dir = "input/data"
  # base_output_dir = "output/results"
  # modalities = ['t1', 't2', 'flair']
  # for modality in modalities:
  #   input_dir = os.path.join(base_input_dir, modality, "images")
  #   output_dir = os.path.join(base_output_dir, modality)
  #   print(f"\nProcessing {modality} modality...")
  #   process_directory(model, input_dir, output_dir, CONNECTIVITY)
