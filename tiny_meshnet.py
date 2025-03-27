from tinygrad import Tensor, nn
from tinygrad.nn.state import torch_load, load_state_dict
import json

KERNEL_SIZE = (3,3,3) # this is a very stupid hack
LAST_KERNEL_SIZE = (1,1,1)

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
      )
    )
    
    # Add bias to the last layer
    self.final_bias = Tensor.zeros(last_config["out_channels"])

  def _forward_impl(self, x):
    # Process all layers except the last one
    for i in range(len(self.model) - 1):
      x = self.model[i](x)
    
    # Apply final layer
    x = self.model[-1](x)
    
    # Add bias
    if self.final_bias is not None:
      x = x + self.final_bias.reshape(1, -1, 1, 1, 1)
    
    return x

  def __call__(self, x):
    """Forward pass"""
    return self._forward_impl(x)

if __name__ == "__main__":
  in_chan = 1
  channel = 15
  n_class = 2
  config  = "./mindgrabAE.json"
  model = MeshNet(
    in_channels=in_chan, 
    n_classes=n_class,
    channels=channel, 
    config_file=config
  )
  state_dict = torch_load("mindgrab.pth")
  #print(state_dict)
  load_state_dict(model,state_dict,strict=True)
  print('loaded state dict properly')

  x = Tensor.randn(1,1,256,256,256)
  out = model(x)
  print(out.shape)
