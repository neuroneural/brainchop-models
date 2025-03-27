from tinygrad import Tensor, nn
import json


class MixedConv3d:
  def __init__(self, *args, **kwargs):
    nondilated_channels = kwargs.pop("nondilated_channels")
    out_channels = kwargs.pop("out_channels")

    self.conv2 = nn.Conv2d(
        *args,
        **kwargs,
        out_channels=out_channels - nondilated_channels,
    )
    padding = kwargs.pop("padding")
    dilation = kwargs.pop("dilation")
    self.conv1 = nn.Conv2d(
        *args,
        **kwargs,
        out_channels=nondilated_channels,
        padding=1,
        dilation=1,
    )

  def forward(self, x):
    x1 = self.conv1(x)
    x2 = self.conv2(x)
    return Tensor.cat(x1, x2, dim=1)


class FusedConv3d:
  def __init__(self, *args, **kwargs):
    super(FusedConv3d, self).__init__()

    self.conv2 = nn.Conv2d(*args, **kwargs)
    padding = kwargs.pop("padding")
    dilation = kwargs.pop("dilation")
    self.conv1 = nn.Conv2d(
        *args,
        **kwargs,
        padding=1,
        dilation=1,
    )

  def forward(self, x):
    x1 = self.conv1(x)
    x2 = self.conv2(x)
    return x1 + x2


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

    super(MeshNet, self).__init__()

    self.layers = []
    for block_kwargs in config["layers"]:
        self.layers.append(
            construct_layer(
                dropout_p=config["dropout_p"],
                bnorm=config["bnorm"],
                gelu=config["gelu"],
                **{**block_kwargs, "bias": False},
            )
        )
    
    # Handle last layer specially
    self.layers[-1] = nn.Conv2d(
        config["layers"][-1]["in_channels"],
        config["layers"][-1]["out_channels"],
        config["layers"][-1]["kernel_size"],
        padding=config["layers"][-1]["padding"],
        stride=config["layers"][-1]["stride"],
        dilation=config["layers"][-1]["dilation"],
    )
    
    # Add bias to the last layer
    self.final_bias = Tensor.zeros(config["layers"][-1]["out_channels"])
    
    self.model = lambda x: self._forward_impl(x)

  def _forward_impl(self, x):
    for _, layer in enumerate(self.layers[:-1]):
      x = layer(x)
    
    # Apply final layer without activation
    x = self.layers[-1](x)
    
    # Add bias
    if self.final_bias is not None:
      x = x + self.final_bias.reshape(1, -1, 1, 1, 1)
    
    return x

  def __call__(self, x):
    """Forward pass"""
    return self.model(x)

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
  x = Tensor.randn(1,1,256,256)
  out = model(x)
  print(out.shape)
  state_dict = nn.state.torch_load("mindgrab.pth")
  print(state_dict)

  
