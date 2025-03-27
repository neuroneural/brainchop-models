from collections import OrderedDict
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp
from torch.utils.checkpoint import checkpoint_sequential
import json
import copy


class MixedConv3d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MixedConv3d, self).__init__()

        nondilated_channels = kwargs.pop("nondilated_channels")
        out_channels = kwargs.pop("out_channels")

        self.conv2 = nn.Conv3d(
            *args,
            **kwargs,
            out_channels=out_channels - nondilated_channels,
        )
        padding = kwargs.pop("padding")
        dilation = kwargs.pop("dilation")
        self.conv1 = nn.Conv3d(
            *args,
            **kwargs,
            out_channels=nondilated_channels,
            padding=1,
            dilation=1,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return torch.cat((x1, x2), dim=1)


class FusedConv3d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FusedConv3d, self).__init__()

        self.conv2 = nn.Conv3d(*args, **kwargs)
        padding = kwargs.pop("padding")
        dilation = kwargs.pop("dilation")
        self.conv1 = nn.Conv3d(
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
    """
    Takes a configuration json for a convolutional neural network of MeshNet architecture and changes it to have the specified number of input channels, output classes, and number of channels that each layer except the input and output layers have.

    Args:
        config (dict): The configuration json for the network.
        in_channels (int): The number of input channels.
        n_classes (int): The number of output classes.
        channels (int): The number of channels that each layer except the input and output layers will have.

    Returns:
        dict: The updated configuration json.
    """
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
    """Constructs a configurable Convolutional block with Batch Normalization and Dropout.

    Args:
    dropout_p (float): Dropout probability. Default is 0.
    bnorm (bool): Whether to include batch normalization. Default is True.
    gelu (bool): Whether to use GELU activation. Default is False.
    *args: Additional positional arguments to pass to nn.Conv3d.
    **kwargs: Additional keyword arguments to pass to nn.Conv3d.

    Returns:
    nn.Sequential: A sequential container of Convolutional block with optional Batch Normalization and Dropout.
    """
    layers = []
    layers.append(nn.Conv3d(*args, **kwargs))
    if bnorm:
        # track_running_stats=False is needed to run the forward mode AD
        # layers.append(
        #     nn.BatchNorm3d(kwargs["out_channels"], track_running_stats=True)
        # )
        layers.append(
            nn.GroupNorm(
                num_groups=kwargs["out_channels"],
                num_channels=kwargs["out_channels"],
                affine=False,
            )
        )

    layers.append(nn.GELU() if gelu else nn.ReLU(inplace=True))
    if dropout_p > 0:
        layers.append(nn.Dropout3d(dropout_p))
    return nn.Sequential(*layers)


def construct_mixedlayer(
    dropout_p=0, bnorm=True, gelu=False, highreslayers=1, *args, **kwargs
):
    """Constructs a configurable Convolutional block with Batch Normalization and Dropout.

    Args:
    dropout_p (float): Dropout probability. Default is 0.
    bnorm (bool): Whether to include batch normalization. Default is True.
    gelu (bool): Whether to use GELU activation. Default is False.
    *args: Additional positional arguments to pass to nn.Conv3d.
    **kwargs: Additional keyword arguments to pass to nn.Conv3d.

    Returns:
    nn.Sequential: A sequential container of Convolutional block with optional Batch Normalization and Dropout.
    """
    layers = []
    if kwargs["dilation"] > 1:
        layers.append(
            MixedConv3d(*args, **kwargs, nondilated_channels=highreslayers)
        )
    else:
        layers.append(nn.Conv3d(*args, **kwargs))
    if bnorm:
        # track_running_stats=False is needed to run the forward mode AD
        layers.append(
            nn.BatchNorm3d(kwargs["out_channels"], track_running_stats=True)
        )
    layers.append(nn.ELU(inplace=True) if gelu else nn.ReLU(inplace=True))
    if dropout_p > 0:
        layers.append(nn.Dropout3d(dropout_p))
    return nn.Sequential(*layers)


def construct_fusedlayer(dropout_p=0, bnorm=True, gelu=False, *args, **kwargs):
    """Constructs a configurable Convolutional block with Batch Normalization and Dropout.

    Args:
    dropout_p (float): Dropout probability. Default is 0.
    bnorm (bool): Whether to include batch normalization. Default is True.
    gelu (bool): Whether to use GELU activation. Default is False.
    *args: Additional positional arguments to pass to nn.Conv3d.
    **kwargs: Additional keyword arguments to pass to nn.Conv3d.

    Returns:
    nn.Sequential: A sequential container of Convolutional block with optional Batch Normalization and Dropout.
    """
    layers = []
    if kwargs["dilation"] > 1:
        layers.append(FusedConv3d(*args, **kwargs))
    else:
        layers.append(nn.Conv3d(*args, **kwargs))
    if bnorm:
        # track_running_stats=False is needed to run the forward mode AD
        layers.append(
            nn.BatchNorm3d(kwargs["out_channels"], track_running_stats=True)
        )
    layers.append(nn.ELU(inplace=True) if gelu else nn.ReLU(inplace=True))
    if dropout_p > 0:
        layers.append(nn.Dropout3d(dropout_p))
    return nn.Sequential(*layers)


def init_weights(model, relu=True):
    """Set weights to be xavier normal for all Convs"""
    for m in model.modules():
        if isinstance(
            m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        ):
            if relu:
                #nn.init.xavier_normal_(
                #    m.weight, gain=nn.init.calculate_gain("relu")
                #)
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            else:
                fan_in = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.in_channels
                )
                nn.init.normal_(
                    m.weight, 0, torch.sqrt(torch.tensor(1.0 / fan_in))
                )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class SequentialConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SequentialConvLayer, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv3d(in_channels, 1, 1) for _ in range(out_channels)]
        )

    def forward(self, x):
        # Size of the input tensor
        batch_size, _, depth, height, width = x.size()

        # Initialize the output cubes
        outB = -10000 * torch.ones(batch_size, 1, depth, height, width).to(
            x.device
        )
        outC = torch.zeros(batch_size, 1, depth, height, width).to(x.device)

        for i, conv in enumerate(self.convs):
            # Apply the current filter
            outA = conv(x)

            # Find where the new filter gives a greater response than the max so far
            greater = outA > outB
            greater = greater.float()

            # Update outB with the max values so far
            outB = (1 - greater) * outB + greater * outA

            # Update outC with the index of the filter with the max response so far
            outC = (1 - greater) * outC + greater * i

        return outC


class MeshNet(nn.Module):
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

        layers = [
            construct_layer(
                dropout_p=config["dropout_p"],
                bnorm=config["bnorm"],
                gelu=config["gelu"],
                # with layer-norm we need no bias as we z-score channels anyway
                **{**block_kwargs, "bias": False},  # **block_kwargs,
            )
            for block_kwargs in config["layers"]
        ]
        # layers[-1] = SequentialConvLayer(
        #    layers[-1][0].in_channels, layers[-1][0].out_channels
        # )
        layers[-1] = layers[-1][0]
        self.model = nn.Sequential(*layers)
        # Add bias to the last layer
        self.model[-1].bias = nn.Parameter(torch.zeros(self.model[-1].out_channels))
        init_weights(self.model)

    def forward(self, x):
        """Forward pass"""
        x = self.model(x)
        return x




class CheckpointMixin:
    def train_forward(self, x):
        y = x
        y.requires_grad_()
        y = checkpoint_sequential(
            self.model, len(self.model), y, preserve_rng_state=False
        )
        return y

    def eval_forward(self, x):
        """Forward pass"""
        self.model.eval()
        with torch.inference_mode():
            x = self.model(x)
        return x

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)


class enMesh_checkpoint(CheckpointMixin, MeshNet):
    pass
