"""ResBlock, Sequential Res-DilatedConv"""


from __future__ import annotations
from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from omegaconf import MISSING, II

from .merge import list_default
from .initializer import init_conv_norm


@dataclass
class ConfResDilatedConv:
    """Configuration of the The ResDilatedConv."""
    channel:   int       = MISSING         # Channel dimension size of input/middle/output
    kernel:    int       = MISSING         # Kernel size of convolution
    dilations: list[int] = list_default(1) # Dilation factors of sequential dilated Convolutions

class ResDilatedConv(nn.Module):
    """Multiple Dilated Conv1d + 1 residual connection, shape preserved."""
    def __init__(self, conf: ConfResDilatedConv):
        super().__init__()

        # For variable-layer convolution with residual connection, ResDilatedConv force below rules:
        channel, stride, padding = conf.channel, 1, "same"
        # kernel size can have variation, but now fixed:
        kernel = conf.kernel
        # Larger LReLU slope
        lrelu_slope = 0.1

        convs: list[nn.Module] = []
        for dilation_i in conf.dilations:
            convs += [nn.LeakyReLU(lrelu_slope)]
            convs += [weight_norm(nn.Conv1d(channel, channel, kernel, stride, dilation=dilation_i, padding=padding))]
        self.convs = nn.Sequential(*convs)
        self.convs.apply(init_conv_norm)

    # Typing of PyTorch forward API is poor.
    def forward(self, ipt: Tensor) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward :: (B, Feat, T) -> (B, Feat, T)"""
        return ipt + self.convs(ipt)

    # TODO: remove_weight_norm

@dataclass
class ConfResBlock:
    """Configuration of `ResBlock`."""
    channel:  int                      = MISSING                          # Channel dimension size of input/middle/output
    kernel:   int                      = MISSING                          # Kernel size of convolutions
    resconvs: list[ConfResDilatedConv] = list_default(ConfResDilatedConv( # Sequence of ResDilatedConv
        channel=II("${..channel}"),                                           # Channel dimension size is preserved in ResDilatedConv
        kernel =II("${..kernel}"),))                                          # Kernel size can have variation, but now fixed

class ResBlock(nn.Module):
    """Residual block containing multiple Res-DilatedConv.
    """
    def __init__(self, conf: ConfResBlock):
        super().__init__()
        self.model = nn.Sequential(*[ResDilatedConv(conf_resconv) for conf_resconv in conf.resconvs])

    def forward(self, ipt: Tensor) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward :: (B, Feat, T) -> (B, Feat, T)"""
        return self.model(ipt)
