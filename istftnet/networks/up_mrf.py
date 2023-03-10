"""The Child sub-module"""


from __future__ import annotations
from dataclasses import dataclass

from torch import Tensor, zeros_like
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from omegaconf import MISSING, II


from .merge import default, list_default
from .res_block import ResBlock, ConfResBlock
from .initializer import init_conv_norm


@dataclass
class ConfMRF:
    """Configuration of `MRF`."""
    channel:   int                = MISSING                    # Channel dimension size of input/middle/output
    resblocks: list[ConfResBlock] = list_default(ConfResBlock( # Different-kernel ResBlocks
        channel=II("${..channel}")))                               # All blocks keep channel dimension size

class MRF(nn.Module):
    """Multi-ReceptiveField (kernel) ResBlocks.
    """
    def __init__(self, conf: ConfMRF):
        super().__init__()
        self.res_blocks = nn.ModuleList([ResBlock(conf_resblock) for conf_resblock in conf.resblocks])
        self.n_blocks = len(conf.resblocks)

    def forward(self, ipt: Tensor) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward :: (B, Feat, T) -> (B, Feat, T)"""
        opt = zeros_like(ipt)
        for res_block in self.res_blocks:
            opt += res_block(ipt)
        return opt / self.n_blocks


@dataclass
class ConfUpMRF:
    """Configuration of `UpMRF`."""
    c_in:      int     = MISSING          # Channel dimension size of UpMRF input
    c_out:     int     = MISSING          # Channel dimension size of UpMRF output
    up_kernel: int     = MISSING          # Kernel size of upsampling conv
    mrf:       ConfMRF = default(ConfMRF( # The MRF
        channel=II("${..c_out}")))            # MRF keep channel size, so should be UpMRF's output channel size

class UpMRF(nn.Module):
    """Upsampling + MultiReceptiveField.
    """
    def __init__(self, conf: ConfUpMRF):
        super().__init__()

        # Upsampling is configured to 'half-overlap'
        kernel, stride = conf.up_kernel, conf.up_kernel // 2
        # Larger LReLU slope
        lrelu_slope = 0.1

        # Upsampling by TConv
        ## TODO: padding
        up_conv = weight_norm(nn.ConvTranspose1d(conf.c_in, conf.c_out, kernel, stride))
        up_conv.apply(init_conv_norm)

        # LReLU-Up-MRF
        self.model = nn.Sequential(*[nn.LeakyReLU(lrelu_slope), up_conv, MRF(conf.mrf)])

    def forward(self, i_pred: Tensor) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward :: (B, Feat=c_in, T=t) -> (B, Feat=c_out, T = t*k/2)"""
        return self.model(i_pred)

    # TODO: remove_weight_norm