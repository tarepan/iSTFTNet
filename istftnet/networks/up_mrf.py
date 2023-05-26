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
    """Configuration of the MRF."""
    feat:   int                = MISSING                       # Feature dimension size of input/middle/output
    resblocks: list[ConfResBlock] = list_default(ConfResBlock( # Different-kernel ResBlocks
        feat=II("${..feat}")))                                     # All blocks keep channel dimension size

class MRF(nn.Module):
    """Multi-ReceptiveField (kernel) ResBlocks.
    """
    def __init__(self, conf: ConfMRF):
        super().__init__()
        self.res_blocks = nn.ModuleList([ResBlock(conf_resblock) for conf_resblock in conf.resblocks])
        self._n_blocks = len(conf.resblocks)

    def forward(self, series: Tensor) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward :: (B, Feat, T) -> (B, Feat, T)"""
        opt = zeros_like(series)
        for res_block in self.res_blocks:
            opt += res_block(series)
        return opt / self._n_blocks


@dataclass
class ConfUpMRF:
    """Configuration of the UpMRF."""
    feat_i:    int     = MISSING          # Feature dimension size of UpMRF input
    feat_o:    int     = MISSING          # Feature dimension size of UpMRF output
    up_kernel: int     = MISSING          # Kernel size of upsampling convT, should be even number
    mrf:       ConfMRF = default(ConfMRF( # The MRF
        feat=II("${..feat_o}")))              # MRF keep feature dim size, so should be UpMRF's output channel size

class UpMRF(nn.Module):
    """Upsampling + MultiReceptiveField.
    """
    def __init__(self, conf: ConfUpMRF):
        super().__init__()

        # Params
        kernel, stride = conf.up_kernel, conf.up_kernel // 2 # half-overlap
        pad = (kernel - stride) // 2
        lrelu_slope = 0.1

        # Validation
        assert kernel % 2 == 0, f"Kernel size should be even-number for half-overlap, but {kernel}."
        assert stride % 2 == 0, f"Currently support only even-number stride for even-padding, but {stride}."

        # Upsampling by TConv
        up_conv = weight_norm(nn.ConvTranspose1d(conf.feat_i, conf.feat_o, kernel, stride, padding=pad))
        up_conv.apply(init_conv_norm)

        # LReLU-Up-MRF
        self.model = nn.Sequential(*[nn.LeakyReLU(lrelu_slope), up_conv, MRF(conf.mrf)])

    def forward(self, i_pred: Tensor) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward :: (B, Feat=i, T=t) -> (B, Feat=o, T = t*k/2)"""
        return self.model(i_pred)

    # TODO: remove_weight_norm
