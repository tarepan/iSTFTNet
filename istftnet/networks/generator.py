"""The Network"""


from __future__ import annotations
from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from omegaconf import MISSING, II


from ..domain import CondSeriesBatched
from .merge import default, list_default
from .up_mrf import UpMRF, ConfUpMRF
from .istft import ISTFT, ConfISTFT
from .initializer import init_conv_norm


@dataclass
class ConfGenerator:
    """Configuration of the iSTFTNet Generator."""
    feat_in:     int             = MISSING                   # Feature dimension size of input
    kernel_pre:  int             = MISSING                   # Kernel size of PreConv
    c_stack_i:   int             = MISSING                   # Channel dimension size of UpMRF stack's input
    upmrfs:      list[ConfUpMRF] = list_default(ConfUpMRF()) # UpMRF stack
    c_stack_o:   int             = MISSING                   # Channel dimension size of UpMRF stack's input
    kernel_post: int             = MISSING                   # Kernel size of PostConv
    istft:       ConfISTFT       = default(ConfISTFT())      # Final iSTFT

class Generator(nn.Module):
    """The iSTFTNet Generator, 'Conv + UpMRF xL + Conv + iSTFT'.
    """
    def __init__(self, conf: ConfGenerator):
        super().__init__()

        # Follow HiFiGAN-official (PyTorch default value, not equal to UpMRF)
        lrelu_slope = 0.01

        # PreConv / PostConv
        pre_conv  = weight_norm(nn.Conv1d(conf.feat_in,   conf.c_stack_i, conf.kernel_pre,  padding="same"))
        post_conv = weight_norm(nn.Conv1d(conf.c_stack_o, 1,              conf.kernel_post, padding="same"))
        ## NOTE: Following PWG-unofficial (In HiFiGAN-official, pre_conv is not manually initialized)
        pre_conv.apply(init_conv_norm)
        post_conv.apply(init_conv_norm)

        # PreConv / UpMRF xL / PostConv / #iSTFT
        layers: list[nn.Module] = []
        layers += [pre_conv]
        layers += [UpMRF(conf_upmrf) for conf_upmrf in conf.upmrfs]
        layers += [nn.LeakyReLU(lrelu_slope)]
        layers += [post_conv]
        layers += [nn.Tanh()]
        # layers += [ISTFT(conf.istft)]
        self.model = nn.Sequential(*layers)

    def forward(self, cond_series: CondSeriesBatched) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward.

        Arguments:
            cond_series - Conditioning series
        Returns:
            :: (B, T) - Predicted waveform
        """
        return self.model(cond_series)

    # TODO: remove_weight_norm
