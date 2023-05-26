"""The Network"""

from dataclasses import dataclass

from torch import Tensor, split, exp, sin
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torchaudio.transforms import InverseSpectrogram
from omegaconf import MISSING


from ..domain import CondSeriesBatched
from .merge import default, list_default
from .up_mrf import UpMRF, ConfUpMRF
from .initializer import init_conv_norm


@dataclass
class ConfISTFT:
    """Configuration of the ISTFT."""
    nfft: int = MISSING # The number of fft bins
    hop:  int = MISSING # iSTFT hop length

@dataclass
class ConfGenerator:
    """Configuration of the iSTFTNet Generator."""
    feat_i:      int             = MISSING                   # Feature dimension size of input
    kernel_pre:  int             = MISSING                   # Kernel size of PreConv
    feat_l0:     int             = MISSING                   # Feature dimension size of layer 0, UpMRF stack's input
    upmrfs:      list[ConfUpMRF] = list_default(ConfUpMRF()) # UpMRF stack
    feat_ln:     int             = MISSING                   # Feature dimension size of layer N, UpMRF stack's output
    kernel_post: int             = MISSING                   # Kernel size of PostConv
    istft:       ConfISTFT       = default(ConfISTFT())      # Final iSTFT

class Generator(nn.Module):
    """The iSTFTNet Generator, 'Conv + UpMRF xL + Conv + iSTFT'.
    """
    def __init__(self, conf: ConfGenerator):
        super().__init__()

        # Rarams
        lrelu_slope = 0.01 # Follow HiFiGAN-official (PyTorch default value, not equal to UpMRF)
        self._freq = conf.istft.nfft // 2 + 1
        # PreConv / PostConv
        pre_conv  = weight_norm(nn.Conv1d(conf.feat_i,    conf.feat_l0, conf.kernel_pre,  padding="same"))
        post_conv = weight_norm(nn.Conv1d(conf.feat_ln, 2 * self._freq, conf.kernel_post, padding="same"))
        ## NOTE: Following PWG-unofficial (In HiFiGAN-official, pre_conv is not manually initialized)
        pre_conv.apply(init_conv_norm)
        post_conv.apply(init_conv_norm)

        # PreConv / UpMRF xL / Pad / PostConv
        layers: list[nn.Module] = []
        layers += [pre_conv]
        layers += [UpMRF(conf_upmrf) for conf_upmrf in conf.upmrfs]
        layers += [nn.LeakyReLU(lrelu_slope)]
        layers += [nn.ReflectionPad1d((1, 0))] # TODO: understand
        layers += [post_conv]
        self.net = nn.Sequential(*layers)

        # iSTFT
        self.istft = InverseSpectrogram(conf.istft.nfft, hop_length=conf.istft.hop)

    def forward(self, cond_series: CondSeriesBatched) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward.

        Arguments:
            cond_series :: (B, Feat,   Frame) - Conditioning series
        Returns:
                        :: (B, Feat=1,     T) - Predicted waveform
        """
        o_net = self.net(cond_series)

        spec, phase = split(o_net, [self._freq]*2, dim=-2)
        spec, phase = exp(spec), sin(phase)
        o_series = self.istft(spec * exp(phase * 1j)).unsqueeze(-2)

        return o_series

    # TODO: remove_weight_norm
