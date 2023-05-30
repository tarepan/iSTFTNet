"""Cloned from https://github.com/rishikksh20/iSTFTNet-pytorch, under Apache-2.0 license."""

import torch
from torch import Tensor, flatten
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm

LRELU_SLOPE = 0.1


class DiscriminatorP(nn.Module):
    """Periodic discriminator."""
    def __init__(self, period: int):
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        # Params
        kernel, stride = 5, 3
        self.period = period
        pad_l = int((kernel - 1)/2)

        self.convs = nn.ModuleList([
            weight_norm(Conv2d(   1,   32, (kernel, 1), (stride, 1), padding=(pad_l, 0))),
            weight_norm(Conv2d(  32,  128, (kernel, 1), (stride, 1), padding=(pad_l, 0))),
            weight_norm(Conv2d( 128,  512, (kernel, 1), (stride, 1), padding=(pad_l, 0))),
            weight_norm(Conv2d( 512, 1024, (kernel, 1), (stride, 1), padding=(pad_l, 0))),
            weight_norm(Conv2d(1024, 1024, (kernel, 1),           1, padding=(    2, 0))),
        ])
        self.conv_post = \
            weight_norm(Conv2d(1024,    1, (     3, 1),           1, padding=(    1, 0)))

    def forward(self, series: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward a batch."""

        fmap: list[Tensor] = []

        # 1d to 2d
        batch, feature, time = series.shape
        ## Padding
        if time % self.period != 0:
            n_pad = self.period - (time % self.period)
            series = F.pad(series, (0, n_pad), "reflect")
            time = time + n_pad
        ## Reshape
        series = series.view(batch, feature, time // self.period, self.period)

        for conv in self.convs:
            series = conv(series)
            series = F.leaky_relu(series, LRELU_SLOPE)
            fmap.append(series)
        series = self.conv_post(series)
        fmap.append(series)

        series = flatten(series, 1, -1)

        return series, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    """MPD, p=2/3/5/7/11."""
    def __init__(self):
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        self.discriminators = nn.ModuleList([
            DiscriminatorP( 2),
            DiscriminatorP( 3),
            DiscriminatorP( 5),
            DiscriminatorP( 7),
            DiscriminatorP(11),
        ])

    def forward(self, series_real: Tensor, series_fake: Tensor) -> tuple[list[Tensor], list[Tensor], list[list[Tensor]], list[list[Tensor]]]:
        """
        Args:
            series_real - real
            series_fake - generated
        Returns:
            d_reals    - MPD(series_real)
            d_fakes    - MPD(series_fake)
            feat_reals - feature map of MPD(series_real)
            feat_fakes - feature map of MPD(series_fake)
        """
        d_reals:    list[Tensor]       = []
        d_fakes:    list[Tensor]       = []
        feat_reals: list[list[Tensor]] = []
        feat_fakes: list[list[Tensor]] = []

        for discriminator in self.discriminators:
            d_real, feat_real = discriminator(series_real)
            d_fake, feat_fake = discriminator(series_fake)
            d_reals.append(d_real)
            d_fakes.append(d_fake)
            feat_reals.append(feat_real)
            feat_fakes.append(feat_fake)

        return d_reals, d_fakes, feat_reals, feat_fakes


class DiscriminatorS(torch.nn.Module):
    """Scaled discriminator."""
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(   1,  128, 15,               padding="same")),
            norm_f(Conv1d( 128,  128, 41, 2, groups= 4, padding=    20)),
            norm_f(Conv1d( 128,  256, 41, 2, groups=16, padding=    20)),
            norm_f(Conv1d( 256,  512, 41, 4, groups=16, padding=    20)),
            norm_f(Conv1d( 512, 1024, 41, 4, groups=16, padding=    20)),
            norm_f(Conv1d(1024, 1024, 41,    groups=16, padding="same")),
            norm_f(Conv1d(1024, 1024,  5,               padding="same")),
        ])
        self.post = \
            norm_f(Conv1d(1024,    1,  3,               padding="same"))

    def forward(self, series: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward a batch."""

        fmap: list[Tensor] = []
        for conv in self.convs:
            series = conv(series)
            series = F.leaky_relu(series, LRELU_SLOPE)
            fmap.append(series)
        series = self.post(series)
        fmap.append(series)

        series = flatten(series, 1, -1)

        return series, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    """MSD x1/x2/x4."""
    def __init__(self):
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True), # x1
            DiscriminatorS(),                       # x2
            DiscriminatorS(),                       # x4
        ])
        self.poolings = nn.ModuleList([
            AvgPool1d(4, 2, padding=2), # x1 -> x2
            AvgPool1d(4, 2, padding=2), # x2 -> x4
        ])

    def forward(self, series_real: Tensor, series_fake: Tensor) -> tuple[list[Tensor], list[Tensor], list[list[Tensor]], list[list[Tensor]]]:
        """Forward a batch."""

        d_reals: list[Tensor] = []
        d_fakes: list[Tensor] = []
        feat_reals: list[list[Tensor]] = []
        feat_fakes: list[list[Tensor]] = []

        for i, discriminator in enumerate(self.discriminators):
            # Downsampling
            if i != 0:
                series_real = self.poolings[i - 1](series_real)
                series_fake = self.poolings[i - 1](series_fake)
            # Discriminator
            d_real, feat_real = discriminator(series_real)
            d_fake, feat_fake = discriminator(series_fake)
            d_reals.append(d_real)
            d_fakes.append(d_fake)
            feat_reals.append(feat_real)
            feat_fakes.append(feat_fake)

        return d_reals, d_fakes, feat_reals, feat_fakes
