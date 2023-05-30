"""Losses."""
from torch import Tensor, zeros, zeros_like, ones_like
import torch.nn.functional as F


def adv_d_loss(d_reals: list[Tensor], d_fakes: list[Tensor]) -> Tensor:
    """Adversarial discriminator loss, L2.

    Args:
        d_reals - Discriminator judges toward real samples
        d_fakes - Discriminator judges toward fake samples
    Returns:
        loss    - Total loss
    """

    loss = zeros([1]).to(d_reals[0].device)

    for d_real, d_fake in zip(d_reals, d_fakes):
        loss_real = F.mse_loss(d_real,  ones_like(d_real))
        loss_fake = F.mse_loss(d_fake, zeros_like(d_fake))
        loss += (loss_real + loss_fake)

    return loss


def adv_g_loss(d_fakes: list[Tensor]) -> Tensor:
    """Adversarial generator loss, L2.

    Args:
        d_fakes - Discriminator judges toward fake samples
    Returns:
        loss    - Total loss
    """

    loss = zeros([1]).to(d_fakes[0].device)

    for d_fake in d_fakes:
        loss += F.mse_loss(d_fake, ones_like(d_fake))

    return loss


def fm_loss(feat_reals: list[list[Tensor]], feat_fakes: list[list[Tensor]]) -> Tensor:
    """Feature matching loss, L1.
    
    Args:
        feat_reals - Discriminator intermediate features toward real samples
        feat_fakes - Discriminator intermediate features toward fake samples
    """

    loss = zeros([1]).to(feat_reals[0][0].device)

    for feat_real, feat_fake in zip(feat_reals, feat_fakes):
        for feat_real_layer_k, feat_fake_layer_k in zip(feat_real, feat_fake):
            loss += F.l1_loss(feat_real_layer_k, feat_fake_layer_k)

    # Comes from rishikksh20/iSTFTNet-pytorch
    loss = 2 * loss

    return loss


def mel_loss(mel_gt: Tensor, mel_pred: Tensor) -> Tensor:
    """Mel loss, L1.
    
    Args:
        mel_gt   - Ground-truth mel-spectrogram
        mel_pred - Predicted mel-spectrogram
    """
    return F.l1_loss(mel_gt, mel_pred)
