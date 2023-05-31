"""Data domain"""


import numpy as np
from numpy.typing import NDArray
from torch import Tensor, tensor


# Raw inputs
Raw  = tuple[NDArray[np.float32], float] # :: (T,) - Raw waveform and its sampling rate

# Statically-preprocessed item
ItemMelIpt = Tensor # :: (Freq, Frame) - Melspectrogram for input
ItemWave   = Tensor # :: (T,)          - Waveform, in range [-0.95, +0.95]
ItemMelOpt = Tensor # :: (Freq, Frame) - Melspectrogram for output/loss
ItemMelWaveMel = tuple[ItemMelIpt, ItemWave, ItemMelOpt] # The item

## For typing
ItemMelIpt_: ItemMelIpt = tensor([[1.,], [1.,]])
ItemWave_:   ItemWave   = tensor([1.,])
ItemMelOpt_: ItemMelOpt = tensor([[1.,], [1.,]])
ItemMelWaveMel_: ItemMelWaveMel = (ItemMelIpt_, ItemWave_, ItemMelOpt_)

# Dynamically-transformed Dataset datum
DatumMelIpt = Tensor # :: (Freq=mel,  Frame=frm) - Clipped melspectrogarm for input
DatumWave   = Tensor # :: (T=frm*hop,)           - Clipped waveform
DatumMelOpt = Tensor # :: (Freq=mel,  Frame=frm) - Clipped melspectrogarm for output/loss
DatumMelWaveMel = tuple[DatumMelIpt, DatumWave, DatumMelOpt] # The datum
