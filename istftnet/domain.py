"""Domain"""

from torch import Tensor # pyright: ignore [reportUnknownVariableType] ; because of PyTorch ; pylint: disable=no-name-in-module


# Data batch
MelIptBatched = Tensor # :: (B=b, Feat=mel, Frame=frm) - Melspectrograms for input
WaveBatched   = Tensor # :: (B=b, T=t)                 - Ground-truth waveform
MelOptBatched = Tensor # :: (B=b, Feat=mel, Frame=frm) - Melspectrograms for output
MelWaveMelBatch = tuple[MelIptBatched, WaveBatched, MelOptBatched] # The batch
