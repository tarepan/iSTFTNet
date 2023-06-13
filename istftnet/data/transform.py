"""Data transformation"""

from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING, II
from torch import Tensor, FloatTensor, tensor, stack, clamp, log # pyright: ignore [reportUnknownVariableType] ; because of PyTorch ; pylint: disable=no-name-in-module
import torch.nn.functional as F
from torchaudio.functional import resample       # pyright: ignore [reportMissingTypeStubs]; because of torchaudio
from torchaudio.transforms import MelSpectrogram # pyright: ignore [reportMissingTypeStubs]; because of torchaudio
import librosa
from librosa.util import normalize # pyright: ignore [reportUnknownVariableType] ; pylint: disable=no-name-in-module; because of librosa
from configen import default

from ..domain import MelIptBatched, WaveBatched, MelOptBatched, MelWaveMelBatch
from .domain import Raw, ItemMelIpt, ItemWave, ItemMelOpt, ItemMelWaveMel, DatumMelWaveMel
from .clip import clip_segment_random, match_length


# [Data transformation]
#
#      load        preprocessing            augmentation              collation
#     -----> raw -----------------> item -----------------> datum -----------------> batch
#                 before training            in Dataset             in DataLoader

###################################################################################################################################
# [Load]

def load_raw(path: Path) -> Raw:
    """Load raw data from the adress."""
    return librosa.load(path, sr=None, mono=True) # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType]; because of librosa

###################################################################################################################################
# [Preprocessing]

@dataclass
class ConfRaw2Wave:
    """Configuration of the raw2wave."""
    sr_target: int = MISSING # Waveform resampling target sampling rate
    use_cuda: bool = MISSING

def raw_to_wave(raw: Raw, conf: ConfRaw2Wave) -> ItemWave:
    """Convert raw data into waveform."""

    wave = FloatTensor(raw[0])
    # Resampling
    wave = wave if not conf.use_cuda else wave.cuda()
    wave = resample(wave, int(raw[1]), conf.sr_target)

    # Scaling - any range to [-0.95, +0.95]
    wave = normalize(wave.cpu().numpy()) * 0.95

    return FloatTensor(wave)


@dataclass
class ConfWave2Mel:
    """Configuration of the wave2mel."""
    n_fft:         int        = MISSING
    hop:           int        = MISSING
    win_size:      int        = MISSING
    sampling_rate: int        = MISSING # Wave sampling rate
    mel:           int        = MISSING # The number of mel frequency bin
    fmin:          int        = MISSING # Minimum frequency
    fmax:          int | None = MISSING # Maximum frequency

def gen_melnizer(conf: ConfWave2Mel) -> MelSpectrogram:
    """Instantiate wave-to-mel class."""
    return MelSpectrogram(conf.sampling_rate, conf.n_fft, conf.win_size, conf.hop, conf.fmin, conf.fmax, n_mels=conf.mel, power=1, norm="slaney", mel_scale="slaney", center=False)


def wave_to_mel_batch(waves: Tensor, melnizer: MelSpectrogram, conf: ConfWave2Mel) -> Tensor:
    """Convert waveforms into melspectrograms.

    Args:
        waves :: (B, T)           - Waveforms    
    Returns:
              :: (B, Freq, Frame) - melspectrogram
    """

    # Centering
    pad = int((conf.n_fft - conf.hop)/2)
    waves = F.pad(waves.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)

    # Transform
    melspec = melnizer(waves)
    logmel = log(clamp(melspec, min=1e-5))

    return logmel


def wave_to_mel(wave: ItemWave, melnizer: MelSpectrogram, conf: ConfWave2Mel) -> ItemMelIpt | ItemMelOpt:
    """Convert a waveform into a melspectrogram."""
    return wave_to_mel_batch(wave.unsqueeze(0), melnizer, conf).squeeze(0)


@dataclass
class ConfPreprocess:
    """Configuration of item-to-datum augmentation."""
    n_fft:             int = MISSING
    hop:               int = MISSING
    win_size:          int = MISSING
    sampling_rate:     int = MISSING # Wave sampling rate
    mel:               int = MISSING # The number of mel frequency bin
    fmin:              int = MISSING # Minimum frequency
    segment_wavescale: int = MISSING # Segment length [samples]
    use_cuda:      bool = MISSING # Whether to use CUDA
    raw2wave:    ConfRaw2Wave = default(ConfRaw2Wave(
        sr_target    =II("..sampling_rate"),
        use_cuda     =II("..use_cuda")))
    wave2melipt: ConfWave2Mel = default(ConfWave2Mel(
        n_fft        =II("..n_fft"),
        hop          =II("..hop"),
        win_size     =II("..win_size"),
        sampling_rate=II("..sampling_rate"),
        mel          =II("..mel"),
        fmin         =II("..fmin")))
    wave2melopt: ConfWave2Mel = default(ConfWave2Mel(
        n_fft        =II("..n_fft"),
        hop          =II("..hop"),
        win_size     =II("..win_size"),
        sampling_rate=II("..sampling_rate"),
        mel          =II("..mel"),
        fmin         =II("..fmin")))

def preprocess(conf: ConfPreprocess, raw: Raw, milnizer_ipt: MelSpectrogram, milnizer_opt: MelSpectrogram) -> ItemMelWaveMel:
    """Preprocess raw data into the item."""

    wave = raw_to_wave(raw, conf.raw2wave)

    if conf.use_cuda:
        wave, milnizer_ipt, milnizer_opt = wave.cuda(), milnizer_ipt.cuda(), milnizer_opt.cuda()

    mel_ipt = wave_to_mel(wave, milnizer_ipt, conf.wave2melipt).cpu()
    mel_opt = wave_to_mel(wave, milnizer_opt, conf.wave2melopt).cpu()
    wave    = wave.cpu()

    mel_ipt, wave, mel_opt = match_length([(mel_ipt, conf.hop), (wave, 1), (mel_opt, conf.hop)], conf.segment_wavescale)

    return mel_ipt, wave, mel_opt

###################################################################################################################################
# [Augmentation]

@dataclass
class ConfAugment:
    """
    Configuration of item-to-datum augmentation.
    Args:
        len_clip - Length of clipping
    """
    hop_mel:           int = MISSING # Hop size of melspectrograms
    segment_wavescale: int = MISSING # Segment length with waveform scale [samples]

def augment(conf: ConfAugment, mode: str, items: ItemMelWaveMel) -> DatumMelWaveMel:
    """Dynamically modify item into datum."""
    mel_ipt, wave, mel_opt = items
    mel_ipt, wave, mel_opt = tensor(mel_ipt), tensor(wave), tensor(mel_opt)

    segment_wavescale = conf.segment_wavescale if mode == "train" else (wave.shape[0] // conf.hop_mel * conf.hop_mel)
    mel_ipt_datum, wave_datum, mel_opt_datum = clip_segment_random([(mel_ipt, conf.hop_mel), (wave, 1), (mel_opt, conf.hop_mel)], segment_wavescale)

    return mel_ipt_datum, wave_datum, mel_opt_datum

###################################################################################################################################
# [collation]

def collate(mode: str, datums: list[DatumMelWaveMel]) -> MelWaveMelBatch:
    """Collation (datum_to_batch) - Bundle multiple datum into a batch."""

    mel_ipt_batched: MelIptBatched = stack([datum[0] for datum in datums])
    wave_batched:    WaveBatched   = stack([datum[1] for datum in datums])
    mel_opt_batched: MelOptBatched = stack([datum[2] for datum in datums])

    return mel_ipt_batched, wave_batched, mel_opt_batched

###################################################################################################################################

@dataclass
class ConfTransform:
    """Configuration of data transform."""
    hop_mel:           int            = MISSING # Hop size of melspectrogram
    segment_wavescale: int            = MISSING # Segment length [sample]
    preprocess:        ConfPreprocess = default(ConfPreprocess(
        hop              =II("..hop_mel"),
        segment_wavescale=II("..segment_wavescale")))
    augment:           ConfAugment    = default(ConfAugment(
        hop_mel          =II("..hop_mel"),
        segment_wavescale=II("..segment_wavescale")))
