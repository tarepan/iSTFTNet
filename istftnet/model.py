"""The model"""

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
import lightning as L                                           # pyright: ignore [reportMissingTypeStubs]
from lightning.pytorch.core.optimizer import LightningOptimizer # pyright: ignore [reportMissingTypeStubs]
from omegaconf import MISSING
from configen import default                                    # pyright: ignore [reportMissingTypeStubs]

from .domain import MelWaveMelBatch
from .data.domain import Raw
from .data.transform import ConfTransform, augment, collate, gen_melnizer, load_raw, preprocess, wave_to_mel_batch
from .networks.generator import Generator, ConfGenerator
from .networks.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from .loss import adv_d_loss, adv_g_loss, fm_loss, mel_loss


@dataclass
class ConfOptim:
    """Configuration of the optimizers."""
    learning_rate:    float = MISSING # Optimizer learning rate
    beta_1:           float = MISSING # Optimizer AdamW beta1
    beta_2:           float = MISSING # Optimizer AdamW beta2
    sched_decay_rate: float = MISSING # ExponentialLR shaduler decay rate (gamma)

@dataclass
class ConfModel:
    """Configuration of the Model.
    """
    sampling_rate: int           = MISSING
    net:           ConfGenerator = default(ConfGenerator())
    optim:         ConfOptim     = default(ConfOptim())
    transform:     ConfTransform = default(ConfTransform())

class Model(L.LightningModule):
    """The model."""

    def __init__(self, conf: ConfModel):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False

        self._conf = conf
        self.generator = Generator(conf.net)
        self.disc_mpd = MultiPeriodDiscriminator()
        self.disc_msd = MultiScaleDiscriminator()
        self.melnizer_ipt = gen_melnizer(conf.transform.preprocess.wave2melipt)
        self.melnizer_opt = gen_melnizer(conf.transform.preprocess.wave2melopt)

    def forward(self, batch: MelWaveMelBatch): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
        """(PL API) Run inference toward a batch :: -> (B, Feat=1, T)."""
        return self.generator(batch[0])

    def training_step(self, batch: MelWaveMelBatch): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
        """(PL API) Train the model with a batch."""

        # Optimizers
        opts : tuple[LightningOptimizer, LightningOptimizer] = self.optimizers() # pyright: ignore [reportGeneralTypeIssues] ; because of L
        opt_g, opt_d = opts                                                      # pylint: disable=unpacking-non-sequence

        # Data
        mel_ipt, wave_gt, mel_gt = batch
        wave_gt = wave_gt.unsqueeze(1)

        # Common_Forward
        wave_pred = self.generator(mel_ipt)

        # D_Forward/Loss
        d_mpd_real, d_mpd_fake, _, _ = self.disc_mpd(wave_gt, wave_pred.detach())
        d_msd_real, d_msd_fake, _, _ = self.disc_msd(wave_gt, wave_pred.detach())
        loss_adv_d_mpd = adv_d_loss(d_mpd_real, d_mpd_fake)
        loss_adv_d_msd = adv_d_loss(d_msd_real, d_msd_fake)
        loss_d = loss_adv_d_mpd + loss_adv_d_msd
        # D_Backward/Optim
        opt_d.zero_grad()                                                        # pyright: ignore [reportGeneralTypeIssues, reportUnknownMemberType] ; because of L
        self.manual_backward(loss_d)
        opt_d.step()

        # G_Loss
        _, d_mpd_fake, feat_mpd_real, feat_mpd_fake = self.disc_mpd(wave_gt, wave_pred)
        _, d_msd_fake, feat_msd_real, feat_msd_fake = self.disc_msd(wave_gt, wave_pred)
        mel_pred = wave_to_mel_batch(wave_pred.squeeze(1), self.melnizer_opt, self._conf.transform.preprocess.wave2melopt)
        ## Adv
        loss_adv_g_mpd = adv_g_loss(d_mpd_fake)
        loss_adv_g_msd = adv_g_loss(d_msd_fake)
        loss_adv_g     =        loss_adv_g_mpd + loss_adv_g_msd
        ## Fm
        loss_fm_mpd    = fm_loss(feat_mpd_real, feat_mpd_fake)
        loss_fm_msd    = fm_loss(feat_msd_real, feat_msd_fake)
        loss_fm        =        loss_fm_mpd + loss_fm_msd
        ## Mel
        loss_mel       = 45.0 * mel_loss(mel_gt, mel_pred)
        ## total
        loss_g         = loss_adv_g + loss_fm + loss_mel
        # G_Backward/Optim
        opt_g.zero_grad()                                                        # pyright: ignore [reportGeneralTypeIssues, reportUnknownMemberType] ; because of L
        self.manual_backward(loss_g)
        opt_g.step()

        # Logging
        self.log_dict({"train/D": loss_d, "train/G": loss_g, "train/G/adv": loss_adv_g, "train/G/fm": loss_fm, "train/G/mel": loss_mel, }) # pyright: ignore [reportUnknownMemberType]

    def validation_step(self, batch: MelWaveMelBatch, batch_idx: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ,unused-argument
        """(PL API) Validate the model with a batch.
        """

        # Data
        mel_ipt, _, mel_gt = batch

        wave_pred = self.generator(mel_ipt).squeeze(1)
        mel_pred = wave_to_mel_batch(wave_pred, self.melnizer_opt, self._conf.transform.preprocess.wave2melopt)

        loss_mel = mel_loss(mel_gt, mel_pred).item()

        # Logging
        self.log_dict({'val/G/mel': loss_mel,}) # pyright: ignore [reportUnknownMemberType]
        ## Audio
        # # [PyTorch](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_audio)
        # #                                                      ::Tensor(1, L)
        self.logger.experiment.add_audio(f"audio_{batch_idx}", wave_pred, global_step=self.global_step, sample_rate=self._conf.sampling_rate)

    # def test_step(self, batch, batch_idx: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
    #     """(PL API) Test a batch. If not provided, test_step == validation_step."""
    #     return anything_for_`test_epoch_end`

    def configure_optimizers(self): # type: ignore ; because of PyTorch-Lightning (no return typing, so inferred as Void)
        """(PL API) Set up a optimizer.
        """
        conf = self._conf.optim

        opt_g = AdamW(self.generator.parameters(),                                             lr=conf.learning_rate, betas=(conf.beta_1, conf.beta_2))
        opt_d = AdamW(itertools.chain(self.disc_mpd.parameters(), self.disc_msd.parameters()), lr=conf.learning_rate, betas=(conf.beta_1, conf.beta_2))
        sched_g = ExponentialLR(opt_g, gamma=conf.sched_decay_rate)
        sched_d = ExponentialLR(opt_d, gamma=conf.sched_decay_rate)

        return [opt_g, opt_d], [sched_g, sched_d]

    # def predict_step(self, batch: HogeFugaBatch, batch_idx: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
    #     """(PL API) Run prediction with a batch. If not provided, predict_step == forward."""
    #     return pred

    def sample(self) -> Raw:
        """Acquire sample input toward preprocess."""

        # Audio Example (librosa is not handled by this template)
        import librosa # pyright: ignore [reportMissingImports, reportUnknownVariableType] ; pylint: disable=import-outside-toplevel,import-error
        path: Path = librosa.example("libri2") # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType]

        return load_raw(self._conf.transform.load, path)

    def load(self, path: Path) -> Raw:
        """Load raw inputs.
        Args:
            path - Path to the input.
        """
        return load_raw(self._conf.transform.load, path)

    def preprocess(self, piyo: Raw, to_device: Optional[str] = None) -> MelWaveMelBatch:
        """Preprocess raw inputs into model inputs for inference."""

        conf = self._conf.transform
        hoge_fuga = preprocess(conf.preprocess, piyo)
        hoge_fuga_datum = augment(conf.augment, hoge_fuga)
        batch = collate([hoge_fuga_datum])

        # To device
        device = torch.device(to_device) if to_device else torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        batch = (batch[0].to(device), batch[1].to(device), batch[2])

        return batch
