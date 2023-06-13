"""Data wrapper by PL-datamodule"""


from typing import Optional
from dataclasses import dataclass

import lightning as L
from omegaconf import MISSING, II
from speechdatasety.helper.loader import generate_loader, ConfLoader # pyright: ignore [reportMissingTypeStubs]
from torch.utils.data import DataLoader
from configen import default

from .domain import DatumMelWaveMel
from .dataset import MelAudioMelDataset, ConfMelAudioMelDataset
from .corpus import prepare_corpora, ConfCorpora


@dataclass
class ConfData:
    """Configuration of the Data.
    """
    adress_data_root: Optional[str]          = MISSING
    corpus:           ConfCorpora            = default(ConfCorpora(
        root                                     =II("..adress_data_root")))
    dataset:          ConfMelAudioMelDataset = default(ConfMelAudioMelDataset(
        adress_data_root                         =II("..adress_data_root")))
    loader:           ConfLoader             = default(ConfLoader())

class Data(L.LightningDataModule):
    """Data wrapper.
    """
    def __init__(self, conf: ConfData):
        super().__init__()
        self._conf = conf

    # def prepare_data(self) -> None:
    #     """(PL-API) Prepare data in dataset.
    #     """
    #     pass

    def setup(self, stage: Optional[str] = None) -> None:
        """(PL-API) Setup train/val/test datasets.
        """

        corpus_train, corpus_val, corpus_test = prepare_corpora(self._conf.corpus)

        if stage == "fit" or stage is None:
            self.dataset_train = MelAudioMelDataset(self._conf.dataset, corpus_train, "train")
            self.dataset_val   = MelAudioMelDataset(self._conf.dataset, corpus_val,   "eval")
        if stage == "test" or stage is None:
            self.dataset_test  = MelAudioMelDataset(self._conf.dataset, corpus_test,  "test")

    def train_dataloader(self) -> DataLoader[DatumMelWaveMel]:
        """(PL-API) Generate training dataloader."""
        return generate_loader(self.dataset_train, self._conf.loader, "train")

    def val_dataloader(self) -> DataLoader[DatumMelWaveMel]:
        """(PL-API) Generate validation dataloader."""
        return generate_loader(self.dataset_val,   self._conf.loader, "val")

    def test_dataloader(self) -> DataLoader[DatumMelWaveMel]:
        """(PL-API) Generate test dataloader."""
        return generate_loader(self.dataset_test,  self._conf.loader, "test")
