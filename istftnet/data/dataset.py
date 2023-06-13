"""Datasets"""


from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from hashlib import md5

from torch.utils.data import Dataset
from tqdm import tqdm
from omegaconf import MISSING
from speechdatasety.interface.speechcorpusy import AbstractCorpus, ItemId               # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.archive import try_to_acquire_archive_contents, save_archive # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.adress import dataset_adress                                 # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.access import generate_saver_loader                          # pyright: ignore [reportMissingTypeStubs]
from configen import default

from ..domain import MelWaveMelBatch
from .domain import ItemMelWaveMel_, DatumMelWaveMel
from .transform import ConfTransform, gen_melnizer, load_raw, preprocess, augment, collate


CorpusItems = Tuple[AbstractCorpus, list[tuple[ItemId, Path]]]


@dataclass
class ConfMelAudioMelDataset:
    """Configuration of HogeFuga dataset.
    Args:
        adress_data_root - Root adress of data
        att1 - Attribute #1
    """
    adress_data_root: Optional[str] = MISSING
    transform:        ConfTransform = default(ConfTransform())

class MelAudioMelDataset(Dataset[DatumMelWaveMel]):
    """The MelIpt/Audio/MelOpt dataset from the corpus.
    """
    def __init__(self, conf: ConfMelAudioMelDataset, items: CorpusItems, mode: str):
        """
        Args:
            conf:  The Configuration
            items: Corpus instance and filtered item information (ItemId/Path pair)
            mode:  train | eval | test
        """

        # Validation
        assert (mode == "train") or (mode == "eval") or (mode == "test"), f"Not supported mode: {mode}"
        # Store parameters
        self._conf = conf
        self._corpus = items[0]
        self._items  = items[1]
        self._mode = mode

        # Calculate data path
        conf_specifier = f"{conf.transform}"
        item_specifier = f"{list(map(lambda item: item[0], self._items))}"
        exp_specifier = md5((conf_specifier+item_specifier).encode()).hexdigest()
        self._adress_archive, self._path_contents = dataset_adress(conf.adress_data_root, self._corpus.__class__.__name__, "MelAudioMel", exp_specifier)
        self._save, self._load = generate_saver_loader(ItemMelWaveMel_, ["mel", "wave", "mel"], self._path_contents)

        # Deploy dataset contents
        ## Try to 'From pre-generated dataset archive'
        contents_acquired = try_to_acquire_archive_contents(self._adress_archive, self._path_contents)
        ## From scratch
        if not contents_acquired:
            print("Dataset archive file is not found.")
            self._generate_dataset_contents()

    def _generate_dataset_contents(self) -> None:
        """Generate dataset with corpus auto-download and static preprocessing.
        """

        print("Generating new dataset...")

        # Lazy contents download
        self._corpus.get_contents()

        # Preprocessing - Load/Transform/Save
        melnizer_ipt = gen_melnizer(self._conf.transform.preprocess.wave2melipt)
        melnizer_opt = gen_melnizer(self._conf.transform.preprocess.wave2melopt)
        for item_id, item_path in tqdm(self._items, desc="Preprocessing", unit="item"):
            raw = load_raw(item_path)
            mel_wave_mel_item = preprocess(self._conf.transform.preprocess, raw, melnizer_ipt, melnizer_opt)
            self._save(item_id, mel_wave_mel_item)

        print("Archiving new dataset...")
        save_archive(self._path_contents, self._adress_archive)
        print("Archived new dataset.")

        print("Generated new dataset.")

    def __getitem__(self, n: int) -> DatumMelWaveMel:
        """(API) Load the n-th datum from the dataset with tranformation.
        """
        item_id = self._items[n][0]
        return augment(self._conf.transform.augment, self._mode, self._load(item_id))

    def __len__(self) -> int:
        return len(self._items)

    def collate_fn(self, items: List[DatumMelWaveMel]) -> MelWaveMelBatch:
        """(API) datum-to-batch function."""
        return collate(self._mode, items)
