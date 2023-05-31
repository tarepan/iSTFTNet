"""Whole Configuration"""


from typing import Optional
from dataclasses import dataclass

from omegaconf import MISSING
from configen import generate_conf_loader # pyright: ignore [reportMissingTypeStubs]
from lightlightning import ConfTrain      # pyright: ignore [reportMissingTypeStubs]

from .data.transform import ConfTransform
from .data.datamodule import ConfData
from .model import ConfModel

"""
feat_i:      80
kernel_pre:   7
feat_l0:    512
upmrfs:
    -
        feat_i:    512
        feat_o:    256
        up_kernel:  16
        mrf:
            resblocks:
                -
                    kernel: 3
                    resconvs:
                        - dilations: [1, 1]
                        - dilations: [3, 1]
                        - dilations: [5, 1]
                -
                    kernel: 7
                    resconvs:
                        - dilations: [1, 1]
                        - dilations: [3, 1]
                        - dilations: [5, 1]
                -
                    kernel: 11
                    resconvs:
                        - dilations: [1, 1]
                        - dilations: [3, 1]
                        - dilations: [5, 1]
    -
        feat_i:    256
        feat_o:    128
        up_kernel:  16
        mrf:
            resblocks:
                -
                    kernel: 3
                    resconvs:
                        - dilations: [1, 1]
                        - dilations: [3, 1]
                        - dilations: [5, 1]
                -
                    kernel: 7
                    resconvs:
                        - dilations: [1, 1]
                        - dilations: [3, 1]
                        - dilations: [5, 1]
                -
                    kernel: 11
                    resconvs:
                        - dilations: [1, 1]
                        - dilations: [3, 1]
                        - dilations: [5, 1]
feat_ln:  128
kernel_post: 7
istft:
    nfft: 16
    hop:   4
"""

CONF_DEFAULT_STR = """
seed: 1234
path_extend_conf: null
transform:
    segment_wavescale: 8192
    hop_mel:            256
    preprocess:
        n_fft:              1024
        win_size:           1024
        sampling_rate:     22050
        mel:                  80
        fmin:                  0
        use_cuda:           true
        wave2melipt:
            fmax:              8000
        wave2melopt:
            fmax:              null
model:
    net:
        dim_i: 1
        dim_o: 1
        child:
            dropout: 0.5
    optim:
        learning_rate: 0.01
        sched_decay_rate: 0.01
        sched_decay_step: 1000
    transform: "${transform}"
data:
    adress_data_root: ""
    corpus:
        train:
            name: "TEST"
            download: False
        val:
            name: "TEST"
            download: False
        test:
            name: "TEST"
            download: False
        n_val: 1
        n_test: 1
    dataset:
        attr1: 1
        transform: "${transform}"
    loader:
        batch_size_train: 1
        batch_size_val: 1
        batch_size_test: 1
        num_workers: null
        pin_memory: null
train:
    gradient_clipping: null
    max_epochs: 10000
    val_interval_epoch: 100
    profiler: null
    ckpt_log:
        dir_root: "."
        name_exp: "default"
        name_version: "version_0"
"""

@dataclass
class ConfGlobal:
    """Configuration of everything."""
    seed: int                       = MISSING # PyTorch-Lightning's seed for every random system
    path_extend_conf: Optional[str] = MISSING # Path of configuration yaml which extends default config
    transform: ConfTransform = ConfTransform()
    model:     ConfModel     = ConfModel()
    data:      ConfData      = ConfData()
    train:     ConfTrain     = ConfTrain()


# Exported
load_conf = generate_conf_loader(CONF_DEFAULT_STR, ConfGlobal)
"""Load configuration type-safely.
"""
