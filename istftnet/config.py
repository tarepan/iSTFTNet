"""Whole Configuration"""


from typing import Optional
from dataclasses import dataclass

from omegaconf import MISSING
from configen import default, generate_conf_loader # pyright: ignore [reportMissingTypeStubs]
from lightlightning import ConfTrain      # pyright: ignore [reportMissingTypeStubs]

from .data.transform import ConfTransform
from .data.datamodule import ConfData
from .model import ConfModel

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
    sampling_rate: 22050
    net:
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
    optim:
        learning_rate:    0.0002
        beta_1:           0.8
        beta_2:           0.99
        sched_decay_rate: 0.999
    transform: "${transform}"
data:
    adress_data_root: ""
    corpus:
        train:
            name: "JVS"
            download: False
        val:
            name: "JVS"
            download: False
        test:
            name: "JVS"
            download: False
        n_val: 5
        n_test: 5
    dataset:
        transform: "${transform}"
    loader:
        batch_size_train:   16
        batch_size_val:      1
        batch_size_test:     1
        num_workers:      null
        pin_memory:       null
train:
    gradient_clipping: null
    max_epochs: 3100
    val_interval_epoch: 5
    profiler: null
    ckpt_log:
        dir_root: "."
        name_exp: "default"
        name_version: "version_0"
"""

@dataclass
class ConfGlobal:
    """Configuration of everything."""
    seed:             int           = MISSING # PyTorch-Lightning's seed for every random system
    path_extend_conf: Optional[str] = MISSING # Path of configuration yaml which extends default config
    transform:        ConfTransform = default(ConfTransform())
    model:            ConfModel     = default(ConfModel())
    data:             ConfData      = default(ConfData())
    train:            ConfTrain     = default(ConfTrain())


# Exported
load_conf = generate_conf_loader(CONF_DEFAULT_STR, ConfGlobal)
"""Load configuration type-safely.
"""
