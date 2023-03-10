import torch.nn as nn


def init_conv_norm(module: nn.Module):
    """Initialize Convolutions with N(0, 0.01).
    
    If called by `module.apply(init_conv_norm)`, all convolution sub-modules are initialized.
    """
    mean, std = 0.0, 0.01

    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        module.weight.data.normal_(mean, std)
