from enum import Enum


class encoder_type_e(Enum):
    DENSE           = 0
    VGG             = 1
    FULLY_CONNECTED = 2
    SEPARABLE       = 3
    TANSFORMER      = 4
    RES_VGG         = 5


class activation_type_e(Enum):
    null    = 0
    ReLU    = 1
    tanh    = 2
    sig     = 3
    lReLU   = 4  # leaky ReLU
    tReLU   = 5  # truncated Relu --> 0 if input < 0 ; 1 if input >1 ; linear else
    # h_sig   = 6  # hard sigmoid --> 0 if x < 3 ; 1 if x > 3 ; linear else
    SELU    = 7


class pool_e(Enum):
    MAX = 0
    AVG = 1


class mode_e(Enum):
    AUTOENCODER = 0
    VAE         = 1


class model_output_e(Enum):
    SENS = 0
    GRID = 1
    BOTH = 2
