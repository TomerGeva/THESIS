from enum import Enum


class encoder_type_e(Enum):
    DENSE           = 0
    VGG             = 1
    FULLY_CONNECTED = 2


class activation_type_e(Enum):
    null    = 0
    ReLU    = 1
    tanh    = 2
    sig     = 3
    lReLU   = 4  # leaky ReLU
    tReLU   = 5  # truncated Relu --> 0 if input < 0 ; 1 if input >1 ; linear else
    # h_sig   = 6  # hard sigmoid --> 0 if x < 3 ; 1 if x > 3 ; linear else
    SELU    = 7
