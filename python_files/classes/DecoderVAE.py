from ConfigVAE import *
import torch.nn as nn
import torch.nn.functional as F
from neural_network_functions import _fc_block


class DecoderVAE(nn.Module):
    """
    This class holds the Variational auto-encoder Decoder part
    """
    def __init__(self, device):
        super(DecoderVAE, self).__init__()
        self.device         = device
        self.description    = DECODER_LAYER_DESCRIPTION
        self.fc_len         = len(DECODER_FC_LAYERS)
        self.layers         = nn.ModuleList()

        # ---------------------------------------------------------
        # Creating the Blocks according to the description
        # ---------------------------------------------------------
        linear_idx = 0
        for ii in range(len(self.description)):
            action = self.description[ii]
            if 'linear' in action:
                if linear_idx == 0:
                    self.layers.append(_fc_block(LATENT_SPACE_DIM,
                                                 DECODER_FC_LAYERS[linear_idx],
                                                 activation=True))
                elif 'last' in action:
                    self.layers.append(_fc_block(DECODER_FC_LAYERS[linear_idx - 1],
                                                 DECODER_FC_LAYERS[linear_idx],
                                                 activation=False))
                else:
                    self.layers.append(_fc_block(DECODER_FC_LAYERS[linear_idx - 1],
                                                 DECODER_FC_LAYERS[linear_idx],
                                                 activation=True))
                linear_idx += 1

    def forward(self, x):
        # ---------------------------------------------------------
        # passing through the fully connected blocks
        # ---------------------------------------------------------
        for ii in range(self.fc_len):
            layer = self.layers[ii]
            x = layer(x)

        return x
