from ConfigVAE import *
import torch.nn as nn
import torch.nn.functional as F
from neural_network_functions import _fc_block


class DecoderVAE(nn.Module):
    """
    This class holds the Variational auto-encoder Decoder part
    """
    def __init__(self, device, topology, latent_dim):
        super(DecoderVAE, self).__init__()
        self.device         = device
        self.topology       = topology
        self.fc_len         = len(topology)
        self.layers         = nn.ModuleList()

        # ---------------------------------------------------------
        # Creating the Blocks according to the description
        # ---------------------------------------------------------
        linear_idx = 0
        action_prev = None
        for ii in range(len(self.topology)):
            action = self.topology[ii]
            if 'linear' in action[0]:
                if linear_idx == 0:
                    action_prev = action
                    self.layers.append(_fc_block(latent_dim,
                                                 action[1],
                                                 activation=True))
                elif 'last' in action[0]:
                    self.layers.append(_fc_block(action_prev[1],
                                                 action[1],
                                                 activation=False))
                    action_prev = action
                else:
                    self.layers.append(_fc_block(action_prev[1],
                                                 action[1],
                                                 activation=True))
                    action_prev = action

                linear_idx += 1

    def forward(self, x):
        # ---------------------------------------------------------
        # passing through the fully connected blocks
        # ---------------------------------------------------------
        for ii in range(self.fc_len):
            layer = self.layers[ii]
            x = layer(x)

        return x
