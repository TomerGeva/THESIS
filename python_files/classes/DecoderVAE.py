from ConfigVAE import *
import torch.nn as nn
import torch.nn.functional as F
from neural_network_block_classes import FullyConnectedBlock


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
                linear_idx += 1
                if action_prev is None:  # First linear layer
                    self.layers.append(FullyConnectedBlock(in_neurons=latent_dim,
                                                           out_neurons=action[1],
                                                           batch_norm=action[2],
                                                           dropout_rate=action[3],
                                                           act=action[4],
                                                           alpha=action[5]
                                                           )
                                       )
                else:
                    self.layers.append(FullyConnectedBlock(in_neurons=action_prev[1],
                                                           out_neurons=action[1],
                                                           batch_norm=action[2],
                                                           dropout_rate=action[3],
                                                           act=action[4],
                                                           alpha=action[5]
                                                           )
                                       )
                action_prev = action

    def forward(self, x):
        # ---------------------------------------------------------
        # passing through the fully connected blocks
        # ---------------------------------------------------------
        for ii in range(self.fc_len):
            layer = self.layers[ii]
            x = layer(x)

        return x
