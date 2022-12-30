import torch.nn as nn
from neural_network_block_classes import FullyConnectedResidualBlock, FullyConnectedBlock


class OmegaModel(nn.Module):
    def __init__(self, device, topology):
        super(OmegaModel, self).__init__()
        self.device = device
        self.topology = topology
        self.layers = nn.ModuleList()
        # ---------------------------------------------------------
        # Creating the Blocks according to the description
        # ---------------------------------------------------------
        linear_len  = 0
        action_prev = None
        for ii in range(len(self.topology)):
            action = self.topology[ii]
            if 'linear' in action[0]:
                linear_len += 1
                if action_prev is not None:  # First linear layer
                    action[1].in_neurons = action_prev[1].out_neurons
                if 'res-linear' in action[0]:
                    self.layers.append(FullyConnectedResidualBlock(action[1]))
                else:
                    self.layers.append(FullyConnectedBlock(action[1]))
                action_prev = action
        self.fc_len = linear_len

    def forward(self, x):
        # ---------------------------------------------------------
        # passing through the fully connected blocks
        # ---------------------------------------------------------
        for ii in range(self.fc_len):
            layer = self.layers[ii]
            x = layer(x)

        return x
