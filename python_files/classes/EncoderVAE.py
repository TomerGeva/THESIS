from ConfigVAE import *
import torch.nn as nn
import torch.nn.functional as F
from neural_network_functions import _conv_block, _fc_block


class EncoderVAE(nn.Module):
    """
    This class holds the Variational auto-encoder Encoder part
    """
    def __init__(self, device, topology):
        super(EncoderVAE, self).__init__()
        self.device         = device
        self.topology       = topology
        self.layers         = nn.ModuleList()

        x_dim, y_dim        = self.compute_dim_sizes()

        self.x_dim          = x_dim
        self.y_dim          = y_dim

        # ---------------------------------------------------------
        # Creating the Blocks according to the description
        # ---------------------------------------------------------
        channels    = 0
        conv_idx    = 0
        maxpool_idx = 0
        linear_idx  = 0
        for ii in range(len(self.topology)):
            action = self.topology[ii]
            if 'conv' in action:
                self.layers.append(_conv_block(in_channels=action[1],
                                               out_channels=action[2],
                                               kernel_size=action[3],
                                               stride=action[4],
                                               padding=action[5],
                                               )
                                   )
                conv_idx += 1
                channels = action[2]
            elif 'pool' in action:
                self.layers.append(nn.MaxPool2d(action[1]))
                maxpool_idx += 1
            elif 'linear' in action:
                if linear_idx == 0:
                    self.midpoint_channels = channels
                    self.layers.append(_fc_block(x_dim * y_dim * channels,
                                                 action[1],
                                                 activation=True))
                elif 'last' in action:
                    self.layers.append(_fc_block(self.topology[ii-1][1],
                                                 action[1],
                                                 activation=False))
                else:
                    self.layers.append(_fc_block(self.topology[ii-1][1],
                                                 action[1],
                                                 activation=True))
                linear_idx += 1

        self.fc_len = linear_idx

    def compute_dim_sizes(self):
        x_dim_size  = XQUANTIZE
        y_dim_size  = YQUANTIZE
        conv_idx    = 0
        maxpool_idx = 0
        for ii in range(len(self.topology)):
            action = self.topology[ii]
            if 'conv' in action[0]:
                x_dim_size = int((x_dim_size - (action[3] - action[4]) + 2 * action[5]) / action[4])
                y_dim_size = int((y_dim_size - (action[3] - action[4]) + 2 * action[5]) / action[4])
                conv_idx += 1
            elif 'pool' in action:
                x_dim_size = int(x_dim_size / action[1])
                y_dim_size = int(y_dim_size / action[1])
                maxpool_idx += 1
        self.conv_len = conv_idx + maxpool_idx
        return x_dim_size, y_dim_size

    def forward(self, x):
        # ---------------------------------------------------------
        # passing through the convolution blocks
        # ---------------------------------------------------------
        for ii in range(self.conv_len):
            layer = self.layers[ii]
            x = layer(x)
        # ---------------------------------------------------------
        # flattening for the FC layers
        # ---------------------------------------------------------
        x = x.view(-1, self.x_dim * self.y_dim * self.midpoint_channels)
        # ---------------------------------------------------------
        # passing through the fully connected blocks
        # ---------------------------------------------------------
        for ii in range(self.fc_len):
            layer = self.layers[ii + self.conv_len]
            x = layer(x)

        return x
