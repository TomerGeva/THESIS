from ConfigVAE import *
import math
import torch.nn as nn
from neural_network_block_classes import ConvBlock, DenseBlock, DenseTransitionBlock, FullyConnectedBlock, Pool2dPadding


class EncoderVAE(nn.Module):
    """
    This class holds the Variational auto-encoder Encoder part
    """
    def __init__(self, device, topology):
        super(EncoderVAE, self).__init__()
        self.device             = device
        self.topology           = topology
        self.layers             = nn.ModuleList()

        x_dim, y_dim, channels  = self.compute_dim_sizes()

        self.x_dim              = x_dim
        self.y_dim              = y_dim
        self.midpoint_channels  = channels

        # ---------------------------------------------------------
        # Creating the Blocks according to the description
        # ---------------------------------------------------------
        channels    = 0
        conv_len    = 0
        linear_len  = 0
        action_prev = None
        for ii in range(len(self.topology)):
            action = self.topology[ii]
            if 'conv' in action[0]:
                conv_len += 1
                channels = action[2]
                self.layers.append(ConvBlock(in_channels=action[1],
                                             out_channels=action[2],
                                             kernel_size=action[3],
                                             stride=action[4],
                                             padding=action[5],
                                             batch_norm=action[6],
                                             dropout_rate=action[7],
                                             act=action[8],
                                             alpha=action[9]
                                             )
                                   )
            elif 'pool' in action[0]:
                conv_len += 1
                self.layers.append(Pool2dPadding(kernel=action[1],
                                                 padding=action[2]))
            elif 'dense' in action[0]:
                conv_len += 1
                self.layers.append(DenseBlock(channels=channels,
                                              depth=action[2],
                                              growth_rate=action[1],
                                              kernel_size=action[3],
                                              stride=action[4],
                                              padding=action[5],
                                              batch_norm=action[6],
                                              dropout_rate=action[7],
                                              act=action[8],
                                              alpha=action[9]
                                              )
                                   )
                channels += action[2] * action[1]
            elif 'transition' in action[0]:
                conv_len += 1
                self.layers.append(DenseTransitionBlock(in_channels=channels,
                                                        reduction_rate=action[1],
                                                        kernel_size=action[2],
                                                        stride=action[3],
                                                        padding=action[4],
                                                        batch_norm=action[5],
                                                        dropout_rate=action[6],
                                                        act=action[7],
                                                        alpha=action[8],
                                                        pool_type=action[9],
                                                        pool_pad=action[10],
                                                        pool_size=action[11]
                                                        )
                                   )
                channels = math.floor(channels * action[1])
            elif 'linear' in action[0]:
                linear_len += 1
                if action_prev is None:  # First linear layer
                    self.layers.append(FullyConnectedBlock(in_neurons=(x_dim * y_dim * channels),
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

        self.conv_len   = conv_len
        self.fc_len     = linear_len

    def compute_dim_sizes(self):
        x_dim_size  = XQUANTIZE
        y_dim_size  = YQUANTIZE
        channels    = 0

        for ii in range(len(self.topology)):
            action = self.topology[ii]
            if 'conv' in action[0]:
                x_dim_size = int((x_dim_size - (action[3] - action[4]) + 2 * action[5]) / action[4])
                y_dim_size = int((y_dim_size - (action[3] - action[4]) + 2 * action[5]) / action[4])
                channels = action[2]
            elif 'pool' in action:
                if type(action[2]) is not tuple:
                    x_dim_size = int((x_dim_size + 2*action[2]) / action[1])
                    y_dim_size = int((y_dim_size + 2*action[2]) / action[1])
                else:
                    x_dim_size = int((x_conv_size + action[2][0] + action[2][1]) / action[1])
                    y_dim_size = int((y_conv_size + action[2][2] + action[2][3]) / action[1])
            elif 'dense' in action[0]:
                channels += action[1] * action[2]
            elif 'transition' in action[0]:
                channels = math.floor(channels * action[1])
                # ------------------------------------------------
                # This account for the conv layer and the pooling
                # ------------------------------------------------
                x_conv_size = int((x_dim_size - (action[2] - action[3]) + 2 * action[4]) / action[3])
                y_conv_size = int((y_dim_size - (action[2] - action[3]) + 2 * action[4]) / action[3])
                if type(action[10]) is not tuple:
                    x_dim_size = int((x_conv_size + 2*action[10]) / action[11])
                    y_dim_size = int((y_conv_size + 2*action[10]) / action[11])
                else:
                    x_dim_size = int((x_conv_size + action[10][0] + action[10][1]) / action[11])
                    y_dim_size = int((y_conv_size + action[10][2] + action[10][3]) / action[11])

        return x_dim_size, y_dim_size, channels

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
