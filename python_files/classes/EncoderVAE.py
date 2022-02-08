from ConfigVAE import *
import math
import torch.nn as nn
from neural_network_block_classes import ConvBlock, SeparableConvBlock, DenseBlock, DenseTransitionBlock, FullyConnectedBlock, PadPool
from auxiliary_functions import compute_output_dim


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
            if 'sep-conv' in action[0]:
                conv_len += 1
                channels = action[1].out_channels
                self.layers.append(SeparableConvBlock(action[1]))
            if 'conv' in action[0]:
                conv_len += 1
                channels = action[1].out_channels
                self.layers.append(ConvBlock(action[1]))
            elif 'pool' in action[0]:
                conv_len += 1
                self.layers.append(PadPool(action[1]))
            elif 'dense' in action[0]:
                conv_len += 1
                action[1].in_channels = channels
                self.layers.append(DenseBlock(action[1]))
                channels += action[1].growth * action[1].depth
            elif 'transition' in action[0]:
                conv_len += 1
                action[1].set_in_out_channels(in_channels=channels)
                self.layers.append(DenseTransitionBlock(action[1])
                                   )
                channels = math.floor(channels * action[1].reduction_rate)
            elif 'linear' in action[0]:
                linear_len += 1
                if action_prev is None:  # First linear layer
                    action[1].in_neurons = x_dim * y_dim * channels
                else:
                    action[1].in_neurons = action_prev[1].out_neurons
                self.layers.append(FullyConnectedBlock(action[1]))
                action_prev = action

        self.conv_len   = conv_len
        self.fc_len     = linear_len

    def compute_dim_sizes(self):
        x_dim_size  = XQUANTIZE
        y_dim_size  = YQUANTIZE
        channels    = 0

        for action in self.topology:
            x_dim_size, y_dim_size, channels = compute_output_dim(x_dim_size, y_dim_size, channels, action)

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
