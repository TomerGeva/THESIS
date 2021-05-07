from ConfigVAE import *
import math
import torch.nn as nn
from neural_network_block_classes import ConvBlock, DenseBlock, DenseTransitionBlock
from neural_network_functions import _fc_block


class DenseEncoderVAE(nn.Module):
    """
    This class holds the Variational auto-encoder Encoder part, with DenseNet implementation
    """
    def __init__(self, device):
        super(DenseEncoderVAE, self).__init__()
        self.device             = device
        self.description        = DENSE_ENCODER_LAYER_DESCRIPTION
        self.layers             = nn.ModuleList()

        x_dim, y_dim, channels  = self.compute_dim_sizes()

        self.x_dim              = x_dim
        self.y_dim              = y_dim
        self.conv_channels      = channels

        # ---------------------------------------------------------
        # Creating the Blocks according to the description
        # ---------------------------------------------------------
        channels    = DENSE_INIT_CONV_LAYER[1]
        conv_len    = 0
        dense_idx   = 0
        trans_idx   = 0
        linear_idx  = 0
        for ii in range(len(self.description)):
            action = self.description[ii]
            if 'conv' in action:
                self.layers.append(ConvBlock(DENSE_INIT_CONV_LAYER[0],
                                             DENSE_INIT_CONV_LAYER[1],
                                             DENSE_INIT_CONV_LAYER[2],
                                             DENSE_INIT_CONV_LAYER[3],
                                             DENSE_INIT_CONV_LAYER[4],
                                             )
                                   )
                conv_len += 1
            elif 'dense' in action:
                self.layers.append(DenseBlock(channels=channels,
                                              depth=DENSE_DEPTHS[dense_idx],
                                              growth_rate=DENSE_GROWTH_RATES[dense_idx],
                                              kernel_size=DENSE_ENCODER_KERNEL_SIZE[dense_idx],
                                              stride=DENSE_ENCODER_STRIDES[dense_idx],
                                              padding=DENSE_ENCODER_PADDING[dense_idx]))
                channels  += DENSE_DEPTHS[dense_idx] * DENSE_GROWTH_RATES[dense_idx]
                dense_idx += 1
                conv_len += 1
            elif 'transition' in action:
                self.layers.append(DenseTransitionBlock(in_channels=channels,
                                                        reduction_rate=DENSE_REDUCTION_RATES[trans_idx],
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=1,
                                                        pool_size=DENSE_ENCODER_MAX_POOL_SIZE[trans_idx]))
                channels = math.floor(channels * DENSE_REDUCTION_RATES[trans_idx])
                trans_idx += 1
                conv_len += 1
            elif 'linear' in action:
                if linear_idx == 0:
                    self.layers.append(_fc_block(x_dim * y_dim * channels,
                                                 ENCODER_FC_LAYERS[linear_idx],
                                                 activation=True))
                elif 'last' in action:
                    self.layers.append(_fc_block(ENCODER_FC_LAYERS[linear_idx - 1],
                                                 ENCODER_FC_LAYERS[linear_idx],
                                                 activation=False))
                else:
                    self.layers.append(_fc_block(ENCODER_FC_LAYERS[linear_idx - 1],
                                                 ENCODER_FC_LAYERS[linear_idx],
                                                 activation=True))
                linear_idx += 1

        self.conv_len = conv_len
        self.fc_len   = linear_idx

    def compute_dim_sizes(self):
        x_dim_size  = XQUANTIZE
        y_dim_size  = YQUANTIZE
        channels    = DENSE_INIT_CONV_LAYER[1]
        dense_idx   = 0
        trans_idx   = 0
        for ii in range(len(self.description)):
            action = self.description[ii]
            if 'conv' in action:
                x_dim_size = int((x_dim_size - (DENSE_INIT_CONV_LAYER[2] - DENSE_INIT_CONV_LAYER[3]) + 2 *
                                  DENSE_INIT_CONV_LAYER[4]) / DENSE_INIT_CONV_LAYER[3])
                y_dim_size = int((y_dim_size - (DENSE_INIT_CONV_LAYER[2] - DENSE_INIT_CONV_LAYER[3]) + 2 *
                                  DENSE_INIT_CONV_LAYER[4]) / DENSE_INIT_CONV_LAYER[3])
            elif 'dense' in action:
                channels += DENSE_GROWTH_RATES[dense_idx] * DENSE_DEPTHS[dense_idx]
                dense_idx += 1
            elif 'transition' in action:
                channels = math.floor(channels * DENSE_REDUCTION_RATES[trans_idx])
                x_dim_size = int(x_dim_size / DENSE_ENCODER_MAX_POOL_SIZE[trans_idx])
                y_dim_size = int(y_dim_size / DENSE_ENCODER_MAX_POOL_SIZE[trans_idx])
                trans_idx += 1

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
        x = x.view(-1, self.x_dim * self.y_dim * self.conv_channels)
        # ---------------------------------------------------------
        # passing through the fully connected blocks
        # ---------------------------------------------------------
        for ii in range(self.fc_len):
            layer = self.layers[ii + self.conv_len]
            x = layer(x)

        return x
