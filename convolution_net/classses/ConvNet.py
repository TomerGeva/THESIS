from convolution_net.Config import *
import torch.nn as nn
from old_python_files.neural_network_functions import _conv_block, _fc_block


class ConvNet(nn.Module):
    def __init__(self, device):
        super(ConvNet, self).__init__()
        self.device      = device
        self.description = LAYER_DESCRIPTION
        self.conv_len    = len(FILTER_NUM) - 1  + len(MAX_POOL_SIZE)
        self.fc_len      = len(FC_LAYERS)
        self.layers      = nn.ModuleList()
        x_dim, y_dim     = self.compute_dim_sizes()
        # ---------------------------------------------------------
        # Creating the Blocks according to the description
        # ---------------------------------------------------------
        conv_idx    = 0
        maxpool_idx = 0
        linear_idx  = 0
        for ii in range(len(self.description)):
            action = self.description[ii]
            if 'conv' in action:
                self.layers.append(_conv_block(FILTER_NUM[conv_idx],
                                               FILTER_NUM[conv_idx + 1],
                                               KERNEL_SIZE[conv_idx],
                                               STRIDES[conv_idx],
                                               PADDING[conv_idx],
                                               )
                                   )
                conv_idx += 1
            elif 'pool' in action:
                self.layers.append(nn.MaxPool2d(MAX_POOL_SIZE[maxpool_idx]))
                maxpool_idx += 1
            elif 'linear' in action:
                if linear_idx == 0:
                    self.layers.append(_fc_block(x_dim * y_dim * FILTER_NUM[-1],
                                                 FC_LAYERS[linear_idx],
                                                 activation=True))
                elif 'last' in action:
                    self.layers.append(_fc_block(FC_LAYERS[linear_idx - 1],
                                                 FC_LAYERS[linear_idx],
                                                 activation=False))
                else:
                    self.layers.append(_fc_block(FC_LAYERS[linear_idx - 1],
                                                 FC_LAYERS[linear_idx],
                                                 activation=True))
                linear_idx += 1

    def compute_dim_sizes(self):
        x_dim_size  = XQUANTIZE
        y_dim_size  = YQUANTIZE
        conv_idx    = 0
        maxpool_idx = 0
        for ii in range(len(self.description)):
            action = self.description[ii]
            if 'conv' in action:
                x_dim_size = int((x_dim_size - (KERNEL_SIZE[conv_idx] - STRIDES[conv_idx]) + 2 * PADDING[conv_idx])
                                 / STRIDES[conv_idx])
                y_dim_size = int((y_dim_size - (KERNEL_SIZE[conv_idx] - STRIDES[conv_idx]) + 2 * PADDING[conv_idx])
                                 / STRIDES[conv_idx])
                conv_idx += 1
            elif 'pool' in action:
                x_dim_size = int(x_dim_size / MAX_POOL_SIZE[maxpool_idx])
                y_dim_size = int(y_dim_size / MAX_POOL_SIZE[maxpool_idx])
                maxpool_idx += 1

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
        x = x.view(x.size(0), -1)
        # ---------------------------------------------------------
        # passing through the fully connected blocks
        # ---------------------------------------------------------
        for ii in range(self.fc_len):
            layer = self.layers[ii + self.conv_len]
            x = layer(x)

        return x
