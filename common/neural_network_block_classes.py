import math
import torch
import torch.nn as nn
from global_const import activation_type_e, pool_e
from global_struct import ConvBlockData
from auxiliary_functions import truncated_relu


class Activator(nn.Module):
    """
    This block implement all possibilities of activators wanted, as defined in the enumerator
    """

    def __init__(self, act_type, alpha=0.01):
        super(Activator, self).__init__()
        self.alpha = alpha
        if act_type == activation_type_e.null:
            self.activator = None
        elif act_type == activation_type_e.ReLU:
            self.activator = nn.ReLU()
        elif act_type == activation_type_e.tanh:
            self.activator = nn.Tanh()
        elif act_type == activation_type_e.sig:
            self.activator = nn.Sigmoid()
        elif act_type == activation_type_e.lReLU:
            self.activator = nn.LeakyReLU(negative_slope=alpha)
        elif act_type == activation_type_e.tReLU:
            self.activator = truncated_relu
        elif act_type == activation_type_e.SELU:
            self.activator = nn.SELU()

    def forward(self, x):
        if self.activator is None:
            return x
        else:
            return self.activator(x)


class Pool2dPadding(nn.Module):
    """
        This class implements max pooling block, with zero padding
    """
    def __init__(self, pool_type, kernel, padding=0):
        super(Pool2dPadding, self).__init__()
        self.kernel = kernel
        self.padding = padding

        if type(padding) is int:
            self.pad = nn.ZeroPad2d(padding) if padding > 0 else None
        else:
            self.pad = nn.ZeroPad2d(padding) if sum(padding) > 0 else None
        if pool_type is pool_e.MAX:
            self.pool = nn.MaxPool2d(kernel_size=kernel) if kernel > 1 else None
        elif pool_type is pool_e.AVG:
            self.pool = nn.AvgPool2d(kernel_size=kernel) if kernel > 1 else None
        else:
            self.pool = None

    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class ConvBlock(nn.Module):
    """
    This class implements a convolution block, support batch morn, dropout and activations
    """

    def __init__(self, conv_data):
        super(ConvBlock, self).__init__()
        self.data = conv_data

        self.conv   = nn.Conv2d(in_channels=conv_data.in_channels,
                                out_channels=conv_data.out_channels,
                                kernel_size=conv_data.kernel,
                                stride=conv_data.stride,
                                padding=conv_data.padding,
                                dilation=conv_data.dilation,
                                bias=conv_data.bias
                                )
        self.bnorm  = nn.BatchNorm2d(num_features=conv_data.out_channels) if conv_data.bnorm is True else None
        self.drop   = nn.Dropout(conv_data.drate) if conv_data.drate > 0 else None
        self.act    = Activator(act_type=conv_data.act, alpha=conv_data.alpha)

    def forward(self, x):
        out = self.conv(x)
        if self.data.drate > 0:
            out = self.drop(out)
        if self.data.bnorm:
            out = self.bnorm(out)
        out = self.act(out)

        return out


class BasicDenseBlock(nn.Module):
    """
    This basic block implements convolution and then concatenation of the input to the output over the channels.
    This block supports batch normalization and / or dropout, and activations
    """
    def __init__(self, basic_dense_data):
        super(BasicDenseBlock, self).__init__()
        self.data = basic_dense_data

        self.conv = ConvBlock(basic_dense_data)

    def forward(self, x):
        out = self.conv(x)

        return torch.cat([x, out], 1)  # concatenating over channel dimension


class DenseBlock(nn.Module):
    """
    This class implements a dense block, with differentiable number of BasicDenseBlocks and custom growth rate.
    All blocks share similar architecture, i.e. kernels, strides, padding, batchnorm and dropout settings
    (may be expanded in the future)
    """
    def __init__(self, dense_data):
        super(DenseBlock, self).__init__()
        self.data = dense_data

        self.module_list    = nn.ModuleList()

        # ---------------------------------------------------------
        # Creating the Blocks according to the inputs
        # ---------------------------------------------------------
        for ii in range(dense_data.depth):
            self.module_list.append(BasicDenseBlock(ConvBlockData(in_channels=dense_data.in_channels+ii*dense_data.growth,
                                                                  out_channels=dense_data.growth,
                                                                  kernel_size=dense_data.kernel,
                                                                  stride=dense_data.stride,
                                                                  padding=dense_data.padding,
                                                                  dilation=dense_data.dilation,
                                                                  bias=dense_data.bias,
                                                                  batch_norm=dense_data.bnorm,
                                                                  dropout_rate=dense_data.drate,
                                                                  activation=dense_data.act,
                                                                  alpha=dense_data.alpha
                                                                  )
                                                    )
                                    )

    def forward(self, x):
        for basic_block in self.module_list:
            x = basic_block(x)
        return x


class DenseTransitionBlock(nn.Module):
    """
    This class implements a transition block, used for pooling as well as convolving to reduce spatial size
    """
    def __init__(self, transition_data):
        super(DenseTransitionBlock, self).__init__()
        self.data = transition_data

        self.conv       = ConvBlock(ConvBlockData(in_channels=transition_data.in_channels,
                                                  out_channels=transition_data.out_channels,
                                                  kernel_size=transition_data.kernel,
                                                  stride=transition_data.stride,
                                                  padding=transition_data.padding,
                                                  dilation=transition_data.dilation,
                                                  bias=transition_data.bias,
                                                  batch_norm=transition_data.bnorm,
                                                  dropout_rate=transition_data.drate,
                                                  activation=transition_data.act,
                                                  alpha=transition_data.alpha
                                                  )
                                    )
        self.padpool    = Pool2dPadding(pool_type=transition_data.pool_type,
                                        kernel=transition_data.pool_size,
                                        padding=transition_data.pool_padding)

    def forward(self, x):
        out = self.conv(x)
        return self.padpool(out)


class FullyConnectedBlock(nn.Module):
    """
        This class implements a fully connected block, support batch morn, ReLU and/or dropout
    """
    def __init__(self, in_neurons, out_neurons, batch_norm=True, dropout_rate=0.0,
                 act=activation_type_e.null, alpha=0.01):
        super(FullyConnectedBlock, self).__init__()
        self.in_neurons  = in_neurons
        self.out_neurons = out_neurons
        self.bnorm = batch_norm
        self.drate = dropout_rate

        self.fc     = nn.Linear(in_features=in_neurons,
                                out_features=out_neurons,
                                bias=(not batch_norm)
                                )
        self.bnorm  = nn.BatchNorm1d(out_neurons)
        self.drop   = nn.Dropout(dropout_rate)
        self.act   = Activator(act_type=act, alpha=alpha)

    def forward(self, x):
        out = self.fc(x)
        if self.bnorm:
            out = self.bnorm(out)
        if self.drate > 0:
            out = self.drop(out)
        out = self.act(out)

        return out
