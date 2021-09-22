import math
import torch
import torch.nn as nn
from global_const import activation_type_e
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


class MaxPool2dPadding(nn.Module):
    """
        This class implements max pooling block, with zero padding
    """
    def __init__(self, kernel, padding=0):
        super(MaxPool2dPadding, self).__init__()
        self.kernel = kernel
        self.padding = padding

        self.pad = nn.ZeroPad2d(padding)
        self.pool = nn.MaxPool2d(kernel_size=kernel)

    def forward(self, x):
        return self.pool(self.pad(x))


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
    def __init__(self, in_channels, reduction_rate, kernel_size, stride, padding, batch_norm=True, dropout_rate=0.0,
                 act=activation_type_e.null, alpha=0.01, pool_pad=0, pool_size=2):
        super(DenseTransitionBlock, self).__init__()
        self.in_channels    = in_channels
        self.out_channels   = math.floor(in_channels * reduction_rate)
        self.kernel         = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.bnorm          = batch_norm
        self.drate          = dropout_rate
        self.activator      = act
        self.alpha          = alpha
        self.pool_size      = pool_size
        self.pool_padding   = pool_pad

        self.conv       = ConvBlock(in_channels=in_channels,
                                    out_channels=self.out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    batch_norm=batch_norm,
                                    dropout_rate=dropout_rate,
                                    act=act, alpha=alpha)
        self.padpool    = MaxPool2dPadding(kernel=pool_size, padding=pool_pad)

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
