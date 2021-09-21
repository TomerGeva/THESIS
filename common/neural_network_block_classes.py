import math
import torch
import torch.nn as nn
from global_const import activation_type_e, pool_e
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
        self.module_list = nn.ModuleList()
        # -----------------------------------------------------------------------------------------
        # Padding
        # -----------------------------------------------------------------------------------------
        if type(padding) is int:
            if padding > 0:
                self.module_list.append(nn.ZeroPad2d(padding))
        else:
            if sum(padding) > 0:
                self.module_list.append(nn.ZeroPad2d(padding))
        # -----------------------------------------------------------------------------------------
        # Pooling
        # -----------------------------------------------------------------------------------------
        if pool_type is pool_e.MAX:
            self.module_list.append(nn.MaxPool2d(kernel_size=kernel))
        elif pool_type is pool_e.AVG:
            self.module_list.append(nn.AvgPool2d(kernel_size=kernel))

    def forward(self, x):
        for module in self.module_list:
            x = module(x)

        return x


class ConvBlock(nn.Module):
    """
    This class implements a convolution block, support batch morn, dropout and activations
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True, dropout_rate=0.0,
                 act=activation_type_e.null, alpha=0.01):
        super(ConvBlock, self).__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel         = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.bnorm          = batch_norm
        self.drate          = dropout_rate

        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          bias=(not batch_norm)
                                          )
                                )
        if self.drate > 0:
            self.module_list.append(nn.Dropout2d(dropout_rate))
        if self.bnorm:
            self.module_list.append(nn.BatchNorm2d(num_features=out_channels))
        self.module_list.append(Activator(act_type=act, alpha=alpha))

    def forward(self, x):
        for module in self.module_list:
            x = module(x)

        return x


class BasicDenseBlock(nn.Module):
    """
    This basic block implements convolution and then concatenation of the input to the output over the channels.
    This block supports batch normalization and / or dropout, and activations
    """
    def __init__(self, in_channels, growth, kernel_size, stride, padding, batch_norm=True, dropout_rate=0.0,
                 act=activation_type_e.null, alpha=0.01):
        super(BasicDenseBlock, self).__init__()
        self.in_channels    = in_channels
        self.growth         = growth
        self.kernel         = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.bnorm          = batch_norm
        self.drate          = dropout_rate

        self.module_list = nn.ModuleList()
        # -----------------------------------------------------------------------------------------
        # Convolution
        # -----------------------------------------------------------------------------------------
        self.module_list.append(nn.Conv2d(in_channels=in_channels,
                                          out_channels=growth,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          bias=(not batch_norm)
                                          )
                                )
        # -----------------------------------------------------------------------------------------
        # Dropout
        # -----------------------------------------------------------------------------------------
        if self.drate > 0:
            self.module_list.append(nn.Dropout2d(dropout_rate))
        # -----------------------------------------------------------------------------------------
        # Concatenation
        # -----------------------------------------------------------------------------------------
        # performed in the forward function
        # -----------------------------------------------------------------------------------------
        # Batch norm
        # -----------------------------------------------------------------------------------------
        if self.bnorm:
            self.module_list.append(nn.BatchNorm2d(num_features=(in_channels + growth)))
        # -----------------------------------------------------------------------------------------
        # Activation
        # -----------------------------------------------------------------------------------------
        self.module_list.append(Activator(act_type=act, alpha=alpha))

    def forward(self, x):
        for module in self.module_list:
            out = module(x)
            if (type(module) is nn.Dropout2d) or (self.drate == 0 and type(module) is nn.Conv2d):
                x = torch.cat([x, out], 1)  # concatenating over channel dimension
            else:
                x = out
        return x


class DenseBlock(nn.Module):
    """
    This class implements a dense block, with differentiable number of BasicDenseBlocks and custom growth rate.
    All blocks share similar architecture, i.e. kernels, strides, padding, batchnorm and dropout settings
    (may be expanded in the future)
    """
    def __init__(self, channels, depth, growth_rate, kernel_size, stride, padding, batch_norm=True, dropout_rate=0.0,
                 act=activation_type_e.null, alpha=0.01):
        super(DenseBlock, self).__init__()
        self.in_channels    = channels
        self.growth         = growth_rate
        self.layers         = depth
        self.kernel         = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.bnorm          = batch_norm
        self.drate          = dropout_rate
        self.activator      = act
        self.alpha          = alpha

        self.module_list    = nn.ModuleList()

        # ---------------------------------------------------------
        # Creating the Blocks according to the inputs
        # ---------------------------------------------------------
        for ii in range(depth):
            self.module_list.append(BasicDenseBlock(in_channels=channels+ii*growth_rate,
                                                    growth=growth_rate,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    batch_norm=batch_norm,
                                                    dropout_rate=dropout_rate,
                                                    act=act,
                                                    alpha=alpha))

    def forward(self, x):
        for basic_block in self.module_list:
            x = basic_block(x)
        return x


class DenseTransitionBlock(nn.Module):
    """
    This class implements a transition block, used for pooling as well as convolving to reduce spatial size
    """
    def __init__(self, in_channels, reduction_rate, kernel_size, stride, padding, batch_norm=True, dropout_rate=0.0,
                 act=activation_type_e.null, alpha=0.01, pool_type=pool_e.MAX, pool_pad=0, pool_size=2):
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
        self.pool_type      = pool_type
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
        self.padpool    = Pool2dPadding(pool_type=pool_type,
                                        kernel=pool_size,
                                        padding=pool_pad)

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

        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Linear(in_features=in_neurons,
                                          out_features=out_neurons,
                                          bias=(not batch_norm)
                                          )
                                )
        if self.bnorm:
            self.module_list.append( nn.BatchNorm1d(out_neurons))
        if self.drate > 0:
            self.module_list.append(nn.Dropout(dropout_rate))
        self.module_list.append(Activator(act_type=act, alpha=alpha))

    def forward(self, x):
        for module in self.module_list:
            x = module(x)

        return x
