from global_const import activation_type_e, pool_e
import math


class ConvBlockData:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=None, batch_norm=True,
                 dropout_rate=0.0, activation=activation_type_e.null, alpha=0.01):
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel         = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.dilation       = dilation
        if bias is None:
            self.bias = not batch_norm
        else:
            self.bias = bias
        self.bnorm          = batch_norm
        self.drate          = dropout_rate
        self.act            = activation
        self.alpha          = alpha


class DenseBlockData:
    def __init__(self, growth_rate, depth, kernel_size, stride, padding, dilation=1, bias=None, batch_norm=True,
                 dropout_rate=0.0, activation=activation_type_e.null, alpha=0.01):
        self.in_channels    = None  # computed during encoder Init
        self.growth         = growth_rate
        self.depth          = depth
        self.kernel         = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.dilation       = dilation
        if bias is None:
            self.bias = not batch_norm
        else:
            self.bias = bias
        self.bnorm          = batch_norm
        self.drate          = dropout_rate
        self.act            = activation
        self.alpha          = alpha


class TransBlockData:
    def __init__(self, reduction_rate, kernel_size, stride, padding, dilation=1, bias=None,
                 batch_norm=True, dropout_rate=0.0, activation=activation_type_e.null, alpha=0.01,
                 pool_type=pool_e.MAX, pool_pad=0, pool_size=2):
        self.in_channels    = None  # computed during encoder Init
        self.reduction_rate = reduction_rate
        self.out_channels   = None
        self.kernel         = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.dilation = dilation
        if bias is None:
            self.bias = not batch_norm
        else:
            self.bias = bias
        self.bnorm          = batch_norm
        self.drate          = dropout_rate
        self.act            = activation
        self.alpha          = alpha
        self.pool_type      = pool_type
        self.pool_size      = pool_size
        self.pool_padding   = pool_pad

    def set_in_out_channels(self, in_channels):
        self.in_channels = in_channels
        self.out_channels = math.floor(in_channels * self.reduction_rate)


class FCBlockData:
    def __init__(self, out_neurons, bias=None, batch_norm=True, dropout_rate=0.0,
                 activation=activation_type_e.null, alpha=0.01):
        self.in_neurons     = None  # computed during encoder Init
        self.out_neurons    = out_neurons
        if bias is None:
            self.bias = not batch_norm
        else:
            self.bias = bias
        self.bnorm          = batch_norm
        self.drate          = dropout_rate
        self.act            = activation
        self.alpha          = alpha
