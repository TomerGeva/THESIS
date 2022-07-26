from global_const import activation_type_e, pool_e
import math


class ConvBlockData:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=True, batch_norm=True,
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


class ResConvBlock2DData:
    def __init__(self, in_channels, out_channels, layers, kernel_size, stride, padding, dilation=1, bias=True, batch_norm=True,
                 dropout_rate=0.0, activation=activation_type_e.null, alpha=0.01):
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.layers         = layers
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


class ConvTransposeBlock2DData:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, output_padding=0, bias=True,
                 batch_norm=True, dropout_rate=0.0, activation=activation_type_e.null, alpha=0.01):
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel         = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.dilation       = dilation
        self.output_padding = output_padding
        if bias is None:
            self.bias = not batch_norm
        else:
            self.bias = bias
        self.bnorm          = batch_norm
        self.drate          = dropout_rate
        self.act            = activation
        self.alpha          = alpha


class DenseBlockData:
    def __init__(self, growth_rate, depth, kernel_size, stride, padding, dilation=1, bias=True, batch_norm=True,
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
    def __init__(self, reduction_rate, kernel_size, stride, padding, dilation=1, bias=True,
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
    def __init__(self, out_neurons, in_neurons=None, bias=True, batch_norm=True, dropout_rate=0.0,
                 activation=activation_type_e.null, alpha=0.01):
        self.in_neurons     = in_neurons  # computed during encoder Init
        self.out_neurons    = out_neurons
        if bias is None:
            self.bias = not batch_norm
        else:
            self.bias = bias
        self.bnorm          = batch_norm
        self.drate          = dropout_rate
        self.act            = activation
        self.alpha          = alpha


class ResFCBlockData:
    def __init__(self, out_neurons, in_neurons=None, layers=2, bias=True, batch_norm=True, dropout_rate=0.0,
                 activation=activation_type_e.null, alpha=0.01):
        self.in_neurons     = in_neurons  # computed during encoder Init
        self.out_neurons    = out_neurons
        self.layers         = layers
        if bias is None:
            self.bias = not batch_norm
        else:
            self.bias = bias
        self.bnorm          = batch_norm
        self.drate          = dropout_rate
        self.act            = activation
        self.alpha          = alpha


class PadPoolData:
    def __init__(self, pool_type, pad=0, kernel=2):
        self.pool_type = pool_type  # MAX or AVG
        self.pad       = pad        # padding, can be either an index or a tuple with size of 4
        self.kernel    = kernel     # pooling kernel size


class AdaPadPoolData:
    def __init__(self, pool_type, pad=0, out_size=1):
        self.pool_type = pool_type  # MAX or AVG
        self.pad       = pad        # padding, can be either an index or a tuple with size of 4
        self.out_size  = out_size   # pooling output size


class SelfAttentionData:
    def __init__(self, patch_size_x, patch_size_y, embed_size):
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.embed_size   = embed_size
        self.out_neurons  = embed_size


class EdgeConvData:
    def __init__(self, k, conv_data, aggregation):
        self.k           = k
        self.conv_data   = conv_data
        self.aggregation = aggregation


class SetAbstractionData:
    def __init__(self, ntag, radius, k, in_channel, out_channels, group_all=False, pnet_kernel=1, residual=True):
        self.ntag_points  = int(math.ceil(ntag))
        self.radius       = radius
        self.k            = k
        self.in_channel   = in_channel
        self.out_channels = out_channels
        self.group_all    = group_all
        self.pnet_kernel  = pnet_kernel
        self.residual     = residual
