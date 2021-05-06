import torch
import torch.nn as nn


class BasicDenseBlock(nn.Module):
    """
    This basic block implements convolution and then concatenation of the input to the output over the channels.
    This block supports batch normalization and / or dropout, and ReLU activation
    """
    def __init__(self, in_channels, growth, kernel_size, stride, padding, batch_norm=True, dropout_rate=0.0, relu=True):
        super(BasicDenseBlock, self).__init__()
        self.in_channels    = in_channels
        self.growth         = growth
        self.kernel         = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.bnorm          = batch_norm
        self.drate          = dropout_rate
        self.relu           = relu

        self.conv  = nn.Conv2d(in_channels, growth, kernel_size, stride, padding)
        self.bnorm = nn.BatchNorm2d(num_features=growth)
        self.drop  = nn.Dropout2d(dropout_rate)
        self.act   = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.bnorm:
            out = self.bnorm(out)
        if self.drate > 0:
            out = self.drop(out)
        if self.relu:
            out = self.act(out)

        return torch.cat([x, out], 1)  # concatenating over channel dimension


class DenseBlock(nn.Module):
    """
    This class implements a dense block, with differentiable number of BasicDenseBlocks and custom growth rate.
    All blocks share similar architecture, i.e. kernels, strides, padding, batchnorm and dropout settings
    (may be expanded in the future)
    """
    def __init__(self, channels, depth, growth_rate, kernel_size, stride, padding, batch_norm=True, dropout_rate=0.0, relu=True):
        super(DenseBlock, self).__init__()
        self.in_channels    = channels
        self.growth         = growth_rate
        self.layers         = depth
        self.kernel         = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.bnorm          = batch_norm
        self.drate          = dropout_rate
        self.relu           = relu

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
                                                    relu=relu))

    def forward(self, x):
        for basic_block in self.module_list:
            x = basic_block(x)
        return x


class DenseTransitionBlock(nn.Module):
    """
    This class implements a transition block, used for pooling as well as convolving to reduce spatial size
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True, dropout_rate=0.0,
                 relu=True, pool_size=2):
        super(DenseTransitionBlock, self).__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel         = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.bnorm          = batch_norm
        self.drate          = dropout_rate
        self.relu           = relu
        self.pool_size      = pool_size

        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bnorm  = nn.BatchNorm2d(num_features=out_channels)
        self.drop   = nn.Dropout2d(dropout_rate)
        self.act    = nn.ReLU()
        self.pool   = nn.MaxPool2d(pool_size)

    def forward(self, x):
        out = self.conv(x)
        if self.bnorm:
            out = self.bnorm(out)
        if self.drate > 0:
            out = self.drop(out)
        if self.relu:
            out = self.act(out)

        return self.pool(out)