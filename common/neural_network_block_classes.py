import torch
import torch.nn as nn
import torch.nn.functional as F
from global_const import activation_type_e, pool_e
from global_struct import ConvBlockData, PadPoolData
from auxiliary_functions import truncated_relu
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


class Activator(nn.Module):
    """
    This block implement all possibilities of activators wanted, as defined in the enumerator
    """

    def __init__(self, act_type, alpha=0.2):
        super(Activator, self).__init__()
        self.alpha    = alpha
        self.act_type = act_type

    def forward(self, x):
        if self.act_type == activation_type_e.null:
            return x
        elif self.act_type == activation_type_e.ReLU:
            return F.relu(x)
        elif self.act_type == activation_type_e.tanh:
            return F.tanh(x)
        elif self.act_type == activation_type_e.sig:
            return F.sigmoid(x)
        elif self.act_type == activation_type_e.lReLU:
            return F.leaky_relu(x, negative_slope=self.alpha)
        elif self.act_type == activation_type_e.tReLU:
            return truncated_relu(x)
        elif self.act_type == activation_type_e.SELU:
            return F.selu(x)
        else:
            raise ValueError('Unknown activation used!')


class PadPool2D(nn.Module):
    """
        This class implements max pooling block, with zero padding
    """

    def __init__(self, padpool_data):
        super(PadPool2D, self).__init__()
        self.kernel = padpool_data.kernel
        self.padding = padpool_data.pad

        if type(self.padding) is int:
            self.pad = nn.ZeroPad2d(self.padding) if self.padding > 0 else None
        else:
            self.pad = nn.ZeroPad2d(self.padding) if sum(self.padding) > 0 else None
        if padpool_data.pool_type is pool_e.MAX:
            self.pool = nn.MaxPool2d(kernel_size=self.kernel) if self.kernel > 1 else None
        elif padpool_data.pool_type is pool_e.AVG:
            self.pool = nn.AvgPool2d(kernel_size=self.kernel) if self.kernel > 1 else None
        else:
            self.pool = None

    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class ConvBlock2D(nn.Module):
    """
    This class implements a convolution block, support batch morn, dropout and activations

    Input ---> conv2D ---> dropout ---> batch_norm ---> activation ---> Output

    """

    def __init__(self, conv_data):
        super(ConvBlock2D, self).__init__()
        self.data = conv_data

        self.conv = nn.Conv2d(in_channels=conv_data.in_channels,
                              out_channels=conv_data.out_channels,
                              kernel_size=conv_data.kernel,
                              stride=conv_data.stride,
                              padding=conv_data.padding,
                              dilation=conv_data.dilation,
                              bias=conv_data.bias
                              )
        self.drop = nn.Dropout(conv_data.drate) if conv_data.drate > 0 else None
        self.bnorm = nn.BatchNorm2d(num_features=conv_data.out_channels) if conv_data.bnorm is True else None
        self.act = Activator(act_type=conv_data.act, alpha=conv_data.alpha)

    def forward(self, x):
        out = self.conv(x)
        if self.data.drate > 0:
            out = self.drop(out)
        if self.data.bnorm:
            out = self.bnorm(out)
        out = self.act(out)

        return out


class ResidualConvBlock2D(nn.Module):
    def __init__(self, conv_data):
        super(ResidualConvBlock2D, self).__init__()
        self.data = conv_data

        self.layers = nn.ModuleList()

        self.transient_conv = nn.Conv2d(in_channels=conv_data.in_channels,
                                        out_channels=conv_data.out_channels,
                                        kernel_size=conv_data.kernel,
                                        stride=conv_data.stride,
                                        padding=conv_data.padding,
                                        dilation=conv_data.dilation,
                                        bias=conv_data.bias
                                        ) if conv_data.in_channels != conv_data.out_channels else None
        self.drop = nn.Dropout(conv_data.drate) if conv_data.drate > 0 else None
        self.bnorm = nn.BatchNorm2d(num_features=conv_data.out_channels) if conv_data.bnorm is True else None
        self.act = Activator(act_type=conv_data.act, alpha=conv_data.alpha)

        for _ in range(conv_data.layers - 1):
            self.layers.append(nn.Conv2d(in_channels=conv_data.in_channels,
                                         out_channels=conv_data.in_channels,
                                         kernel_size=conv_data.kernel,
                                         stride=conv_data.stride,
                                         padding=conv_data.padding,
                                         dilation=conv_data.dilation,
                                         bias=conv_data.bias
                                         ))
        self.layers.append(nn.Conv2d(in_channels=conv_data.in_channels,
                                     out_channels=conv_data.out_channels,
                                     kernel_size=conv_data.kernel,
                                     stride=conv_data.stride,
                                     padding=conv_data.padding,
                                     dilation=conv_data.dilation,
                                     bias=conv_data.bias
                                     ))

    def forward(self, x):
        # ===========================================
        # short path
        # ===========================================
        if self.transient_conv is not None:
            out_short = self.transient_conv(x)
        else:
            out_short = x
        # ===========================================
        # Long path
        # ===========================================
        for ii in range(len(self.layers)):
            layer = self.layers[ii]
            x = self.act(layer(x))
        out_total = x + out_short
        # ===========================================
        # Other functionalities
        # ===========================================
        if self.data.drate > 0:
            out_total = self.drop(out_total)
        if self.data.bnorm:
            out_total = self.bnorm(out_total)
        out_total = self.act(out_total)

        return out_total


class SeparableConvBlock2D(nn.Module):
    """
    This class implements a convolution block, support batch morn, dropout and activations

    Input ---> conv2D --->  conv2D ---> dropout ---> batch_norm ---> activation ---> Output
            Channel-wise  Point-wise
    """

    def __init__(self, conv_data):
        super(SeparableConvBlock2D, self).__init__()
        self.data = conv_data

        self.conv_cw = nn.Conv2d(in_channels=conv_data.in_channels,
                                 out_channels=conv_data.in_channels,
                                 kernel_size=conv_data.kernel,
                                 stride=conv_data.stride,
                                 padding=conv_data.padding,
                                 dilation=conv_data.dilation,
                                 bias=conv_data.bias,
                                 groups=conv_data.in_channels
                                 )
        self.conv_pw = nn.Conv2d(in_channels=conv_data.in_channels,
                                 out_channels=conv_data.out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 dilation=1,
                                 bias=conv_data.bias,
                                 groups=1
                                 )
        self.drop = nn.Dropout(conv_data.drate) if conv_data.drate > 0 else None
        self.bnorm = nn.BatchNorm2d(num_features=conv_data.out_channels) if conv_data.bnorm is True else None
        self.act = Activator(act_type=conv_data.act, alpha=conv_data.alpha)

    def forward(self, x):
        out = self.conv_cw(x)
        out = self.conv_pw(out)
        if self.data.drate > 0:
            out = self.drop(out)
        if self.data.bnorm:
            out = self.bnorm(out)
        out = self.act(out)

        return out


class ConvTransposeBlock2D(nn.Module):
    """
   This class implements a convolution block, support batch morn, dropout and activations

   Input ---> convTranspose2D ---> dropout ---> batch_norm ---> activation ---> Output

   """

    def __init__(self, conv_data):
        super(ConvTransposeBlock2D, self).__init__()
        self.data = conv_data

        self.conv = nn.ConvTranspose2d(in_channels=conv_data.in_channels,
                                       out_channels=conv_data.out_channels,
                                       kernel_size=conv_data.kernel,
                                       stride=conv_data.stride,
                                       padding=conv_data.padding,
                                       output_padding=conv_data.output_padding,
                                       dilation=conv_data.dilation,
                                       bias=conv_data.bias
                                       )
        self.drop = nn.Dropout(conv_data.drate) if conv_data.drate > 0 else None
        self.bnorm = nn.BatchNorm2d(num_features=conv_data.out_channels) if conv_data.bnorm is True else None
        self.act = Activator(act_type=conv_data.act, alpha=conv_data.alpha)

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

    Input ---> conv2D ---> dropout  ---> batch_norm ---> activation ---> concatenation ---> Output

    """

    def __init__(self, basic_dense_data):
        super(BasicDenseBlock, self).__init__()
        self.data = basic_dense_data

        self.conv = nn.Conv2d(in_channels=basic_dense_data.in_channels,
                              out_channels=basic_dense_data.out_channels,
                              kernel_size=basic_dense_data.kernel,
                              stride=basic_dense_data.stride,
                              padding=basic_dense_data.padding,
                              dilation=basic_dense_data.dilation,
                              bias=basic_dense_data.bias
                              )
        self.drop = nn.Dropout(basic_dense_data.drate) if basic_dense_data.drate > 0 else None
        self.bnorm = nn.BatchNorm2d(
            num_features=basic_dense_data.out_channels) if basic_dense_data.bnorm is True else None
        self.act = Activator(act_type=basic_dense_data.act, alpha=basic_dense_data.alpha)

    def forward(self, x):
        out = self.conv(x)
        if self.data.drate > 0:
            out = self.drop(out)
        if self.data.bnorm is True:
            out = self.bnorm(out)
        self.act(out)

        out = torch.cat([x, out], 1)  # concatenating over channel dimension
        return out


class DenseBlock(nn.Module):
    """
    This class implements a dense block, with differentiable number of BasicDenseBlocks and custom growth rate.
    All blocks share similar architecture, i.e. kernels, strides, padding, batchnorm and dropout settings
    (may be expanded in the future)
    """

    def __init__(self, dense_data):
        super(DenseBlock, self).__init__()
        self.data = dense_data

        self.module_list = nn.ModuleList()

        # ---------------------------------------------------------
        # Creating the Blocks according to the inputs
        # ---------------------------------------------------------
        for ii in range(dense_data.depth):
            self.module_list.append(
                BasicDenseBlock(ConvBlockData(in_channels=dense_data.in_channels + ii * dense_data.growth,
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

        self.conv = ConvBlock2D(ConvBlockData(in_channels=transition_data.in_channels,
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
        self.padpool = PadPool2D(PadPoolData(pool_type=transition_data.pool_type,
                                             kernel=transition_data.pool_size,
                                             pad=transition_data.pool_padding
                                             )
                                 )

    def forward(self, x):
        out = self.conv(x)
        return self.padpool(out)


class FullyConnectedBlock(nn.Module):
    """
        This class implements a fully connected block, support batch morn, ReLU and/or dropout
    """

    def __init__(self, fc_data):
        super(FullyConnectedBlock, self).__init__()
        self.data = fc_data

        self.fc = nn.Linear(in_features=fc_data.in_neurons,
                            out_features=fc_data.out_neurons,
                            bias=fc_data.bias
                            )
        self.bnorm = nn.LayerNorm(fc_data.out_neurons) if fc_data.bnorm is True else None
        # self.bnorm = nn.BatchNorm1d(fc_data.out_neurons) if fc_data.bnorm is True else None
        self.drop = nn.Dropout(fc_data.drate) if fc_data.drate > 0 else None
        self.act = Activator(act_type=fc_data.act, alpha=fc_data.alpha)

    def forward(self, x):
        out = self.fc(x)
        if self.data.drate > 0:
            out = self.drop(out)
        if self.data.bnorm:
            out = self.bnorm(out)
        out = self.act(out)

        return out


class FullyConnectedResidualBlock(nn.Module):
    def __init__(self, fc_data):
        super(FullyConnectedResidualBlock, self).__init__()
        self.data = fc_data
        self.layers = nn.ModuleList()
        self.transient_fc = nn.Linear(in_features=fc_data.in_neurons, out_features=fc_data.out_neurons, bias=fc_data.bias) if fc_data.in_neurons != fc_data.out_neurons else None
        self.bnorm = nn.LayerNorm(fc_data.out_neurons) if fc_data.bnorm is True else None
        # self.bnorm = nn.BatchNorm1d(fc_data.out_neurons) if fc_data.bnorm is True else None
        self.drop = nn.Dropout(fc_data.drate) if fc_data.drate > 0 else None
        self.act = Activator(act_type=fc_data.act, alpha=fc_data.alpha)

        for _ in range(fc_data.layers - 1):
            self.layers.append(nn.Linear(in_features=fc_data.in_neurons, out_features=fc_data.in_neurons))
        self.layers.append(nn.Linear(in_features=fc_data.in_neurons, out_features=fc_data.out_neurons))

    def forward(self, x):
        # ===========================================
        # short path
        # ===========================================
        if self.transient_fc is not None:
            out_short = self.transient_fc(x)
        else:
            out_short = x
        # ===========================================
        # Long path
        # ===========================================
        for ii in range(len(self.layers)):
            layer = self.layers[ii]
            x = self.act(layer(x))
        out_total = x + out_short
        # ===========================================
        # Other functionalities
        # ===========================================
        if self.data.drate > 0:
            out_total = self.drop(out_total)
        if self.data.bnorm:
            out_total = self.bnorm(out_total)
        out_total = self.act(out_total)

        return out_total


class SelfAttentionExperiment(nn.Module):
    def __init__(self, self_attention_data):
        super(SelfAttentionExperiment, self).__init__()
        self.data = self_attention_data
        self.patch_size_x = self_attention_data.patch_size_x
        self.patch_size_y = self_attention_data.patch_size_y

    def flatten_patches(self, batch):
        """
        :param batch:
        :return: converts the image batch to a batch of flattened patches, to which we can perform the attention
        """
        return rearrange(batch, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=self.patch_size_y, s2=self.patch_size_x)

    def forward(self, x):
        # converting to flattened patches
        x_patches = self.flatten_patches(x)

        energy    = torch.einsum("bql,bkl->bqk", [x_patches, x_patches])
        attention = torch.softmax(energy / (self.patch_size_y ** 0.5), dim=2)


class SelfAttentionBlock(nn.Module):  # single head attention
    def __init__(self, self_attention_data):
        super(SelfAttentionBlock, self).__init__()
        self.data = self_attention_data
        self.patch_size_x = self_attention_data.patch_size_x
        self.patch_size_y = self_attention_data.patch_size_y
        self.embed_size   = self_attention_data.embed_size
        self.patch_embed  = nn.Sequential(
            nn.Conv2d(1, self.embed_size,
                      kernel_size=(self.patch_size_y, self.patch_size_x),
                      stride=(self.patch_size_y, self.patch_size_x)),
            Rearrange('b e h w -> b (h w) e')
        )
        self.projection = nn.Linear(self.embed_size, self.embed_size)

    def flatten_patches_embed(self, batch):
        """
        :param batch:
        :return: converts the image batch to a batch of flattened patches, to which we can perform the attention
        """
        return self.patch_embed(batch)

    def forward(self, x):
        # converting to flattened patches
        x_patches = self.flatten_patches_embed(x)

        energy    = torch.einsum("bql,bkl->bqk", x_patches, x_patches)
        attention = torch.softmax(energy / (self.patch_size_y ** 0.5), dim=-1)
        out       = torch.einsum('bnl,blp -> bnp', attention, x_patches)  # (batch, number of patches, patch length)
        out       = reduce(out, 'b n p -> b p', reduction='mean')
        out       = self.projection(out)
        return out
