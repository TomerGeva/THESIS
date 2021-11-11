from torch import clamp
import torch.nn as nn
import math


# ==================================================================================================================
# Truncated ReLU activation function
# ==================================================================================================================
def truncated_relu(x):
    return clamp(x, min=0, max=1)


# ==================================================================================================================
# Function used to initialize the weights of the network before the training
# ==================================================================================================================
def initialize_weights(net, mean, std):
    """
    :param net: the model which is being normalized
    :param mean: the target mean of the weights
    :param std: the target standard deviation of the weights
    :return: nothing, just adjusts the weights
    """
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            nn.init.normal_(module.weight.data, mean, std)


# ==================================================================================================================
# Function used to computed the dimensions between each layer
# ==================================================================================================================
def compute_output_dim(x_dim, y_dim, ch_num, action):
    """
    Function used to compute the output dimensions of a NN layer
    :param x_dim: x dimension size of the input
    :param y_dim: y dimension size of the input
    :param ch_num: channel number of the input
    :param action: layer description
    :return:
    """
    if 'conv' in action[0]:  # :   [                       K_tilde                                ]         S                          P                    S
        x_dim_size = int((x_dim - ((action[1].dilation * action[1].kernel - (action[1].dilation-1)) - action[1].stride) + 2 * action[1].padding) / action[1].stride)
        y_dim_size = int((y_dim - ((action[1].dilation * action[1].kernel - (action[1].dilation-1)) - action[1].stride) + 2 * action[1].padding) / action[1].stride)
        channels = action[1].out_channels
    elif 'pool' in action:
        if type(action[1].pool_padding) is not tuple:
            x_dim_size = int((x_dim + 2 * action[1].pool_padding) / action[1].kernel)
            y_dim_size = int((y_dim + 2 * action[1].pool_padding) / action[1].kernel)
        else:
            x_dim_size = int((x_dim + 2 * action[1].pool_padding[0] + 2 * action[1].pool_padding[1]) / action[1].pool_size)
            y_dim_size = int((y_dim + 2 * action[1].pool_padding[2] + 2 * action[1].pool_padding[3]) / action[1].pool_size)
        channels = ch_num
    elif 'dense' in action[0]:
        channels = ch_num + action[1].growth * action[1].depth
        x_dim_size = x_dim
        y_dim_size = y_dim
    elif 'transition' in action[0]:
        channels = math.floor(ch_num * action[1].reduction_rate)
        # ------------------------------------------------
        # This account for the conv layer and the pooling
        # ------------------------------------------------
        x_conv_size = int((x_dim - (action[1].kernel - action[1].stride) + 2 * action[1].padding) / action[1].stride)
        y_conv_size = int((y_dim - (action[1].kernel - action[1].stride) + 2 * action[1].padding) / action[1].stride)
        if type(action[1].pool_padding) is not tuple:
            x_dim_size = int((x_conv_size + 2 * action[1].pool_padding) / action[1].pool_size)
            y_dim_size = int((y_conv_size + 2 * action[1].pool_padding) / action[1].pool_size)
        else:
            x_dim_size = int((x_conv_size + 2 * action[1].pool_padding[0] + 2 * action[1].pool_padding[1]) / action[1].pool_size)
            y_dim_size = int((y_conv_size + 2 * action[1].pool_padding[2] + 2 * action[1].pool_padding[3]) / action[1].pool_size)
    elif 'convTrans' in action[0]:
        channels = action[1].out_channels
        x_dim_size = (x_dim - 1) * action[1].stride - (2 * action[1].padding) + action[1].dilation * (action[1].kernel - 1) + action[1].output_padding + 1
        y_dim_size = (y_dim - 1) * action[1].stride - (2 * action[1].padding) + action[1].dilation * (action[1].kernel - 1) + action[1].output_padding + 1
    else:  # linear case
        x_dim_size = x_dim
        y_dim_size = y_dim
        channels   = ch_num

    return x_dim_size, y_dim_size, channels
