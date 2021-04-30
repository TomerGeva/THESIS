import torch.nn as nn


# ==================================================================================================================
# Convolution block, with a convolution layer, possible batch normalization and ReLU activation
# ==================================================================================================================
def _conv_block(in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.ReLU(),
        )


# ==================================================================================================================
# Linear block, with a fully connected layer and ReLU activation
# ==================================================================================================================
def _fc_block(in_size, out_size, activation=True):
    if activation:
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU()
        )
    else:
        return nn.Linear(in_size, out_size)


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
