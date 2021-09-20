from torch import clamp
import torch.nn as nn


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
