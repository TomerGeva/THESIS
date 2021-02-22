from Config import *
import torch.nn as nn
import pandas   as pd
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
from torch.autograd          import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data        import Dataset, DataLoader, random_split
from functions import *


class ConvNet(nn.Module):
    def __init__(self, device):
        super(ConvNet, self).__init__()
        self.device = device
        self.description = CONV_DESCRIPTION
        self.conv_len = len(FILTER_NUM) - 1  # THS INPUT IS NOT INCLUDED, THUS THE REDUCTION OF 1
        self.fc_len = len(FC_LAYERS)
        self.layers = nn.ModuleList()
        # ---------------------------------------------------------
        # Creating the convolution blocks
        # ---------------------------------------------------------
        for ii in range(self.conv_len):
            self.layers.append(self._conv_block(FILTER_NUM[ii],
                                                FILTER_NUM[ii + 1],
                                                KERNEL_SIZE[ii],
                                                STRIDES[ii],
                                                PADDING[ii],
                                                MAX_POOL_SIZE[ii])
                               )

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding, maxpool=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(maxpool)
        )

