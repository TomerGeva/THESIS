# import torch
# import numpy as np
from Config import *
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# defining the network
# ============================================================
class Net(nn.Module):
    # --------------------------------------------------------
    # input is a 2500X2500X1 signed matrix {-1, 1}
    # --------------------------------------------------------
    def __init__(self, device):
        super(Net, self).__init__()
        # a variable describing the forward process
        self.description = CONV_DESCRIPTION
        # computation location
        self.device      = device
        #                       in_channels | out_channels | kernel_size    | stride
        self.conv1 = nn.Conv2d(1            , FILTER_NUM[0], KERNEL_SIZE[0], STRIDES[0])
        self.conv2 = nn.Conv2d(FILTER_NUM[0], FILTER_NUM[1], KERNEL_SIZE[1], STRIDES[1])
        self.conv3 = nn.Conv2d(FILTER_NUM[1], FILTER_NUM[2], KERNEL_SIZE[2], STRIDES[2])
        self.conv4 = nn.Conv2d(FILTER_NUM[2], FILTER_NUM[3], KERNEL_SIZE[3], STRIDES[3])
        self.conv5 = nn.Conv2d(FILTER_NUM[3], FILTER_NUM[4], KERNEL_SIZE[4], STRIDES[4])
        self.conv6 = nn.Conv2d(FILTER_NUM[4], FILTER_NUM[5], KERNEL_SIZE[5], STRIDES[5])
        self.conv7 = nn.Conv2d(FILTER_NUM[5], FILTER_NUM[6], KERNEL_SIZE[6], STRIDES[6])
        # max-pooling to avoid over-fitting
        self.pool  = nn.MaxPool2d(MAX_POOL_SIZE)
        # performing affine operation: y = Wx + b
        x_dim, y_dim = self.compute_dim_sizes()
        self.fc1 = nn.Linear(FILTER_NUM[-1] * x_dim * y_dim, FC_LAYERS[0])
        self.fc2 = nn.Linear(FC_LAYERS[0], FC_LAYERS[1])
        self.fc3 = nn.Linear(FC_LAYERS[1], FC_LAYERS[2])
        self.fc4 = nn.Linear(FC_LAYERS[2], FC_LAYERS[3])

    def compute_dim_sizes(self):
        x_dim_size = XQUANTIZE
        y_dim_size = YQUANTIZE
        counter    = 0
        for action in range(len(self.description)):
            if self.description[action] == 'conv':
                x_dim_size = int((x_dim_size - (KERNEL_SIZE[counter] - STRIDES[counter])) / STRIDES[counter])
                y_dim_size = int((y_dim_size - (KERNEL_SIZE[counter] - STRIDES[counter])) / STRIDES[counter])
                counter += 1
            elif self.description[action] == 'pool':
                x_dim_size = int(x_dim_size / MAX_POOL_SIZE)
                y_dim_size = int(y_dim_size / MAX_POOL_SIZE)

        return x_dim_size, y_dim_size

    def forward(self, x):
        #                                               Convolutional section
        x = F.relu(self.conv1(x))                           # data_compression
        x = F.relu(self.conv2(x))                           # { first stage
        x = self.pool(F.relu(self.conv3(x)))                # }
        x = F.relu(self.conv4(x))                           # { second stage
        x = self.pool(F.relu(self.conv5(x)))                # }
        x = self.pool(F.relu(self.conv6(x)))                # third stage
        x = F.relu(self.conv7(x))                           # fourth stage
        x_dim, y_dim = self.compute_dim_sizes()       # Fully Connected section
        x = x.view(-1, FILTER_NUM[-1] * x_dim * y_dim)  # reshaping
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x