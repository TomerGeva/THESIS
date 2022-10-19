import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_network_block_classes import ConvBlock1D, FullyConnectedBlock, PadPool1D, AdaPadPool1D


class ModPointNet(nn.Module):
    def __init__(self, device, topology):
        super(ModPointNet, self).__init__()
        self.device   = device
        self.topology = topology
        self.layers   = nn.ModuleList()
        # ---------------------------------------------------------
        # Creating the Blocks according to the description
        # ---------------------------------------------------------
        conv_len = 0
        linear_len = 0
        for ii in range(len(self.topology)):
            action = self.topology[ii]
            if 'conv1d' in action[0]:
                conv_len += 1
                self.layers.append(ConvBlock1D(action[1]))
            elif 'adapool1d' in action[0]:
                conv_len += 1
                self.layers.append(AdaPadPool1D(action[1]))
            elif 'pool1d' in action[0]:
                conv_len += 1
                self.layers.append(PadPool1D(action[1]))
            elif 'linear' in action[0]:
                linear_len += 1
                self.layers.append(FullyConnectedBlock(action[1]))

        self.squeeze  = conv_len > 0
        self.conv_len = conv_len
        self.fc_len   = linear_len

    def forward(self, x):
        # ---------------------------------------------------------
        # passing through the convolution blocks
        # ---------------------------------------------------------
        for ii in range(self.conv_len):
            layer = self.layers[ii]
            x = layer(x)
        # ---------------------------------------------------------
        # flattening for the FC layers
        # ---------------------------------------------------------
        if self.squeeze:
            x = x.squeeze()
        # ---------------------------------------------------------
        # passing through the fully connected blocks
        # ---------------------------------------------------------
        for ii in range(self.fc_len):
            layer = self.layers[ii + self.conv_len]
            x = layer(x)
        return x


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x
