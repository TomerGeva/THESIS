import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_network_block_classes import ConvBlock1D, FullyConnectedBlock, FullyConnectedResidualBlock, PadPool1D, AdaPadPool1D
from neural_network_block_classes import EdgeConv, ModEdgeConv, PointNetSetAbstraction


class ModDGCNN(nn.Module):
    def __init__(self, device, topology, concat_edgeconv, flatten_type):
        super(ModDGCNN, self).__init__()
        self.device       = device
        self.topology     = topology
        self.layers       = nn.ModuleList()
        self.concat_edge  = concat_edgeconv
        self.flatten_type = flatten_type
        # ---------------------------------------------------------
        # Creating the Blocks according to the description
        # ---------------------------------------------------------
        edgeconv_len = 0
        conv_len     = 0
        linear_len   = 0
        for ii in range(len(self.topology)):
            action = self.topology[ii]
            if 'edgeconv' in action[0]:
                edgeconv_len += 1
                self.layers.append(EdgeConv(action[1]))
            elif 'conv1d' in action[0]:
                conv_len += 1
                self.layers.append(ConvBlock1D(action[1]))
            elif 'adapool1d' in action[0]:
                conv_len += 1
                self.layers.append(AdaPadPool1D(action[1]))
            elif 'pool1d' in action[0]:
                conv_len += 1
                self.layers.append(PadPool1D(action[1]))
            elif 'res-linear' in action[0]:
                linear_len += 1
                self.layers.append(FullyConnectedResidualBlock(action[1]))
            elif 'linear' in action[0]:
                linear_len += 1
                self.layers.append(FullyConnectedBlock(action[1]))
        self.squeeze = (conv_len > 0) or (edgeconv_len > 0)
        self.edgeconv_len = edgeconv_len
        self.conv_len     = conv_len
        self.fc_len       = linear_len

    def forward(self, x):
        batch_size    = x.size(0)
        edgeconv_list = []
        # ---------------------------------------------------------
        # passing through the edge - convolution blocks
        # ---------------------------------------------------------
        for ii in range(self.edgeconv_len):
            layer = self.layers[ii]
            x     = layer(x)
            edgeconv_list.append(x)
        if self.concat_edge:
            x = torch.cat(edgeconv_list, dim=1)
        # ---------------------------------------------------------
        # passing through the convolution blocks
        # ---------------------------------------------------------
        for ii in range(self.conv_len):
            layer = self.layers[ii + self.edgeconv_len]
            x = layer(x)
        # ---------------------------------------------------------
        # flattening
        # ---------------------------------------------------------
        if self.flatten_type == 'max':
            x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        elif self.flatten_type == 'avg':
            x = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        elif self.flatten_type == 'both':
            x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
            x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
            x = torch.cat((x1, x2), 1)
        # ---------------------------------------------------------
        # squeezing for the FC layers
        # ---------------------------------------------------------
        if self.squeeze:
            x = x.squeeze()
        # ---------------------------------------------------------
        # passing through the fully connected blocks
        # ---------------------------------------------------------
        for ii in range(self.fc_len):
            layer = self.layers[ii + self.conv_len + self.edgeconv_len]
            x = layer(x)
        return x


class ModDGCNN2(nn.Module):
    def __init__(self, device, topology, flatten_type):
        super(ModDGCNN2, self).__init__()
        self.device          = device
        self.topology        = topology
        self.edgeconv_layers = nn.ModuleList()
        self.conv_layers     = nn.ModuleList()
        self.fc_layers       = nn.ModuleList()
        # self.layers        = nn.ModuleList()
        self.flatten_type = flatten_type
        # ---------------------------------------------------------
        # Creating the Blocks according to the description
        # ---------------------------------------------------------
        edgeconv_len = 0
        conv_len = 0
        linear_len = 0
        for ii in range(len(self.topology)):
            action = self.topology[ii]
            if 'modedgeconv' in action[0]:
                edgeconv_len += 1
                self.edgeconv_layers.append(ModEdgeConv(action[1]))
            elif 'edgeconv' in action[0]:
                edgeconv_len += 1
                self.edgeconv_layers.append(EdgeConv(action[1]))
            elif 'sg_pointnet' in action[0]:
                edgeconv_len += 1
                self.edgeconv_layers.append(PointNetSetAbstraction(action[1]))
            elif 'conv1d' in action[0]:
                conv_len += 1
                self.conv_layers.append(ConvBlock1D(action[1]))
            elif 'adapool1d' in action[0]:
                conv_len += 1
                self.conv_layers.append(AdaPadPool1D(action[1]))
            elif 'pool1d' in action[0]:
                conv_len += 1
                self.conv_layers.append(PadPool1D(action[1]))
            elif 'res-linear' in action[0]:
                linear_len += 1
                self.fc_layers.append(FullyConnectedResidualBlock(action[1]))
            elif 'linear' in action[0]:
                linear_len += 1
                self.fc_layers.append(FullyConnectedBlock(action[1]))
        self.squeeze = (conv_len > 0) or (edgeconv_len > 0)
        self.edgeconv_len = edgeconv_len
        self.conv_len = conv_len
        self.fc_len = linear_len

    def forward(self, x, points=None):
        """
        :param x: data per point. Size is B X D X N
        :param points: coordinates. If None, copies the input. Size is B X F X N ; F = 2, 3 - spatial coordinates
        :return:
        """
        batch_size    = x.size(0)
        if points is None:
            points = x.contiguous()
        # ---------------------------------------------------------
        # passing through the edge - convolution blocks
        # ---------------------------------------------------------
        for layer in self.edgeconv_layers:
            if type(layer) is PointNetSetAbstraction:
                points, x = layer(points, x)
            else:
                x = layer(points, x)
        # ---------------------------------------------------------
        # passing through the convolution blocks
        # ---------------------------------------------------------
        for layer in self.conv_layers:
            x = layer(x)
        # ---------------------------------------------------------
        # flattening
        # ---------------------------------------------------------
        if self.flatten_type == 'max':
            x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        elif self.flatten_type == 'avg':
            x = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        elif self.flatten_type == 'both':
            x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
            x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
            x = torch.cat((x1, x2), 1)
        # ---------------------------------------------------------
        # squeezing for the FC layers
        # ---------------------------------------------------------
        if self.squeeze:
            x = x.squeeze()
        # ---------------------------------------------------------
        # passing through the fully connected blocks
        # ---------------------------------------------------------
        for layer in self.fc_layers:
            x = layer(x)
        return x
