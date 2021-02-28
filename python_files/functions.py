# ***************************************************************************************************
# THIS FILE HOLDS THE FUNCTIONS NEEDED TO MANIPULATE THE DATABASE ON WHICH THE NETWORK TRAINS
# ***************************************************************************************************
from Config import *
import torch
import numpy  as np
import torch.nn as nn
from torch.autograd import Variable


# ==================================================================================================================
# Testing the MSE of the net
# ==================================================================================================================
def accuracy_test(net, loader):
    total = 0
    MSE   = 0
    with torch.no_grad():
        for sample in loader:
            grids = Variable(sample['grid'].float()).to(net.device)
            sensitivities = sample['sensitivity'].to(net.device)
            outputs = net(grids)
            MSE   += (sensitivities - outputs).pow(2).sum()
            total += sensitivities.size(0)

    return (MSE / total).item()


# ==================================================================================================================
# Auxiliary functions and classes for database manipulation
# ==================================================================================================================
def micrometer2pixel(arr):
    """
    This function is used to convert the coordinates from micro meter to pixel values
    :param arr: (N,2) array holding the coordinates in microns
    :return: array sized (N, 2) with the coordinates in pixel values
    """
    grid_coords = []
    for ii in range(len(arr)):
        x = float(arr[ii, 0])
        y = float(arr[ii, 1])
        x_grid = int(round(((x - XRANGE[0]) / XRANGE[1]) * (XQUANTIZE - 1), 0))
        y_grid = int(round(((y - YRANGE[0]) / YRANGE[1]) * (YQUANTIZE - 1), 0))
        grid_coords.append(np.array([x_grid, y_grid]))

    return np.array(grid_coords)


def points2mat(arr):
    """
    THIS FILE HOLDS THE FUNCTION WHICH TAKES AN ARRAY OF POINTS AND CONVERTS IT TO A MATRIX, WHERE:
    FOR EACH (X,Y) OF THE MATRIX:
        IF (X,Y) IS IN THE ARRAY, THE INDEX WILL BE 1
        OTHERWISE, IT WILL BE 0
    :param: arr: a 2-D array which holds the coordinates of the scatterers
    :return: xQuantize X yQuantize grid simulating the array
    """
    grid_array = np.zeros([XQUANTIZE, YQUANTIZE])
    grid_array[arr[:, 1], arr[:, 0]] = 255
    return grid_array


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
