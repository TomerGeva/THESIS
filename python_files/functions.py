# ***************************************************************************************************
# THIS FILE HOLDS THE FUNCTIONS NEEDED TO MANIPULATE THE DATABASE ON WHICH THE NETWORK TRAINS
# ***************************************************************************************************
from ConfigVAE import *
import os
import torch
import numpy  as np
from ModVAE import ModVAE
from TrainerVAE import TrainerVAE
from torch.autograd import Variable


# ==================================================================================================================
# Saving and loading trained networks
# ==================================================================================================================
def load_state_train(data_path, device=None):
    """
    :param data_path: path to the saved data regarding the network
    :param device: allocation to either cpu of cuda:0
    :return: the function loads the data into and returns the saves network and trainer
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # -------------------------------------
    # loading the dictionary
    # -------------------------------------
    checkpoint = torch.load(data_path, map_location=device)

    # -------------------------------------
    # arranging the data
    # -------------------------------------
    mod_vae = ModVAE(device=device)
    mod_vae.to(device)  # allocating the computation to the CPU or GPU
    mod_vae.load_state_dict(checkpoint['vae_state_dict'])

    trainer = TrainerVAE(mod_vae, lr=checkpoint['lr'], mom=MOM, beta=BETA)
    trainer.start_epoch = checkpoint['epoch']
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return mod_vae, trainer


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
