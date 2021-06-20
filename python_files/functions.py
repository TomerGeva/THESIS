# ***************************************************************************************************
# THIS FILE HOLDS THE FUNCTIONS NEEDED TO MANIPULATE THE DATABASE ON WHICH THE NETWORK TRAINS
# ***************************************************************************************************
from ConfigVAE import *
import os
import torch
import numpy  as np
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


def save_state_train(trainer, logdir, vae, epoch, lr, filename=None):
    """Saving model and optimizer to drive, as well as current epoch and loss
    # When saving a general checkpoint, to be used for either inference or resuming training, you must save more
    # than just the model’s state_dict.
    # It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are
    # updated as the model trains.
    """
    if filename is None:
        name = 'VAE_model_data_lr_ +' + str(lr) + '_epoch_' + str(epoch) + '.tar'
        path = os.path.join(logdir, name)
    else:
        path = os.path.join(logdir, filename)

    data_to_save = {'epoch': epoch,
                    'vae_state_dict': vae.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'lr': lr
                    }
    torch.save(data_to_save, path)


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
