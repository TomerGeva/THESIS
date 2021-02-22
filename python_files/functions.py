# ***************************************************************************************************
# THIS FILE HOLDS THE FUNCTIONS NEEDED TO MANIPULATE THE DATABASE ON WHICH THE NETWORK TRAINS
# ***************************************************************************************************
from abc import ABC

from Config import *
import numpy  as np
import pandas as pd
import os
import csv
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from itertools        import cycle, islice
from torch.autograd   import Variable
from torch.utils.data import IterableDataset, DataLoader, random_split


# **********************************************************************************************************************
# THESE FUNCTIONS ARE OBSOLETE, BUT FOR NOW I SAVE THEM HERE
# **********************************************************************************************************************
def file2array(filename):
    """
    :param filename: path to the .csv file with the array data
    :return: function returns a dictionary with the following information:
        [array]       - key to a NX2 array which holds the scatterers' location in the medium
        [sensitivity] - key to the resulting sensitivity of the array
        [scat_num]    - key to the index of the scatterer which produces the maximal sensitivity
    """
    result_dict = {}
    grid_coords = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_line = True
        for row in csv_reader:
            if first_line:
                first_line = False
            elif row[4] == '0+0i' and row[5] == '0+0i' and row[6] == '0+0i':
                # last line, extracting the sensitivity
                result_dict['sensitivity'] = float(row[1])
                result_dict['scat_num'] = int(row[0])
            else:  # converting locations to grid coordinates
                x = float(row[0])
                y = float(row[1])
                x_grid = ((x - xRange[0])/xRange[1]) * (xQuantize - 1)
                y_grid = ((y - yRange[0])/yRange[1]) * (yQuantize - 1)
                grid_coords.append(np.array([x_grid, y_grid]))
        result_dict['array'] = np.array(grid_coords)
        return result_dict


def array2mat(arr):
    """
    THIS FILE HOLDS THE FUNCTION WHICH TAKES AN ARRAY OF POINTS AND CONVERTS IT TO A MATRIX, WHERE:
    FOR EACH (X,Y) OF THE MATRIX:
        IF (X,Y) IS IN THE ARRAY, THE INDEX WILL BE 1
        OTHERWISE, IT WILL BE 0
    :param arr: a 2-D array which holds the coordinates of the scatterers
    :return: xQuantize X yQuantize grid simulating the array
    """
    grid_array = torch.ones([1, 1, xQuantize, yQuantize]) * -1
    grid_array[0, 0, arr[:, 1], arr[:, 0]] = 1
    return grid_array


def gather_data(path):
    """
    :param path: holds the path to the folder which holds the csv files of the database
    :return:this function goes through all the '.csv' files and extracts the grid points, and the resulting sensitivity.
    the function returns a dictionary with the following keys:
    [scat_arrays] - contains the grid coordinates of the scatterers. Each array hold a set of scatterer locations
    [sensitivity] - contains the matching sensitivity of the respective array of scatterers
    [scat_num]    - contains the number of scatterer that produces the maximal sensitivity for the respective array
    [size]        - hold the size of the database
    """
    # ==================================================================
    # Internal Variables
    # ==================================================================
    file_count = 1
    array_dict = {}
    sensitivity_dict = {}
    scat_num_dict = {}

    # ==================================================================
    # Passing through all the files, and getting the data
    # ==================================================================
    files = os.listdir(path)
    for file in files:
        if file_count > 1000:
            break
        print('processing file number ' + str(file_count))
        fullfile  = path + '\\' + file
        file_dict = file2array(fullfile)

        array_dict[file_count]       = file_dict['array']
        sensitivity_dict[file_count] = file_dict['sensitivity']
        scat_num_dict[file_count]    = file_dict['scat_num']
        file_count += 1

    # ==================================================================
    # Saving and returning the variables
    # ==================================================================
    database = {
        "scat_arrays": array_dict       ,
        "sensitivity": sensitivity_dict ,
        "scat_num"   : scat_num_dict    ,
        "size"        : file_count - 1
    }
    return database


def plot_hist(sensitivity_dict, bins=100, absolute=True):
    """
    :param sensitivity_dict: Dictionary containing the sensitivities in the values
    :param bins: Number of bins in the histogram. default 100
    :param absolute: if True, uses absolute values
    :return: The function plots a histogram of the sensitivities
    """
    sens_vals = list(sensitivity_dict.values())

    if absolute:
        sens_vals = np.abs(sens_vals)

    plt.hist(sens_vals, bins=bins)


# **********************************************************************************************************************
# THESE FUNCTIONS ARE GOOD, AND I USE THEM
# **********************************************************************************************************************
# ==================================================================================================================
# Testing the MSE of the net
# ==================================================================================================================
def accuracy_test(net, loader):
    total = 0
    MSE   = 0
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
        x_grid = int(round(((x - XRANGE[0]) / XRANGE[1]) * (xQuantize - 1), 0))
        y_grid = int(round(((y - YRANGE[0]) / YRANGE[1]) * (yQuantize - 1), 0))
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
    grid_array = np.zeros([xQuantize, yQuantize])
    grid_array[arr[:, 1], arr[:, 0]] = 255
    return grid_array


class ConvertToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # in this case there is only one channel, C = 1, thus we use expand_dims instead of transpose
        sample = np.expand_dims(sample, axis=0)

        return torch.from_numpy(sample)


class ScattererCoordinateIterableDataset(IterableDataset):
    """
        This is a dataset used to store the coordinate array of the scatterers, and the corresponding sensitivity.

        * A sample of the dataset will be a tuple ('grid', 'sensitivity')
            ** the database will be an iterable
        * Optional future feature: Out dataset will take additional arguments 'transform_sens' and 'transform_grid so
          that any required processing can be applied on the sample.
    """
    def __init__(self, csv_file, transform_sens=None, transform_grid=None):
        """
        Args:
        :param csv_file: path to the file with all the database
        :param transform_sens: transformation flag of the sensitivity
        :param transform_grid: transformation flag of the grid
        """
        self.file_path = csv_file
        self.transform_sens = transform_sens
        self.transform_grid = transform_grid

    def parse_file(self):
        with open(self.file_path, 'r') as file_obj:
            for line in file_obj:
                yield line

    def get_stream(self):
        return cycle(self.parse_file())

    def __len__(self):
        counter = 0
        with open(self.file_path, 'r') as file_obj:
            for line in file_obj:
                counter += 1
        return counter

    def __iter__(self):
        # ---------------------------------------------------------------------
        # Fetching a line from the database
        # ---------------------------------------------------------------------
        line = self.get_stream()
        # ---------------------------------------------------------------------
        # Extracting the sensitivity
        # ---------------------------------------------------------------------
        sensitivity = line[0] / NORM_FACT
        if self.transform_sens is not None:
            sensitivity = self.transform_sens(sensitivity)
        # ---------------------------------------------------------------------
        # Extracting the coordinates
        # ---------------------------------------------------------------------
        points = line[1:]
        points = np.array([points])
        points = points.astype('float').reshape(-1, 2)
        # ---------------------------------------------------------------------
        # converting the coordinates to a 2D grid
        # ---------------------------------------------------------------------
        pixel_points = micrometer2pixel(points)
        grid_array   = points2mat(pixel_points)  # places with cylinder are 255, without are zeros

        # ---------------------------------------------------------------------
        # If needed, performing the transformation
        # ---------------------------------------------------------------------
        if self.transform is not None:
            grid_array = self.transform(grid_array)

        return tuple((grid_array, sensitivity))


# ============================================================
# defining function which manipulate the classes above
# ============================================================
def import_data_sets(batch_size, test_rate):
    """
    This function imports the train and test database
    :param batch_size: size of each batch in the databases
    :param test_rate: percentage of the total dataset which will be dedicated to taining
    :return: two datasets, training and test
    """
    # --------------------------------------------------------
    # Transformations - defining useful transformations
    # --------------------------------------------------------
    transform_grid = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([TRANSFORM_NORM] * IMG_CHANNELS, [TRANSFORM_NORM] * IMG_CHANNELS)
        ])
    # --------------------------------------------------------
    # Importing complete dataset
    # --------------------------------------------------------
    data = ScattererCoordinateIterableDataset(csv_file=PATH_DATABASE,
                                              transform_sens=ConvertToTensor(),
                                              transform_grid=transform_grid)

    # --------------------------------------------------------
    # Computing the lengths
    # --------------------------------------------------------
    length    = round(len(data))
    train_len = round(length * (1 - test_rate))
    test_len  = length - train_len

    # --------------------------------------------------------
    # Splitting randomly into two sets
    # --------------------------------------------------------
    train_data, test_data = random_split(data, [train_len, test_len])

    # --------------------------------------------------------
    # Creating the loaders
    # --------------------------------------------------------
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader  = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = import_data_sets(32, 0.15)

    for batch in islice(train_loader, 8):
        print('hi')


