from Config import *
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
import pandas   as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import IterableDataset, DataLoader, random_split
from functions import *


# ==============================================================
# Defining the database class, dealing with all the data
# ==============================================================
class ScattererCoordinateDataset(IterableDataset):
    """
        This is a dataset used to store the coordinate array of the scatterers, and the corresponding sensitivity.
        * A sample of the dataset will be a tuple ('2D-grid', 'sensitivity')
            ** the database will be an iterable
        * Optional future feature: Out dataset will take additional argument 'transform' so that any required processing
          can be applied on the sample.
    """
    def __init__(self, csv_file, transform=None):
        """
        Args:
        :param csv_file: path to the file with all the database
        :param transform: transformation flag of the data
        """
        self.transform = transform
        # --------------------------------------------------
        # Reading the database csv file
        # --------------------------------------------------
        csv_data = pd.read_csv(csv_file)
        # --------------------------------------------------
        # Extracting the sensitivities
        # --------------------------------------------------
        sensitivities      = csv_data.iloc[:, 0] / NORM_FACT
        self.sensitivities = sensitivities.values.tolist()
        # --------------------------------------------------
        # Extracting the coordinations as pixel locations
        # --------------------------------------------------
        self.grids_array = []
        for ii in range(len(csv_data)):
            points = csv_data.iloc[ii, 1:]
            points = np.array([points])
            points = points.astype('float').reshape(-1, 2)
            # ++++++++++++++++++++++++++++++++++++++++++
            # Converting from micrometer to pixel
            # ++++++++++++++++++++++++++++++++++++++++++
            pixel_points = micrometer2pixel(points)
            # ++++++++++++++++++++++++++++++++++++++++++
            # Converting from pixel to actual grid
            # ++++++++++++++++++++++++++++++++++++++++++
            grid_array = points2mat(pixel_points)
            self.grids_array.append(grid_array)
        # --------------------------------------------------
        # Expanding dimensions as a pre-requisite of pytorch
        # --------------------------------------------------
        self.grids_array = [np.expand_dims(grid, axis=0) for grid in self.grids_array]

    def __len__(self):
        return len(self.sensitivities)

    def __iter__(self):
        return iter(tuple(zip(self.grids_array, self.sensitivities)))



if __name__ == '__main__':
    database = ScattererCoordinateDataset(csv_file=path_database)
    print('hi')
