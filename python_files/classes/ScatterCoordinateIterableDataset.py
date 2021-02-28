from Config import *
import numpy as np
import torch
import torchvision.transforms as transforms
from itertools import cycle, islice
from torch.utils.data import IterableDataset, DataLoader, random_split
from functions import micrometer2pixel, points2mat


# ============================================================
# The dataset class
# ============================================================
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

    def __len__(self):
        counter = 0
        with open(self.file_path, 'r') as file_obj:
            for line in file_obj:
                counter += 1
        return counter


# ============================================================
# Auxiliary class used to convert samples to tensors
# ============================================================
class ConvertToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # in this case there is only one channel, C = 1, thus we use expand_dims instead of transpose
        sample = np.expand_dims(sample, axis=0)

        return torch.from_numpy(sample)


# ============================================================
# defining function which manipulate the classes above
# THIS FUNCTION CURRENTLY DOES NOT WORK!!!!!!!!!!!!!!!
# DO NOT USE THIS ITERABLE, USE MAP DATASET INSTEAD
# ============================================================
def import_data_sets_iterable(batch_size, test_rate):
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
    train_loader, test_loader = import_data_sets_iterable(32, 0.15)

    for batch in islice(train_loader, 8):
        print('hi')
