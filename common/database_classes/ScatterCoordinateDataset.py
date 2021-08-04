from ConfigVAE import *
import os
import torch
import pandas as pd
from functions import micrometer2pixel, points2mat
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize


# ============================================================
# defining the dataset class
# ============================================================
class ScattererCoordinateDataset(Dataset):
    """
    This is a dataset used to store the coordinate array of the scatterers, and the corresponding sensitivity.
    A sample of the dataset will be a dictionary {'grid': 2D array, 'sensitivity': target sensitivity}
    Out dataset will take additional argument 'transform' so that any required processing can be applied on the sample.
    """
    def __init__(self, csv_file, transform=None):
        """
        Args:
        :param csv_file: logdir to the file with all the database
        :param transform: transformation flag of the data
        """
        import os
        dirname  = os.path.dirname(__file__)
        filepath = os.path.join(dirname, csv_file)
        self.csv_data = pd.read_csv(filepath)
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # ---------------------------------------------
        # extracting the sensitivity
        # ---------------------------------------------
        sensitivity = self.csv_data.iloc[idx, 0] / NORM_FACT

        # ---------------------------------------------
        # extracting the points
        # ---------------------------------------------
        points = self.csv_data.iloc[idx, 1:]
        points = np.array([points])
        points = points.astype('float').reshape(-1, 2)

        # ---------------------------------------------
        # converting points from micro meter to pixels
        # ---------------------------------------------
        pixel_points = micrometer2pixel(points)

        # ---------------------------------------------
        # converting the points to a 2-D array
        # ---------------------------------------------
        grid_array = points2mat(pixel_points)

        # -------------------------------------------
        # creating the sample dict
        # -------------------------------------------
        sample = {'grid': grid_array,
                  'sensitivity': sensitivity}

        # -------------------------------------------
        # transforming sample if given
        # -------------------------------------------
        if self.transform:
            sample = self.transform(sample)

        return sample


# ============================================================
# defining transform class, converting 2D arrays to tensors
# ============================================================
class ToTensorMap(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        super(ToTensorMap, self).__init__()
        self.trans_grids = Compose([
            ToTensor(),
            Normalize(mean=[TRANSFORM_MEAN]*IMG_CHANNELS, std=[TRANSFORM_STD]*IMG_CHANNELS)
        ])
        self.trans_sens = ToTensor()

    def __call__(self, sample):
        grid, sensitivity = sample['grid'], sample['sensitivity']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # in this case there is only one channel, C = 1, thus we use expand_dims instead of transpose
        grid        = self.trans_grids(grid)
        sensitivity = np.expand_dims(sensitivity, axis=0)
        return {'grid': grid,
                'sensitivity': torch.from_numpy(np.array(sensitivity))}


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
    # Importing complete dataset
    # --------------------------------------------------------
    data = ScattererCoordinateDataset(csv_file=PATH_DATABASE, transform=ToTensorMap())

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
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
