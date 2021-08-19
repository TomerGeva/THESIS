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
    def __init__(self, csv_files, transform=None):
        """
        Args:
        :param csv_files: logdir to the file with all the database
        :param transform: transformation flag of the data
        """
        import os
        dirname  = os.path.dirname(__file__)
        self.csv_data = []
        self.csv_lens = []
        for csv_file in csv_files:
            filepath = os.path.join(dirname, csv_file)
            self.csv_data.append(pd.read_csv(filepath))
            self.csv_lens.append(len(self.csv_data[-1]))
        self.transform = transform

    def __len__(self):
        return sum(self.csv_lens)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # ---------------------------------------------
        # Extracting row from specific file
        # ---------------------------------------------
        counter = 0
        for file_idx in range(len(self.csv_lens)):
            if idx < counter + self.csv_lens[file_idx]:
                row_idx = idx - counter
                break
            else:
                counter += self.csv_lens[file_idx]

        # ---------------------------------------------
        # extracting the sensitivity
        # ---------------------------------------------
        sensitivity = self.csv_data[file_idx].iloc[row_idx, 0] / NORM_FACT

        # ---------------------------------------------
        # extracting the points
        # ---------------------------------------------
        points = self.csv_data[file_idx].iloc[row_idx, 1:]
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
    # Importing complete dataset and creating dataloaders
    # --------------------------------------------------------
    data_train = ScattererCoordinateDataset(csv_files=PATH_DATABASE_TRAIN, transform=ToTensorMap())
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

    test_loaders = {}
    for dataset in PATH_DATABASE_TEST:
        data_list = [dataset]
        temp_data = ScattererCoordinateDataset(csv_files=data_list, transform=ToTensorMap())
        test_loaders[dataset[-17:-4]] = DataLoader(temp_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loaders
