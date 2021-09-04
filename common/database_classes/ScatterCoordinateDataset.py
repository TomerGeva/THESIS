from ConfigVAE import *
import os
import torch
import pandas as pd
import random as rnd
from functions import micrometer2pixel, points2mat
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize


# ======================================================================================================================
# Defining the dataset class
# ======================================================================================================================
class ScattererCoordinateDataset(Dataset):
    """
    This is a dataset used to store the coordinate array of the scatterers, and the corresponding sensitivity.
    A sample of the dataset will be a dictionary {'grid': 2D array, 'sensitivity': target sensitivity}
    Out dataset will take additional argument 'transform' so that any required processing can be applied on the sample.
    """
    def __init__(self, csv_files, transform=None, case=train, mix_factor=0):
        """
        Args:
        :param csv_files: logdir to the file with all the database
        :param transform: transformation flag of the data
        :param      case: train of test database
        :param mix_factor: mixup parameter, should be between 0 and 1
        """
        dirname  = os.path.dirname(__file__)
        self.csv_data = []
        self.csv_lens = []
        self.cumsum   = []
        for csv_file in csv_files:
            filepath = os.path.join(dirname, csv_file)
            self.csv_data.append(pd.read_csv(filepath))
            self.csv_lens.append(len(self.csv_data[-1]))
            if len(self.cumsum) == 0:
                self.cumsum.append(len(self.csv_data[-1]))
            else:
                self.cumsum.append(self.cumsum[-1] + len(self.csv_data[-1]))
        self.transform    = transform
        self.case         = case
        self.mixup_factor = mix_factor

    def __len__(self):
        return sum(self.csv_lens)

    def __getitem__(self, idx, mixup=True):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # ----------------------------------------------------------------------------------------------------------
        # Extracting row from specific file
        # ----------------------------------------------------------------------------------------------------------
        counter = 0
        for file_idx in range(len(self.csv_lens)):
            if idx < counter + self.csv_lens[file_idx]:
                row_idx = idx - counter
                break
            else:
                counter += self.csv_lens[file_idx]
        del counter
        # ----------------------------------------------------------------------------------------------------------
        # Extracting the sensitivity
        # ----------------------------------------------------------------------------------------------------------
        # print('file_idx ' + str(file_idx) + ' , row_idx ' + str(row_idx))
        sensitivity = self.csv_data[file_idx].iloc[row_idx, 0]

        # ----------------------------------------------------------------------------------------------------------
        # extracting the points
        # ----------------------------------------------------------------------------------------------------------
        points = self.csv_data[file_idx].iloc[row_idx, 1:]
        points = np.array([points])
        points = points.astype('float').reshape(-1, 2)

        # ----------------------------------------------------------------------------------------------------------
        # Converting points from micro meter to pixels
        # ----------------------------------------------------------------------------------------------------------
        pixel_points = micrometer2pixel(points)

        # ----------------------------------------------------------------------------------------------------------
        # Converting the points to a 2-D array
        # ----------------------------------------------------------------------------------------------------------
        grid_array = points2mat(pixel_points)

        # ----------------------------------------------------------------------------------------------------------
        # Creating the sample dict
        # ----------------------------------------------------------------------------------------------------------
        sample = {'grid': grid_array,
                  'sensitivity': sensitivity}

        # ----------------------------------------------------------------------------------------------------------
        # Transforming sample if given
        # ----------------------------------------------------------------------------------------------------------
        if self.transform:
            sample = self.transform(sample)

        # ----------------------------------------------------------------------------------------------------------
        # Doing mixup, if indicated and possible
        # ----------------------------------------------------------------------------------------------------------
        if mixup and self.case == 'train' and len(self.csv_lens) > 1:
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Randomly deciding to do mixup or not
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            eps = rnd.random()
            if eps >= 0:  # 0.11:
                return sample
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Raffling a different group
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            groups = list(range(0, len(self.csv_lens)))
            for ii, size in enumerate(self.cumsum):
                if idx <= size:
                    groups.remove(ii)
                    break
            group = rnd.choice(groups)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Raffling an index from the group
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            index = rnd.randint(0, self.csv_lens[group] - 1)
            # print('doing mixup, chosen row index ' + str(index))
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Fetching the data of that index
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if group == 0:  # If group is 0, no need to add anything to the index
                full_idx = index
            else:  # adding the cumsum of the previous group
                full_idx = self.cumsum[group-1] + index
            mix_sample = self.__getitem__(full_idx, mixup=False)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Performing the mixup
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            sample['grid'] = sample['grid'] * self.mixup_factor + mix_sample['grid'] * (1 - self.mixup_factor)
            sample['sensitivity'] = sample['sensitivity'] * self.mixup_factor + mix_sample['sensitivity'] * (1 - self.mixup_factor)

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
            Normalize(mean=[GRID_MEAN]*IMG_CHANNELS, std=[GRID_STD]*IMG_CHANNELS)
        ])
        self.trans_sens = Compose([
            Normalize(mean=[SENS_MEAN]*IMG_CHANNELS, std=[SENS_STD]*IMG_CHANNELS)
        ])

    def __call__(self, sample):
        grid, sensitivity = sample['grid'], sample['sensitivity']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # in this case there is only one channel, C = 1, thus we use expand_dims instead of transpose
        grid        = self.trans_grids(grid)
        sensitivity = np.expand_dims(sensitivity, axis=0)
        # sensitivity = np.sign(sensitivity) * (abs(sensitivity) - SENS_MEAN) / SENS_STD
        sensitivity /= SENS_STD
        sensitivity = torch.from_numpy(np.array(sensitivity))
        return {'grid': grid,
                'sensitivity': sensitivity}


# ============================================================
# defining function which manipulate the classes above
# ============================================================
def import_data_sets(batch_size, mixup_factor):
    """
    This function imports the train and test database
    :param batch_size: size of each batch in the databases
    :param mixup_factor: for the training dataset, possibility of performing mixup, with the mixup factor
    :return: two datasets, training and test
    """
    # --------------------------------------------------------
    # Importing complete dataset and creating dataloaders
    # --------------------------------------------------------
    data_train = ScattererCoordinateDataset(csv_files=PATH_DATABASE_TRAIN, transform=ToTensorMap(), case='train', mix_factor=mixup_factor)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)

    test_loaders = {}
    for dataset in PATH_DATABASE_TEST:
        data_list = [dataset]
        temp_data = ScattererCoordinateDataset(csv_files=data_list, transform=ToTensorMap(), case='test')
        test_loaders[dataset[-17:-4]] = DataLoader(temp_data, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, test_loaders
