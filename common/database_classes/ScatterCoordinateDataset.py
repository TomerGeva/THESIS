from ConfigVAE import *
import os
import torch
import pandas as pd
import random as rnd
import cv2
from auxiliary_functions import create_circle_kernel
from database_functions import DatabaseFunctions
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
    def __init__(self, csv_files, transform=None, abs_sens=True, dilation=0):
        """
        Args:
        :param csv_files: logdir to the file with all the database
        :param transform: transformation flag of the data
        :param abs_sens: if true, doing abs on the sensitivity
        :param dilation: amount of dilation done for the cylinder locations
        :param norm_sens: (mean, std) with which to normalize the sensitivity
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
        self.abs_sens     = abs_sens
        self.dilation     = dilation
        self.kernel       = create_circle_kernel(radius=self.dilation) if dilation > 0 else 0
        self.dbf          = DatabaseFunctions()

    def __len__(self):
        # return 2 * sum(self.csv_lens)
        return sum(self.csv_lens)

    def __getitem__(self, idx, mixup=True):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx_new = idx - sum(self.csv_lens) if idx >= sum(self.csv_lens) else idx
        # ----------------------------------------------------------------------------------------------------------
        # Extracting row from specific file
        # ----------------------------------------------------------------------------------------------------------
        counter = 0
        for file_idx in range(len(self.csv_lens)):
            if idx_new < counter + self.csv_lens[file_idx]:
                row_idx = idx_new - counter
                break
            else:
                counter += self.csv_lens[file_idx]
        del counter
        # ----------------------------------------------------------------------------------------------------------
        # Extracting the sensitivity
        # ----------------------------------------------------------------------------------------------------------
        # print('file_idx ' + str(file_idx) + ' , row_idx ' + str(row_idx))
        sensitivity = abs(self.csv_data[file_idx].iloc[row_idx, 0]) if self.abs_sens else self.csv_data[file_idx].iloc[row_idx, 0]

        # ----------------------------------------------------------------------------------------------------------
        # extracting the points
        # ----------------------------------------------------------------------------------------------------------
        points = self.csv_data[file_idx].iloc[row_idx, 1:]
        points = np.array([points])
        points = points.astype('float').reshape(-1, 2)

        # ----------------------------------------------------------------------------------------------------------
        # Converting points from micro meter to pixels
        # ----------------------------------------------------------------------------------------------------------
        pixel_points = self.dbf.micrometer2pixel(points)

        # ----------------------------------------------------------------------------------------------------------
        # Converting the points to a 2-D array
        # ----------------------------------------------------------------------------------------------------------
        if idx >= sum(self.csv_lens):
            pixel_points = np.fliplr(pixel_points)
        grid_array = self.dbf.points2mat(pixel_points)
        # ----------------------------------------------------------------------------------------------------------
        # Dilating the cylinder locations
        # ----------------------------------------------------------------------------------------------------------
        if self.dilation > 0:
            grid_array = cv2.dilate(grid_array.astype(np.uint8), self.kernel, iterations=1)
        # ----------------------------------------------------------------------------------------------------------
        # Creating the sample dict
        # ----------------------------------------------------------------------------------------------------------
        sample = {'grid': grid_array,
                  'coordinate_target': points,
                  'sensitivity': sensitivity}
        # ----------------------------------------------------------------------------------------------------------
        # Transforming sample if given
        # ----------------------------------------------------------------------------------------------------------
        if self.transform:
            sample = self.transform(sample)

        return sample


# ============================================================
# defining transform class, converting 2D arrays to tensors
# ============================================================
class ToTensorMap(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, norm_sens=(0, 1), norm_grid=(0, 1), grid_channels=1):
        super(ToTensorMap, self).__init__()
        self.norm_sens_mean = norm_sens[0]
        self.norm_sens_std  = norm_sens[1]
        self.norm_grid_mean = norm_grid[0]
        self.norm_grid_std  = norm_grid[1]
        self.trans_grids = Compose([
            ToTensor(),
            Normalize(mean=[self.norm_grid_mean]*grid_channels, std=[self.norm_grid_st]*grid_channels)
        ])
        self.to_tensor = ToTensor()

    def __call__(self, sample):
        grid, sensitivity, pixel_points = sample['grid'], sample['sensitivity'], sample['coordinate_target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # in this case there is only one channel, C = 1, thus we use expand_dims instead of transpose
        grid_normalized = self.trans_grids(grid)
        sensitivity  = np.expand_dims(sensitivity, axis=0)
        if self.norm_sens_mean != 0 or self.norm_sens_std != 1:
            sensitivity  = (abs(sensitivity) - SENS_MEAN) / SENS_STD if ABS_SENS else sensitivity / SENS_STD
        sensitivity  = torch.from_numpy(np.array(sensitivity))
        pixel_points = torch.from_numpy(np.reshape(pixel_points, -1))
        return {'grid_target': self.to_tensor(grid),
                'grid_in': grid_normalized,
                'sensitivity': sensitivity,
                'coordinate_target': pixel_points}


# ======================================================================================================================
# Defining functions which manipulate the classes above
# ======================================================================================================================
def import_data_sets_pics(paths_train, paths_test, batch_size, abs_sens=True, dilation=0, norm_sens=(0, 1), norm_grid=(0, 1), num_workers=1):
    """
    This function imports the train and test database
    :param paths_train:
    :param paths_test:
    :param batch_size: size of each batch in the databases
    :param abs_sens: if true, doing absolute value over teh sensitivity
    :param dilation: amount of dilation done for the cylinder locations
    :param norm_sens: (mean, std) with which to normalize the sensitivity
    :param norm_grid:
    :param num_workers:
    :return: two datasets, training and test
    """
    # --------------------------------------------------------------------------------------------------------------
    # Importing complete dataset and creating train dataloader
    # --------------------------------------------------------------------------------------------------------------
    data_train = ScattererCoordinateDataset(csv_files=paths_train,
                                            transform=ToTensorMap(norm_sens=norm_sens, norm_grid=norm_grid),
                                            abs_sens=abs_sens,
                                            dilation=dilation,)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # --------------------------------------------------------------------------------------------------------------
    # Importing complete dataset and creating test dataloaders
    # --------------------------------------------------------------------------------------------------------------
    threshold_list  = []
    test_loaders    = {}
    for ii, dataset in enumerate(paths_test):
        data_list = [dataset]
        temp_data = ScattererCoordinateDataset(csv_files=data_list,
                                               transform=ToTensorMap(norm_sens=norm_sens, norm_grid=norm_grid),
                                               abs_sens=abs_sens,
                                               dilation=dilation)
        # ******************************************************************************************************
        # extracting the data-loader key from the name
        # ******************************************************************************************************
        file_name_list  = dataset.split('\\')[-1].split('_')
        if 'lt' in file_name_list:
            loader_key  = '0_to_' + file_name_list[-2]
        else:
            loader_key  = file_name_list[-2] + '_to_' + get_next_threshold(paths_test, ii)
            threshold_list.append(eval(file_name_list[-2]))
        # ******************************************************************************************************
        # creating data-loader
        # ******************************************************************************************************
        test_loaders[loader_key] = DataLoader(temp_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # --------------------------------------------------------------------------------------------------------------
    # normalizing thresholds
    # --------------------------------------------------------------------------------------------------------------
    if norm_sens[0] != 0 or norm_sens[1] != 1:
        threshold_list = [(ii - norm_sens[0]) / norm_sens[1] for ii in threshold_list] if abs_sens else [ii / norm_sens[1] for ii in threshold_list]

    return train_loader, test_loaders, threshold_list


def import_data_set_test(path, batch_size, mixup_factor=0, mixup_prob=0, abs_sens=True, dilation=0, shuffle=False):
    """
    This function imports the train and test database
    :param path: path to the database location
    :param batch_size: size of each batch in the databases
    :param mixup_factor: for the training dataset,  the mixup factor
    :param mixup_prob: for the training dataset, probability of performing mixup with the mixup factor
    :param abs_sens: if true, doing absolute value over teh sensitivity
    :param dilation: amount of dilation done for the cylinder locations
    :param shuffle: Boolean stating if we want to shuffle the database or not
    :return: two datasets, training and test
    """
    # --------------------------------------------------------------------------------------------------------------
    # Importing complete dataset and creating train dataloader
    # --------------------------------------------------------------------------------------------------------------
    dataset = ScattererCoordinateDataset(csv_files=path,
                                         transform=ToTensorMap(),
                                         abs_sens=abs_sens,
                                         dilation=dilation)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    # ******************************************************************************************************
    # extracting the data-loader key from the name
    # ******************************************************************************************************
    file_name_list = path[0].split('\\')[-1].split('_')
    if 'lt' in file_name_list:
        loader_key = '0_to_' + file_name_list[-2]
    else:
        loader_key = file_name_list[-2] + '_to_' + get_next_threshold(PATH_DATABASE_TEST, int(file_name_list[-2][0]))
    return {loader_key: data_loader}


# ======================================================================================================================
# axillary unctions
# ======================================================================================================================
def get_next_threshold(name_list, ii):
    """
    :param name_list: list of database names
    :param ii: locaation of current database
    :return: getting the next threshold from the list
    """
    if ii == len(name_list) - 1:  # last name in the list
        return 'inf'
    return name_list[ii+1].split('\\')[-1].split('_')[-2]
