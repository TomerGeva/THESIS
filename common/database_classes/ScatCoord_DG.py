import os
import torch
import pandas as pd
import numpy as np
from database_functions import DatabaseFunctions
from torch.utils.data import Dataset, DataLoader


# ======================================================================================================================
# Defining the dataset class
# ======================================================================================================================
class ScatCoordDG(Dataset):
    """
    This is a dataset used to store the coordinate array of the scatterers, and the corresponding sensitivity.
    A sample of the dataset will be a dictionary {'grid': 2D array, 'sensitivity': target sensitivity}
    Out dataset will take additional argument 'transform' so that any required processing can be applied on the sample.
    """
    def __init__(self, csv_files, transform=None, abs_sens=True, double_size=False):
        """
        Args:
        :param csv_files: logdir to the file with all the database
        :param transform: transformation flag of the data
        :param abs_sens: if true, doing abs on the sensitivity
        :param double_size: if True, also transposes the data and changes sign of the sensitivity
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
        self.double_size  = double_size
        self.dbf          = DatabaseFunctions()

    def __len__(self):
        if self.double_size:
            return 2 * sum(self.csv_lens)
        else:
            return sum(self.csv_lens)

    def __getitem__(self, idx):
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
        # Creating the sample dict
        # ----------------------------------------------------------------------------------------------------------
        sample = {'coordinate_target': points,
                  'sensitivity': sensitivity}
        # ----------------------------------------------------------------------------------------------------------
        # Transforming sample if given
        # ----------------------------------------------------------------------------------------------------------
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ============================================================
# defining transform class, converting 2D arrays to tensors
# ============================================================
class ToTensorCoord(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        super(ToTensorCoord, self).__init__()

    def __call__(self, sample):
        sensitivity, points = sample['sensitivity'], sample['coordinate_target']
        sensitivity         = torch.from_numpy(np.expand_dims(sensitivity, axis=0))
        points              = torch.from_numpy(np.reshape(points, -1))
        return {'sensitivity': sensitivity,
                'coordinate_target': points}


# ======================================================================================================================
# Defining functions which manipulate the classes above
# ======================================================================================================================
def import_data_sets_coord(path_list_train, path_list_test, batch_size, abs_sens=True, num_workers=1):
    """
    This function imports the train and test database
    :param path_list_train: list of paths to be used in the train dataloader
    :param path_list_test: list of paths to be used in the test dataloader
    :param batch_size: size of each batch in the databases
    :param abs_sens: if true, doing absolute value over teh sensitivity
    :param num_workers: number of workers in the dataloader
    :return: two datasets, training and test
    """
    # --------------------------------------------------------------------------------------------------------------
    # Importing complete dataset and creating train dataloader
    # --------------------------------------------------------------------------------------------------------------
    data_train = ScatCoordDG(csv_files=path_list_train,
                             transform=None,  # ToTensorCoord(),
                             abs_sens=abs_sens)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    # --------------------------------------------------------------------------------------------------------------
    # Importing complete dataset and creating test dataloaders
    # --------------------------------------------------------------------------------------------------------------
    threshold_list  = []
    test_loaders    = {}
    for ii, dataset in enumerate(path_list_test):
        data_list = [dataset]
        temp_data = ScatCoordDG(csv_files=data_list,
                                transform=None,  # ToTensorCoord(),
                                abs_sens=abs_sens,)
        # ******************************************************************************************************
        # extracting the data-loader key from the name
        # ******************************************************************************************************
        file_name_list  = dataset.split('\\')[-1].split('_')
        if 'lt' in file_name_list:
            loader_key  = '0_to_' + file_name_list[-2]
        else:
            loader_key  = file_name_list[-2] + '_to_' + get_next_threshold(path_list_test, ii)
            threshold_list.append(eval(file_name_list[-2]))
        # ******************************************************************************************************
        # creating data-loader
        # ******************************************************************************************************
        test_loaders[loader_key] = DataLoader(temp_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # --------------------------------------------------------------------------------------------------------------
    # normalizing thresholds
    # --------------------------------------------------------------------------------------------------------------
    # threshold_list = [(ii - SENS_MEAN) / SENS_STD for ii in threshold_list] if abs_sens else [ii / SENS_STD for ii in threshold_list]

    return train_loader, test_loaders, threshold_list


# def import_data_set_test(path, batch_size, mixup_factor=0, mixup_prob=0, abs_sens=True, dilation=0, shuffle=False):
#     """
#     This function imports the train and test database
#     :param path: path to the database location
#     :param batch_size: size of each batch in the databases
#     :param mixup_factor: for the training dataset,  the mixup factor
#     :param mixup_prob: for the training dataset, probability of performing mixup with the mixup factor
#     :param abs_sens: if true, doing absolute value over teh sensitivity
#     :param dilation: amount of dilation done for the cylinder locations
#     :param shuffle: Boolean stating if we want to shuffle the database or not
#     :return: two datasets, training and test
#     """
#     # --------------------------------------------------------------------------------------------------------------
#     # Importing complete dataset and creating train dataloader
#     # --------------------------------------------------------------------------------------------------------------
#     dataset = ScatCoordDG(csv_files=path,
#                           transform=ToTensorCoord(),
#                           abs_sens=abs_sens)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
#
#     # ******************************************************************************************************
#     # extracting the data-loader key from the name
#     # ******************************************************************************************************
#     file_name_list = path[0].split('\\')[-1].split('_')
#     if 'lt' in file_name_list:
#         loader_key = '0_to_' + file_name_list[-2]
#     else:
#         loader_key = file_name_list[-2] + '_to_' + get_next_threshold(PATH_DATABASE_TEST, int(file_name_list[-2][0]))
#     return {loader_key: data_loader}


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