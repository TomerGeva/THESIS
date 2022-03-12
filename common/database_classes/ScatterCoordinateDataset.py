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
    def __init__(self, csv_files, transform=None, case='train', mix_factor=0, mix_prob=0, abs_sens=True, dilation=0):
        """
        Args:
        :param csv_files: logdir to the file with all the database
        :param transform: transformation flag of the data
        :param      case: train or test database
        :param mix_factor: mixup parameter, should be between 0 and 1
        :param mix_prob: probability to do mixup when calling __get_item__(), should be between 0 and 1
        :param abs_sens: if true, doing abs on the sensitivity
        :param dilation: amount of dilation done for the cylinder locations
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
        self.mixup_prob   = mix_prob
        self.abs_sens     = abs_sens
        self.dilation     = dilation
        self.kernel       = create_circle_kernel(radius=self.dilation) if dilation > 0 else 0
        self.dbf          = DatabaseFunctions()

    def __len__(self):
        return 2 * sum(self.csv_lens)

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
        # training has 50% chance to flip, test checks un-flipped and flipped
        if idx >= sum(self.csv_lens):
            # print(self.case, idx)
            # print(idx, idx_new, sum(self.csv_lens))
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
                  'sensitivity': sensitivity}
        # ----------------------------------------------------------------------------------------------------------
        # Transforming sample if given
        # ----------------------------------------------------------------------------------------------------------
        if self.transform:
            sample = self.transform(sample)
        """
        # ----------------------------------------------------------------------------------------------------------
        # Doing mixup, if indicated and possible
        # ----------------------------------------------------------------------------------------------------------
        if mixup and self.case == 'train' and len(self.csv_lens) > 1 and self.mixup_prob > 0:
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Randomly deciding to do mixup or not
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            eps = rnd.random()
            if eps >= self.mixup_prob:  # 0.1:
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
        """
        # if self.case == 'test':
        #     sample['coordinate_target'] = pixel_points
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
        self.to_tensor = ToTensor()
        self.trans_sens = Compose([
            Normalize(mean=[SENS_MEAN]*IMG_CHANNELS, std=[SENS_STD]*IMG_CHANNELS)
        ])

    def __call__(self, sample):
        grid, sensitivity = sample['grid'], sample['sensitivity']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # in this case there is only one channel, C = 1, thus we use expand_dims instead of transpose
        grid_normalized = self.trans_grids(grid)
        sensitivity = np.expand_dims(sensitivity, axis=0)
        sensitivity = (abs(sensitivity) - SENS_MEAN) / SENS_STD if ABS_SENS else sensitivity / SENS_STD
        sensitivity = torch.from_numpy(np.array(sensitivity))
        return {'grid_target': self.to_tensor(grid),
                'grid_in': grid_normalized,
                'sensitivity': sensitivity}


# ======================================================================================================================
# defining functions which manipulate the classes above
# ======================================================================================================================
def import_data_sets(batch_size, mixup_factor=0, mixup_prob=0, abs_sens=True, dilation=0):
    """
    This function imports the train and test database
    :param batch_size: size of each batch in the databases
    :param mixup_factor: for the training dataset,  the mixup factor
    :param mixup_prob: for the training dataset, probability of performing mixup with the mixup factor
    :param abs_sens: if true, doing absolute value over teh sensitivity
    :param dilation: amount of dilation done for the cylinder locations
    :return: two datasets, training and test
    """
    # --------------------------------------------------------------------------------------------------------------
    # Importing complete dataset and creating train dataloader
    # --------------------------------------------------------------------------------------------------------------
    data_train = ScattererCoordinateDataset(csv_files=PATH_DATABASE_TRAIN,
                                            transform=ToTensorMap(),
                                            case='train',
                                            mix_factor=mixup_factor,
                                            mix_prob=mixup_prob,
                                            abs_sens=abs_sens,
                                            dilation=dilation,)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # --------------------------------------------------------------------------------------------------------------
    # Importing complete dataset and creating test dataloaders
    # --------------------------------------------------------------------------------------------------------------
    threshold_list  = []
    test_loaders    = {}
    for ii, dataset in enumerate(PATH_DATABASE_TEST):
        data_list = [dataset]
        temp_data = ScattererCoordinateDataset(csv_files=data_list,
                                               transform=ToTensorMap(),
                                               case='test',
                                               abs_sens=abs_sens,
                                               dilation=dilation)
        # ******************************************************************************************************
        # extracting the data-loader key from the name
        # ******************************************************************************************************
        file_name_list  = dataset.split('\\')[-1].split('_')
        if 'lt' in file_name_list:
            loader_key  = '0_to_' + file_name_list[-2]
        else:
            loader_key  = file_name_list[-2] + '_to_' + get_next_threshold(PATH_DATABASE_TEST, ii)
            threshold_list.append(eval(file_name_list[-2]))
        # ******************************************************************************************************
        # creating data-loader
        # ******************************************************************************************************
        test_loaders[loader_key] = DataLoader(temp_data, batch_size=batch_size, shuffle=False,
                                              num_workers=NUM_WORKERS, pin_memory=True)

    # --------------------------------------------------------------------------------------------------------------
    # normalizing thresholds
    # --------------------------------------------------------------------------------------------------------------
    threshold_list = [(ii - SENS_MEAN) / SENS_STD for ii in threshold_list] if abs_sens else [ii / SENS_STD for ii in threshold_list]

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
                                         case='test',
                                         mix_factor=mixup_factor,
                                         mix_prob=mixup_prob,
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
