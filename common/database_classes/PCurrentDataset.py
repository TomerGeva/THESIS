import os
import numpy as np
import torch
import pandas as pd
from database_functions import DatabaseFunctions
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize


class PCurrentsDataset(Dataset):
    def __init__(self, csv_file, transform=None, omega_factor=1e2):
        """
        Args:
        :param csv_file: logdir to the file with all the database
        :param transform: transformation flag of the data
        """
        dirname  = os.path.dirname(__file__)
        filepath = os.path.join(dirname, csv_file)
        self.csv_data = pd.read_csv(filepath)
        # self.csv_lens.append(len(self.csv_data[-1]))
        self.transform    = transform
        self.dbf          = DatabaseFunctions()

        self.omega_factor = omega_factor

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        omega     = self.csv_data.iloc[item, 0] * self.omega_factor
        pcurrents = self.csv_data.iloc[item, 1:]

        sample = {'omega': torch.from_numpy(np.array([omega])),
                  'pcurrents': torch.from_numpy(np.array(pcurrents))}
        # sample = {'omega': omega,
        #           'pcurrents': pcurrents}
        # if self.transform:
        #     sample = self.transform(sample)

        return sample


# ======================================================================================================================
# Defining functions which manipulate the classes above
# ======================================================================================================================
def import_pcurrents_dataset(batch_size, path_train, path_test, path_valid=None, omega_Factor=1, num_workers=1):
    # --------------------------------------------------------------------------------------------------------------
    # Importing complete dataset and creating train dataloader
    # --------------------------------------------------------------------------------------------------------------
    data_train = PCurrentsDataset(path_train, omega_factor=omega_Factor)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    # --------------------------------------------------------------------------------------------------------------
    # Same for test and valid
    # --------------------------------------------------------------------------------------------------------------
    data_test   = PCurrentsDataset(path_test, omega_factor=omega_Factor)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if path_valid is not None:
        data_valid   = PCurrentsDataset(path_valid, omega_factor=omega_Factor)
        valid_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        return train_loader, valid_loader, test_loader
    return train_loader, test_loader


