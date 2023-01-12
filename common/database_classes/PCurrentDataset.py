import os
import numpy as np
import torch
import pandas as pd
from database_functions import DatabaseFunctions
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize


class PCurrentsDataset(Dataset):
    def __init__(self, csv_file, transform=None, omega_factor=1e2, shot_noise=False, sampling_rate=1):
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

        self.omega_factor     = omega_factor
        self.shot_noise       = shot_noise
        self.sampling_rate    = sampling_rate  # in Hz
        self.elec_amp_per_sec = 6.24e18
        self.curr_per_part    = 1e12 * 1 / self.elec_amp_per_sec * self.sampling_rate  # in [pico ampere / Hz]

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        omega     = self.csv_data.iloc[item, 0] * self.omega_factor
        pcurrents = self.csv_data.iloc[item, 1:]  # that is in pico ampere
        # -------------------------------------------------------------------------------
        # Adding shot noise if needed
        # -------------------------------------------------------------------------------
        if self.shot_noise:
            # transforming to complex numbers to extract the phases and abs value
            temp        = np.array(pcurrents).reshape((-1, 2))
            pcurr_comp  = temp[:, 0] + 1j * temp[:, 1]
            pcurr_abs   = np.abs(pcurr_comp)    # in pico ampere
            pcurr_phase = np.angle(pcurr_comp)  # in radians
            # Generating noised currents in abs value - assuming same phase (no phase noise atm)
            n_of_particles          = self.elec_amp_per_sec * pcurr_abs * 1e-12 / self.sampling_rate
            n_of_particles_snoised  = torch.poisson(torch.tensor(n_of_particles)).numpy()
            pcurr_abs_snoised       = n_of_particles_snoised * self.curr_per_part
            # Converting back to vector of real values double length for the net
            pcurr_comp_snoised = pcurr_abs_snoised * np.exp(1j * pcurr_phase)
            temp               = np.array([np.real(pcurr_comp_snoised), np.imag(pcurr_comp_snoised)])
            pcurrents          = np.reshape(temp.T, -1)

        sample = {'omega': torch.from_numpy(np.array([omega])),
                  'pcurrents': torch.from_numpy(np.array(pcurrents))}
        # sample = {'omega': omega,
        #           'pcurrents': pcurrents}
        # if self.transform:
        #     sample = self.transform(sample)

        return sample


class PCurrentOmegaDataset(Dataset):
    def __init__(self, csv_file, omega, length, transform=None, omega_factor=1e2, shot_noise=False, sampling_rate=1, is_complex=False):
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

        self.omega_factor     = omega_factor
        self.shot_noise       = shot_noise
        self.sampling_rate    = sampling_rate  # in Hz
        self.elec_amp_per_sec = 6.24e18
        self.curr_per_part    = 1e12 * 1 / self.elec_amp_per_sec * self.sampling_rate  # in [pico ampere / Hz]
        self.is_complex       = is_complex

        self.omega  = omega
        self.length = length
        # Searching for the wanted omega
        for ii in range(0, len(self.csv_data)):
            omega_cand = self.csv_data.iloc[ii, 0]
            if np.isclose(omega, omega_cand):
                self.omega_idx = ii
                return
        raise ValueError('Omega is not in the database!')

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        omega     = self.csv_data.iloc[self.omega_idx, 0] * self.omega_factor
        pcurrents = self.csv_data.iloc[self.omega_idx, 1:]  # that is in pico ampere
        if not self.is_complex:
            pcurrents = np.array(pcurrents).reshape((-1, 2))
            pcurrents = np.array(pcurrents[:, 0] ** 2 + pcurrents[:, 1] ** 2)
        # -------------------------------------------------------------------------------
        # Adding shot noise if needed
        # -------------------------------------------------------------------------------
        if self.shot_noise:
            if self.is_complex:
                # transforming to complex numbers to extract the phases and abs value
                temp        = np.array(pcurrents).reshape((-1, 2))
                pcurr_comp  = temp[:, 0] + 1j * temp[:, 1]
                pcurr_abs   = np.abs(pcurr_comp)    # in pico ampere
                pcurr_phase = np.angle(pcurr_comp)  # in radians
                # Generating noised currents in abs value - assuming same phase (no phase noise atm)
                n_of_particles          = self.elec_amp_per_sec * pcurr_abs * 1e-12 / self.sampling_rate
                n_of_particles_snoised  = torch.poisson(torch.tensor(n_of_particles)).numpy()
                pcurr_abs_snoised       = n_of_particles_snoised * self.curr_per_part
                # Converting back to vector of real values double length for the net
                pcurr_comp_snoised = pcurr_abs_snoised * np.exp(1j * pcurr_phase)
                temp               = np.array([np.real(pcurr_comp_snoised), np.imag(pcurr_comp_snoised)])
                pcurrents          = np.reshape(temp.T, -1)
            else:
                # Generating noised currents in abs value
                n_of_particles         = self.elec_amp_per_sec * pcurrents * 1e-12 / self.sampling_rate
                n_of_particles_snoised = torch.poisson(torch.tensor(n_of_particles)).numpy()
                pcurrents              = n_of_particles_snoised * self.curr_per_part

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
def import_pcurrents_dataset(batch_size, path_train, path_test=None, path_valid=None, omega_factor=1, shot_noise=False, sampling_rate=1, shuffle=True, num_workers=1):
    # --------------------------------------------------------------------------------------------------------------
    # Importing complete dataset and creating train dataloader
    # --------------------------------------------------------------------------------------------------------------
    data_train = PCurrentsDataset(path_train, omega_factor=omega_factor, shot_noise=shot_noise, sampling_rate=sampling_rate)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    # --------------------------------------------------------------------------------------------------------------
    # Same for test and valid
    # --------------------------------------------------------------------------------------------------------------
    if path_test is not None:
        data_test   = PCurrentsDataset(path_test, omega_factor=omega_factor, shot_noise=shot_noise, sampling_rate=sampling_rate)
        test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        if path_valid is not None:
            data_valid   = PCurrentsDataset(path_valid, omega_factor=omega_factor, shot_noise=shot_noise, sampling_rate=sampling_rate)
            valid_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
            return train_loader, valid_loader, test_loader
        return train_loader, test_loader
    return train_loader


def import_pcurrent_omega_dataset(batch_size, path, omega, length,  omega_factor=1, shot_noise=False, sampling_rate=1, num_workers=1):
    # --------------------------------------------------------------------------------------------------------------
    # Importing complete dataset and creating train dataloader
    # --------------------------------------------------------------------------------------------------------------
    data_train   = PCurrentOmegaDataset(path, omega, length, omega_factor=omega_factor, shot_noise=shot_noise, sampling_rate=sampling_rate)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader
