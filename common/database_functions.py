# ***************************************************************************************************
# THIS FILE HOLDS THE FUNCTIONS NEEDED TO MANIPULATE THE DATABASE ON WHICH THE NETWORK TRAINS
# ***************************************************************************************************
from ConfigVAE import *
import os
import csv
import torch
import torch.nn as nn
import numpy as np
from ModVAE import ModVAE
from TrainerVAE import TrainerVAE
from DecoderVAE import DecoderVAE


class ModelManipulationFunctions:
    def __init__(self):
        pass

    @staticmethod
    def load_state_train(data_path, device=None, thresholds=None):
        """
        :param data_path: path to the saved data regarding the network
        :param device: allocation to either cpu of cuda:0
        :param thresholds: test group thresholds
        :return: the function loads the data into and returns the saves network and trainer
        """
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # -------------------------------------
        # loading the dictionary
        # -------------------------------------
        checkpoint = torch.load(data_path, map_location=device)

        # -------------------------------------
        # arranging the data
        # -------------------------------------
        encoder_topology = checkpoint['encoder_topology']
        decoder_topology = checkpoint['decoder_topology']
        latent_dim = checkpoint['latent_dim']
        encoder_type = checkpoint['encoder_type']
        try:
            mode = checkpoint['mode']
            model_out = checkpoint['model_out']
        except:
            mode = mode_e.AUTOENCODER
            model_out = model_output_e.SENS

        mod_vae = ModVAE(device=device,
                         encoder_topology=encoder_topology,
                         decoder_topology=decoder_topology,
                         latent_space_dim=latent_dim,
                         encoder_type=encoder_type,
                         mode=mode,
                         model_out=model_out)
        mod_vae.to(device)  # allocating the computation to the CPU or GPU
        mod_vae.load_state_dict(checkpoint['vae_state_dict'])

        trainer = TrainerVAE(mod_vae,
                             lr=checkpoint['lr'],
                             mom=MOM,
                             beta_dkl=checkpoint['beta_dkl'],
                             beta_grid=checkpoint['beta_grid'],
                             sched_step=SCHEDULER_STEP,
                             sched_gamma=SCHEDULER_GAMMA,
                             grad_clip=GRAD_CLIP,
                             group_thresholds=thresholds,
                             group_weights=MSE_GROUP_WEIGHT,
                             abs_sens=ABS_SENS)
        trainer.epoch = checkpoint['epoch']
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return mod_vae, trainer

    @staticmethod
    def load_decoder(data_path=None, device=None):
        """
        :param data_path: path to the saved data regarding the network
        :param device: allocation to either cpu of cuda:0
        :return: the function returns the trained decoder
        """
        pff = PathFindingFunctions()
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if data_path is None:
            results_dir = os.path.join(os.path.abspath(os.getcwd()), '..\\results')
            data_path = pff.get_latest_model(pff.get_latest_dir(results_dir))

        # --------------------------------------------------------------------------------------------------------------
        # loading the dictionary
        # --------------------------------------------------------------------------------------------------------------
        checkpoint = torch.load(data_path, map_location=device)
        # --------------------------------------------------------------------------------------------------------------
        # arranging the data
        # --------------------------------------------------------------------------------------------------------------
        encoder_topology = checkpoint['encoder_topology']
        decoder_topology = checkpoint['decoder_topology']
        latent_dim = checkpoint['latent_dim']
        encoder_type = checkpoint['encoder_type']
        try:
            mode = checkpoint['mode']
            model_out = checkpoint['model_out']
        except:
            mode = mode_e.AUTOENCODER
            model_out = model_output_e.SENS

        mod_vae = ModVAE(device=device,
                         encoder_topology=encoder_topology,
                         decoder_topology=decoder_topology,
                         latent_space_dim=latent_dim,
                         encoder_type=encoder_type)
        mod_vae.load_state_dict(checkpoint['vae_state_dict'])
        # --------------------------------------------------------------------------------------------------------------
        # Extracting the decoder
        # --------------------------------------------------------------------------------------------------------------
        decoder = DecoderVAE(device=device, topology=decoder_topology, latent_dim=latent_dim, model_out=model_out)
        decoder.load_state_dict(mod_vae.decoder.state_dict())
        decoder.to(decoder.device)

        return decoder, latent_dim

    @staticmethod
    def initialize_weights(net, mean, std):
        """
        :param net: the model which is being normalized
        :param mean: the target mean of the weights
        :param std: the target standard deviation of the weights
        :return: nothing, just adjusts the weights
        """
        for module in net.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight.data, mean, std)
                if isinstance(module, nn.Linear):
                    pass

    @staticmethod
    def copy_net_weights(source_net, target_net):
        """
        :param source_net: We copy the weights from this network
        :param target_net: We copy the weights to this network
        :return: Nothing
        """
        for module_source, module_target in zip(source_net.encoder.modules(), target_net.encoder.modules()):
            if isinstance(module_source, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.BatchNorm1d, nn.ConvTranspose2d)) and isinstance(module_target, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.BatchNorm1d, nn.ConvTranspose2d)):
                if type(module_source) == type(module_target):
                    if module_source.weight.shape == module_target.weight.shape:
                        module_target.weight.data = module_source.weight.data
                        module_target.bias        = module_source.bias
                        if isinstance(module_source, (nn.BatchNorm2d, nn.BatchNorm1d)):
                            module_target.running_mean = module_source.running_mean
                            module_target.running_var = module_source.running_var

        for module_source, module_target in zip(source_net.decoder.modules(), target_net.decoder.modules()):
            if isinstance(module_source, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.BatchNorm1d, nn.ConvTranspose2d)) and isinstance(module_target, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.BatchNorm1d, nn.ConvTranspose2d)):
                if type(module_source) == type(module_target):
                    if module_source.weight.shape == module_target.weight.shape:
                        module_target.weight.data = module_source.weight.data
                        module_target.bias        = module_source.bias
                        if isinstance(module_source, (nn.BatchNorm2d, nn.BatchNorm1d)):
                            module_target.running_mean = module_source.running_mean
                            module_target.running_var = module_source.running_var

    @staticmethod
    def slice_grid(grid, threshold):
        """
        :param grid: 2D np array of a raw model output
        :param threshold: threshold for the slicing
        :return: 2D numpy array of the sliced array
        """
        return (grid > threshold).astype(float)

    @staticmethod
    def get_nof_params(model):
        """Return the number of trainable model parameters.
        Args:
            model: nn.Module.
        Returns:
            The number of model parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DatabaseFunctions:
    def __init__(self):
        pass

    @staticmethod
    def micrometer2pixel(arr):
        """
        This function is used to convert the coordinates from micro meter to pixel values
        :param arr: (N,2) array holding the coordinates in microns
        :return: array sized (N, 2) with the coordinates in pixel values
        """
        grid_coords = [np.zeros([2, ]).astype(int)] * len(arr)
        for ii in range(len(arr)):
            x = float(arr[ii, 0])
            y = float(arr[ii, 1])
            x_grid = int(round(((x - XRANGE[0]) / XRANGE[1]) * (XQUANTIZE - 1), 0))
            y_grid = int(round(((y - YRANGE[0]) / YRANGE[1]) * (YQUANTIZE - 1), 0))
            grid_coords[ii] = np.array([x_grid, y_grid])

        return np.array(grid_coords)

    @staticmethod
    def points2mat(arr):
        """
        THIS FILE HOLDS THE FUNCTION WHICH TAKES AN ARRAY OF POINTS AND CONVERTS IT TO A MATRIX, WHERE:
        FOR EACH (X,Y) OF THE MATRIX:
            IF (X,Y) IS IN THE ARRAY, THE INDEX WILL BE 1
            OTHERWISE, IT WILL BE 0
        :param: arr: a 2-D array which holds the coordinates of the scatterers
        :return: xQuantize X yQuantize grid simulating the array
        """
        grid_array = np.zeros([XQUANTIZE, YQUANTIZE])
        grid_array[arr[:, 1], arr[:, 0]] = 255
        return grid_array.astype(np.uint8)

    @staticmethod
    def check_array_validity(scat_locations, x_rate, y_rate, dmin):
        """
        :param scat_locations: 2D array containing the scatterer centers such that each row shows as follows:
                               [x_coordinate, y_coordinate, scale]
                               The size of the array is NX3
        :param x_rate: conversion rate pixel to micrometer in the x direction
        :param y_rate: conversion rate pixel to micrometer in the y direction
        :param dmin: minimal allowed distance in micro-meters
        :return: The function checks the distance between each scatterer and the other scatterers, computes the distance
                 to the closest cylinder and checks that it is above the threshold. if the closest cylinder is below the
                 threshold,we discard one of the cylinders based on their scale (bigger is better)
        """
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        valid_array = np.zeros_like(scat_locations)
        indexes     = np.array(list(range(scat_locations.shape[0])))
        rates       = np.array([x_rate, y_rate])
        counter     = 0
        # ==============================================================================================================
        # for each coordinate running the following
        # ==============================================================================================================
        for ii, coordinate in enumerate(scat_locations):
            # ------------------------------------------------------------------------------------------------------
            # Computing distance to all other cylinders
            # ------------------------------------------------------------------------------------------------------
            loc         = coordinate[0:2]
            diffs       = np.sqrt(np.sum(np.power((scat_locations[:, 0:2] - loc) * rates, 2), axis=1))
            diffs       = np.delete(diffs, ii)
            # ------------------------------------------------------------------------------------------------------
            # if min distance is bigger than threshold, add coordinate, else checking for the largest scale
            # ------------------------------------------------------------------------------------------------------
            if np.min(diffs) >= dmin:
                valid_array[counter] = coordinate
                counter += 1
            else:
                candidates = scat_locations[np.delete(indexes, ii)]
                candidates = candidates[diffs < dmin]
                if coordinate[2] > np.max(candidates[:, 2]):
                    valid_array[counter] = coordinate
                    counter += 1
        return valid_array[:counter]

    @staticmethod
    def find_differences(inputs, target, x_rate, y_rate, dmin):
        """
        :param inputs: N X 2 array of coordinates
        :param target: M X 2 array of coordinates
        :param x_rate: conversion ratio pixel/micrometer in the x dimension
        :param y_rate: conversion ratio pixel/micrometer in the y dimension
        :param dmin: minimal distance between scatterers
        :return: function returns two arrays:
                    1. input_unique - K x 2 array with coordinates unique to input
                    2. target_unique - Q X 2 array with coordinates unique to target
                Decision rule is as follows:
                1. Iterating over the target coordinates, computing distances between each coordinate and the input
                   coordinates.
                   1.1. If closest distance is smaller than dmin / 2, not unique to target
                   1.2. else, unique to target
                2. Iterating over the input coordinates, computing distances between each coordinate and the input
                   coordinates.
                   2.1. If closest distance is smaller than dmin / 2, not unique to input
                   2.2. else, unique to input
        """
        rates = np.array([x_rate, y_rate])
        target_unique = []
        target_approx = []
        inputs_unique = []
        inputs_approx = []
        commons       = []
        # ==============================================================================================================
        # For each coordinate in the target coordinates, running the following
        # ==============================================================================================================
        for ii, coordinate in enumerate(target):
            # ------------------------------------------------------------------------------------------------------
            # Computing distance to all other cylinders
            # ------------------------------------------------------------------------------------------------------
            loc = coordinate[0:2]
            diffs = np.sqrt(np.sum(np.power((inputs[:, 0:2] - loc) * rates, 2), axis=1))
            # ------------------------------------------------------------------------------------------------------
            # if min distance is bigger than threshold, add coordinate
            # ------------------------------------------------------------------------------------------------------
            if np.min(diffs) > dmin / 2:
                target_unique.append(list(coordinate))
            elif np.min(diffs) > 0:
                target_approx.append(list(coordinate))
            else:
                commons.append(list(coordinate))
        # ==============================================================================================================
        # For each coordinate in the input coordinates, running the following
        # ==============================================================================================================
        for ii, coordinate in enumerate(inputs):
            # ------------------------------------------------------------------------------------------------------
            # Computing distance to all other cylinders
            # ------------------------------------------------------------------------------------------------------
            loc = coordinate[0:2]
            diffs = np.sqrt(np.sum(np.power((target[:, 0:2] - loc) * rates, 2), axis=1))
            # ------------------------------------------------------------------------------------------------------
            # if min distance is bigger than threshold, add coordinate
            # ------------------------------------------------------------------------------------------------------
            if np.min(diffs) > dmin:
                inputs_unique.append(list(coordinate))
            elif np.min(diffs) > 0:
                inputs_approx.append(list(coordinate))
        return np.array(inputs_unique), np.array(target_unique), np.array(inputs_approx), np.array(target_approx), np.array(commons)

    @staticmethod
    def save_array(scat_locations, sensitivity, path, name=None, target_sensitivity=None):
        """
        :param scat_locations: NX3 array with N scatterer coordiantes:
                               [x_coord, y_coord, scale]
        :param sensitivity: matching sensitivity of the array
        :param path: path to save the data
        :param name:optional, name of the csv file
        :param target_sensitivity
        :return:
        """
        titles = ['x_coordinate', 'y_coordinate', 'scale', 'value']
        sens_row = ['sensitivity', sensitivity] if target_sensitivity is None else ['sensitivity', sensitivity, 'target', target_sensitivity]
        filename = 'scatterer_coordinates.csv' if name is None else name
        with open(os.path.join(path, filename), 'w', newline='') as f:
            writer   = csv.writer(f)
            writer.writerow(titles)
            writer.writerow(sens_row)
            for row in scat_locations:
                writer.writerow(row)


# ==================================================================================================================
# Misc functions
# ==================================================================================================================
class PathFindingFunctions:
    def __init__(self):
        pass

    @staticmethod
    def get_latest_dir(path):
        """
        :param path: a path to a directory with sub directories
        :return: the name of the newest directory
        """
        all_subdirs     = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        latest_subdir   = max(all_subdirs, key=os.path.getmtime)
        return latest_subdir

    @staticmethod
    def get_latest_model(path):
        all_subfiles    = [os.listdir(d) for d in os.listdir(path) if 'tar' in d]
        latest_subfile  = max(all_subfiles, key=os.path.getmtime)
        return latest_subfile

    @staticmethod
    def get_full_path(path, epoch=None):
        save_files = [os.path.join(path, d) for d in os.listdir(path) if "epoch" in d]
        if epoch is None:
            epoch_nums = [int(file.split(sep='_')[-1][0:-4]) for file in save_files[1:]]
            epoch = max(epoch_nums)
        chosen_file = [d for d in save_files if np.all((str(epoch) in d.split('\\')[-1], d[-3:] == 'tar'))][0]
        return chosen_file
