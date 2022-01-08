# ***************************************************************************************************
# THIS FILE HOLDS THE FUNCTIONS NEEDED TO MANIPULATE THE DATABASE ON WHICH THE NETWORK TRAINS
# ***************************************************************************************************
from ConfigVAE import *
import os
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
        grid_coords = []
        for ii in range(len(arr)):
            x = float(arr[ii, 0])
            y = float(arr[ii, 1])
            x_grid = int(round(((x - XRANGE[0]) / XRANGE[1]) * (XQUANTIZE - 1), 0))
            y_grid = int(round(((y - YRANGE[0]) / YRANGE[1]) * (YQUANTIZE - 1), 0))
            grid_coords.append(np.array([x_grid, y_grid]))

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
