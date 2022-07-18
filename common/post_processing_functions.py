from ConfigVAE import *
import os
import json
import math
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
from global_const import encoder_type_e
from ScatterCoordinateDataset import import_data_sets_pics, import_data_set_test
from database_functions import PathFindingFunctions, ModelManipulationFunctions
from auxiliary_functions import PlottingFunctions
from roc_det_functions import RocDetFunctions
from blob_detection_functions import BlobDetectionFunctions
from database_functions import DatabaseFunctions
from time import time


class PostProcessing:
    def __init__(self):
        pass

    @staticmethod
    def log_to_plot(path, spacing=1, save_plt=True, plt_joined=True):
        """
        :param path: path to a result folder
        :param spacing: epoch distance between plots
        :param save_plt:
        :param plt_joined:
        :return: the function reads the log and creates a plot of RMS loss, with all the test databases documented
        """
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        filename    = os.path.join(path, 'logger_vae.txt')
        fileID      = open(filename, 'r')
        lines       = fileID.readlines()
        fileID.close()

        reached_start  = False
        epoch_list     = []
        keys_list      = []
        test_wmse      = {}
        test_grid      = {}

        train_label     = None
        train_mse_loss  = []
        train_dkl_loss  = []
        train_grid_loss = []
        train_tot_loss  = []
        # ==============================================================================================================
        # Going over lines, adding to log
        # ==============================================================================================================
        for line in lines:
            # ------------------------------------------------------------------------------------------------------
            # Getting to beginning of training
            # ------------------------------------------------------------------------------------------------------
            if not reached_start and 'Beginning Training' not in line:
                continue
            elif not reached_start:
                reached_start = True
                continue
            # ------------------------------------------------------------------------------------------------------
            # Reached beginning, going over cases
            # ------------------------------------------------------------------------------------------------------
            words = list(filter(None, line.split(sep=' ')))
            if 'Epoch' in line:
                try:
                    epoch_list.append(int(words[4]))
                except ValueError:
                    epoch_list.append(int(words[4][:-1]))
            elif 'train' in line.lower():
                if train_label is None:
                    train_label = words[3]
                train_mse_loss.append(float(words[7]))
                train_dkl_loss.append(float(words[9]))
                train_grid_loss.append(float(words[13]))
                train_tot_loss.append(float(words[16]))
            elif 'MSE' in line:
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # one of the test databases
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                temp_key = words[3]
                if temp_key not in keys_list:
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # if key does not exist, creates a new list
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    keys_list.append(temp_key)
                    test_wmse[temp_key] = []
                    test_grid[temp_key] = []
                test_wmse[temp_key].append(float(words[10]))
                test_grid[temp_key].append(float(words[13]))

        # ==============================================================================================================
        # Plotting the results
        # ==============================================================================================================
        epoch_len = len(epoch_list)
        plt.rcParams["figure.figsize"] = (20, 10)
        # ------------------------------------------------------------------------------------------------------
        # Sensitivity plot
        # ------------------------------------------------------------------------------------------------------
        sens_plt = plt.figure()
        if NORM_SENS:
            plt.plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) * SENS_STD for x in train_mse_loss[0:epoch_len:spacing]], '-o', label=train_label)
            for test_db in keys_list:
                plt.plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) * SENS_STD for x in test_wmse[test_db][0:epoch_len:spacing]], '-o', label=test_db)
        else:
            plt.plot(epoch_list[0:epoch_len:spacing],
                     [math.sqrt(x) for x in train_mse_loss[0:epoch_len:spacing]], '-o', label=train_label)
            for test_db in keys_list:
                plt.plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) for x in test_wmse[test_db][0:epoch_len:spacing]], '-o', label=test_db)
        plt.xlabel('Epoch')
        plt.ylabel('RMS Loss')
        plt.title('RMS loss vs Epoch number')
        plt.legend()
        plt.grid()
        # ------------------------------------------------------------------------------------------------------
        # D_kl plot
        # ------------------------------------------------------------------------------------------------------
        dkl_plt = plt.figure()
        plt.plot(epoch_list[0:epoch_len:spacing], train_dkl_loss[0:epoch_len:spacing], '-o', label=train_label)
        plt.xlabel('Epoch')
        plt.ylabel('D_kl')
        plt.title('D_kl loss vs Epoch number')
        plt.grid()
        # ------------------------------------------------------------------------------------------------------
        # Grid loss plot
        # ------------------------------------------------------------------------------------------------------
        grid_plt = plt.figure()
        plt.plot(epoch_list[0:epoch_len:spacing], train_grid_loss[0:epoch_len:spacing], '-o', label=train_label)
        for test_db in keys_list:
            plt.plot(epoch_list[0:epoch_len:spacing], test_grid[test_db][0:epoch_len:spacing], '-o', label=test_db)
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Reconstruction loss vs Epoch number')
        plt.legend()
        plt.grid()
        # ------------------------------------------------------------------------------------------------------
        # joined
        # ------------------------------------------------------------------------------------------------------
        if plt_joined:
            _, ax = plt.subplots(3, 1)
            if NORM_SENS:
                ax[0].plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) * SENS_STD for x in train_mse_loss[0:epoch_len:spacing]], '-o', label=train_label)
                for test_db in keys_list:
                    ax[0].plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) * SENS_STD for x in test_wmse[test_db][0:epoch_len:spacing]], '-o', label=test_db)
            else:
                ax[0].plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) for x in train_mse_loss[0:epoch_len:spacing]], '-o', label=train_label)
                for test_db in keys_list:
                    ax[0].plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) for x in test_wmse[test_db][0:epoch_len:spacing]], '-o', label=test_db)
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('RMS Loss')
            ax[0].set_title('RMS loss vs Epoch number')
            ax[0].legend()
            ax[0].grid()

            ax[1].plot(epoch_list[0:epoch_len:spacing], train_dkl_loss[0:epoch_len:spacing], '-o', label=train_label)
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('D_kl')
            ax[1].set_title('D_kl loss vs Epoch number')
            ax[1].grid()

            ax[2].plot(epoch_list[0:epoch_len:spacing], train_grid_loss[0:epoch_len:spacing], '-o', label=train_label)
            for test_db in keys_list:
                ax[2].plot(epoch_list[0:epoch_len:spacing], test_grid[test_db][0:epoch_len:spacing], '-o', label=test_db)
            ax[2].set_xlabel('Epoch')
            ax[2].set_ylabel('Cross Entropy Loss')
            ax[2].set_title('Reconstruction loss vs Epoch number')
            ax[2].legend()
            ax[2].grid()
        # ==============================================================================================================
        # Saving
        # ==============================================================================================================
        if save_plt and (path is not None):
            # ------------------------------------------------------------------------------------------------------
            # Setting filename
            # ------------------------------------------------------------------------------------------------------
            filename_sens = f'sensitivity_loss_training.png'
            filename_dkl  = f'dkl_loss_training.png'
            filename_grid = f'grid_loss_training.png'
            # ------------------------------------------------------------------------------------------------------
            # Creating directory if not exists
            # ------------------------------------------------------------------------------------------------------
            if not os.path.isdir(os.path.join(path, FIG_DIR)):
                os.makedirs(os.path.join(path, FIG_DIR))
            # ------------------------------------------------------------------------------------------------------
            # Saving
            # ------------------------------------------------------------------------------------------------------
            sens_plt.savefig(os.path.join(path, FIG_DIR, filename_sens))
            dkl_plt.savefig(os.path.join(path, FIG_DIR, filename_dkl))
            grid_plt.savefig(os.path.join(path, FIG_DIR, filename_grid))
        plt.show()

    @staticmethod
    def load_and_pass(path, epoch, key='4e+03_to_inf'):
        pf = PlottingFunctions()
        pff = PathFindingFunctions()
        mff = ModelManipulationFunctions()
        sigmoid = torch.nn.Sigmoid()
        # ==============================================================================================================
        # Extracting the full file path
        # ==============================================================================================================
        chosen_file = pff.get_full_path(path, epoch)
        # ==============================================================================================================
        # Loading the needed models and data
        # ==============================================================================================================
        train_loader, test_loaders, _ = import_data_sets_pics(PATH_DATABASE_TRAIN,
                                                              PATH_DATABASE_TEST,
                                                              BATCH_SIZE,
                                                              abs_sens=ABS_SENS,
                                                              dilation=DILATION)
        mod_vae, trainer = mff.load_state_train(chosen_file)
        # ==============================================================================================================
        # Extracting statistics
        # ==============================================================================================================
        mod_vae.eval()
        with torch.no_grad():
            test_loader_iter = iter(test_loaders[key])
            mu_means         = np.zeros((mod_vae.latent_dim, test_loader_iter.__len__()))
            std_means        = np.zeros((mod_vae.latent_dim, test_loader_iter.__len__()))
            for ii in range(len(test_loaders[key])):
                # ------------------------------------------------------------------------------
                # Working with iterables, much faster
                # ------------------------------------------------------------------------------
                try:
                    sample_batched = next(test_loader_iter)
                except StopIteration:
                    break
                # ------------------------------------------------------------------------------
                # Extracting the grids and sensitivities, passing through the model
                # ------------------------------------------------------------------------------
                target_sens = sample_batched['sensitivity'].float().to(mod_vae.device)
                if mod_vae.encoder_type == encoder_type_e.FULLY_CONNECTED:
                    target_grids = Variable(sample_batched['coordinate_target'].float()).to(mod_vae.device)
                    grids        = Variable(sample_batched['coordinate_target'].float()).to(mod_vae.device)
                else:
                    target_grids = sample_batched['grid_target'].float().to(mod_vae.device)
                    grids = Variable(sample_batched['grid_in'].float()).to(mod_vae.device)

                out_grid, out_sens, mu, logvar = mod_vae(grids)
                # ------------------------------------------------------------------------------
                # Logging mean mu and mean std values
                # ------------------------------------------------------------------------------
                mu_means[:, ii] = np.mean(mu.cpu().detach().numpy(), axis=0)
                std_means[:, ii] = np.exp(np.mean(logvar.cpu().detach().numpy(), axis=0))

    @staticmethod
    def load_model_plot_roc_det(path, epoch, key='3e+05_to_inf'):
        """
        :return: This function:
                    1. Loads a saved model in path, epoch
                    2. Loads the dataloader with the given key
                    3. passes all the database through the model and gathers:
                        i. True positive rates
                        ii. False positive rates
                        iii. False negative rates
                    4. Plotting and saving the MROC and MDET curves
        """
        pf  = PlottingFunctions()
        pff = PathFindingFunctions()
        mmf = ModelManipulationFunctions()
        rdf = RocDetFunctions()
        # ==============================================================================================================
        # Extracting the full file path
        # ==============================================================================================================
        chosen_file = pff.get_full_path(path, epoch)
        # ==============================================================================================================
        # Loading the needed models and data
        # ==============================================================================================================
        _, test_loaders, _ = import_data_sets_pics(PATH_DATABASE_TRAIN,
                                                   PATH_DATABASE_TEST,
                                                   BATCH_SIZE,
                                                   abs_sens=ABS_SENS,
                                                   dilation=DILATION)
        test_loader = test_loaders[key]
        # test_loader = test_loaders['3e+05_to_inf']
        # test_loader = test_loaders['2e+05_to_3e+05']
        # test_loader = test_loaders['1e+05_to_2e+05']
        # test_loader = test_loaders['0_to_1e+05']
        mod_vae, _  = mmf.load_state_train(chosen_file)
        # ==============================================================================================================
        # Getting the ROC and DET curves
        # ==============================================================================================================
        len_a = 200
        thr_a = 0.1
        len_b = 45
        thr_b = 1
        threshold_num = [(ii*thr_a)/len_a for ii in list(range(len_a))] + [thr_a+((ii*(thr_b-thr_a))/len_b) for ii in list(range(len_b+1))]
        tpr, fpr, fnr = rdf.get_roc_det_curve(mod_vae, test_loader, threshold_num=threshold_num)
        # ==============================================================================================================
        # Saving the data
        # ==============================================================================================================
        # ----------------------------------------------------------------------------------------------------------
        # Creating directory is does not exists
        # ----------------------------------------------------------------------------------------------------------
        if not os.path.isdir(os.path.join(path, PP_DATA)):
            os.makedirs(os.path.join(path, PP_DATA))
        # ----------------------------------------------------------------------------------------------------------
        # Creating filename and full path
        # ----------------------------------------------------------------------------------------------------------
        output_filename = key + f'_tpr_fpr_npr_epoch_{epoch}.json'
        output_filepath = os.path.join(path, 'post_processing', output_filename)
        # ----------------------------------------------------------------------------------------------------------
        # creating a dictionary with the data
        # ----------------------------------------------------------------------------------------------------------
        all_output_data = [{
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'thresholds': threshold_num
        }]
        # ----------------------------------------------------------------------------------------------------------
        # Saving
        # ----------------------------------------------------------------------------------------------------------
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_output_data, f, indent=4)
        # ==============================================================================================================
        # Plotting
        # ==============================================================================================================
        data_dict = {
            key: {
                'true_positive_rate': tpr,
                'false_positive_rate': fpr,
                'false_negative_rate': fnr,
                'thresholds': threshold_num}}
        pf.plot_roc_curve(data_dict, name_prefixes=[key], save_plt=True, path=path, epoch=epoch)
        pf.plot_det_curve(data_dict, name_prefixes=[key], save_plt=True, path=path, epoch=epoch)

    @staticmethod
    def get_latent_statistics(path, epoch, save_plt=True):
        """
        :param path: path to a model training results folder
        :param epoch: wanted epoch to load
        :param save_plt:
        :return: the function prints out plot of the statistics regarding the latent space
        """
        pf  = PlottingFunctions()
        pff = PathFindingFunctions()
        mff = ModelManipulationFunctions()
        sigmoid = torch.nn.Sigmoid()
        # ==============================================================================================================
        # Extracting the full file path
        # ==============================================================================================================
        chosen_file = pff. get_full_path(path, epoch)
        # ==============================================================================================================
        # Loading the needed models and data
        # ==============================================================================================================
        train_loader, test_loaders, _ = import_data_sets_pics(PATH_DATABASE_TRAIN,
                                                              PATH_DATABASE_TEST,
                                                              BATCH_SIZE,
                                                              abs_sens=ABS_SENS,
                                                              dilation=DILATION)
        mod_vae, trainer = mff.load_state_train(chosen_file)
        # ==============================================================================================================
        # Extracting statistics
        # ==============================================================================================================
        key = '4e+03_to_inf'  # '3e+05_to_inf' '2e+05_to_3e+05' '1e+05_to_2e+05' '0_to_1e+05'
        test_loader_iter = iter(test_loaders[key])
        mu_means  = np.zeros((mod_vae.latent_dim, test_loader_iter.__len__()))
        std_means = np.zeros((mod_vae.latent_dim, test_loader_iter.__len__()))
        mod_vae.eval()
        with torch.no_grad():
            for ii in range(len(test_loader_iter)):
                # ------------------------------------------------------------------------------
                # Working with iterables, much faster
                # ------------------------------------------------------------------------------
                try:
                    sample_batched = next(test_loader_iter)
                except StopIteration:
                    break
                # ------------------------------------------------------------------------------
                # Extracting the grids and sensitivities, passing through the model
                # ------------------------------------------------------------------------------
                grids         = Variable(sample_batched['grid_in'].float()).to(mod_vae.device)
                sensitivities = Variable(sample_batched['sensitivity'].float()).to(mod_vae.device)

                grid_outs, outputs, mu, logvar = mod_vae(grids)
                # ------------------------------------------------------------------------------
                # Logging mean mu and mean std values
                # ------------------------------------------------------------------------------
                mu_means[:, ii] = np.mean(mu.cpu().detach().numpy(), axis=0)
                std_means[:, ii] = np.exp(np.mean(logvar.cpu().detach().numpy(), axis=0))
                # ------------------------------------------------------------------------------
                # Plotting manually
                # ------------------------------------------------------------------------------
                plt.figure()
                plt.imshow(1 - np.squeeze(sample_batched['grid_target'][0, 0, :, :].cpu().detach().numpy()), cmap='gray')
                plt.title("Target Output - Model Input")
                plt.figure()
                plt.imshow(np.squeeze(1 - sigmoid(grid_outs[0, 0, :, :]).cpu().detach().numpy()), cmap='gray')
                plt.title("Model output - Raw")
                plt.figure()
                plt.imshow(np.where(np.squeeze(1 - sigmoid(grid_outs[0, 0, :, :]).cpu().detach().numpy()) >= 0.5, 1, 0), cmap='gray')
                plt.title("Model output - After Step at 0.5")
                plt.figure()
                plt.imshow(np.where(np.squeeze(1 - sigmoid(grid_outs[0, 0, :, :]).cpu().detach().numpy()) >= 0.9, 1, 0), cmap='gray')
                plt.title("Model output - After Step at 0.1")
                # mu_temp = mu.cpu().detach().numpy()
                # var_temp = np.exp(logvar.cpu().detach().numpy())
                # target = sensitivities.cpu().detach().numpy()
                # output = outputs.cpu().detach().numpy()
                # pf.plot_latent(mu_temp, var_temp, target, output)
                # for jj in range(20):
                #     mu_temp     = mu[jj, :].cpu().detach().numpy()
                #     var_temp    = np.exp(logvar[jj, :].cpu().detach().numpy())
                #     target      = sensitivities[jj, :].cpu().detach().numpy()
                #     output      = outputs[jj, :].cpu().detach().numpy()
                #     plot_latent(mu_temp, var_temp, target, output)
                print('hi')
        # ==============================================================================================================
        # Saving
        # ==============================================================================================================
        output_filename = key + f'_latent_statistics_epoch_{epoch}.json'
        output_filepath = os.path.join(path, 'post_processing', output_filename)
        # ----------------------------------------------------------------------------------------------------------
        # creating a dictionary with the data
        # ----------------------------------------------------------------------------------------------------------
        all_output_data = [{
            'mean_expectation': [list(mu_means[:, ii]) for ii in range(mu_means.shape[1])],
            'mean_std': [list(std_means[:, ii]) for ii in range(std_means.shape[1])],
        }]
        # ----------------------------------------------------------------------------------------------------------
        # Saving
        # ----------------------------------------------------------------------------------------------------------
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_output_data, f, indent=4)
        # ==============================================================================================================
        # Plotting statistics
        # ==============================================================================================================
        mu_dim = np.mean(mu_means, axis=1)
        std_dim = np.mean(std_means, axis=1)
        latent = plt.figure()
        latent.set_size_inches(18.5, 10.5)
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(mu_dim, 'o')
        plt.title('Expectation mean per index, latent space')
        plt.xlabel('index')
        plt.ylabel('mean')
        plt.grid()
        ax2 = plt.subplot(2, 1, 2)
        plt.plot(std_dim, 'o')
        plt.title('Variance mean per index, latent space')
        plt.xlabel('index')
        plt.ylabel('mean')
        plt.grid()
        # ==============================================================================================================
        # Saving
        # ==============================================================================================================
        if save_plt and (path is not None) and (epoch is not None):
            # ------------------------------------------------------------------------------------------------------
            # Setting filename
            # ------------------------------------------------------------------------------------------------------
            filename = f'latent_statistics_{epoch}.png'
            filename = key + '_' + filename
            # ------------------------------------------------------------------------------------------------------
            # Creating directory if not exists
            # ------------------------------------------------------------------------------------------------------
            if not os.path.isdir(os.path.join(path, FIG_DIR)):
                os.makedirs(os.path.join(path, FIG_DIR))
            # ------------------------------------------------------------------------------------------------------
            # Saving
            # ------------------------------------------------------------------------------------------------------
            latent.savefig(os.path.join(path, FIG_DIR, filename))
        plt.show()
        pass

    @staticmethod
    def load_data_plot_latent_statistics(path, epoch, prefix_list, save_plt=True):
        """
        :param path:
        :param epoch:
        :param prefix_list:
        :param save_plt
        :return:
        """
        # ==============================================================================================================
        # Loading the saved data
        # ==============================================================================================================
        main_dict = {}
        for prefix in prefix_list:
            filename = prefix + f'_latent_statistics_epoch_{epoch}.json'
            filepath = os.path.join(path, PP_DATA, filename)
            with open(filepath, mode='r', encoding='utf-8') as json_f:
                results_dict = json.load(json_f)[-1]
            main_dict[prefix] = results_dict
        # ==============================================================================================================
        # Plotting
        # ==============================================================================================================
        latent = plt.figure()
        latent.set_size_inches(18.5, 10.5)
        ax1    = plt.subplot(2, 1, 1)
        plt.title('Expectation mean per index, latent space')
        plt.xlabel('index')
        plt.ylabel('mean')
        plt.grid()
        ax2    = plt.subplot(2, 1, 2)
        plt.title('Variance mean per index, latent space')
        plt.xlabel('index')
        plt.ylabel('mean')
        plt.grid()
        for prefix in prefix_list:
            # ------------------------------------------------------------------------------------------------------
            # Extracting data
            # ------------------------------------------------------------------------------------------------------
            mu_means  = np.array(main_dict[prefix]['mean_expectation'])
            std_means = np.array(main_dict[prefix]['mean_std'])
            # ------------------------------------------------------------------------------------------------------
            # Plotting it
            # ------------------------------------------------------------------------------------------------------
            mu_dim  = np.mean(mu_means, axis=0)
            std_dim = np.mean(std_means, axis=0)
            ax1.plot(mu_dim, 'o', label=prefix)
            ax2.plot(std_dim, 'o', label=prefix)
        plt.legend()

        # ==============================================================================================================
        # Saving
        # ==============================================================================================================
        if save_plt and (path is not None) and (epoch is not None):
            # ------------------------------------------------------------------------------------------------------
            # Setting filename
            # ------------------------------------------------------------------------------------------------------
            filename = f'_latent_statistics_{epoch}.png'
            filename = prefix_list[0] + '_' + filename if len(prefix_list) == 1 else 'combined_' + filename
            # ------------------------------------------------------------------------------------------------------
            # Creating directory if not exists
            # ------------------------------------------------------------------------------------------------------
            if not os.path.isdir(os.path.join(path, FIG_DIR)):
                os.makedirs(os.path.join(path, FIG_DIR))
            # ------------------------------------------------------------------------------------------------------
            # Saving
            # ------------------------------------------------------------------------------------------------------
            latent.savefig(os.path.join(path, FIG_DIR, filename))
        plt.show()

    @staticmethod
    def load_data_plot_roc_det(path, epoch, prefix_list, threshold_list=None):
        """
        :param path: Path to the saved data
        :param epoch: the epoch in question
        :param prefix_list: self explanatory
        :param threshold_list: list of thresholds to scatter
        :return: The function:
                    1. loads the data in the given path according to the given prefixes
                    2. plots the ROC and DET curves
        """
        thr = threshold_list
        pf = PlottingFunctions()
        # ==============================================================================================================
        # Loading the saved data
        # ==============================================================================================================
        main_dict = {}
        for prefix in prefix_list:
            filename = prefix + f'_tpr_fpr_npr_epoch_{epoch}.json'
            filepath = os.path.join(path, PP_DATA, filename)
            with open(filepath, mode='r', encoding='utf-8') as json_f:
                results_dict = json.load(json_f)[-1]
            main_dict[prefix] = results_dict
        # ==============================================================================================================
        # Plotting
        # ==============================================================================================================
        pf.plot_roc_curve(main_dict, name_prefixes=prefix_list, thresholds=thr, save_plt=True, path=path, epoch=epoch)
        pf.plot_det_curve(main_dict, name_prefixes=prefix_list, thresholds=thr, save_plt=True, path=path, epoch=epoch)

    @staticmethod
    def load_model_compare_blobs(path, epoch, key='3e+05_to_inf',
                                 sigma_0=0.3, scale=1.15, k=15, peak_threshold=3, kernel_size=25):
        """
        :param path: path to saved model
        :param epoch: epoch of saved model
        :param key: name ot test lodar
        :param sigma_0: initial scale for the scale space
        :param scale: multiplication factor for adjacent scales
        :param k: size of the scale dimension in the scale space
        :param peak_threshold: threshold for local maxima classification
        :param kernel_size: size of the gaussian kernel
        :return: Function loads a model and a test loader, passes a single grid through the model and computes the
                 maximum locations for both the input and the output. Plots the differences and saves the reconstructed
                 inputs in csv file
        """
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        sigmoid = torch.nn.Sigmoid()
        pf  = PlottingFunctions()
        dbf = DatabaseFunctions()
        pff = PathFindingFunctions()
        mmf = ModelManipulationFunctions()
        bdf = BlobDetectionFunctions(peak_threshold=peak_threshold,
                                     kernel_size=kernel_size,
                                     sigma_0=sigma_0,
                                     scale=scale,
                                     k=k)
        x_rate = (XRANGE[1] - XRANGE[0] + 1) / XQUANTIZE
        y_rate = (YRANGE[1] - YRANGE[0] + 1) / YQUANTIZE
        dmin   = DMIN
        threshold = 0.5
        # ==============================================================================================================
        # Extracting the full file path
        # ==============================================================================================================
        chosen_file = pff.get_full_path(path, epoch)
        # ==============================================================================================================
        # Loading the needed models and data
        # ==============================================================================================================
        test_loaders = import_data_set_test([PATH_DATABASE_TEST[-2]], batch_size=1,
                                            mixup_factor=MIXUP_FACTOR,
                                            mixup_prob=MIXUP_PROB,
                                            abs_sens=ABS_SENS,
                                            dilation=DILATION,
                                            shuffle=True)
        test_loader = test_loaders[key]
        mod_vae, _ = mmf.load_state_train(chosen_file)
        mod_vae.eval()
        # ==============================================================================================================
        # No grad for speed
        # ==============================================================================================================
        with torch.no_grad():
            loader_iter = iter(test_loader)
            sample = next(loader_iter)
            # ------------------------------------------------------------------------------
            # Extracting the grids and sensitivities
            # ------------------------------------------------------------------------------
            grids = Variable(sample['grid_in'].float()).to(mod_vae.device)
            origin_points = np.squeeze(sample['coordinate_target'].detach().numpy()).astype(int)
            sens_target   = (sample['sensitivity'].detach().numpy() * SENS_STD) + SENS_MEAN
            # ------------------------------------------------------------------------------
            # Forward pass
            # ------------------------------------------------------------------------------
            grid_out, sens_out, _, _ = mod_vae(grids)
            grid_out = np.squeeze(sigmoid(grid_out).cpu().detach().numpy())
            grid_out_sliced = mmf.slice_grid(grid_out, threshold)
        # ==============================================================================================================
        # Creating scale space and DoG space
        # ==============================================================================================================
        print('Computing scale space . . . ')
        scale_space         = bdf.create_scale_space(grid_out)
        scale_space_sliced  = bdf.create_scale_space(grid_out_sliced)
        print('Computing Difference of Gaussians space . . . ')
        dog_space           = bdf.create_dog_space(scale_space)
        dog_space_sliced    = bdf.create_dog_space(scale_space_sliced)
        print('Finding local maxima . .. ')
        local_max           = bdf.extract_local_maxima(dog_space)
        local_max_sliced    = bdf.extract_local_maxima(dog_space_sliced)
        # ==============================================================================================================
        # Removing the cylinders based on the minimal distance and blob size
        # ==============================================================================================================
        print('Making array valid . . .')
        valid_array         = dbf.check_array_validity(local_max, x_rate=x_rate, y_rate=y_rate, dmin=dmin)
        valid_array_sliced  = dbf.check_array_validity(local_max_sliced, x_rate=x_rate, y_rate=y_rate, dmin=dmin)
        print('Valid array saved to ' + os.path.join(path, PP_DATA))
        dbf.save_array(valid_array, (sens_out.item() * SENS_STD) + SENS_MEAN, os.path.join(path, PP_DATA), name='scatter_raw7.csv', target_sensitivity=sens_target[0, 0])
        dbf.save_array(valid_array_sliced, (sens_out.item() * SENS_STD) + SENS_MEAN, os.path.join(path, PP_DATA), name='scatter_sliced7.csv', target_sensitivity=sens_target[0, 0])
        # ==============================================================================================================
        # Getting differences between the original and reconstructed coordinates
        # ==============================================================================================================
        print('Computing differences . . .')
        model_unique, origin_unique, model_approx, origin_approx, commons                                    = dbf.find_differences(valid_array, origin_points, x_rate, y_rate, dmin)
        model_unique_sliced, origin_unique_sliced, model_approx_sliced, origin_approx_sliced, commons_sliced = dbf.find_differences(valid_array_sliced, origin_points, x_rate, y_rate, dmin)
        # ==============================================================================================================
        # Plotting
        # ==============================================================================================================
        plt.figure()
        leg = []
        plt.imshow(1-grid_out, cmap='gray')
        if commons.shape[0] > 0:
            plt.scatter(commons[:, 0], commons[:, 1])
            leg.append('Similar')
        if origin_approx.shape[0] > 0:
            plt.scatter(origin_approx[:, 0], origin_approx[:, 1])
            leg.append('neighboring, target')
        if model_approx.shape[0] > 0:
            plt.scatter(model_approx[:, 0], model_approx[:, 1])
            leg.append('neighboring, model')
        if origin_unique.shape[0] > 0:
            plt.scatter(origin_unique[:, 0], origin_unique[:, 1], marker='d')
            leg.append('target unique')
        if model_unique.shape[0] > 0:
            plt.scatter(model_unique[:, 0], model_unique[:, 1], marker='^')
            leg.append('model unique')
        plt.legend(leg)
        plt.title('Raw Output: Original: ' + str(origin_points.shape[0]) +
                  ' Reconstructed: ' + str(valid_array.shape[0]) +
                  ', Unique Original: ' + str(origin_unique.shape[0]) +
                  ', Unique Reconstructed: ' + str(model_unique.shape[0]))

        plt.figure()
        leg = []
        plt.imshow(1 - grid_out_sliced, cmap='gray')
        if commons_sliced.shape[0] > 0:
            plt.scatter(commons_sliced[:, 0], commons_sliced[:, 1])
            leg.append('Similar')
        if origin_approx_sliced.shape[0] > 0:
            plt.scatter(origin_approx_sliced[:, 0], origin_approx_sliced[:, 1])
            leg.append('neighboring, target')
        if model_approx_sliced.shape[0] > 0:
            plt.scatter(model_approx_sliced[:, 0], model_approx_sliced[:, 1])
            leg.append('neighboring, model')
        if origin_unique_sliced.shape[0] > 0:
            plt.scatter(origin_unique_sliced[:, 0], origin_unique_sliced[:, 1], marker='d')
            leg.append('target unique')
        if model_unique_sliced.shape[0] > 0:
            plt.scatter(model_unique_sliced[:, 0], model_unique_sliced[:, 1], marker='^')
            leg.append('model unique')
        plt.legend(leg)
        plt.title('Slicer at ' + str(threshold) + ': Original: ' + str(origin_points.shape[0]) +
                  ' Reconstructed: ' + str(valid_array_sliced.shape[0]) +
                  ', Unique Original: ' + str(origin_unique_sliced.shape[0]) +
                  ', Unique Reconstructed: ' + str(model_unique_sliced.shape[0]))

        print(f'Target sensitivity: {sens_target[0,0]} ; Predicted sensitivity: {(sens_out.item() * SENS_STD) + SENS_MEAN}')
        plt.show()


class PostProcessingDG:
    def __init__(self):
        pass

    @staticmethod
    def log_to_plot(path, spacing=1, save_plt=True):
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        filename = os.path.join(path, 'logger_vae.txt')
        fileID = open(filename, 'r')
        lines = fileID.readlines()
        fileID.close()

        reached_start = False
        epoch_list = []
        keys_list = []
        test_wmse = {}
        test_mse  = {}

        train_label = None
        train_mse_loss = []
        # ==============================================================================================================
        # Going over lines, adding to log
        # ==============================================================================================================
        for line in lines:
            # ------------------------------------------------------------------------------------------------------
            # Getting to beginning of training
            # ------------------------------------------------------------------------------------------------------
            if not reached_start and 'Beginning Training' not in line:
                continue
            elif not reached_start:
                reached_start = True
                continue
            # ------------------------------------------------------------------------------------------------------
            # Reached beginning, going over cases
            # ------------------------------------------------------------------------------------------------------
            words = list(filter(None, line.split(sep=' ')))
            if 'Epoch' in line:
                try:
                    epoch_list.append(int(words[4]))
                except ValueError:
                    epoch_list.append(int(words[4][:-1]))
            elif 'train' in line.lower():
                if train_label is None:
                    train_label = words[3]
                train_mse_loss.append(float(words[8]))
            elif 'group' in line.lower():
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # one of the test databases
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                temp_key = words[3]
                if temp_key not in keys_list:
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # if key does not exist, creates a new list
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    keys_list.append(temp_key)
                    test_wmse[temp_key] = []
                    test_mse[temp_key]  = []
                test_wmse[temp_key].append(float(words[8]))
                test_mse[temp_key].append(float(words[11]))
        # ==============================================================================================================
        # Plotting the results
        # ==============================================================================================================
        epoch_len = len(epoch_list)
        plt.rcParams["figure.figsize"] = (18, 9)
        # ------------------------------------------------------------------------------------------------------
        # Sensitivity plot
        # ------------------------------------------------------------------------------------------------------
        sens_plt = plt.figure()
        plt.plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) for x in train_mse_loss[0:epoch_len:spacing]], '-o', label=train_label)
        for test_db in keys_list:
            plt.plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) for x in test_mse[test_db][0:epoch_len:spacing]], '-o', label=test_db+'_unweighted')
        plt.plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) for x in test_wmse['test_total'][0:epoch_len:spacing]], '-o', label=test_db)
        plt.xlabel('Epoch')
        plt.ylabel('RMS Loss')
        plt.title('RMS loss vs Epoch number')
        plt.legend()
        plt.grid()
        # ==============================================================================================================
        # Saving
        # ==============================================================================================================
        if save_plt and (path is not None):
            # ------------------------------------------------------------------------------------------------------
            # Setting filename
            # ------------------------------------------------------------------------------------------------------
            filename_sens = f'sensitivity_loss_training.png'
            # ------------------------------------------------------------------------------------------------------
            # Creating directory if not exists
            # ------------------------------------------------------------------------------------------------------
            if not os.path.isdir(os.path.join(path, FIG_DIR)):
                os.makedirs(os.path.join(path, FIG_DIR))
            # ------------------------------------------------------------------------------------------------------
            # Saving
            # ------------------------------------------------------------------------------------------------------
            sens_plt.savefig(os.path.join(path, FIG_DIR, filename_sens))
        plt.show()


if __name__ == '__main__':
    # 14_7_2021_0_47 # 12_7_2021_15_22 # 15_7_2021_9_7  # 4_8_2021_8_30
    # 24_8_2021_8_51   - with mixup 0.00 - 25k database
    # 23_8_2021_8_16   - with mixup 0.01 - 25k database
    # 24_8_2021_16_57  - without mixup - 25k database
    # 30_8_2021_16_57  - without mixup - 30k database
    # 1_9_2021_16_31   - without mixup - 30k database
    # 2_9_2021_14_2    - without mixup - 30k singed database - latent space 90   ---> BAD CONFIG
    # 5_9_2021_13_46   - without mixup - 30k signed database - latent space 50   ---> BAD CONFIG
    # 5_9_2021_22_54   - without mixup - 30k signed database - with dropout 0.5  ---> BAD CONFIG
    # 6_9_2021_7_29    - without mixup - 30k signed database - with dropout 0.01
    # 7_9_2021_10_28   - without mixup - 30k unsigned database - with dropout 0.01
    # 8_9_2021_8_44    - without mixup - 30k unsigned database - with dropout 0.2 and 0.01
    # 9_9_2021_7_21    - without mixup - 30k unsigned database - with dropout 0.25
    # 15_9_2021_10_6   - without mixup - 30k unsigned database - weighted MSE [1, 2, 4] with uncorrected log
    # 16_9_2021_12_27  - without mixup - 30k unsigned database - weighted MSE [1, 2, 4]
    # 18_9_2021_8_7    - without mixup - 30p5k unsigned database - weighted MSE [1, 4, 12] lr 1e-4
    # 19_9_2021_7_14   - without mixup - 30p5k unsigned database - weighted MSE [1, 2, 12] lr 1e-4 BEST RESULTS SO FAR
    """
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # TO WORK WITH MODELS OLDER MODELS ABOVE, WE NEED TO ROLLBACK TO REVISION 5ecc9855a264e09cfb5aa92dcd8ba3729f818193
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    """
    # 22_9_2021_21_52 - without mixup, 30p5k unsigned database - weighted MSE [1, 2, 12] lr 3e-, without bias in conv and FC
    # 23_9_2021_23_29 - without mixup, 30p5k unsigned database - weighted MSE [1, 1.5, 12] lr 3e-4
    # 25_9_2021_17_24 - without mixup, 30p5k unsigned database - weighted MSE [1, 1.5, 3, 12] lr 1e-4
    # 26_9_2021_11_25 - without mixup, 30p5k unsigned database - weighted MSE [1, 1.5, 2, 12] lr 1e-4 BEST RESULTS SO FAR NEW CONFIGS
    # 28_9_2021_7_12  - without mixup, 30p5k unsigned database - weighted MSE [1, 3, 2, 12] lr 1e-4
    # 30_9_2021_8_45  - without mixup, 30p5k unsigned database - weighted MSE [1, 1.5, 2, 12] lr 1e-4 beta 1e-2
    # 2_10_2021_9_40  - without mixup, 30p5k unsigned database - weighted MSE [1, 1.5, 2, 12] lr 1e-4 beta 1e-1
    # 4_10_2021_11_50 - without mixup, 30p5k unsigned database - weighted MSE [1, 1.5, 2, 20] lr 1e-4 beta 1e-1
    # 11_10_2021_15_49 - without mixup, 30p5k unsigned database - weighted MSE [1, 2, 2, 20] lr 1e-4 beta 1
    # ---------------- BEGINNING LATENT SPACE REDUCTION ----------------------
    # 17_10_2021_6_51 - without mixup, 30p5k unsigned database - weighted MSE [1, 2, 2, 20] lr 1e-4 beta 1 less layers, Latent size 90  42000!!!
    # 24_10_2021_7_1 - without mixup, 30p5k unsigned database - weighted MSE [1, 2, 2, 20] lr 1e-4 beta 1 less layers Latent size 70
    # 24_10_2021_15_19 - without mixup, 30p5k unsigned database - weighted MSE [1, 2, 2, 20] lr 1e-4 beta 1 less layers Latent size 50 - 40500!!!
    # 25_10_2021_11_7 - without mixup, 30p5k unsigned database - weighted MSE [1, 2, 2, 20] lr 1e-4 beta 1 less layers Latent size 40 - 40500
    # 26_10_2021_9_50 - without mixup, 30p5k unsigned database - weighted MSE [1, 2, 2, 20] lr 1e-4 beta 1 less layers Latent size 30 - 44700
    # 28_10_2021_14_11 - without mixup, 30p5k unsigned database - weighted MSE [1, 2, 2, 20] lr 1e-4 beta 1 less layers Latent size 20 -42500
    # 30_10_2021_15_50 - without mixup, 30p5k unsigned database - weighted MSE [1, 2, 2, 20] lr 1e-4 beta 1 less layers Latent size 110 - 39300
    # 31_10_2021_18_13 - without mixup, 30p5k unsigned database - weighted MSE [1, 2, 2, 20] lr 1e-4 beta 1 less layers Latent size 10 - 43800
    # 2_11_2021_8_9 -  without mixup, 30p5k unsigned database - weighted MSE [1, 2, 2, 20] lr 1e-4 beta 1 less layers Latent size 5 - 37300
    # 18_11_2021_8_21 - normal regerssion
    # 23_11_2021_17_48 - VGG
    # 24_11_2021_21_34 - VGG with more channels - latent space 100  - ALL with beta_grid = 1.6e-5
    # 1_12_2021_10_30  - VGG with more channels - latent space 50 epoch 20
    # 5_12_2021_8_9    - VGG with more channels - latent space 25 epoch 20
    # 7_12_2021_8_7    - VGG with more channels - latent space 15
    # 8_12_2021_9_17   - VGG with more channels - latent space 50
    # 9_12_2021_19_3   - VGG with more channels - latent space 50
    # 12_12_2021_23_5 + 15_12_2021_23_46 - VGG -latent space 50, scatterer dilation of 3 - GOOD RESULTS with mistake padding in the last padpool layer. This resulted in information leaking from the last 200 pixels and that is why the network was not able to reconstruct
    # 20_12_2021_11_38 + 23_12_2021_8_20 - VGG with more channels - latent space 25 After fixing the reconstruction problem
    # 26_12_2021_8_41
    # 2_1_2022_7_50 -  VGG latent space 50, scatterer dilation of 3  after padding fix
    # 12_1_2022_6_51 - VGG latent space 50, scaterrer dilation of 4 after padding fix, GREAT RESULTS!
    # c_path = '..\\results\\12_12_2021_23_5'
    # 12_1_2022_6_51 + 16_1_2022_21_39 - The model that worked!
    # 10_2_2022_16_45 + 13_2_2022_21_4 - The model that worked + transpose training

    c_epoch = 240
    # c_path = '..\\results\\16_1_2022_21_39'
    # c_path = '..\\results\\10_2_2022_16_45'
    # c_path = '..\\results\\10_2_2022_16_45_plus_13_2_2022_21_4'
    # c_path = '..\\results\\5_4_2022_10_27'
    # c_path = '..\\results\\13_4_2022_22_58'
    # c_path = '..\\results\\15_5_2022_17_9'
    # c_path = '..\\results\\6_6_2022_19_7'
    # c_path = '..\\results\\14_6_2022_16_8'
    c_path = '..\\results_vae\\9_7_2022_22_38'
    c_path2 = '..\\results_dg\\12_7_2022_14_46'

    pp = PostProcessing()
    pp2 = PostProcessingDG()

    threshold_list = [0.1, 0.2, 0.5]
    # prefix_list    = ['3e+05_to_inf', '2e+05_to_3e+05', '1e+05_to_2e+05', '0_to_1e+05']
    # prefix_list    = ['1e+05_to_2e+05']
    prefix_list    = ['4e+03_to_inf']
    # pp.load_data_plot_roc_det(c_path, c_epoch, prefix_list)

    # pp.load_model_compare_blobs(c_path, c_epoch, key='2e+05_to_3e+05', peak_threshold=3.3)
    # pp.load_model_plot_roc_det(c_path, c_epoch, key='0_to_1e+05')
    # pp.log_to_plot(c_path, spacing=10)
    # pp.get_latent_statistics(c_path, c_epoch)
    # pp.load_data_plot_latent_statistics(c_path, c_epoch, prefix_list)

    # pp.load_and_pass(c_path, c_epoch, key=prefix_list[0])
    # pp.log_to_plot(c_path)

    pp2.log_to_plot(c_path2)

