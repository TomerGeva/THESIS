import os
import json
import math
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
from global_const import encoder_type_e
from ScatterCoordinateDataset import import_data_sets_pics, import_data_set_test
from PCurrentDataset import import_pcurrents_dataset, import_pcurrent_omega_dataset
from ScatCoord_DG import import_data_sets_coord
from database_functions import PathFindingFunctions, ModelManipulationFunctions
from auxiliary_functions import PlottingFunctions
from roc_det_functions import RocDetFunctions
from blob_detection_functions import BlobDetectionFunctions
from database_functions import DatabaseFunctions
from time import time


class PostProcessingVAE:
    def __init__(self):
        pass

    @staticmethod
    def log_to_plot(path, spacing=1, save_plt=True, plt_joined=True, unwighted_plot=False):
        """
        :param path: path to a result folder
        :param spacing: epoch distance between plots
        :param save_plt:
        :param plt_joined:
        :return: the function reads the log and creates a plot of RMS loss, with all the test databases documented
        """
        def mse_plot(epoch_vec, train_vec, test_dict, key_list, spacing, norm, weights):
            epoch_len = len(epoch_vec)
            fig = plt.figure()
            if norm:
                plt.plot(epoch_list[0:epoch_len:spacing],
                         [math.sqrt(x) * SENS_STD for x in train_vec[0:epoch_len:spacing]], '-o',
                         label=train_label)
                for test_db in key_list:
                    plt.plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x / weights[test_db]) * SENS_STD for x in test_dict[test_db][0:epoch_len:spacing]], '-o', label=test_db)
            else:
                plt.plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) for x in train_vec[0:epoch_len:spacing]], '-o', label=train_label)
                for ii, test_db in enumerate(key_list):
                    plt.plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x / weights[test_db]) for x in test_dict[test_db][0:epoch_len:spacing]], '-o', label=test_db)
            plt.xlabel('Epoch')
            plt.ylabel('RMS Loss')
            plt.title('RMS loss vs Epoch number')
            plt.legend()
            plt.grid()
            return fig
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        filename    = os.path.join(path, 'logger_vae.txt')
        # filename    = os.path.join(path, 'rework_log.txt')
        fileID      = open(filename, 'r')
        lines       = fileID.readlines()
        fileID.close()

        reached_start  = False
        epoch_list     = []
        keys_list      = []
        weights        = {}
        test_wmse      = {}
        test_mse       = {}
        test_grid      = {}

        train_label     = None
        train_mse_loss  = []
        train_mse_loss_unweighted  = []
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
            elif 'train_weighted ' in line.lower():
                if train_label is None:
                    train_label = words[3]
                train_mse_loss.append(float(words[7]))
                train_dkl_loss.append(float(words[9]))
                train_grid_loss.append(float(words[13]))
                train_tot_loss.append(float(words[16]))
            elif 'train' in line.lower():
                train_mse_loss_unweighted.append(float(words[7]))
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
                    test_mse[temp_key]  = []
                    test_grid[temp_key] = []
                    weights[temp_key] = np.max([float(words[-2]), 1.0])
                test_wmse[temp_key].append(float(words[7]))
                test_mse[temp_key].append(float(words[10]))
                test_grid[temp_key].append(float(words[13]))
        # ==============================================================================================================
        # Plotting the results
        # ==============================================================================================================
        epoch_len = len(epoch_list)
        plt.rcParams["figure.figsize"] = (18, 9)
        # ------------------------------------------------------------------------------------------------------
        # Sensitivity plot
        # ------------------------------------------------------------------------------------------------------
        # sens_plt = plt.figure()
        # if NORM_SENS:
        #     plt.plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) * SENS_STD for x in train_mse_loss[0:epoch_len:spacing]], '-o', label=train_label)
        #     for test_db in keys_list:
        #         plt.plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) * SENS_STD for x in test_mse[test_db][0:epoch_len:spacing]], '-o', label=test_db)
        # else:
        #     plt.plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) for x in train_mse_loss[0:epoch_len:spacing]], '-o', label=train_label)
        #     for ii, test_db in enumerate(keys_list):
        #         plt.plot(epoch_list[0:epoch_len:spacing], [math.sqrt(x) for x in test_mse[test_db][0:epoch_len:spacing]], '-o', label=test_db)
        #
        # plt.xlabel('Epoch')
        # plt.ylabel('RMS Loss')
        # plt.title('RMS loss vs Epoch number')
        # plt.legend()
        # plt.grid()
        sens_plt = mse_plot(epoch_list, train_mse_loss, test_wmse, keys_list, spacing=spacing, norm=False, weights=weights)
        if unwighted_plot:
            sens_unweighted_plt = mse_plot(epoch_list, train_mse_loss_unweighted, test_mse, keys_list, spacing=spacing, norm=NORM_SENS, weights=weights)
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
        norm_grid = (GRID_MEAN, GRID_STD) if NORM_GRID else (0, 1)
        norm_sens = (SENS_MEAN, SENS_STD) if NORM_SENS else (0, 1)
        train_loader, test_loaders, thresholds = import_data_sets_pics(PATH_DATABASE_TRAIN,
                                                                       PATH_DATABASE_TEST,
                                                                       BATCH_SIZE,
                                                                       abs_sens=ABS_SENS,
                                                                       dilation=DILATION,
                                                                       norm_sens=norm_sens,
                                                                       norm_grid=norm_grid,
                                                                       num_workers=NUM_WORKERS)
        mod_vae, trainer = mff.load_state_train(chosen_file, thresholds=thresholds)
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
                temp = trainer.compute_loss(target_sens, out_sens, mu, logvar, target_grids, out_grid)
                print(f'SENS Normalized RMS over single batch --> {np.sqrt(temp[0].item() / BATCH_SIZE)}')
                # ------------------------------------------------------------------------------
                # plotting sens
                # ------------------------------------------------------------------------------
                plt.figure()
                plt.plot(target_sens.cpu().detach().numpy()*SENS_STD)
                plt.plot(out_sens.cpu().detach().numpy()*SENS_STD)
                _, ax = plt.subplots(1, 2)
                ax[0].contourf(target_grids[0, 0].cpu().detach().numpy(), cmap='gray', vmax=1, vmin=0)
                ax[0].grid()
                ax[1].contourf(sigmoid(out_grid[0, 0]).cpu().detach().numpy(), cmap='gray', vmax=1, vmin=0)
                ax[1].grid()
                plt.show()
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

    @staticmethod
    def recompute_log(path):
        from classes.LoggerVAE import LoggerVAE
        # pf = PlottingFunctions()
        # pff = PathFindingFunctions()
        mff = ModelManipulationFunctions()
        norm_grid = (GRID_MEAN, GRID_STD) if NORM_GRID else (0, 1)
        norm_sens = (SENS_MEAN, SENS_STD) if NORM_SENS else (0, 1)
        train_loader, test_loaders, thresholds = import_data_sets_pics(PATH_DATABASE_TRAIN,
                                                                       PATH_DATABASE_TEST,
                                                                       BATCH_SIZE,
                                                                       abs_sens=ABS_SENS,
                                                                       dilation=DILATION,
                                                                       norm_sens=norm_sens,
                                                                       norm_grid=norm_grid,
                                                                       num_workers=NUM_WORKERS)
        # ==============================================================================================================
        # Setting the logger
        # ==============================================================================================================
        logdir = path
        logger = LoggerVAE(logdir=logdir, filename='rework_log.txt')
        logger.start_log()
        # ==============================================================================================================
        # Getting all the save files
        # ==============================================================================================================
        save_files = [os.path.join(path, d) for d in os.listdir(path) if "epoch" in d]
        for chosen_file in save_files:
            mod_vae, trainer = mff.load_state_train(chosen_file, thresholds=thresholds)
            epoch = int(chosen_file.split('_')[-1][:-4])
            # ==========================================================================================================
            # Passing the train database
            # ==========================================================================================================
            t = time()
            # ------------------------------------------------------------------------------------------------------
            # Training a single epoch
            # ------------------------------------------------------------------------------------------------------
            train_sens_mse, train_kl_div, train_grid_mse, train_loss, _, _ = trainer.run_single_epoch(mod_vae, train_loader)
            # ------------------------------------------------------------------------------------------------------
            # Logging
            # ------------------------------------------------------------------------------------------------------
            logger.log_epoch(epoch, t)
            logger.log_epoch_results_train('train_weighted', train_sens_mse, train_kl_div, train_grid_mse, train_loss)
            with torch.no_grad():
                # --------------------------------------------------------------------------------------------------
                # Testing accuracy at the end of the epoch, and logging with LoggerVAE
                # --------------------------------------------------------------------------------------------------
                test_sens_mse_vec            = []
                test_sens_mse_vec_unweighted = []
                test_grid_mse_vec            = []
                test_counters_vec            = []
                test_costs_vec               = []
                for key in test_loaders:
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # Getting the respective group weight
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    test_mse_weight = trainer.get_test_group_weight(key)
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # Testing the results of the current group
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    test_sens_mse, test_grid_mse, test_counter, test_cost, test_sens_mse_unweighted = trainer.test_model(mod_vae, test_loaders[key])
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # Logging
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    logger.log_epoch_results_test(key, test_sens_mse, test_grid_mse, test_mse_weight, test_sens_mse_unweighted)
                    test_sens_mse_vec.append(test_sens_mse)
                    test_grid_mse_vec.append(test_grid_mse)
                    test_counters_vec.append(test_counter)
                    test_costs_vec.append(test_cost)
                    test_sens_mse_vec_unweighted.append(test_sens_mse_unweighted)
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Computing total cost for all test loaders and logging with LoggerVAE
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                test_sens_mse            = 0.0
                test_sens_mse_unweighted = 0.0
                test_grid_mse            = 0.0
                test_counter             = 0
                test_loss                = 0.0
                for sens_mse, grid_mse, count, loss, sens_mse_unweighted in zip(test_sens_mse_vec,
                                                                                test_grid_mse_vec,
                                                                                test_counters_vec, test_costs_vec,
                                                                                test_sens_mse_vec_unweighted):
                    test_sens_mse += (sens_mse * count)
                    test_sens_mse_unweighted += (sens_mse_unweighted * count)
                    test_grid_mse += (grid_mse * count)
                    test_loss += (loss * count)
                    test_counter += count
                test_sens_mse = test_sens_mse / test_counter
                test_sens_mse_unweighted = test_sens_mse_unweighted / test_counter
                test_grid_mse = test_grid_mse / test_counter
                test_loss = test_loss / test_counter
                logger.log_epoch_results_test('test_total', test_sens_mse, test_grid_mse, 0, test_sens_mse_unweighted)


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
        train_loader, test_loaders, _ = import_data_sets_coord(PATH_DATABASE_TRAIN,
                                                               PATH_DATABASE_TEST,
                                                               BATCH_SIZE,
                                                               abs_sens=ABS_SENS,
                                                               coord_mean=COORD_MEAN,
                                                               coord_scale=COORD_SCALE,
                                                               num_workers=NUM_WORKERS
                                                               )
        model, trainer = mff.load_state_train_pcloud(chosen_file)
        # ==============================================================================================================
        # Extracting statistics
        # ==============================================================================================================
        model.eval()
        with torch.no_grad():
            test_loader_iter = iter(test_loaders[key])
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
                sens_targets = sample_batched['sensitivity'].float().to(model.device)
                coordinates = sample_batched['coordinate_target'].float().to(model.device)
                # ------------------------------------------------------------------------------
                # Forward pass
                # ------------------------------------------------------------------------------
                sens_outputs = model(coordinates)
                print('hi')


class PostProcessingCNN:
    def __init__(self):
        pass

    @staticmethod
    def log_to_plot(path, spacing=1, save_plt=True):
        def plot_fig(train_loss, test_losses, title, xlabel, ylabel, div_weights=False, log_scale=False, scale=1):
            fig = plt.figure()
            if log_scale:
                if div_weights:
                    plt.semilogy(epoch_list[0:epoch_len:spacing], [scale * math.sqrt(x / train_weights[ii]) for ii, x in enumerate(train_loss[0:epoch_len:spacing])], '-o', label=train_label)
                    for test_db in keys_list:
                        plt.semilogy(epoch_list[0:epoch_len:spacing], [scale * math.sqrt(x / weights[test_db]) for x in test_losses[test_db][0:epoch_len:spacing]], '-o', label=test_db)
                else:
                    plt.semilogy(epoch_list[0:epoch_len:spacing], [scale * math.sqrt(x) for x in train_loss[0:epoch_len:spacing]], '-o', label=(train_label + '_weighted'))
                    for test_db in keys_list:
                        plt.semilogy(epoch_list[0:epoch_len:spacing], [scale * math.sqrt(x) for x in test_losses[test_db][0:epoch_len:spacing]], '-o', label=test_db)
            else:
                if div_weights:
                    plt.plot(epoch_list[0:epoch_len:spacing], [scale * math.sqrt(x / train_weights[ii]) for ii, x in enumerate(train_loss[0:epoch_len:spacing])], '-o', label=train_label)
                    for test_db in keys_list:
                        plt.plot(epoch_list[0:epoch_len:spacing], [scale * math.sqrt(x / weights[test_db]) for x in test_losses[test_db][0:epoch_len:spacing]], '-o', label=test_db)
                else:
                    plt.plot(epoch_list[0:epoch_len:spacing], [scale * math.sqrt(x) for x in train_loss[0:epoch_len:spacing]], '-o', label=(train_label + '_weighted'))
                    for test_db in keys_list:
                        plt.plot(epoch_list[0:epoch_len:spacing], [scale * math.sqrt(x) for x in test_losses[test_db][0:epoch_len:spacing]], '-o', label=test_db)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.grid()
            return fig
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        filename = os.path.join(path, 'logger_cnn.txt')
        fileID = open(filename, 'r')
        lines = fileID.readlines()
        fileID.close()

        reached_start = False
        epoch_list = []
        keys_list  = []
        weights    = {}
        test_nwmse  = {}
        test_wmse   = {}
        test_mse    = {}

        train_label = None
        train_nwmse_loss = []
        train_wmse_loss  = []
        train_mse_loss   = []
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
                train_nwmse_loss.append(float(words[6]))
                train_wmse_loss.append(float(words[8]))
                train_mse_loss.append(float(words[10]))
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
                    test_nwmse[temp_key] = []
                    test_wmse[temp_key]  = []
                    test_mse[temp_key]   = []
                    weights[temp_key]    = np.max([float(words[-2]), 1.0])
                test_nwmse[temp_key].append(float(words[6]))
                test_wmse[temp_key].append(float(words[8]))
                test_mse[temp_key].append(float(words[10]))
        # ==============================================================================================================
        # Computing train weights
        # ==============================================================================================================
        train_weights = [train_wmse_loss[ii] / train_mse_loss[ii] for ii in range(len(train_wmse_loss))]
        # ==============================================================================================================
        # Plotting the results
        # ==============================================================================================================
        epoch_len = len(epoch_list)
        plt.rcParams["figure.figsize"] = (18, 9)
        # ------------------------------------------------------------------------------------------------------
        # Sensitivity plot - weighted
        # ------------------------------------------------------------------------------------------------------
        sens_plt_nwmse = plot_fig(train_nwmse_loss, test_nwmse, title='Normalized Weighted RMS loss vs Epoch number',
                                  xlabel='Epoch', ylabel='Normalized Weighted RMS Loss', div_weights=False, log_scale=True)
        sens_plt_nmse  = plot_fig(train_nwmse_loss, test_nwmse, title='Normalized RMS loss vs Epoch number',
                                  xlabel='Epoch', ylabel='Normalized RMS Loss', div_weights=True, log_scale=True)
        # ------------------------------------------------------------------------------------------------------
        # Sensitivity plot - unweighted
        # ------------------------------------------------------------------------------------------------------
        scale = SENS_STD if NORM_SENS else 1
        sens_plt_wmse = plot_fig(train_wmse_loss, test_wmse, title='Weighted RMS loss vs Epoch number',
                                 xlabel='Epoch', ylabel='Weighted RMS Loss', div_weights=False, scale=scale)
        sens_plt_mse  = plot_fig(train_mse_loss, test_mse, title='RMS loss vs Epoch number',
                                 xlabel='Epoch', ylabel='RMS Loss', div_weights=False, scale=scale)
        # ==============================================================================================================
        # Saving
        # ==============================================================================================================
        if save_plt and (path is not None):
            # ------------------------------------------------------------------------------------------------------
            # Setting filename
            # ------------------------------------------------------------------------------------------------------
            filename_sens_nw = f'sensitivity_loss_training_normalized_weights.png'
            filename_sens_n  = f'sensitivity_loss_training_normalized.png'
            filename_sens_w  = f'sensitivity_loss_training_weighted.png'
            filename_sens    = f'sensitivity_loss_training.png'
            # ------------------------------------------------------------------------------------------------------
            # Creating directory if not exists
            # ------------------------------------------------------------------------------------------------------
            if not os.path.isdir(os.path.join(path, FIG_DIR)):
                os.makedirs(os.path.join(path, FIG_DIR))
            # ------------------------------------------------------------------------------------------------------
            # Saving
            # ------------------------------------------------------------------------------------------------------
            sens_plt_nwmse.savefig(os.path.join(path, FIG_DIR, filename_sens_nw))
            sens_plt_nmse.savefig(os.path.join(path, FIG_DIR, filename_sens_n))
            sens_plt_wmse.savefig(os.path.join(path, FIG_DIR, filename_sens_w))
            sens_plt_mse.savefig(os.path.join(path, FIG_DIR, filename_sens))
        plt.show()

    @staticmethod
    def load_and_pass(path, epoch, key='4e+03_to_inf'):
        pf = PlottingFunctions()
        pff = PathFindingFunctions()
        mff = ModelManipulationFunctions()
        # ==============================================================================================================
        # Extracting the full file path
        # ==============================================================================================================
        chosen_file = pff.get_full_path(path, epoch)
        # ==============================================================================================================
        # Loading the needed models and data
        # ==============================================================================================================
        norm_grid = (GRID_MEAN, GRID_STD) if NORM_GRID else (0, 1)
        norm_sens = (SENS_MEAN, SENS_STD) if NORM_SENS else (0, 1)
        train_loader, test_loaders, _ = import_data_sets_pics(PATH_DATABASE_TRAIN,
                                                              PATH_DATABASE_TEST,
                                                              BATCH_SIZE,
                                                              abs_sens=ABS_SENS,
                                                              dilation=DILATION,
                                                              norm_grid=norm_grid,
                                                              norm_sens=norm_sens,
                                                              num_workers=NUM_WORKERS)
        model, trainer = mff.load_state_train_cnn(chosen_file)
        # ==============================================================================================================
        # Extracting statistics
        # ==============================================================================================================
        model.eval()
        with torch.no_grad():
            test_loader_iter = iter(test_loaders[key])
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
                sens_targets = sample_batched['sensitivity'].float().to(model.device)
                grids        = Variable(sample_batched['grid_in'].float()).to(model.device)
                # ------------------------------------------------------------------------------
                # Forward pass
                # ------------------------------------------------------------------------------
                sens_outputs = model(grids)
                plt.figure()
                scale = SENS_STD if NORM_SENS else 1
                plt.plot(sens_targets.cpu().detach().numpy() * scale)
                plt.plot(sens_outputs.cpu().detach().numpy() * scale)
                print('hi')


class PostProcessingOmega:

    @staticmethod
    def log_to_plot(path, spacing=1, save_plt=True, scale=1):
        def plot_fig(train_loss, test_loss, title, xlabel, ylabel, log_scale=False, rms_scale=1):
            fig = plt.figure()
            if log_scale:
                plt.semilogy(epoch_list[0:epoch_len:spacing], [rms_scale * math.sqrt(x) for x in train_loss[0:epoch_len:spacing]], '-o', label=train_label)
                plt.semilogy(epoch_list[0:epoch_len:spacing], [rms_scale * math.sqrt(x) for x in test_loss[0:epoch_len:spacing]], '-o',  label=test_label)
            else:
                plt.plot(epoch_list[0:epoch_len:spacing], [rms_scale * math.sqrt(x) for x in train_loss[0:epoch_len:spacing]], '-o', label=train_label)
                plt.plot(epoch_list[0:epoch_len:spacing], [rms_scale * math.sqrt(x) for x in test_loss[0:epoch_len:spacing]], '-o', label=test_label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.grid()
            return fig

        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        filename = os.path.join(path, 'logger_omega.txt')
        fileID = open(filename, 'r')
        lines = fileID.readlines()
        fileID.close()

        reached_start   = False
        epoch_list      = []
        test_label      = None
        test_mse_loss   = []
        test_nmse_loss  = []
        train_label     = None
        train_mse_loss  = []
        train_nmse_loss = []
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
                train_mse_loss.append(float(words[6]))
                train_nmse_loss.append(float(words[8]))
            elif 'test' in line.lower():
                if test_label is None:
                    test_label = words[3]
                test_mse_loss.append(float(words[6]))
                test_nmse_loss.append(float(words[8]))
        # ==============================================================================================================
        # Plotting the results
        # ==============================================================================================================
        epoch_len = len(epoch_list)
        plt.rcParams["figure.figsize"] = (18, 9)
        # ------------------------------------------------------------------------------------------------------
        # RMS plot
        # ------------------------------------------------------------------------------------------------------
        plt_rms = plot_fig(train_mse_loss, test_mse_loss,
                           title='RMS Vs Epoch',
                           xlabel='Epoch',
                           ylabel='RMS',
                           log_scale=True,
                           rms_scale=scale)
        plt_nrms = plot_fig(train_nmse_loss, test_nmse_loss,
                           title='NRMS Vs Epoch',
                           xlabel='Epoch',
                           ylabel='NRMS',
                           log_scale=True,
                           rms_scale=1)
        # ==============================================================================================================
        # Saving
        # ==============================================================================================================
        if save_plt and (path is not None):
            # ------------------------------------------------------------------------------------------------------
            # Setting filename
            # ------------------------------------------------------------------------------------------------------
            filename_rms  = f'rms_loss.png'
            filename_nrms = f'nrms_loss.png'
            # ------------------------------------------------------------------------------------------------------
            # Creating directory if not exists
            # ------------------------------------------------------------------------------------------------------
            if not os.path.isdir(os.path.join(path, FIG_DIR)):
                os.makedirs(os.path.join(path, FIG_DIR))
            # ------------------------------------------------------------------------------------------------------
            # Saving
            # ------------------------------------------------------------------------------------------------------
            plt_rms.savefig(os.path.join(path, FIG_DIR, filename_rms))
            plt_nrms.savefig(os.path.join(path, FIG_DIR, filename_nrms))
        plt.show()

    @staticmethod
    def load_and_pass(path, epoch, spacing=1, save_plt=False):
        pf = PlottingFunctions()
        pff = PathFindingFunctions()
        mff = ModelManipulationFunctions()
        # ==============================================================================================================
        # Extracting the full file path
        # ==============================================================================================================
        chosen_file = pff.get_full_path(path, epoch)
        # ==============================================================================================================
        # Loading the needed models and data
        # ==============================================================================================================
        train_loader = import_pcurrents_dataset(BATCH_SIZE, PATH_DATABASE_TRAIN,
                                                omega_factor=OMEGA_FACTOR,
                                                shot_noise=SHOT_NOISE,
                                                sampling_rate=SAMPLING_RATE,
                                                shuffle=False,
                                                num_workers=1)
        # loader_list = [train_loader, test_loader]
        loader_list = [train_loader]
        model, trainer = mff.load_state_train_omega(chosen_file)
        print('Loaded model, trainer and datasets')
        # ==============================================================================================================
        # Extracting statistics
        # ==============================================================================================================
        omega_outputs = torch.tensor([]).to(model.device)
        omega_targets = torch.tensor([]).to(model.device)
        model.eval()
        with torch.no_grad():
            for loader in loader_list:
                loader_iter = iter(loader)
                for ii in range(len(loader)):
                    # ------------------------------------------------------------------------------
                    # Working with iterables, much faster
                    # ------------------------------------------------------------------------------
                    try:
                        sample_batched = next(loader_iter)
                    except StopIteration:
                        break
                    # ------------------------------------------------------------------------------
                    # Extracting the grids and sensitivities, passing through the model
                    # ------------------------------------------------------------------------------
                    omega_targets_batch = sample_batched['omega'].float().to(model.device)
                    pcurrents           = Variable(sample_batched['pcurrents'].float()).to(model.device)
                    # ------------------------------------------------------------------------------
                    # Forward pass + Docking
                    # ------------------------------------------------------------------------------
                    omega_outputs_batch = model(pcurrents)
                    omega_outputs       = torch.cat((omega_outputs, omega_outputs_batch))
                    omega_targets       = torch.cat((omega_targets, omega_targets_batch))
        print('Finished passing through the datasets')
        # ==============================================================================================================
        # Plotting the results
        # ==============================================================================================================
        plt.rcParams["figure.figsize"] = (18, 9)
        omega_targets = omega_targets.cpu().detach().numpy() / trainer.omega_factor
        omega_outputs = omega_outputs.cpu().detach().numpy() / trainer.omega_factor
        error         = np.abs(omega_targets - omega_outputs)

        omega_plot = plt.figure()
        plt.plot(omega_targets[0:len(omega_targets):spacing], omega_targets[0:len(omega_targets):spacing], '.', label='target')
        plt.plot(omega_targets[0:len(omega_targets):spacing], omega_outputs[0:len(omega_outputs):spacing], '.', label='result')
        plt.title('Omega results - Train & test')
        plt.xlabel('Omega, normalized')
        plt.ylabel('Omega, normalized')
        plt.legend()
        plt.grid()

        error_plot = plt.figure()
        plt.semilogy(omega_targets[0:len(omega_targets):spacing], error[0:len(error):spacing], '.')
        # plt.plot(omega_targets[0:len(omega_targets):spacing], 20 * np.log10(error[0:len(error):spacing]), '.')
        plt.title('Omega results - Train & test')
        plt.xlabel('Omega, normalized')
        # plt.ylabel('Error [dB]')
        plt.ylabel('Error')
        plt.grid()

        if save_plt and (path is not None):
            # ------------------------------------------------------------------------------------------------------
            # Setting filename
            # ------------------------------------------------------------------------------------------------------
            filename_omega = f'omega_inference.png'
            filename_error = f'omega_error.png'
            # ------------------------------------------------------------------------------------------------------
            # Creating directory if not exists
            # ------------------------------------------------------------------------------------------------------
            if not os.path.isdir(os.path.join(path, FIG_DIR)):
                os.makedirs(os.path.join(path, FIG_DIR))
            # ------------------------------------------------------------------------------------------------------
            # Saving
            # ------------------------------------------------------------------------------------------------------
            omega_plot.savefig(os.path.join(path, FIG_DIR, filename_omega))
            error_plot.savefig(os.path.join(path, FIG_DIR, filename_error))
        plt.show()

    @staticmethod
    def generate_sens_pdf(db_path, run_path, epoch, omegas=None, save_plt=False):
        import pandas as pd
        pff = PathFindingFunctions()
        mff = ModelManipulationFunctions()
        elec_amp_per_sec = 6.24e18
        pdf_count        = 100000
        # ==============================================================================================================
        # Extracting the full file path
        # ==============================================================================================================
        chosen_file = pff.get_full_path(run_path, epoch)
        model, trainer = mff.load_state_train_omega(chosen_file)
        sampling_rate = trainer.sampling_rate  # (100 * trainer.sampling_rate)  # TODO: remove the 100
        curr_per_part = 1e12 * 1 / elec_amp_per_sec * sampling_rate
        # ==============================================================================================================
        # Extracting the most sensitive slope and 0 current
        # ==============================================================================================================
        csv_data = pd.read_csv(db_path)
        omega_0_row = len(csv_data) // 2
        omega_1p_row = omega_0_row + 1
        omega_1n_row = omega_0_row - 1
        omega_0  = csv_data.iloc[omega_0_row, 0]
        omega_1p = csv_data.iloc[omega_1p_row, 0]
        omega_1n = csv_data.iloc[omega_1n_row, 0]
        omega_0_current  = csv_data.iloc[omega_0_row, [1, 2]].to_numpy()
        omega_1p_current = csv_data.iloc[omega_1p_row, [1, 2]].to_numpy()
        omega_1n_current = csv_data.iloc[omega_1n_row, [1, 2]].to_numpy()
        omega_0_current_abs  = np.sqrt(omega_0_current[0] ** 2 + omega_0_current[1] ** 2)
        omega_1p_current_abs = np.sqrt(omega_1p_current[0] ** 2 + omega_1p_current[1] ** 2)
        omega_1n_current_abs = np.sqrt(omega_1n_current[0] ** 2 + omega_1n_current[1] ** 2)
        omega_0_slope        = (omega_1p_current_abs - omega_1n_current_abs) / (omega_1p - omega_1n)
        # ==============================================================================================================
        # Creating the single scatterer pdf
        # ==============================================================================================================
        # Generating noised current in abs value - assuming same phase (no phase noise atm)
        n_of_particles         = elec_amp_per_sec * omega_0_current_abs * 1e-12 / sampling_rate
        n_of_particles_snoised = np.random.poisson(n_of_particles, pdf_count)
        noised_current_abs     = n_of_particles_snoised * curr_per_part
        sensitivity_estimator  = (noised_current_abs - omega_0_current_abs) / omega_0_slope
        sens_est_std           = np.std(sensitivity_estimator)
        # ==============================================================================================================
        # Plotting sens estimator
        # ==============================================================================================================
        plt.rcParams["figure.figsize"] = (18, 9)
        sensitivity_estimator_fig = plt.figure()
        plt.hist(sensitivity_estimator, 50, density=True)
        plt.xlabel('Omega, normalized')
        plt.title('Sensitivity Estimator PDF, at Omega = {0:.3e} ; std = {1:.3e}'.format(omega_0, sens_est_std))
        plt.grid()
        if save_plt:
            # ------------------------------------------------------------------------------------------------------
            # Setting filename
            # ------------------------------------------------------------------------------------------------------
            sensitivity_pdf = f'sinsitivity_pdf.png'
            # ------------------------------------------------------------------------------------------------------
            # Creating directory if not exists
            # ------------------------------------------------------------------------------------------------------
            if not os.path.isdir(os.path.join(run_path, FIG_DIR)):
                os.makedirs(os.path.join(run_path, FIG_DIR))
            # ------------------------------------------------------------------------------------------------------
            # Saving
            # ------------------------------------------------------------------------------------------------------
            sensitivity_estimator_fig.savefig(os.path.join(run_path, FIG_DIR, sensitivity_pdf))
        plt.show()

    @staticmethod
    def generate_multi_sens_pdf(db_path, run_path, epoch, omegas=None, save_plt=False):
        import pandas as pd
        pff = PathFindingFunctions()
        mff = ModelManipulationFunctions()
        elec_amp_per_sec = 6.24e18
        pdf_count        = 100000
        # ==============================================================================================================
        # Extracting the full file path
        # ==============================================================================================================
        chosen_file = pff.get_full_path(run_path, epoch)
        model, trainer = mff.load_state_train_omega(chosen_file)
        sampling_rate  = trainer.sampling_rate
        curr_per_part  = 1e12 * 1 / elec_amp_per_sec * sampling_rate
        # ==============================================================================================================
        # Extracting the most sensitive slope and 0 current
        # ==============================================================================================================
        csv_data = pd.read_csv(db_path)
        omega_0_row  = len(csv_data) // 2
        omega_1p_row = omega_0_row + 1
        omega_1n_row = omega_0_row - 1
        omega_0  = csv_data.iloc[omega_0_row, 0]
        omega_1p = csv_data.iloc[omega_1p_row, 0]
        omega_1n = csv_data.iloc[omega_1n_row, 0]
        omega_0_current  = csv_data.iloc[omega_0_row, 1:].to_numpy()
        omega_1p_current = csv_data.iloc[omega_1p_row, 1:].to_numpy()
        omega_1n_current = csv_data.iloc[omega_1n_row, 1:].to_numpy()
        # ----------------------------------------------------------------------------------------------------------
        # Creating the currents in abs value
        # ----------------------------------------------------------------------------------------------------------
        omega_0_current       = np.reshape(omega_0_current, [-1, 2])
        omega_0_current_comp  = omega_0_current[:, 0] + 1j * omega_0_current[:, 1]
        omega_1p_current      = np.reshape(omega_1p_current, [-1, 2])
        omega_1p_current_comp = omega_1p_current[:, 0] + 1j * omega_1p_current[:, 1]
        omega_1n_current      = np.reshape(omega_1n_current, [-1, 2])
        omega_1n_current_comp = omega_1n_current[:, 0] + 1j * omega_1n_current[:, 1]
        pcurr_comp     = np.array([omega_1n_current_comp, omega_0_current_comp, omega_1p_current_comp]).T
        pcurr_abs      = np.abs(pcurr_comp / pcurr_comp[:, 1][:, None])
        omega_0_slopes = (pcurr_abs[:, 2] - pcurr_abs[:, 0]) / (omega_1p - omega_1n)
        omega_mat      = np.array([omega_0_slopes, np.abs(omega_0_current_comp)])
        # ==============================================================================================================
        # Creating the multi- scatterer pdf
        # ==============================================================================================================
        # Generating noised current in abs value - assuming same phase (no phase noise atm)
        n_of_particles = elec_amp_per_sec * np.abs(omega_0_current_comp) * 1e-12 / sampling_rate
        # ----------------------------------------------------------------------------------------------------------
        # Generating the noised samples
        # ----------------------------------------------------------------------------------------------------------
        noised_current_abs = np.zeros([len(n_of_particles), pdf_count])
        for ii in range(len(n_of_particles)):
            n_of_particle_ii_noised = np.random.poisson(n_of_particles[ii], pdf_count)
            noised_current_abs[ii]  = n_of_particle_ii_noised * curr_per_part
        # ----------------------------------------------------------------------------------------------------------
        # Created omega_mat, where the 1st col are the slopes, the 2nd col are the intersections, trying to estimate
        # a vector with length 2: [Omega, 1]
        # ----------------------------------------------------------------------------------------------------------
        estimator, _, _, _ = np.linalg.lstsq(omega_mat.T, noised_current_abs, rcond=-1)  # vector of 2, the first is the wanted omega, the second is the constant 1
        sensitivity_estimator = estimator[0, :] / estimator[1, :]
        sens_est_std          = np.std(sensitivity_estimator)
        # ==============================================================================================================
        # Plotting sens estimator
        # ==============================================================================================================
        plt.rcParams["figure.figsize"] = (18, 9)
        sensitivity_estimator_fig = plt.figure()
        plt.hist(sensitivity_estimator, 100, density=True)
        plt.xlabel('Omega, normalized')
        plt.title('Multi-Sensitivity Estimator PDF, at Omega = {0:.3e} ; std = {1:.3e}'.format(omega_0, sens_est_std))
        plt.grid()
        if save_plt:
            # ------------------------------------------------------------------------------------------------------
            # Setting filename
            # ------------------------------------------------------------------------------------------------------
            sensitivity_pdf = f'multi_sens_pdf.png'
            # ------------------------------------------------------------------------------------------------------
            # Creating directory if not exists
            # ------------------------------------------------------------------------------------------------------
            if not os.path.isdir(os.path.join(run_path, FIG_DIR)):
                os.makedirs(os.path.join(run_path, FIG_DIR))
            # ------------------------------------------------------------------------------------------------------
            # Saving
            # ------------------------------------------------------------------------------------------------------
            sensitivity_estimator_fig.savefig(os.path.join(run_path, FIG_DIR, sensitivity_pdf))
        plt.show()

    @staticmethod
    def generate_nn_pdf(db_path, run_path,  epoch, omega=0, save_plt=False, wanted_sampling_rate=100):
        pff = PathFindingFunctions()
        mff = ModelManipulationFunctions()
        elec_amp_per_sec = 6.24e18
        pdf_count        = 100000
        # ==============================================================================================================
        # Extracting the full file path
        # ==============================================================================================================
        chosen_file    = pff.get_full_path(run_path, epoch)
        model, trainer = mff.load_state_train_omega(chosen_file)
        sampling_rate  = trainer.sampling_rate  # TODO: remove the 100
        averaging_num  = int(sampling_rate / wanted_sampling_rate)
        # ==============================================================================================================
        # Creating the NN PDF
        # ==============================================================================================================
        data_loader = import_pcurrent_omega_dataset(100, db_path, omega, int(pdf_count*averaging_num), OMEGA_FACTOR,
                                                    shot_noise=True,
                                                    sampling_rate=sampling_rate,
                                                    num_workers=NUM_WORKERS)
        loader_iter = iter(data_loader)
        nn_estimator = []
        model.eval()
        with torch.no_grad():
            for _ in range(len(data_loader)):
                # ------------------------------------------------------------------------------
                # Working with iterables, much faster
                # ------------------------------------------------------------------------------
                try:
                    sample = next(loader_iter)
                except StopIteration:
                    break
                # ------------------------------------------------------------------------------
                # Extracting the grids and sensitivities
                # ------------------------------------------------------------------------------
                # omega_targets = sample['omega'].float().to(model.device)
                pcurrents = Variable(sample['pcurrents'].float()).to(model.device)
                # ------------------------------------------------------------------------------
                # Forward pass
                # ------------------------------------------------------------------------------
                omega_outputs = model(pcurrents)
                nn_estimator.append(omega_outputs.T.cpu().detach().numpy().tolist()[0])
        nn_estimator = np.reshape(np.array(nn_estimator), [-1, averaging_num]) / trainer.omega_factor
        nn_estimator = np.mean(nn_estimator, axis=1)
        nn_est_std   = np.std(nn_estimator)
        plt.rcParams["figure.figsize"] = (18, 9)
        nn_estimator_fig = plt.figure()
        plt.hist(nn_estimator, 100, density=True)
        plt.xlabel('Omega, normalized')
        plt.title('NN Estimator PDF, at Omega = {0:.3e} ; std = {1:.3e}'.format(omega, nn_est_std))
        plt.grid()
        if save_plt:
            # ------------------------------------------------------------------------------------------------------
            # Setting filename
            # ------------------------------------------------------------------------------------------------------
            sinsitivity_pdf = f'nn_pdf.png'
            # ------------------------------------------------------------------------------------------------------
            # Creating directory if not exists
            # ------------------------------------------------------------------------------------------------------
            if not os.path.isdir(os.path.join(run_path, FIG_DIR)):
                os.makedirs(os.path.join(run_path, FIG_DIR))
            # ------------------------------------------------------------------------------------------------------
            # Saving
            # ------------------------------------------------------------------------------------------------------
            nn_estimator_fig.savefig(os.path.join(run_path, FIG_DIR, sinsitivity_pdf))
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
    # from ConfigVAE import *
    # from ConfigDG import *
    # from ConfigCNN import *
    from config_omega import *
    c_epoch = 300
    # c_path = '..\\results\\16_1_2022_21_39'
    # c_path = '..\\results\\10_2_2022_16_45'
    # c_path = '..\\results\\10_2_2022_16_45_plus_13_2_2022_21_4'
    # c_path = '..\\results\\5_4_2022_10_27'
    # c_path = '..\\results\\13_4_2022_22_58'
    # c_path = '..\\results\\15_5_2022_17_9'
    # c_path = '..\\results\\6_6_2022_19_7'
    # c_path = '..\\results\\14_6_2022_16_8'
    # c_path = '..\\results_vae\\20_9_2022_10_39'
    # c_path = '..\\results_vae\\25_8_2022_15_6'
    # c_path = '..\\results_vae\\5_8_2022_8_41'
    c_path_vae   = '..\\results_vae\\13_10_2022_11_14'
    c_path_dg    = '..\\results_dg\\16_12_2022_14_6'
    c_path_cnn   = '..\\results_cnn\\18_12_2022_9_16'
    # c_path_omega = '..\\results_omega\\7_1_2023_22_51'
    c_path_omega = '..\\results_omega\\6_1_2023_8_21'

    pp_vae = PostProcessingVAE()
    pp_dg  = PostProcessingDG()
    pp_cnn = PostProcessingCNN()
    pp_omega = PostProcessingOmega()

    threshold_list = [0.1, 0.2, 0.5]
    # prefix_list    = ['3e+05_to_inf', '2e+05_to_3e+05', '1e+05_to_2e+05', '0_to_1e+05']
    # prefix_list    = ['1e+05_to_2e+05']
    # prefix_list    = ['4e+03_to_inf', '0_to_2e+03']
    # pp_vae.load_data_plot_roc_det(c_path, c_epoch, prefix_list)
    # pp_vae.load_model_compare_blobs(c_path, c_epoch, key='2e+05_to_3e+05', peak_threshold=3.3)
    # pp_vae.load_model_plot_roc_det(c_path, c_epoch, key='0_to_1e+05')
    # pp_vae.log_to_plot(c_path, spacing=10)
    # pp_vae.get_latent_statistics(c_path, c_epoch)
    # pp_vae.load_data_plot_latent_statistics(c_path, c_epoch, prefix_list)

    # pp.recompute_log(c_path)
    # pp.load_and_pass(c_path, c_epoch, key=prefix_list[0])
    # pp_vae.log_to_plot(c_path, plt_joined=False, spacing=10)

    # pp_dg.log_to_plot(c_path_dg, spacing=1)
    # pp_dg.load_and_pass(c_path2, c_epoch)
    # prefix_list    = ['3e+02_to_inf', '2e+02_to_3e+02', '0_to_1e+02']

    # pp_cnn.log_to_plot(c_path_cnn, spacing=1)
    # pp_cnn.load_and_pass(c_path_cnn, c_epoch, key=prefix_list[0])

    print('hi')
    # pp_omega.log_to_plot(c_path_omega, spacing=1, scale=1/OMEGA_FACTOR)
    # pp_omega.load_and_pass(c_path_omega, c_epoch, spacing=100, save_plt=True)
    # pp_omega.generate_sens_pdf(PATH_DATABASE, c_path_omega, c_epoch, save_plt=True)
    pp_omega.generate_multi_sens_pdf(PATH_DATABASE, c_path_omega, c_epoch, save_plt=True)
    # pp_omega.generate_nn_pdf(PATH_DATABASE, c_path_omega, c_epoch, save_plt=True)
