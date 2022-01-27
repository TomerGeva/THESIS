import numpy as np
from ConfigVAE import *
import os
import json
import math
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from ScatterCoordinateDataset import import_data_sets
from database_functions import PathFindingFunctions, ModelManipulationFunctions
from auxiliary_functions import PlottingFunctions


class PostProcessing:
    def __init__(self):
        pass

    @staticmethod
    def log_to_plot(path):
        """
        :param path: path to a result folder
        :return: the function reads the log and creates a plot of RMS loss, with all the test databases documented
        """
        # ====================================================================================================
        # Local variables
        # ====================================================================================================
        filename    = os.path.join(path, 'logger_vae.txt')
        fileID      = open(filename, 'r')
        lines       = fileID.readlines()
        fileID.close()

        reached_start  = False
        epoch_list     = []
        keys_list      = []
        test_results   = {}

        train_label    = None
        train_mse_loss = []
        train_dkl_loss = []
        train_tot_loss = []
        # ====================================================================================================
        # Going over lines, adding to log
        # ====================================================================================================
        for line in lines:
            # --------------------------------------------------------------------------------------------
            # Getting to beginning of training
            # --------------------------------------------------------------------------------------------
            if not reached_start and 'Beginning Training' not in line:
                continue
            elif not reached_start:
                reached_start = True
                continue
            # --------------------------------------------------------------------------------------------
            # Reached beginning, going over cases
            # --------------------------------------------------------------------------------------------
            words = list(filter(None, line.split(sep=' ')))
            if 'Epoch' in line:
                epoch_list.append(int(words[-1]))
            elif 'train' in line.lower():
                if train_label is None:
                    train_label = words[3]
                train_mse_loss.append(float(words[7]))
                train_dkl_loss.append(float(words[9]))
                train_tot_loss.append(float(words[16]))
            elif 'MSE' in line:
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # one of the test databases
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                temp_key = words[3]
                if temp_key in keys_list:
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # if key already exists, appends the result
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    test_results[temp_key].append(float(words[10]))
                else:
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # if key does not exist, creates a new list
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    keys_list.append(temp_key)
                    test_results[temp_key] = [float(words[10])]

        # ====================================================================================================
        # Plotting the results
        # ====================================================================================================
        plt.plot(epoch_list, [math.sqrt(x) * SENS_STD for x in train_mse_loss], '-o', label=train_label)
        for test_db in keys_list:
            if '15_22' in path:
                plt.plot(epoch_list[0:-1], [math.sqrt(x) * SENS_STD for x in test_results[test_db]], '-o', label=test_db)
            else:
                plt.plot(epoch_list, [math.sqrt(x) * SENS_STD for x in test_results[test_db]], '-o', label=test_db)
        plt.xlabel('Epoch')
        plt.ylabel('RMS Loss')
        plt.title('RMS loss vs Epoch number')
        plt.legend()
        plt.grid()
        plt.show()

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
        moc = ModelOutputComputation()
        # ==============================================================================================================
        # Extracting the full file path
        # ==============================================================================================================
        chosen_file = pff.get_full_path(path, epoch)
        # ==============================================================================================================
        # Loading the needed models and data
        # ==============================================================================================================
        _, test_loaders, _ = import_data_sets(BATCH_SIZE,
                                              mixup_factor=MIXUP_FACTOR,
                                              mixup_prob=MIXUP_PROB,
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
        tpr, fpr, fnr = moc.get_roc_det_curve(mod_vae, test_loader, threshold_num=200)
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
            'false_negative_rate': fnr
        }]
        # ----------------------------------------------------------------------------------------------------------
        # Saving
        # ----------------------------------------------------------------------------------------------------------
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_output_data, f, indent=4)
        # ==============================================================================================================
        # Plotting
        # ==============================================================================================================
        pf.plot_roc_curve(fpr, tpr, save_plt=True, path=path, epoch=epoch, name_prefix=key)
        pf.plot_det_curve(fpr, fnr, save_plt=True, path=path, epoch=epoch, name_prefix=key)

    @staticmethod
    def get_latent_statistics(path, epoch):
        """
        :param path: path to a model training results folder
        :param epoch: wanted epoch to load
        :return: the function prints out plot of the statistics regarding the latent space
        """
        pf  = PlottingFunctions()
        pff = PathFindingFunctions()
        mff = ModelManipulationFunctions()
        sigmoid = torch.nn.Sigmoid()
        # ======================================================================================
        # Extracting the full file path
        # ======================================================================================
        chosen_file = pff. get_full_path(path, epoch)
        # ======================================================================================
        # Loading the needed models and data
        # ======================================================================================
        train_loader, test_loaders, _ = import_data_sets(BATCH_SIZE, dilation=DILATION)
        mod_vae, trainer = mff.load_state_train(chosen_file)

        # ======================================================================================
        # Extracting statistics
        # ======================================================================================
        test_loader_iter = iter(test_loaders['3e+05_to_inf'])
        # test_loader_iter = iter(test_loaders['2e+05_to_3e+05'])
        # test_loader_iter = iter(test_loaders['1e+05_to_2e+05'])
        # test_loader_iter = iter(test_loaders['0_to_1e+05'])
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

                mu_temp = mu.cpu().detach().numpy()
                var_temp = np.exp(logvar.cpu().detach().numpy())
                target = sensitivities.cpu().detach().numpy()
                output = outputs.cpu().detach().numpy()
                pf.plot_latent(mu_temp, var_temp, target, output)
                # for jj in range(20):
                #     mu_temp     = mu[jj, :].cpu().detach().numpy()
                #     var_temp    = np.exp(logvar[jj, :].cpu().detach().numpy())
                #     target      = sensitivities[jj, :].cpu().detach().numpy()
                #     output      = outputs[jj, :].cpu().detach().numpy()
                #     plot_latent(mu_temp, var_temp, target, output)

        # ======================================================================================
        # Plotting statistics
        # ======================================================================================
        mu_dim = np.mean(mu_means, axis=1)
        std_dim = np.mean(std_means, axis=1)
        plt.figure()
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
        plt.show()
        pass

    @staticmethod
    def load_data_plot_roc_det(path, epoch, prefix_list):
        """
        :param path: Path to the saved data
        :param epoch: the epoch in question
        :param prefix_list: self explanatory
        :return: The function:
                    1. loads the data in the given path according to the given prefixes
                    2. plots the ROC and DET curves
        """
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
        pf.plot_roc_curve(main_dict, name_prefixes=prefix_list, save_plt=True, path=path, epoch=epoch)
        pf.plot_det_curve(main_dict, name_prefixes=prefix_list, save_plt=True, path=path, epoch=epoch)


class ModelOutputComputation:
    def __init__(self):
        pass

    @staticmethod
    def slice_batch(samples, threshold):
        """
        :param samples: 4-D torch tensor of model output grids
        :param threshold: decision threshold, should be between 0 and 1
        :return: converts the 4-D tensor to 3-D numpy matrix (grids have 1 channel) and slices the decision at the
                 threshold
        """
        sigmoid = torch.nn.Sigmoid()
        return np.where(np.squeeze(sigmoid(samples[:, 0, :, :]).cpu().detach().numpy()) >= threshold, 1, 0)

    @staticmethod
    def get_tpr_fpr_fnr(sample, target):
        """
        :param sample: a sample 3-D matrix of scatterer grids after slicing at a certain threshold
        :param target: target grid
        :return: function computed the true positive ratio (TPR), false positive ratio (FPR) and false negative ratio
                 (FNR) and returns them
                                TP                              FP                              FN
                    TPR = -------------             FPR = -------------             FNR = ---------------
                           TP   +   FN                     FP   +   TN                      FN   +  TP
        """
        tpr = np.sum(sample[target == 1]) / np.sum(target)
        fpr = np.sum(sample[target == 0]) / np.sum(1 - target)
        fnr = 1 - tpr

        return tpr, fpr, fnr

    def get_roc_det_curve(self, model, loader, threshold_num=20):
        """
        :param model: trained model
        :param loader: dataloader to use
        :param threshold_num: number of threshold to compute the ROC curve with
        :return: true positive and false positive vectors used to plot the ROC curve
        """
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        ratio_tp = np.zeros([threshold_num + 1, ])
        ratio_fp = np.zeros([threshold_num + 1, ])
        ratio_fn = np.zeros([threshold_num + 1, ])
        thresholds = [ii / threshold_num for ii in list(range(threshold_num+1))]
        counter = 0
        # ==============================================================================================================
        # No grad for speed
        # ==============================================================================================================
        model.eval()
        with torch.no_grad():
            loader_iter = iter(loader)
            for _ in range(len(loader)):
                # ------------------------------------------------------------------------------
                # Working with iterables, much faster
                # ------------------------------------------------------------------------------
                try:
                    sample = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    sample = next(loader_iter)
                # ------------------------------------------------------------------------------
                # Extracting the grids and sensitivities
                # ------------------------------------------------------------------------------
                grids           = Variable(sample['grid_in'].float()).to(model.device)
                grid_targets    = np.squeeze(sample['grid_target'].detach().numpy()).astype(int)
                batch_size      = sample['sensitivity'].shape[0]
                counter         += batch_size
                # ------------------------------------------------------------------------------
                # Forward pass
                # ------------------------------------------------------------------------------
                grid_out, _, _, _ = model(grids)
                # ------------------------------------------------------------------------------
                # Slicing and getting TP, FP for each threshold
                # ------------------------------------------------------------------------------
                for (ii, threshold) in enumerate(thresholds):
                    samples       = self.slice_batch(grid_out, threshold)
                    tpr, fpr, fnr = self.get_tpr_fpr_fnr(samples, grid_targets)
                    ratio_tp[ii] += tpr * batch_size
                    ratio_fp[ii] += fpr * batch_size
                    ratio_fn[ii] += fnr * batch_size
        # ==============================================================================================================
        # Normalizing
        # ==============================================================================================================
        ratio_tp = [ii / counter for ii in ratio_tp]
        ratio_fp = [ii / counter for ii in ratio_fp]
        ratio_fn = [ii / counter for ii in ratio_fn]
        return ratio_tp, ratio_fp, ratio_fn


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
    c_path = '..\\results\\16_1_2022_21_39'
    c_epoch = 700
    pp = PostProcessing()
    prefix_list = ['3e+05_to_inf', '2e+05_to_3e+05', '1e+05_to_2e+05', '0_to_1e+05']
    # prefix_list = ['0_to_1e+05']
    pp.load_data_plot_roc_det(c_path, c_epoch, prefix_list)

    # pp.load_model_plot_roc_det(c_path, c_epoch)
    # pp.log_to_plot(c_path)
    # pp.get_latent_statistics(c_path, c_epoch)
    # pp.log_to_plot(c_path)


