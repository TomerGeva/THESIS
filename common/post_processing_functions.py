from ConfigVAE import *
import os
import math
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from ScatterCoordinateDataset import import_data_sets
from database_functions import load_state_train
from auxiliary_functions import get_full_path, plot_latent


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


def load_and_batch(path, epoch):
    """
    :return: This function loads a saves model, and tests the MSE of the target error
    """
    # ======================================================================================
    # Extracting the full file path
    # ======================================================================================
    chosen_file = get_full_path(path, epoch)
    # ======================================================================================
    # Loading the needed models and data
    # ======================================================================================
    train_loader, test_loaders, _   = import_data_sets(BATCH_SIZE)
    mod_vae, trainer                = load_state_train(chosen_file)

    smapled_batch   = next(iter(test_loaders['3e+05_to_inf']))
    grids           = Variable(smapled_batch['grid_in'].float()).to(mod_vae.device)
    sensitivities   = Variable(smapled_batch['sensitivity'].float()).to(mod_vae.device)

    mod_vae.eval()
    _, outputs, mu, logvar = mod_vae(grids)
    print('Outputs: ' + str(outputs))
    print('Targets: ' + str(sensitivities))


def get_latent_statistics(path, epoch):
    """
    :param path: path to a model training results folder
    :param epoch: wanted epoch to load
    :return: the function prints out plot of the statistics regarding the latent space
    """
    sigmoid = torch.nn.Sigmoid()
    # ======================================================================================
    # Extracting the full file path
    # ======================================================================================
    chosen_file = get_full_path(path, epoch)
    # ======================================================================================
    # Loading the needed models and data
    # ======================================================================================
    train_loader, test_loaders, _ = import_data_sets(BATCH_SIZE, dilation=DILATION)
    mod_vae, trainer = load_state_train(chosen_file)

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
        grids = Variable(sample_batched['grid_in'].float()).to(mod_vae.device)
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
        plot_latent(mu_temp, var_temp, target, output)
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


def plot_grid_histogram(grid, bins=10):
    plt.hist(np.array(grid).ravel(), bins=bins, density=True)


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
    c_path = '..\\results\\2_1_2022_7_50'
    c_epoch = 160

    # log_to_plot(c_path)

    get_latent_statistics(c_path, c_epoch)

    log_to_plot(c_path)

    # load_and_batch(c_path, c_epoch)

