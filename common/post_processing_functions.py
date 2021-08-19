from ConfigVAE import *
import os
import math
import matplotlib.pyplot as plt
from torch.autograd import Variable
from ScatterCoordinateDataset import import_data_sets
from functions import load_state_train


def log_to_plot(path):
    """
    :param path: path to a result folder
    :return: the function reads the log and creates a plot of MSE loss, D_KL and total cost
    """
    # ====================================================================================================
    # Local variables
    # ====================================================================================================
    filename    = os.path.join(path, 'logger.txt')
    fileID      = open(filename, 'r')
    lines       = fileID.readlines()
    fileID.close()

    reached_start  = False
    epoch_list     = []
    train_mse_loss = []
    train_dkl_loss = []
    train_tot_loss = []
    test_mse_loss  = []
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
        elif 'Train' in line:
            train_mse_loss.append(float(words[7]))
            train_dkl_loss.append(float(words[9]))
            train_tot_loss.append(float(words[12]))
        elif 'Test' in line:
            test_mse_loss.append(float(words[7]))
    # ====================================================================================================
    # Plotting the results
    # ====================================================================================================
    plt.plot(epoch_list, [math.sqrt(x) * NORM_FACT for x in train_mse_loss], '-o', label='Train RMS loss')
    if '15_22' in path:
        plt.plot(epoch_list[0:-1], [math.sqrt(x) * NORM_FACT for x in test_mse_loss], '-o', label='Test RMS loss')
    else:
        plt.plot(epoch_list, [math.sqrt(x) * NORM_FACT for x in test_mse_loss], '-o', label='Test RMS loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & test loss vs Epoch number')
    plt.legend()
    plt.grid()
    plt.show()


def load_and_batch(path, epoch):
    """
    :return: This function loads a saves model, and tests the MSE of the target error
    """
    data_path                   = os.path.join(path, 'VAE_model_data_lr_0.0003_epoch_' + str(epoch) + '.tar')
    train_loader, test_loader   = import_data_sets(BATCH_SIZE, 0.15)
    mod_vae, trainer            = load_state_train(data_path)

    smapled_batch   = next(iter(test_loader))
    grids           = Variable(smapled_batch['grid'].float()).to(mod_vae.device)
    sensitivities   = Variable(smapled_batch['sensitivity'].float()).to(mod_vae.device)

    mod_vae.eval()
    outputs, mu, logvar = mod_vae(grids)
    print('Outputs: ' + str(outputs))
    print('Targets: ' + str(sensitivities))


def plot_grid_histogram(grid):
    plt.hist(np.array(grid).ravel(), bins=10, density=True)


if __name__ == '__main__':
    # 14_7_2021_0_47 # 12_7_2021_15_22 # 15_7_2021_9_7
    c_path = '..\\results\\4_8_2021_8_30'
    c_epoch = 20

    load_and_batch(c_path, c_epoch)

    log_to_plot(c_path)
