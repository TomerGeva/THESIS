from ConfigVAE import *
import os
import math
import matplotlib.pyplot as plt

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
    plt.plot(epoch_list[0:-1], [math.sqrt(x) * NORM_FACT for x in test_mse_loss], '-o', label='Test RMS loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & test loss vs Epoch number')
    plt.legend()
    plt.grid()
    plt.show()
    print('hi')


if __name__ == '__main__':
    log_to_plot('..\\results\\12_7_2021_15_22')
