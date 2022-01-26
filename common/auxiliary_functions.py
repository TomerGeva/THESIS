from ConfigVAE import SENS_STD, SENS_MEAN
from torch import clamp
import scipy.stats as sp
import matplotlib.pyplot as plt
import numpy as np
import math
import os


# ==================================================================================================================
# Truncated ReLU activation function
# ==================================================================================================================
def truncated_relu(x):
    return clamp(x, min=0, max=1)


# ==================================================================================================================
# Function used to create a circular kernel
# ==================================================================================================================
def create_circle_kernel(radius=1):
    kernel = np.zeros((2*radius+1, 2*radius+1))
    xx, yy = np.meshgrid(np.arange(2*radius+1), np.arange(2*radius+1))
    r = np.sqrt((xx - radius)**2 + (yy - radius)**2)
    kernel[r <= radius] = 1
    # kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=0)
    return kernel.astype(np.uint8)


# ==================================================================================================================
# Function used to computed the dimensions between each layer
# ==================================================================================================================
def compute_output_dim(x_dim, y_dim, ch_num, action):
    """
    Function used to compute the output dimensions of a NN layer
    :param x_dim: x dimension size of the input
    :param y_dim: y dimension size of the input
    :param ch_num: channel number of the input
    :param action: layer description
    :return:
    """
    if 'convTrans' in action[0]:
        channels = action[1].out_channels
        x_dim_size = (x_dim - 1) * action[1].stride - (2 * action[1].padding) + action[1].dilation * (action[1].kernel - 1) + action[1].output_padding + 1
        y_dim_size = (y_dim - 1) * action[1].stride - (2 * action[1].padding) + action[1].dilation * (action[1].kernel - 1) + action[1].output_padding + 1
    elif 'pool' in action:
        if type(action[1].pad) is not tuple:
            x_dim_size = int((x_dim + 2 * action[1].pad) / action[1].kernel)
            y_dim_size = int((y_dim + 2 * action[1].pad) / action[1].kernel)
        else:
            x_dim_size = int((x_dim + action[1].pad[0] + action[1].pad[1]) / action[1].kernel)
            y_dim_size = int((y_dim + action[1].pad[2] + action[1].pad[3]) / action[1].kernel)
        channels = ch_num
    elif 'dense' in action[0]:
        channels = ch_num + action[1].growth * action[1].depth
        x_dim_size = x_dim
        y_dim_size = y_dim
    elif 'transition' in action[0]:
        channels = math.floor(ch_num * action[1].reduction_rate)
        # ------------------------------------------------
        # This account for the conv layer and the pooling
        # ------------------------------------------------
        x_conv_size = int((x_dim - (action[1].kernel - action[1].stride) + 2 * action[1].padding) / action[1].stride)
        y_conv_size = int((y_dim - (action[1].kernel - action[1].stride) + 2 * action[1].padding) / action[1].stride)
        if type(action[1].pool_padding) is not tuple:
            x_dim_size = int((x_conv_size + 2 * action[1].pool_padding) / action[1].pool_size)
            y_dim_size = int((y_conv_size + 2 * action[1].pool_padding) / action[1].pool_size)
        else:
            x_dim_size = int((x_conv_size + 2 * action[1].pool_padding[0] + 2 * action[1].pool_padding[1]) / action[1].pool_size)
            y_dim_size = int((y_conv_size + 2 * action[1].pool_padding[2] + 2 * action[1].pool_padding[3]) / action[1].pool_size)
    elif 'conv' in action[0]:  # :   [                       K_tilde                                ]         S                          P                    S
        x_dim_size = int((x_dim - ((action[1].dilation * action[1].kernel - (action[1].dilation-1)) - action[1].stride) + 2 * action[1].padding) / action[1].stride)
        y_dim_size = int((y_dim - ((action[1].dilation * action[1].kernel - (action[1].dilation-1)) - action[1].stride) + 2 * action[1].padding) / action[1].stride)
        channels = action[1].out_channels
    else:  # linear case
        x_dim_size = x_dim
        y_dim_size = y_dim
        channels   = ch_num

    return x_dim_size, y_dim_size, channels


# ==================================================================================================================
# Function used to plot the 2d grid of arrays
# ==================================================================================================================
class PlottingFunctions:
    def __init__(self):
        pass

    @staticmethod
    def plot_grid(grid):
        """
        :param grid: 1 X 1 X 2500 X 2500 grid tensor
        :return: plot the grid after a step function in the middle
        """
        grid_np = grid.cpu().squeeze().detach().numpy()
        plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        grid_np = (grid_np - np.min(grid_np)) / (np.max(grid_np) - np.min(grid_np)) * 255
        imgplot = plt.imshow(grid_np, cmap='gray', vmin=0, vmax=255)
        plt.title('Grid Raw')
        ax2 = plt.subplot(1, 2, 2)
        grid_np = np.where(grid_np > 127, 255, 0)
        imgplot = plt.imshow(grid_np, cmap='gray', vmin=0, vmax=255)
        plt.title('Grid After Step Function')
        plt.show()

    @staticmethod
    def plot_latent(mu, var, target, output):
        """
        :param mu: expectation vector
        :param var: variance vector
        :param output: output sensitivity
        :param target: target sensitivity
        :return: plots
        """
        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(mu.T, 'o')
        plt.title('Expectation per index, latent space')  # , target sensitivity {0:.2f} ' .format(target[0] * SENS_STD + SENS_MEAN))
        plt.xlabel('index')
        plt.ylabel('amplitude')
        plt.grid()
        ax2 = plt.subplot(2, 1, 2)
        plt.plot(var.T, 'o')
        plt.title('Variance per index, latent space')  # , output sensitivity {0:.2f} ' .format(output[0] * SENS_STD + SENS_MEAN))
        plt.xlabel('index')
        plt.ylabel('amplitude')
        plt.grid()

    @staticmethod
    def plot_grid_histogram(grid, bins=10):
        plt.hist(np.array(grid).ravel(), bins=bins, density=True)

    @staticmethod
    def plot_roc_curve(fpr, tpr, save_plt=False, path=None, epoch=None, name_prefix=None):
        modified_roc = plt.figure()
        plt.grid()
        plt.plot(fpr, tpr, linewidth=2)
        plt.title('Modified ROC Curve for Grid Reconstruction', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        if save_plt and (path is not None) and (epoch is not None):
            filename = f'modified_roc_{epoch}.png'
            if name_prefix is not None:
                filename = name_prefix + '_' + filename
            if not os.path.isdir(os.path.join(path, 'figures')):
                os.makedirs(os.path.join(path, 'figures'))
            modified_roc.savefig(os.path.join(path, 'figures', filename))

    @staticmethod
    def plot_det_curve(fpr, fnr, save_plt=False, path=None, epoch=None, name_prefix=None):
        fpr_std_scale = sp.norm.ppf(fpr)
        fnr_std_scale = sp.norm.ppf(fnr)
        modified_det = plt.figure()
        plt.grid()
        plt.plot(fpr_std_scale, fnr_std_scale, linewidth=2)
        plt.title('Modified DET Curve for Grid Reconstruction', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('False Negative Rate', fontsize=12)
        ticks = [0.0001, 0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
        tick_labels = ticks
        # tick_labels = ['{:.0%}'.format(s) if (100 * s).is_integer() else '{:.1%}'.format(s) for s in ticks]
        tick_locations = sp.norm.ppf(ticks)
        axes = modified_det.gca()
        axes.set_xticks(tick_locations)
        axes.set_xticklabels(tick_labels)
        axes.set_yticks(tick_locations)
        axes.set_yticklabels(tick_labels)
        axes.set_ylim(-3, 3)
        if save_plt and (path is not None) and (epoch is not None):
            filename = f'modified_det_{epoch}.png'
            if name_prefix is not None:
                filename = name_prefix + '_' + filename
            if not os.path.isdir(os.path.join(path, 'figures')):
                os.makedirs(os.path.join(path, 'figures'))
            modified_det.savefig(os.path.join(path, 'figures', filename))
