from ConfigVAE import SENS_STD, SENS_MEAN, FIG_DIR, PP_DATA
import torch
from torch import clamp
import scipy.stats as sp
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import datetime


# ==================================================================================================================
# Functions used by the networks
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
    elif 'modedgeconv' in action[0]:
        x_dim_size = x_dim
        y_dim_size = y_dim
        channels = ch_num
    elif 'sg_pointnet' in action[0]:
        x_dim_size = x_dim
        y_dim_size = y_dim
        channels = ch_num
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
        # THIS HOLDS FOR SEPARABLE CONVOLUTION AND REGULAR CONVOLUTION
        x_dim_size = int((x_dim - ((action[1].dilation * action[1].kernel - (action[1].dilation-1)) - action[1].stride) + 2 * action[1].padding) / action[1].stride)
        y_dim_size = int((y_dim - ((action[1].dilation * action[1].kernel - (action[1].dilation-1)) - action[1].stride) + 2 * action[1].padding) / action[1].stride)
        channels = action[1].out_channels
    elif 'linear' in action[0]:  # linear case
        x_dim_size = x_dim
        y_dim_size = y_dim
        channels   = ch_num
    elif 'transformer' in action[0]:
        x_dim_size = 1
        y_dim_size = 1
        channels   = x_dim // action[1].patch_size_x * y_dim // action[1].patch_size_y

    else:
        raise ValueError('Invalid layer description')
    return x_dim_size, y_dim_size, channels


def truncated_relu(x):
    return clamp(x, min=0, max=1)


# ==================================================================================================================
# Functions used by the trainers
# ==================================================================================================================
def weighted_mse(targets, outputs, weights=None, thresholds=None):
    """
    :param targets: model targets
    :param outputs: model outputs
    :param weights: weights of the mse according to the groups
    :param thresholds: the thresholds between the different groups
    :return:
    """
    # ==================================================================================================================
    # Getting the weight vector
    # ==================================================================================================================
    if (weights is None) or (thresholds is None):
        weight_vec = torch.ones_like(targets)
    else:
        weight_vec = ((targets < thresholds[0]) * weights[0]).type(torch.float)
        for ii in range(1, len(thresholds)):
            weight_vec += torch.logical_and(thresholds[ii - 1] <= targets, targets < thresholds[ii]) * weights[ii]
        weight_vec += (targets >= thresholds[-1]) * weights[-1]
    # ==================================================================================================================
    # Computing weighted MSE as a sum, not mean
    # ==================================================================================================================
    return 0.5 * torch.sum((outputs - targets).pow(2) * weight_vec / torch.abs(targets))
    # return 0.5 * torch.sum((outputs - targets).pow(2) * weight_vec)


def grid_mse(targets, outputs):
    return 0.5 * torch.sum(torch.pow(targets - outputs, 2.0))


def hausdorf_distance(X, Y, reduction='sum'):
    """
    :param X: B X N X 2 tensor or B X 2N tensor
    :param Y: B X N X 2 tensor or B X 2N tensor
    :param reduction: sum or mean over the batch
    :return: Function computes the Hausdorf distance between set X and set Y
    """
    if len(X.size()) == 2:
        X = X.view(X.size()[0], -1, 2).contiguous()
    if len(Y.size()) == 2:
        Y = Y.view(Y.size()[0], -1, 2).contiguous()
    dx = (X[:, :, 0][:, :, None] - Y[:, :, 0].view(Y.size()[0], 1, -1).contiguous()) ** 2
    dy = (X[:, :, 1][:, :, None] - Y[:, :, 1].view(Y.size()[0], 1, -1).contiguous()) ** 2
    distance = torch.sqrt(dx + dy)
    dxy      = torch.min(distance, dim=1, keepdim=False).values
    dyx      = torch.min(distance, dim=2, keepdim=False).values
    hausdorf = torch.max(torch.cat((dxy, dyx), 1), 1).values
    # hausdorf = torch.cat((dxy, dyx), 1)
    if reduction == 'sum':
        return torch.sum(hausdorf)
    elif reduction == 'mean':
        return torch.mean(hausdorf)


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
# Function used to plot the 2d grid of arrays
# ==================================================================================================================
class PlottingFunctions:
    def __init__(self):
        pass

    @staticmethod
    def plot_grid(grid):
        """
        :param grid: 1 X 1 X 2500 X 2500 grid tensor
        :return: plots the grid before and after a step function
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
    def plot_roc_curve(data_dict, name_prefixes, thresholds=None, save_plt=False, path=None, epoch=None):
        """
        :param data_dict: dictionary holding the rates for all the prefixes
        :param name_prefixes: prefix list, the prefixes are the keys for the data_dict. THIS musT BE A LIST
        :param thresholds: if not None, plots a circle around the thresholds. Should be a list
        :param save_plt: weather to save the plot
        :param path: if save_plt, uses this path
        :param epoch: if save_plt, uses this epoch number
        :return: Plots ROC curve
        """
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        modified_roc = plt.figure()
        fpr_scatter = [] if thresholds is None else np.zeros([len(thresholds), len(name_prefixes)])
        tpr_scatter = [] if thresholds is None else np.zeros([len(thresholds), len(name_prefixes)])
        # ==============================================================================================================
        # For each prefix, we plot a curve in the figure
        # ==============================================================================================================
        for ii, name_prefix in enumerate(name_prefixes):
            # ------------------------------------------------------------------------------------------------------
            # Extracting data
            # ------------------------------------------------------------------------------------------------------
            fpr = data_dict[name_prefix]['false_positive_rate']
            tpr = data_dict[name_prefix]['true_positive_rate']
            # ------------------------------------------------------------------------------------------------------
            # Computing Area under Modified ROC - AuMC
            # ------------------------------------------------------------------------------------------------------
            fpr_np = np.array(fpr)
            tpr_np = np.array(tpr)
            dx = fpr_np[1:] - fpr_np[:-1]
            y = (tpr_np[1:] + tpr_np[:-1]) / 2
            aumc = abs(np.sum(y * dx))
            # ------------------------------------------------------------------------------------------------------
            # Plotting the scattered thresholds
            # ------------------------------------------------------------------------------------------------------
            if thresholds is not None:
                indices = [0] * len(thresholds)  # np.zeros_like(thresholds).astype(int)
                data_thresholds = data_dict[name_prefix]['thresholds']
                for jj, threshold in enumerate(thresholds):
                    try:
                        indices[jj] = data_thresholds.index(threshold)
                    except ValueError:
                        indices[jj] = (np.abs(np.array(data_thresholds) - threshold)).argmin()
                tpr_scatter[:, ii] = tpr_np[indices]
                fpr_scatter[:, ii] = fpr_np[indices]
            # ------------------------------------------------------------------------------------------------------
            # Plotting
            # ------------------------------------------------------------------------------------------------------
            leg = name_prefix + ' AuMC value: {:.3}'.format(aumc)
            plt.plot(fpr, tpr, linewidth=2, label=leg)
        # ==============================================================================================================
        # Plotting thresholds scatter if needed
        # ==============================================================================================================
        if thresholds is not None:
            for jj, threshold in enumerate(thresholds):
                plt.plot(fpr_scatter[jj, :], tpr_scatter[jj, :], 'o', label=f'threshold = {threshold}')
        # ==============================================================================================================
        # General Plotting
        # ==============================================================================================================
        plt.grid()
        plt.title('Modified ROC Curve for Grid Reconstruction', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.legend()
        modified_roc.set_size_inches((10, 10))
        # ==============================================================================================================
        # Saving
        # ==============================================================================================================
        if save_plt and (path is not None) and (epoch is not None):
            # ------------------------------------------------------------------------------------------------------
            # Setting filename
            # ------------------------------------------------------------------------------------------------------
            filename = f'modified_roc_{epoch}.png'
            filename = name_prefixes[0] + '_' + filename if len(name_prefixes) == 1 else 'combined_' + filename
            # ------------------------------------------------------------------------------------------------------
            # Creating directory if not exists
            # ------------------------------------------------------------------------------------------------------
            if not os.path.isdir(os.path.join(path, FIG_DIR)):
                os.makedirs(os.path.join(path, FIG_DIR))
            # ------------------------------------------------------------------------------------------------------
            # Saving
            # ------------------------------------------------------------------------------------------------------
            modified_roc.savefig(os.path.join(path, FIG_DIR, filename))

    @staticmethod
    def plot_det_curve(data_dict, name_prefixes, thresholds=None, save_plt=False, path=None, epoch=None):
        """
        :param data_dict: dictionary holding the rates for all the prefixes
        :param name_prefixes: prefix list, the prefixes are the keys for the data_dict. THIS musT BE A LIST
        :param thresholds: if not None, plots a circle around the thresholds. Should be a list
        :param save_plt: weather to save the plot
        :param path: if save_plt, uses this path
        :param epoch: if save_plt, uses this epoch number
        :return: Plots ROC curve
        """
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        modified_det = plt.figure()
        fpr_scatter = [] if thresholds is None else np.zeros([len(thresholds), len(name_prefixes)])
        fnr_scatter = [] if thresholds is None else np.zeros([len(thresholds), len(name_prefixes)])
        # ==============================================================================================================
        # For each prefix, we plot a curve in the figure
        # ==============================================================================================================
        for ii, name_prefix in enumerate(name_prefixes):
            # ------------------------------------------------------------------------------------------------------
            # Changing to normal distribution coordinates
            # ------------------------------------------------------------------------------------------------------
            fnr = data_dict[name_prefix]['false_negative_rate']
            fpr = data_dict[name_prefix]['false_positive_rate']
            fpr_std_scale = sp.norm.ppf(fpr)
            fnr_std_scale = sp.norm.ppf(fnr)
            # ------------------------------------------------------------------------------------------------------
            # Plotting curve
            # ------------------------------------------------------------------------------------------------------
            plt.plot(fpr_std_scale, fnr_std_scale, linewidth=2, label=name_prefix)
            # ------------------------------------------------------------------------------------------------------
            # saving the wanted threshold plots
            # ------------------------------------------------------------------------------------------------------
            if thresholds is not None:
                indices = [0] * len(thresholds)  # np.zeros_like(thresholds).astype(int)
                data_thresholds = data_dict[name_prefix]['thresholds']
                for jj, threshold in enumerate(thresholds):
                    try:
                        indices[jj] = data_thresholds.index(threshold)
                    except ValueError:
                        indices[jj] = (np.abs(np.array(data_thresholds) - threshold)).argmin()
                fnr_scatter[:, ii] = np.array(fnr_std_scale)[indices]
                fpr_scatter[:, ii] = np.array(fpr_std_scale)[indices]
        # ==============================================================================================================
        # Plotting thresholds scatter
        # ==============================================================================================================
        if thresholds is not None:
            for jj, threshold in enumerate(thresholds):
                plt.plot(fpr_scatter[jj, :], fnr_scatter[jj, :], 'o', label=f'threshold = {threshold}')  #  label='_nolegend_'
        # ==============================================================================================================
        # Changing to normal distribution presentation
        # ==============================================================================================================
        ticks = [1e-4, 0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
        # tick_labels = ticks
        tick_labels = ['{:.2f}'.format(s) if (100 * s).is_integer() else '{:.3f}'.format(s) if (1000 * s).is_integer() else '{:.0e}'.format(s) for s in ticks]
        tick_locations = sp.norm.ppf(ticks)
        # ==============================================================================================================
        # General plotting
        # ==============================================================================================================
        modified_det.set_size_inches((10, 10))
        plt.grid()
        plt.title('Modified DET Curve for Grid Reconstruction', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('False Negative Rate', fontsize=12)
        plt.legend()
        # ----------------------------------------------------------------------------------------------------------
        # Fixing the axes to normal deviate
        # ----------------------------------------------------------------------------------------------------------
        axes = modified_det.gca()
        axes.set_xticks(tick_locations)
        axes.set_xticklabels(tick_labels)
        axes.set_yticks(tick_locations)
        axes.set_yticklabels(tick_labels)
        axes.set_ylim(-3, 3)
        axes.set_xlim(-4, 3)
        # ==============================================================================================================
        # Saving
        # ==============================================================================================================
        if save_plt and (path is not None) and (epoch is not None):
            # ------------------------------------------------------------------------------------------------------
            # Setting filename
            # ------------------------------------------------------------------------------------------------------
            filename = f'modified_det_{epoch}.png'
            filename = name_prefixes[0] + '_' + filename if len(name_prefixes) == 1 else 'combined_' + filename
            # ------------------------------------------------------------------------------------------------------
            # Creating directory if not exists
            # ------------------------------------------------------------------------------------------------------
            if not os.path.isdir(os.path.join(path, FIG_DIR)):
                os.makedirs(os.path.join(path, FIG_DIR))
            # ------------------------------------------------------------------------------------------------------
            # Saving
            # ------------------------------------------------------------------------------------------------------
            modified_det.savefig(os.path.join(path, FIG_DIR, filename))

    @staticmethod
    def scat_coord(target, output):
        """
        :param target:
        :param output: Function receives a target coordinate vector and output coordinate vactor, scatters them both on
                       the same plot
        :return:
        """
        target_re = np.reshape(target.cpu().detach().numpy(), [-1, 2])
        output_re = np.reshape(output.cpu().detach().numpy(), [-1, 2])
        plt.figure()
        plt.scatter(target_re[:, 0], target_re[:, 1], label='Target Grid')
        plt.scatter(output_re[:, 0], output_re[:, 1], label='Output Grid')
        plt.title('Coordinate scatter results')
        plt.legend()
        plt.grid()
        plt.show()


# ==================================================================================================================
# Init the files and folders
# ==================================================================================================================
def _init_(path):
    # --------------------------------------------------
    # Creating new folder name
    # --------------------------------------------------
    time_data = datetime.datetime.now()
    time_list = [time_data.day, time_data.month, time_data.year, time_data.hour, time_data.minute]
    time_string = '_'.join([str(ii) for ii in time_list])
    del time_data, time_list
    logdir = os.path.join(path, time_string)
    # --------------------------------------------------
    # Setting folders
    # --------------------------------------------------
    try:
        os.makedirs(logdir)
        print('{0:s} {1:s}'.format(' Created new directory ', logdir))
    except OSError:
        pass
    if not os.path.exists(os.path.join(path, time_string, 'figures')):
        os.makedirs(os.path.join(path, time_string, 'figures'))

    return logdir
