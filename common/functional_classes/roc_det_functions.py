import numpy as np
import torch
from torch.autograd import Variable
from numba import jit


class RocDetFunctions:
    """
    This class holds the functions needed to generate the probability rates needed for the ROC curve and DET curve.
    The plotting function are under the PlottingFunction class.
    """
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
        return np.where(samples >= threshold, 1, 0)

    @staticmethod
    @jit(nopython=True)
    def slice_batch_jit(samples, threshold):
        """
        :param samples: 4-D torch tensor of model output grids
        :param threshold: decision threshold, should be between 0 and 1
        :return: converts the 4-D tensor to 3-D numpy matrix (grids have 1 channel) and slices the decision at the
                 threshold
        """
        return np.where(samples >= threshold, 1, 0)

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
        :param threshold_num: number of threshold to compute the ROC curve with. if int, creates equally-spaced intervals else this is a list, takes it as it is
        :return: true positive and false positive vectors used to plot the ROC curve
        """
        sigmoid = torch.nn.Sigmoid()
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        ratio_tp = np.zeros([threshold_num + 1, ]) if type(threshold_num) is int else np.zeros_like(threshold_num)
        ratio_fp = np.zeros([threshold_num + 1, ]) if type(threshold_num) is int else np.zeros_like(threshold_num)
        ratio_fn = np.zeros([threshold_num + 1, ]) if type(threshold_num) is int else np.zeros_like(threshold_num)
        counter = 0
        thresholds = [ii / threshold_num for ii in list(range(threshold_num + 1))] if type(threshold_num) is int else threshold_num
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
                grid_out          = np.squeeze(sigmoid(grid_out).cpu().detach().numpy())
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
