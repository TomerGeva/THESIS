from torch.autograd import Variable
from global_const import encoder_type_e
from time import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from auxiliary_functions import weighted_mse
from GPUtil import showUtilization as gpu_usage  # for debug only


class TrainerDG:
    """
    This class holds the trainer for the DG model
    """

    def __init__(self, net, lr=1e-2, mom=0.9,
                 sched_step=20, sched_gamma=0.5, grad_clip=5,
                 group_thresholds=None, group_weights=None,
                 abs_sens=True,
                 training=True, optimize_time=False):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # -------------------------------------
        # cost function
        # -------------------------------------
        self.abs_sens = abs_sens
        self.sensitivity_loss = weighted_mse
        # -------------------------------------
        # optimizer
        # -------------------------------------
        # self.optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mom)
        self.optimizer = optim.Adam(net.parameters(), lr=lr)
        self.learning_rate = lr
        self.mom = mom
        self.grad_clip = grad_clip
        # -------------------------------------
        # Scheduler, reduces learning rate
        # -------------------------------------
        self.optimize_time = optimize_time
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=sched_step,
                                                   gamma=sched_gamma,
                                                   verbose=True)
        # -------------------------------------
        # Initializing the start epoch to zero
        # if not zero, the model is pre-trained
        # -------------------------------------
        self.epoch = 0
        # -------------------------------------
        # Misc training parameters
        # -------------------------------------
        self.cost = []
        # MSE parameters
        self.group_th = group_thresholds
        self.group_weights = group_weights
        self.training = training

    def compute_loss(self, sens_targets, sens_outputs):
        if self.training:
            sens_mse_loss = self.sensitivity_loss(sens_targets, sens_outputs, self.group_weights, self.group_th)
        else:
            sens_mse_loss = self.sensitivity_loss(sens_targets, sens_outputs)

        return sens_mse_loss

    def run_single_epoch(self, model, loader):
        loss    = 0.0
        counter = 0
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
            sens_targets = sample['sensitivity'].float().to(model.device)
            coordinates = sample['coordinate_target'].float().to(model.device)
            # ------------------------------------------------------------------------------
            # Forward pass
            # ------------------------------------------------------------------------------
            sens_outputs = model(coordinates)
            # ------------------------------------------------------------------------------
            # loss computations
            # ------------------------------------------------------------------------------
            sens_mse_loss = self.compute_loss(sens_targets, sens_outputs)
            loss    += sens_mse_loss.item()
            counter += sens_targets.size(0)
            # ------------------------------------------------------------------------------
            # Back propagation
            # ------------------------------------------------------------------------------
            if self.training:
                for param in model.parameters():  # zero gradients
                    param.grad = None
                sens_mse_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optimizer.step()
        return loss / counter

    def test_model(self, model, loader):
        # ==========================================================================================
        # Begin of testing
        # ==========================================================================================
        model.eval()
        self.trainer_eval()
        with torch.no_grad():
            test_loss = self.run_single_epoch(model, loader)
        model.train()
        self.trainer_train()
        return test_loss

    def train(self, model, train_loader, test_loaders, logger, epochs, save_per_epochs=1):
        """
        :param           model: DG model to train
        :param    train_loader: holds the training database
        :param    test_loaders: holds the testing databases
        :param          logger: logging the results
        :param          epochs: number of epochs to train
        :param save_per_epochs: flag indicating if you want to save
        :return: The function trains the network, and saves the trained network
        """
        # ==========================================================================================
        # Init Log
        # ==========================================================================================
        logger.filename = 'logger_vae.txt'
        logger.start_log()
        logger.log_model_arch(model)
        # ==========================================================================================
        # Begin of training
        # ==========================================================================================
        if self.optimize_time:
            # torch.backends.cudnn.benchmark  = True
            # torch.backends.cudnn.enabled    = True
            torch.autograd.set_detect_anomaly(False)  # Warns about gradients getting Nan of Inf
            torch.autograd.profiler.profile(False)  # Logs time spent on CPU and GPU
            torch.autograd.profiler.emit_nvtx(False)  # Used for NVIDIA visualizations
        init_mse = 1e5  # if mse of last group is better, save the epoch results
        logger.log_title('Beginning Training! ! ! ! number of epochs: {}'.format(epochs))
        model.train()
        for epoch in range(self.epoch, epochs):
            # ----------------------------------------------------------------------------------
            # Training single epoch
            # ----------------------------------------------------------------------------------
            train_loss = self.run_single_epoch(model, train_loader)
            # ----------------------------------------------------------------------------------
            # Logging training results
            # ----------------------------------------------------------------------------------
            logger.log_epoch_results_train('train_weighted', train_loss)
            # ----------------------------------------------------------------------------------
            # Testing accuracy
            # ----------------------------------------------------------------------------------





