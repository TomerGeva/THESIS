from ConfigVAE import *
from torch.autograd import Variable
from global_const import encoder_type_e
import os
import torch
import torch.nn as nn
import torch.optim as optim


class TrainerLatent:
    """
    This class holds the Trainer for the Variational autoencoder
    """
    def __init__(self, input_vec, lr=1e-2, mom=0.9, beta=1, sched_step=20, sched_gamma=0.5, grad_clip=5, training=True,
                 abs_sens=True, sens_std=SENS_STD, sens_mean=SENS_MEAN):
        # -------------------------------------
        # optimizer
        # -------------------------------------
        # self.optimizer = optim.SGD(input_vec.parameters(), lr=lr, momentum=mu)
        self.optimizer = optim.Adam([input_vec], lr=lr)
        # -------------------------------------
        # Scheduler, reduces learning rate
        # -------------------------------------
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
        self.learning_rate  = lr
        self.mom            = mom
        self.beta           = beta
        self.grad_clip      = grad_clip
        self.training       = training
        self.abs_sens       = abs_sens
        self.sens_std       = sens_std
        self.sens_mean      = sens_mean

    def optimize_input(self, input_vec, decoder, steps, logger, save_per_epoch=1):
        """
        :param input_vec: The input vector we want to optimize
        :param decoder: the trained decoder predicting the sensitivity from the latent space
        :param steps: number of optimization steps the trainer performs
        :param logger: the logger for the training
        :param save_per_epoch:  flag indicating if you want to save
        :return: the function changes the input to yield maximal sensitivity
        """
        # ==========================================================================================
        # Init Log
        # ==========================================================================================
        logger.start_log()
        logger.log_start_setup(input_vec.size)
        # ==========================================================================================
        # Begin of training
        # ==========================================================================================
        # input_vec = Variable(input_vec).to(decoder.device)
        # input_vec.requires_grad_(True)
        # input_vec.to(decoder.device)
        sigmoid = nn.Sigmoid()
        decoder.eval()
        for step in range(steps):
            # ------------------------------------------------------------------------------
            # Forward pass
            # ------------------------------------------------------------------------------
            _, sensitivity, _, _ = decoder(input_vec)
            # ------------------------------------------------------------------------------
            # Normalizing and documenting training results with LoggerLatent
            # ------------------------------------------------------------------------------
            self.optimizer.zero_grad()
            sensitivity = ((sensitivity * self.sens_std) + self.sens_mean) if self.abs_sens else sensitivity * self.sens_std
            logger.log_step(step, sensitivity)
            # ------------------------------------------------------------------------------
            # Backward computations
            # ------------------------------------------------------------------------------
            cost = 1 / torch.abs(sensitivity)
            cost.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(input_vec, self.grad_clip)
            self.optimizer.step()
            # ------------------------------------------------------------------------------
            # Advancing the scheduler of the lr
            # ------------------------------------------------------------------------------
            # self.scheduler.step()
            # ------------------------------------------------------------------------------
            # Saving the training state
            # save every x steps and on the last epoch
            # ------------------------------------------------------------------------------
        return input_vec
