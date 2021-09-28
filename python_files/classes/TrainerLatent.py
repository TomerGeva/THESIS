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
    def __init__(self, input_vec, lr=1e-2, mom=0.9, beta=1, sched_step=20, sched_gamma=0.5, grad_clip=5, training=True):
        # -------------------------------------
        # optimizer
        # -------------------------------------
        # self.optimizer = optim.SGD(input_vec.parameters(), lr=lr, momentum=mu)
        self.optimizer = optim.Adam(input_vec.parameters(), lr=lr)
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
        self.learning_rate = lr
        self.mom = mom
        self.beta = beta
        self.grad_clip = grad_clip
        self.training = training

    def optimize_input(self, input_vec, decoder, steps, save_per_epoch=1):
        """
        :param input_vec: The input vector we want to optimize
        :param decoder: the trained decoder predicting the sensitivity from the latent space
        :param steps: number of optimization steps the trainer performs
        :param save_per_epoch:  flag indicating if you want to save
        :return: the function changes the input to yield maximal sensitivity
        """
        for step in range(steps):
            # ------------------------------------------------------------------------------
            # Forward pass
            # ------------------------------------------------------------------------------
            sensitivity = decoder(input_vec)
            # ------------------------------------------------------------------------------
            # Backward computations
            # ------------------------------------------------------------------------------
            cost = -1 * torch.abs(sensitivity)
            cost.backward()
            nn.utils.clip_grad_norm_(input_vec, self.grad_clip)
            self.optimizer.step()
            # ------------------------------------------------------------------------------
            # Advancing the scheduler of the lr
            # ------------------------------------------------------------------------------
            self.scheduler.step()
            # ------------------------------------------------------------------------------
            # Saving the training state
            # save every x steps and on the last epoch
            # ------------------------------------------------------------------------------
        pass
