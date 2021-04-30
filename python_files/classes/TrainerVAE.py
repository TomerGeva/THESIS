from Config import *
import torch.nn as nn
import torch
import torch.optim as optim
from functions import *


class TrainerVAE:
    """
    This class holds the Trainer for the Variational autoencoder
    """
    def __init__(self, net, lr=1e-2, mom=0.9):
        # -------------------------------------
        # cost function
        # -------------------------------------
        self.reconstruction_loss = nn.MSELoss()
        self.d_kl                = d_kl()
        # -------------------------------------
        # optimizer
        # -------------------------------------
        # self.optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mu)
        self.optimizer = optim.Adam(net.parameters(), lr=lr)
        # -------------------------------------
        # Initializing the start epoch to zero
        # if not zero, the model is pre-trained
        # -------------------------------------
        self.epoch = 0
        # -------------------------------------
        # Misc training parameters
        # -------------------------------------
        self.loss           = []
        self.learning_rate  = lr
        self.mom            = mom

    def train(self, net, train_loader, test_loader, logger, save_per_epochs=1):
        """
        :param net: Net class object, which is the net we want to train
        :param train_loader: holds the training database
        :param test_loader: holds the testing database
        :param logger: logging the results
        :param save_per_epochs: flag indicating if you want to save
        :return: the function trains the network, and saves the trained network
        """
        losses = []
        accuracies_train = []
        accuracies_test = []
        print("Started training, learning rate: {}".format(self.learning_rate))
        # ----------------------------------------------
        # Begin of training
        # ----------------------------------------------
        net.eval()
        for epoch in range(self.epoch, EPOCH_NUM):
            train_loss = 0.0
            print("Starting Epoch #" + str(epoch + 1))
            for i, sample_batched in enumerate(train_loader):
                # ++++++++++++++++++++++++++++++++++++++
                # Extracting the grids and sensitivities
                # ++++++++++++++++++++++++++++++++++++++
                grids         = Variable(sample_batched['grid'].float()).to(net.device)
                sensitivities = Variable(sample_batched['sensitivity'].float()).to(net.device)

                # ++++++++++++++++++++++++++++++++++++++
                # Feed forward
                # ++++++++++++++++++++++++++++++++++++++
                self.optimizer.zero_grad()
                outputs = net(grids)

                # ++++++++++++++++++++++++++++++++++++++
                # Computing the loss
                # ++++++++++++++++++++++++++++++++++++++
                loss = self.criterion(outputs, sensitivities)

                # ++++++++++++++++++++++++++++++++++++++
                # Back propagation
                # ++++++++++++++++++++++++++++++++++++++
                loss.backward()
                self.optimizer.step()

                # ++++++++++++++++++++++++++++++++++++++
                # Documenting the loss
                # ++++++++++++++++++++++++++++++++++++++
                losses.append(loss.data.item())
                train_loss += loss.item() * grids.size(0)
            self.loss = train_loss / len(train_loader.dataset)
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Testing accuracy at the end of the epoch
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            accuracies_train.append(self.loss)
            accuracies_test.append(accuracy_test(net, test_loader))

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Documenting with tensorboard
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            logger.logger.add_scalars(logger.logger_tag + "_accuracy",
                                      {"Train_MSE_learning_rate_{}".format(self.learning_rate): accuracies_train[-1],
                                       "Test_MSE_learning_rate_{}".format(self.learning_rate): accuracies_test[-1]},
                                      epoch + 1)
            logger.logger.add_scalars(logger.logger_tag + "_loss",
                                      {"learning_rate_{}".format(self.learning_rate): self.loss},
                                      epoch + 1)

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Saving the training state
            # save every x epochs and on the last epoch
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Printing log to screen
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            print("Epoch: {}/{} \tTraining loss: {:.10f} \tTrain MSE: {:.6f} \tTest MSE: {:.6f}".format(
                epoch + 1, EPOCH_NUM, self.loss,
                accuracies_train[-1], accuracies_test[-1]))


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# AUXILIARY FUNCTIONS
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def d_kl(mu, logvar):
    """
    :param mu: expectation vector, assuming a tensor
    :param logvar: log variance vector, assuming a tensor
    :return: The function computes and returns the Kullback-Leiber divergence of a multivariable independant
             normal distribution form the normal distribution N(0, I)
    """
    return torch.sum(0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar, dim=1))