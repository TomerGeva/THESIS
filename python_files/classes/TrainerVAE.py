import torch.nn as nn
import torch.optim as optim
from LoggerVAE import LoggerVAE
from functions import *


class TrainerVAE:
    """
    This class holds the Trainer for the Variational autoencoder
    """
    def __init__(self, net, lr=1e-2, mom=0.9, beta=1):
        # -------------------------------------
        # cost function
        # -------------------------------------
        self.reconstruction_loss = nn.MSELoss()
        self.d_kl                = d_kl
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
        self.cost           = []
        self.learning_rate  = lr
        self.mom            = mom
        self.beta           = beta

    def compute_loss(self, targets, outputs, mu, logvar):
        mse_loss = self.reconstruction_loss(targets, outputs) * targets.size()[0]
        kl_div   = self.d_kl(mu, logvar)
        return mse_loss, kl_div, mse_loss + self.beta * kl_div

    def test_model(self, mod_vae, loader):
        """
        :param mod_vae: VAE we want to test
        :param loader: lodaer with the test database
        :return: test parameters
        """
        # ==========================================================================================
        # Init variables
        # ==========================================================================================
        test_cost       = 0.0
        test_mse_cost   = 0.0
        test_kl_div     = 0.0

        # ==========================================================================================
        # Begin of testing
        # ==========================================================================================
        mod_vae.eval()
        with torch.no_grad():
            for sample in loader:
                # ------------------------------------------------------------------------------
                # Extracting the grids and sensitivities
                # ------------------------------------------------------------------------------
                grids = Variable(sample['grid'].float()).to(mod_vae.device)
                sensitivities = sample['sensitivity'].to(mod_vae.device)

                # ------------------------------------------------------------------------------
                # Forward pass
                # ------------------------------------------------------------------------------
                outputs, mu, logvar = mod_vae(grids)

                # ------------------------------------------------------------------------------
                # Cost computations
                # ------------------------------------------------------------------------------
                mse_loss, kl_div, cost = self.compute_loss(sensitivities, outputs, mu, logvar)

                test_mse_cost   += mse_loss
                test_kl_div     += sensitivities.size(0)
                test_cost       += cost

        mod_vae.train()
        return test_mse_cost, test_kl_div, test_cost

    def train(self, mod_vae, train_loader, test_loader, logger, save_per_epochs=1):
        """
        :param         mod_vae: Modified VAE which we want to train
        :param    train_loader: holds the training database
        :param     test_loader: holds the testing database
        :param          logger: logging the results
        :param save_per_epochs: flag indicating if you want to save
        :return: The function trains the network, and saves the trained network
        """
        # ==========================================================================================
        # Init Log
        # ==========================================================================================
        logger.start_log()
        logger.log_model_arch(mod_vae)

        # ==========================================================================================
        # Begin of training
        # ==========================================================================================
        logger.log_title('Beginning Training! ! ! ! number of epochs: {}' .format(EPOCH_NUM))
        mod_vae.train()
        for epoch in range(self.epoch, EPOCH_NUM):
            train_cost      = 0.0
            train_mse_cost  = 0.0
            train_kl_div    = 0.0
            for i, sample_batched in enumerate(train_loader):
                # ------------------------------------------------------------------------------
                # Extracting the grids and sensitivities
                # ------------------------------------------------------------------------------
                grids         = Variable(sample_batched['grid'].float()).to(mod_vae.device)
                sensitivities = Variable(sample_batched['sensitivity'].float()).to(mod_vae.device)

                # ------------------------------------------------------------------------------
                # Forward pass
                # ------------------------------------------------------------------------------
                outputs, mu, logvar = mod_vae(grids)

                # ------------------------------------------------------------------------------
                # Backward computations
                # ------------------------------------------------------------------------------
                mse_loss, kl_div, cost = self.compute_loss(sensitivities, outputs, mu, logvar)
                train_cost      += cost.item()
                train_mse_cost  += mse_loss.item()
                train_kl_div    += kl_div.item()

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Back propagation
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

            self.cost = train_cost / len(train_loader.dataset)
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Testing accuracy at the end of the epoch
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            test_mse_cost, test_kl_div, test_cost = self.test_model(mod_vae, test_loader)

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Documenting with LoggerVAE
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            logger.log_epoch_results(epoch, train_mse_cost, train_kl_div, train_cost, test_mse_cost, test_kl_div, test_cost)

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Saving the training state
            # save every x epochs and on the last epoch
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


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
