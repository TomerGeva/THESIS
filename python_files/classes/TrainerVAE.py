from ConfigVAE import *
from torch.autograd import Variable
import os
import torch
import torch.nn as nn
import torch.optim as optim


class TrainerVAE:
    """
    This class holds the Trainer for the Variational autoencoder
    """
    def __init__(self, net, lr=1e-2, mom=0.9, beta=1, sched_step=20, sched_gamma=0.5, grad_clip=5):
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
        # Scheduler, reduces learning rate
        # -------------------------------------
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=sched_step, gamma=sched_gamma, verbose=True)
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
        self.grad_clip      = grad_clip

    def compute_loss(self, targets, outputs, mu, logvar):
        mse_loss = self.reconstruction_loss(targets, outputs) * targets.size()[0]  # nn.MSEloss returns the mean, therefore we multipy by the batch size
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

    def train(self, mod_vae, train_loader, test_loaders, logger, save_per_epochs=1):
        """
        :param         mod_vae: Modified VAE which we want to train
        :param    train_loader: holds the training database
        :param    test_loaders: holds the testing databases
        :param          logger: logging the results
        :param save_per_epochs: flag indicating if you want to save
        :return: The function trains the network, and saves the trained network
        """
        # ==========================================================================================
        # Init Log
        # ==========================================================================================
        logger.start_log()
        # logger.log_model_arch(mod_vae)
        logger.log_dense_model_arch(mod_vae)

        # ==========================================================================================
        # Begin of training
        # ==========================================================================================
        logger.log_title('Beginning Training! ! ! ! number of epochs: {}' .format(EPOCH_NUM))
        mod_vae.train()
        train_loader_iter = iter(train_loader)
        for epoch in range(self.epoch, EPOCH_NUM):
            train_cost      = 0.0
            train_mse_cost  = 0.0
            train_kl_div    = 0.0
            counter         = 0
            for _ in range(len(train_loader)):
                # ------------------------------------------------------------------------------
                # Working with iterables, much faster
                # ------------------------------------------------------------------------------
                try:
                    sample_batched = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(train_loader)
                    sample_batched = next(train_loader_iter)
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
                counter         += sensitivities.size(0)

                # ------------------------------------------------------------------------------
                # Back propagation
                # ------------------------------------------------------------------------------
                self.optimizer.zero_grad()
                cost.backward()
                nn.utils.clip_grad_norm_(mod_vae.parameters(), self.grad_clip)
                self.optimizer.step()

            self.cost = train_cost / len(train_loader.dataset)
            # ------------------------------------------------------------------------------
            # Normalizing and documenting training results with LoggerVAE
            # ------------------------------------------------------------------------------
            logger.log_epoch(epoch)
            train_cost = train_cost / counter
            train_mse_cost = train_mse_cost / counter
            train_kl_div = train_kl_div / counter
            logger.log_epoch_results('train', train_mse_cost, train_kl_div, train_cost)

            # ------------------------------------------------------------------------------
            # Testing accuracy at the end of the epoch, and logging with LoggerVAE
            # ------------------------------------------------------------------------------
            test_mse_costs  = []
            test_counters   = []
            test_costs      = []
            for key in test_loaders:
                test_mse_cost, test_counter, test_cost = self.test_model(mod_vae, test_loaders[key])
                test_mse_cost = test_mse_cost / test_counter
                test_cost     = test_cost / test_counter
                logger.log_epoch_results(key, test_mse_cost, 0, test_cost)

                test_mse_costs.append(test_mse_cost)
                test_counters.append(test_counter)
                test_costs.append(test_cost)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Computing total cost for all test loaders and logging with LoggerVAE
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            test_mse_cost   = 0
            test_counter    = 0
            test_cost       = 0
            for mse, count, cost in zip(test_mse_costs, test_counters, test_costs):
                test_mse_cost   += (mse * count)
                test_cost       += (cost * count)
                test_counter    += count
            test_mse_cost   = test_mse_cost / test_counter
            test_cost       = test_cost / test_counter
            logger.log_epoch_results('test_total', test_mse_cost, 0, test_cost)
            # ------------------------------------------------------------------------------
            # Advancing the scheduler of the lr
            # ------------------------------------------------------------------------------
            self.scheduler.step()

            # ------------------------------------------------------------------------------
            # Saving the training state
            # save every x epochs and on the last epoch
            # ------------------------------------------------------------------------------
            if epoch % save_per_epochs == 0 or epoch == EPOCH_NUM-1:
                self.save_state_train(logger.logdir, mod_vae, epoch, self.learning_rate, self.mom, self.beta, SENS_STD)

    def save_state_train(self, logdir, vae, epoch, lr, mom, beta, norm_fact, filename=None):
        """Saving model and optimizer to drive, as well as current epoch and loss
        # When saving a general checkpoint, to be used for either inference or resuming training, you must save more
        # than just the model’s state_dict.
        # It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are
        # updated as the model trains.
        """
        if filename is None:
            name = 'VAE_model_data_lr_' + str(lr) + '_epoch_' + str(epoch) + '.tar'
            path = os.path.join(logdir, name)
        else:
            path = os.path.join(logdir, filename)

        data_to_save = {'epoch': epoch,
                        'vae_state_dict':       vae.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr':                   lr,
                        'mom':                  mom,
                        'beta':                 beta,
                        'norm_fact':            norm_fact,
                        'encoder_topology':     vae.encoder.topology,
                        'decoder_topology':     vae.decoder.topology,
                        'latent_dim':           vae.latent_dim,
                        'encoder_type':         vae.encoder_type
                        }
        torch.save(data_to_save, path)


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
