from ConfigVAE import *
from torch.autograd import Variable
from global_const import encoder_type_e
from time import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from GPUtil import showUtilization as gpu_usage  # for debug only


class TrainerVAE:
    """
    This class holds the Trainer for the Variational auto-encoder
    """
    def __init__(self, net, lr=1e-2, mom=0.9,
                 beta_dkl=1, beta_grid=1,
                 sched_step=20, sched_gamma=0.5, grad_clip=5,
                 group_thresholds=None, group_weights=None,
                 abs_sens=True,
                 training=True, optimize_time=False,
                 grid_pos_weight=1,
                 xquantize=2500, yquantize=2500, coord2map_sigma=1, n=1):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # -------------------------------------
        # cost function
        # -------------------------------------
        self.abs_sens            = abs_sens
        self.sensitivity_loss    = weighted_mse
        self.d_kl                = d_kl
        if net.encoder_type == encoder_type_e.FULLY_CONNECTED:
            self.reconstruction_loss = coord_reconstruction_loss
        else:
            self.reconstruction_loss = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.ones([xquantize, yquantize], device=device) * grid_pos_weight)
        self.encoder_type    = net.encoder_type
        self.coord2map_sigma = coord2map_sigma
        self.n               = n
        # -------------------------------------
        # optimizer
        # -------------------------------------
        # self.optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mom)
        self.optimizer = optim.Adam(net.parameters(), lr=lr)
        self.learning_rate  = lr
        self.mom            = mom
        self.grad_clip      = grad_clip
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
        self.cost            = []
        # MSE parameters
        self.group_th        = group_thresholds
        self.group_weights   = group_weights
        # Cost function summation parameters
        self.beta_dkl        = beta_dkl
        self.beta_grid       = beta_grid
        self.grid_pos_weight = grid_pos_weight
        self.training        = training

    def compute_loss(self, sens_targets, sens_outputs, mu, logvar, grid_targets=0, grid_outputs=0, model_out=model_output_e.BOTH):
        if self.training:
            sens_mse_loss = self.sensitivity_loss(sens_targets, sens_outputs, self.group_weights, self.group_th)
        else:
            sens_mse_loss = self.sensitivity_loss(sens_targets, sens_outputs)
        kl_div          = self.d_kl(mu, logvar)
        if model_out is model_output_e.SENS:
            grid_mse_loss = torch.zeros(1).to(kl_div.device)
        else:
            if self.encoder_type == encoder_type_e.FULLY_CONNECTED:
                grid_mse_loss = self.reconstruction_loss(grid_targets, grid_outputs, self.n, self.sigma, self.xquantize, self.yquantize)
            else:
                grid_mse_loss = self.reconstruction_loss(grid_outputs, grid_targets)

        return sens_mse_loss, kl_div, grid_mse_loss, sens_mse_loss + (self.beta_dkl * kl_div) + (self.beta_grid * grid_mse_loss)

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
        test_sens_mse   = 0.0
        test_grid_mse   = 0.0
        counter         = 0

        # ==========================================================================================
        # Begin of testing
        # ==========================================================================================
        mod_vae.eval()
        self.trainer_eval()
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
                grids           = Variable(sample['grid_in'].float()).to(mod_vae.device)
                sensitivities   = sample['sensitivity'].float().to(mod_vae.device)
                grid_targets    = sample['grid_target'].float().to(mod_vae.device)

                # ------------------------------------------------------------------------------
                # Forward pass
                # ------------------------------------------------------------------------------
                grid_out, sens_out, mu, logvar = mod_vae(grids)

                # ------------------------------------------------------------------------------
                # Cost computations
                # ------------------------------------------------------------------------------
                sens_mse_loss, kl_div, grid_mse_loss, cost = self.compute_loss(sensitivities, sens_out, mu, logvar, grid_targets, grid_out, model_out=mod_vae.model_out)

                test_sens_mse   += sens_mse_loss.item()
                counter         += sensitivities.size(0)
                test_grid_mse   += grid_mse_loss.item()
                test_cost       += cost.item()

        mod_vae.train()
        self.trainer_train()
        return test_sens_mse, test_grid_mse, counter, test_cost

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
        logger.filename = 'logger_vae.txt'
        logger.start_log()
        if mod_vae.encoder_type == encoder_type_e.DENSE:
            logger.log_dense_model_arch(mod_vae)
        else:  # if mod_vae.encoder_type == encoder_type_e.VGG:
            logger.log_model_arch(mod_vae)

        # ==========================================================================================
        # Begin of training
        # ==========================================================================================
        if self.optimize_time:
            # torch.backends.cudnn.benchmark  = True
            # torch.backends.cudnn.enabled    = True
            torch.autograd.set_detect_anomaly(False)  # Warns about gradients getting Nan of Inf
            torch.autograd.profiler.profile(False)    # Logs time spent on CPU and GPU
            torch.autograd.profiler.emit_nvtx(False)  # Used for NVIDIA visualizations
        mse_last_group = 1e5  # if mse of last group is better, save the epoch results
        logger.log_title('Beginning Training! ! ! ! number of epochs: {}' .format(EPOCH_NUM))
        mod_vae.train()
        train_loader_iter = iter(train_loader)
        for epoch in range(self.epoch, EPOCH_NUM):
            train_cost      = 0.0
            train_sens_mse  = 0.0
            train_kl_div    = 0.0
            train_grid_mse  = 0.0
            counter         = 0
            t = time()
            for _ in range(len(train_loader)):
                # ------------------------------------------------------------------------------
                # Working with iterables, much faster
                # ------------------------------------------------------------------------------
                try:
                    sample_batched = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(train_loader)
                    sample_batched    = next(train_loader_iter)
                # ------------------------------------------------------------------------------
                # Extracting the grids and sensitivities
                # ------------------------------------------------------------------------------
                grids         = Variable(sample_batched['grid_in'].float()).to(mod_vae.device)
                sensitivities = Variable(sample_batched['sensitivity'].float()).to(mod_vae.device)
                grid_targets  = Variable(sample_batched['grid_target'].float()).to(mod_vae.device)
                # ------------------------------------------------------------------------------
                # Forward pass
                # ------------------------------------------------------------------------------
                grid_out, sens_out, mu, logvar = mod_vae(grids)
                # ------------------------------------------------------------------------------
                # Backward computations
                # ------------------------------------------------------------------------------
                sens_mse_loss, kl_div, grid_mse_loss, cost = self.compute_loss(sensitivities, sens_out, mu, logvar, grid_targets, grid_out, model_out=mod_vae.model_out)
                train_cost      += cost.item()
                train_sens_mse  += sens_mse_loss.item()
                train_kl_div    += kl_div.item()
                train_grid_mse  += grid_mse_loss.item()
                counter         += sensitivities.size(0)
                # ------------------------------------------------------------------------------
                # Back propagation
                # ------------------------------------------------------------------------------
                for param in mod_vae.parameters():  # zero gradients
                    param.grad = None
                # gpu_usage()  # DEBUG
                if mod_vae.mode is mode_e.AUTOENCODER and mod_vae.model_out is model_output_e.SENS:
                    sens_mse_loss.backward()
                elif mod_vae.mode is mode_e.AUTOENCODER and mod_vae.model_out is model_output_e.GRID:
                    grid_mse_loss.backward()
                else:
                    cost.backward()
                nn.utils.clip_grad_norm_(mod_vae.parameters(), self.grad_clip)
                self.optimizer.step()

            self.cost = train_cost / len(train_loader.dataset)
            # ------------------------------------------------------------------------------
            # Normalizing and documenting training results with LoggerVAE
            # ------------------------------------------------------------------------------
            logger.log_epoch(epoch, t)
            train_cost      = train_cost / counter
            train_sens_mse  = train_sens_mse / counter
            train_kl_div    = train_kl_div / counter
            train_grid_mse  = train_grid_mse / counter
            logger.log_epoch_results_train('train_weighted', train_sens_mse, train_kl_div, train_grid_mse, train_cost)
            # ------------------------------------------------------------------------------
            # Testing accuracy at the end of the epoch, and logging with LoggerVAE
            # ------------------------------------------------------------------------------
            test_sens_mse_vec = []
            test_grid_mse_vec = []
            test_counters_vec = []
            test_costs_vec    = []
            for key in test_loaders:
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Getting the respective group weight
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                test_mse_weight = self.get_test_group_weight(key)
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Testing the results of the current group
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                test_sens_mse, test_grid_mse, test_counter, test_cost = self.test_model(mod_vae, test_loaders[key])
                test_sens_mse = test_sens_mse / test_counter
                test_grid_mse = test_grid_mse / test_counter
                test_cost     = test_cost / test_counter
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Logging
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                logger.log_epoch_results_test(key, test_sens_mse, test_grid_mse, test_mse_weight)
                test_sens_mse_vec.append(test_sens_mse)
                test_grid_mse_vec.append(test_grid_mse)
                test_counters_vec.append(test_counter)
                test_costs_vec.append(test_cost)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Computing total cost for all test loaders and logging with LoggerVAE
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            test_sens_mse   = 0.0
            test_grid_mse   = 0.0
            test_counter    = 0
            test_cost       = 0.0
            for sens_mse, grid_mse, count, cost in zip(test_sens_mse_vec, test_grid_mse_vec, test_counters_vec, test_costs_vec):
                test_sens_mse += (sens_mse * count)
                test_grid_mse += (grid_mse * count)
                test_cost     += (cost * count)
                test_counter  += count
            test_sens_mse   = test_sens_mse / test_counter
            test_grid_mse   = test_grid_mse / test_counter
            test_cost       = test_cost / test_counter
            logger.log_epoch_results_test('test_total', test_sens_mse, test_grid_mse, 0)
            # ------------------------------------------------------------------------------
            # Advancing the scheduler of the lr
            # ------------------------------------------------------------------------------
            self.scheduler.step()
            # ------------------------------------------------------------------------------
            # Saving the training state
            # save every x epochs and on the last epoch
            # ------------------------------------------------------------------------------
            if epoch % save_per_epochs == 0 or epoch == EPOCH_NUM-1 or test_sens_mse_vec[-1] < mse_last_group:
                self.save_state_train(logger.logdir, mod_vae, epoch, self.learning_rate, self.mom, self.beta_dkl, self.beta_grid, SENS_STD)
                if test_sens_mse_vec[-1] < mse_last_group:
                    mse_last_group = test_sens_mse_vec[-1]

    def save_state_train(self, logdir, vae, epoch, lr, mom, beta_dkl, beta_grid, norm_fact, filename=None):
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
                        'beta_dkl':             beta_dkl,
                        'beta_grid':            beta_grid,
                        'norm_fact':            norm_fact,
                        'encoder_topology':     vae.encoder.topology,
                        'decoder_topology':     vae.decoder.topology,
                        'latent_dim':           vae.latent_dim,
                        'encoder_type':         vae.encoder_type,
                        'mode':                 vae.mode,
                        'model_out':            vae.model_out
                        }
        torch.save(data_to_save, path)

    def trainer_eval(self, net=None):
        self.training = False
        if net is not None:
            net.eval()

    def trainer_train(self, net=None):
        self.training = True
        if net is not None:
            net.train()

    def get_test_group_weight(self, test_group):
        low_threshold = eval(test_group.split('_')[0])
        low_threshold = (abs(low_threshold) - SENS_MEAN) / SENS_STD if self.abs_sens else low_threshold / SENS_STD
        for ii, th in enumerate(self.group_th):
            if low_threshold < th:
                return self.group_weights[ii]
        return self.group_weights[-1]


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
    return 0.5 * torch.sum((outputs - targets).pow(2) * weight_vec)


def coord_reconstruction_loss(targets, outputs, n, sigma, xquantize, yquantize):
    """
    :param targets: B X 2N where 0 dim is batch, 1 dim is [x1,y1,x2,y2,...] coord vec
    :param outputs: B X 2N where 0 dim is batch, 1 dim is [x1,y1,x2,y2,...] coord vec
    :param n: power of the gaussian. if n=1 we have the standard Gaussian, if n approaches infinity the gaussian
              approaches the box function
    :param sigma: denominator of the gaussian, controls the width of the gaussian
    :param xquantize: grid size in the x axis
    :param yquantize: grid size in the y axis
    :return:  computes a map out of the coordinates, then computes a loss for all of them
    """
    mse_sum = 0
    for ii in range(targets.size()[0]):  # going over all the batch
        target_map = coord2map(outputs[ii, :], n, sigma, xquantize, yquantize)
        output_map = coord2map(outputs[ii, :], n, sigma, xquantize, yquantize)
        mse_sum += grid_mse(target_map, output_map)
    return mse_sum / targets.size()[0]


def coord2map(coordinates_vec, n, sigma, xquantize, yquantize):
    """
    :param coordinates_vec: 1D vector of coordinates [x1,y1,x2,y2,...]
    :param n: power of the gaussian. if n=1 we have the standard Gaussian, if n approaches infinity the gaussian
              approaches the box function
    :param sigma: denominator of the gaussian, controls the width of the gaussian
    :param xquantize: grid size in the x axis
    :param yquantize: grid size in the y axis
    :return: Creates
    """
    # =================================================
    # Local variables
    # =================================================
    grid  = torch.zeros([yquantize, xquantize])
    x_vec = np.arange(0, xquantize)
    y_vec = np.arange(0, yquantize)
    xgrid, ygrid = np.meshgrid(x_vec, y_vec)
    xgrid, ygrid = torch.tensor(xgrid), torch.Tensor(ygrid)

    for ii in range(len(coordinates_vec) // 2):
        grid += np.exp(-1*((xgrid - coordinates_vec[2*ii])**(2*n) + (ygrid - coordinates_vec[2*ii+1])**(2*n)) / (2*sigma**2))
    return grid


def grid_mse(targets, outputs):
    return 0.5 * torch.sum((targets - outputs).pow(2))

