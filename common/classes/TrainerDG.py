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
        return loss / counter, counter

    def test_model(self, model, loader):
        # ==========================================================================================
        # Begin of testing
        # ==========================================================================================
        model.eval()
        self.trainer_eval()
        with torch.no_grad():
            test_loss, counter = self.run_single_epoch(model, loader)
        model.train()
        self.trainer_train()
        return test_loss, counter

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
            t = time()
            # ----------------------------------------------------------------------------------
            # Training single epoch
            # ----------------------------------------------------------------------------------
            train_loss = self.run_single_epoch(model, train_loader)
            # ----------------------------------------------------------------------------------
            # Logging training results
            # ----------------------------------------------------------------------------------
            logger.log_epoch(epoch, t)
            logger.log_epoch_results_train('train_weighted', train_loss)
            # ----------------------------------------------------------------------------------
            # Testing accuracy
            # ----------------------------------------------------------------------------------
            test_sens_mse_vec = []
            test_counters_vec = []
            for key in test_loaders:
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Testing the results of the current group
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                test_loss, counter = self.test_model(model, test_loaders[key])
                test_sens_mse_vec.append(test_loss)
                test_counters_vec.append(counter)
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Getting the respective group weight, logging
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                test_mse_weight = self.get_test_group_weight(key)
                logger.log_epoch_results_test(key, test_loss, test_mse_weight)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Computing total cost for all test loaders and logging with LoggerVAE
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            test_sens_mse = 0.0
            test_counter  = 0
            for sens_mse, count in zip(test_sens_mse_vec, test_counters_vec):
                test_sens_mse += (sens_mse * count)
                test_counter  += count
            test_sens_mse   = test_sens_mse / test_counter
            logger.log_epoch_results_test('test_total', test_sens_mse, 0)
            # ------------------------------------------------------------------------------
            # Advancing the scheduler of the lr
            # ------------------------------------------------------------------------------
            self.scheduler.step()
            # ------------------------------------------------------------------------------
            # Saving the training state
            # save every x epochs and on the last epoch
            # ------------------------------------------------------------------------------
            if epoch % save_per_epochs == 0 or epoch == epochs - 1 or test_sens_mse_vec[-1] < init_mse:
                self.save_state_train(logger.logdir, model, epoch, self.learning_rate, self.mom, self.beta_dkl, self.beta_grid)
                if test_sens_mse_vec[-1] < init_mse:
                    init_mse = test_sens_mse_vec[-1]

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

    def trainer_eval(self, model=None):
        self.training = False
        if model is not None:
            model.eval()

    def trainer_train(self, model=None):
        self.training = True
        if model is not None:
            model.train()

    def get_test_group_weight(self, test_group):
        low_threshold = eval(test_group.split('_')[0])
        for ii, th in enumerate(self.group_th):
            if low_threshold < th:
                return self.group_weights[ii]
        return self.group_weights[-1]


