from ConfigCNN import *
from torch.autograd import Variable
from time import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from auxiliary_functions import weighted_mse, grid_mse, hausdorf_distance
from GPUtil import showUtilization as gpu_usage  # for debug only


class TrainerCNN:
    def __init__(self,
                 model,
                 lr=1e-2,
                 mom=0.9,
                 sched_step=20,
                 sched_gamma=0.5,
                 grad_clip=5,
                 group_thresholds=None,
                 group_weights=None,
                 abs_sens=True,
                 norm_sens=(0, 1),
                 training=True,
                 optimize_time=False,
                 xquantize=600,
                 yquantize=600):
        # -------------------------------------
        # cost function
        # -------------------------------------
        self.abs_sens         = abs_sens
        self.norm_sens        = norm_sens
        self.sensitivity_loss = weighted_mse
        self.model_type       = model.model_type
        self.xquantize        = xquantize
        self.yquantize        = yquantize
        # -------------------------------------
        # optimizer
        # -------------------------------------
        # self.optimizer      = optim.SGD(net.parameters(), lr=lr, momentum=mom)
        self.optimizer      = optim.Adam(model.parameters(), lr=lr)
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
        self.cost = []
        # MSE parameters
        self.group_th = group_thresholds
        self.group_weights = group_weights

        self.training        = training

    def compute_loss(self, sens_targets, sens_outputs):
        sens_mse_loss_weighted_normalized = self.sensitivity_loss(sens_targets, sens_outputs, self.group_weights, self.group_th, normalize=True)
        sens_mse_loss_weighted            = self.sensitivity_loss(sens_targets, sens_outputs, self.group_weights, self.group_th, normalize=False)
        sens_mse_loss                     = self.sensitivity_loss(sens_targets, sens_outputs)
        return sens_mse_loss_weighted_normalized, sens_mse_loss_weighted, sens_mse_loss

    def run_single_epoch(self, model, loader):
        # ==========================================================================================
        # Init variables
        # ==========================================================================================
        loss_sens_weighted_normalized = 0.0
        loss_sens_weighted            = 0.0
        counter                       = 0
        loss_sens                     = 0.0
        loader_iter = iter(loader)
        for _ in range(len(loader)):
            # ------------------------------------------------------------------------------
            # Working with iterables, much faster
            # ------------------------------------------------------------------------------
            try:
                sample = next(loader_iter)
            except StopIteration:
                break
            # ------------------------------------------------------------------------------
            # Extracting the grids and sensitivities
            # ------------------------------------------------------------------------------
            sens_targets = sample['sensitivity'].float().to(model.device)
            grids        = Variable(sample['grid_in'].float()).to(model.device)
            # ------------------------------------------------------------------------------
            # Forward pass
            # ------------------------------------------------------------------------------
            sens_outputs = model(grids)
            # ------------------------------------------------------------------------------
            # Cost computations
            # ------------------------------------------------------------------------------
            mse_loss_wn, mse_loss_w, mse_loss = self.compute_loss(sens_targets, sens_outputs)
            loss_sens_weighted_normalized += mse_loss_wn.item()
            loss_sens_weighted            += mse_loss_w.item()
            loss_sens                     += mse_loss.item()
            counter                       += sens_targets.size(0)
            # ------------------------------------------------------------------------------
            # Back propagation
            # ------------------------------------------------------------------------------
            if self.training:
                for param in model.parameters():  # zero gradients
                    param.grad = None
                # gpu_usage()  # DEBUG
                mse_loss_w.backward()
                # mse_loss_unweighted.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optimizer.step()

        return loss_sens_weighted_normalized / counter, loss_sens_weighted / counter, loss_sens / counter, counter

    def test_model(self, model, loader):
        # ==========================================================================================
        # Init variables
        # ==========================================================================================
        model.eval()
        self.trainer_eval()
        # ==========================================================================================
        # Begin of testing
        # ==========================================================================================
        with torch.no_grad():
            test_sens_mse_wn, test_sens_mse_w, test_sens_mse, counter = self.run_single_epoch(model, loader)
        model.train()
        self.trainer_train()
        return test_sens_mse_wn, test_sens_mse_w, test_sens_mse, counter

    def trainer_eval(self, model=None):
        self.training = False
        if model is not None:
            model.eval()

    def trainer_train(self, model=None):
        self.training = True
        if model is not None:
            model.train()

    def train(self, model, train_loader, test_loaders, logger, save_per_epochs=1):
        # ==========================================================================================
        # Init Log
        # ==========================================================================================
        logger.filename = 'logger_cnn.txt'
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
        logger.log_title('Beginning Training! ! ! ! number of epochs: {}' .format(EPOCH_NUM))
        model.train()
        for epoch in range(self.epoch, EPOCH_NUM):
            t = time()
            # ------------------------------------------------------------------------------
            # Training a single epoch
            # ------------------------------------------------------------------------------
            train_sens_mse_nw, train_sens_mse_w, train_sens_mse, counter = self.run_single_epoch(model, train_loader)
            # ------------------------------------------------------------------------------
            # Logging
            # ------------------------------------------------------------------------------
            logger.log_epoch(epoch, t)
            logger.log_epoch_results_train('train', train_sens_mse_nw, train_sens_mse_w, train_sens_mse)
            # ------------------------------------------------------------------------------
            # Testing accuracy at the end of the epoch, and logging with LoggerVAE
            # ------------------------------------------------------------------------------
            test_sens_mse_vec_weighted_normalized = []
            test_sens_mse_vec_weighted            = []
            test_sens_mse_vec                     = []
            test_counters_vec                     = []
            for key in test_loaders:
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Getting the respective group weight
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                test_mse_weight = self.get_test_group_weight(key)
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Testing the results of the current group
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                test_sens_mse_wn, test_sens_mse_w, test_sens_mse, test_counter = self.test_model(model, test_loaders[key])
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Logging
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                logger.log_epoch_results_test(key, test_sens_mse_wn, test_sens_mse_w, test_sens_mse, test_mse_weight)
                test_sens_mse_vec_weighted_normalized.append(test_sens_mse_wn)
                test_sens_mse_vec_weighted.append(test_sens_mse_w)
                test_sens_mse_vec.append(test_sens_mse)
                test_counters_vec.append(test_counter)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Computing total cost for all test loaders and logging with LoggerVAE
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            test_sens_mse_wn = 0.0
            test_sens_mse_w  = 0.0
            test_sens_mse    = 0.0
            test_counter     = 0
            for sens_mse_wn, sens_mse_w, sens_mse, count in zip(test_sens_mse_vec_weighted_normalized, test_sens_mse_vec_weighted,test_sens_mse_vec, test_counters_vec):
                test_sens_mse_wn += (sens_mse_wn * count)
                test_sens_mse_w  += (sens_mse_w * count)
                test_sens_mse    += (sens_mse * count)
                test_counter     += count
            test_sens_mse_wn = test_sens_mse_wn / test_counter
            test_sens_mse_w  = test_sens_mse_w / test_counter
            test_sens_mse    = test_sens_mse / test_counter
            logger.log_epoch_results_test('test_total', test_sens_mse_wn, test_sens_mse_w, test_sens_mse, 0)
            # ------------------------------------------------------------------------------
            # Advancing the scheduler of the lr
            # ------------------------------------------------------------------------------
            self.scheduler.step()
            # ------------------------------------------------------------------------------
            # Saving the training state
            # save every x epochs and on the last epoch
            # ------------------------------------------------------------------------------
            if epoch % save_per_epochs == 0 or epoch == EPOCH_NUM - 1:
                self.save_state_train(logger.logdir, model, epoch, self.learning_rate, self.mom, SENS_STD)

    def save_state_train(self, logdir, model, epoch, lr, mom, norm_fact, filename=None):
        """Saving model and optimizer to drive, as well as current epoch and loss
        # When saving a general checkpoint, to be used for either inference or resuming training, you must save more
        # than just the model’s state_dict.
        # It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are
        # updated as the model trains.
        """
        if filename is None:
            name = 'CNN_model_data_lr_' + str(lr) + '_epoch_' + str(epoch) + '.tar'
            path = os.path.join(logdir, name)
        else:
            path = os.path.join(logdir, filename)

        data_to_save = {'epoch': epoch,
                        'model_state_dict':     model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr':                   lr,
                        'mom':                  mom,
                        'norm_fact':            norm_fact,
                        'topology':             model.topology,
                        'model_type':           model.model_type,
                        }
        torch.save(data_to_save, path)

    def get_test_group_weight(self, test_group):
        low_threshold = eval(test_group.split('_')[0])
        if self.norm_sens != (0, 1):
            low_threshold = (abs(low_threshold) - self.norm_sens[0]) / self.norm_sens[1] if self.abs_sens else low_threshold / self.norm_sens[1]
        for ii, th in enumerate(self.group_th):
            if low_threshold < th:
                return self.group_weights[ii]
        return self.group_weights[-1]

