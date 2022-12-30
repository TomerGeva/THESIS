from torch.autograd import Variable
from time import time
import os
import torch
import torch.nn as nn
import torch.optim as optim


class TrainerOmega:
    def __init__(self,
                 model,
                 num_epochs,
                 lr=1e-2,
                 mom=0.9,
                 sched_step=20,
                 sched_gamma=0.5,
                 grad_clip=5,
                 training=True,
                 optimize_time=False,
                 xquantize=2500,
                 yquantize=2500):
        # -------------------------------------
        # cost function
        # -------------------------------------
        self.loss      = nn.MSELoss(reduction='sum')
        self.xquantize = xquantize
        self.yquantize = yquantize
        # -------------------------------------
        # optimizer
        # -------------------------------------
        # self.optimizer      = optim.SGD(net.parameters(), lr=lr, momentum=mom)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.learning_rate = lr
        self.mom = mom
        self.grad_clip = grad_clip
        # -------------------------------------
        # Scheduler, reduces learning rate
        # -------------------------------------
        self.optimize_time = optimize_time
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=sched_step,
                                                   gamma=sched_gamma)
        # -------------------------------------
        # Initializing the start epoch to zero
        # if not zero, the model is pre-trained
        # -------------------------------------
        self.epoch      = 0
        self.num_epochs = num_epochs
        # -------------------------------------
        # Misc training parameters
        # -------------------------------------
        self.cost       = []
        self.training   = training

    def compute_loss(self, omega_targets, omega_outputs):
        return self.loss(omega_targets, omega_outputs)

    def run_single_epoch(self, model, loader):
        # ==========================================================================================
        # Init variables
        # ==========================================================================================
        loss    = 0
        counter = 0
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
            omega_targets = sample['omega'].float().to(model.device)
            pcurrents     = Variable(sample['pcurrents'].float()).to(model.device)
            # ------------------------------------------------------------------------------
            # Forward pass
            # ------------------------------------------------------------------------------
            omega_outputs = model(pcurrents)
            # ------------------------------------------------------------------------------
            # Cost computations
            # ------------------------------------------------------------------------------
            batch_loss = self.loss(omega_targets, omega_outputs)
            loss += batch_loss.item()
            counter += omega_targets.size(0)
            # ------------------------------------------------------------------------------
            # Back propagation
            # ------------------------------------------------------------------------------
            if self.training:
                for param in model.parameters():  # zero gradients
                    param.grad = None
                # gpu_usage()  # DEBUG
                batch_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optimizer.step()
        return loss / counter, counter

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
            test_loss, counter = self.run_single_epoch(model, loader)
        model.train()
        self.trainer_train()
        return test_loss, counter

    def trainer_eval(self, model=None):
        self.training = False
        if model is not None:
            model.eval()

    def trainer_train(self, model=None):
        self.training = True
        if model is not None:
            model.train()

    def train(self, model, logger, train_loader, test_loader, valid_loader=None, save_per_epochs=1):
        # ==========================================================================================
        # Init Log
        # ==========================================================================================
        logger.filename = 'logger_omega.txt'
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
        logger.log_title('Beginning Training! ! ! ! number of epochs: {}' .format(self.num_epochs))
        model.train()
        for epoch in range(self.epoch, self.num_epochs):
            t = time()
            # ------------------------------------------------------------------------------
            # Training a single epoch + logging
            # ------------------------------------------------------------------------------
            train_loss, _ = self.run_single_epoch(model, train_loader)
            logger.log_epoch(epoch, t)
            logger.log_epoch_results('train', train_loss)
            # ------------------------------------------------------------------------------
            # Testing accuracy at the end of the epoch and logging
            # ------------------------------------------------------------------------------
            if valid_loader is not None:
                valid_loss, _ = self.test_model(model, valid_loader)
                logger.log_epoch_results('valid', valid_loss)
            test_loss, _ = self.test_model(model, test_loader)
            logger.log_epoch_results('test', test_loss)
            # ------------------------------------------------------------------------------
            # Advancing the scheduler of the lr
            # ------------------------------------------------------------------------------
            self.scheduler.step()
            # ------------------------------------------------------------------------------
            # Saving the training state
            # save every x epochs and on the last epoch
            # ------------------------------------------------------------------------------
            if epoch % save_per_epochs == 0 or epoch == self.num_epochs - 1:
                self.save_state_train(logger.logdir, model, epoch, self.learning_rate, self.mom)

    def save_state_train(self, logdir, model, epoch, lr, mom, filename=None):
        """Saving model and optimizer to drive, as well as current epoch and loss
        # When saving a general checkpoint, to be used for either inference or resuming training, you must save more
        # than just the model’s state_dict.
        # It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are
        # updated as the model trains.
        """
        if filename is None:
            name = 'Omega_model_data_lr_' + str(lr) + '_epoch_' + str(epoch) + '.tar'
            path = os.path.join(logdir, name)
        else:
            path = os.path.join(logdir, filename)

        data_to_save = {'epoch': epoch,
                        'model_state_dict':     model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr':                   lr,
                        'mom':                  mom,
                        'topology':             model.topology,
                        }
        torch.save(data_to_save, path)
