import os
import math
from ConfigVAE import *
from LoggerGeneric import LoggerGeneric
from database_functions import ModelManipulationFunctions


class LoggerDG(LoggerGeneric):
    def __init__(self, logdir=None, filename=None, write_to_file=True):
        super().__init__(logdir=logdir, filename=filename, write_to_file=write_to_file)

    # ==================================================================================================================
    # Basic logging functions
    # ==================================================================================================================
    def _get_result_string_train(self, loss_sens):
        temp_str = 'Sensitivity loss weighted: {0:^' + str(self.result_space) + 'f}'
        return temp_str.format(loss_sens)

    def _get_result_string_test(self, loss_sens, weight):
        temp_str = 'Sensitivity loss weighted: {0:^' + str(self.result_space) + \
                   'f} Sensitivity loss {1:^' + str(self.result_space) + \
                   'f} Group weight: {2:^' + str(self.result_space) + 'f}'
        return temp_str.format(loss_sens * weight, loss_sens, weight)

    def _get_layer_log_string(self, action):
        if 'conv1d' in action[0]:
            self.log_line(self.get_header(action[0]) + self._get_conv_layer_string(action[1].in_channels,
                                                                                   action[1].out_channels,
                                                                                   action[1].kernel,
                                                                                   action[1].stride,
                                                                                   action[1].padding,
                                                                                   action[1].bnorm,
                                                                                   action[1].drate,
                                                                                   action[1].act,
                                                                                   0,
                                                                                   0))
        elif 'pool1d' in action[0]:
            self.log_line(self.get_header(action[0]) + self._get_pool_layer_string(action[1].kernel,  # kernel
                                                                                   0,
                                                                                   0))
        elif 'linear' in action[0]:
            self.log_line(self.get_header(action[0]) + self._get_linear_layer_string(action[1].in_neurons,
                                                                                     action[1].out_neurons,
                                                                                     action[1].bnorm,
                                                                                     action[1].drate,
                                                                                     action[1].act))

    # ==================================================================================================================
    # Logging functions
    # ==================================================================================================================
    def log_epoch_results_train(self, header, loss_sens):
        self.log_line(self.get_header(header) + self._get_result_string_train(loss_sens))
        self.end_log()
        if self.write_to_file:
            full_path = os.path.join(self.logdir, self.filename)
            self.fileID = open(full_path, 'a')

    def log_epoch_results_test(self, header, loss_sens, weight):
        self.log_line(self.get_header(header) + self._get_result_string_test(loss_sens, weight))
        self.end_log()
        if self.write_to_file:
            full_path = os.path.join(self.logdir, self.filename)
            self.fileID = open(full_path, 'a')

    def log_model_arch(self, model):
        mmf = ModelManipulationFunctions()
        # ----------------------------------------------------------------------------------------------------------
        # Logging titles
        # ----------------------------------------------------------------------------------------------------------====
        self.log_title('Model architecture')
        self.log_line('Number of parameters: ' + str(mmf.get_nof_params(model)))
        # ----------------------------------------------------------------------------------------------------------
        # Logging layers
        # ----------------------------------------------------------------------------------------------------------====
        for action in model.topology:
            self._get_layer_log_string(action)
