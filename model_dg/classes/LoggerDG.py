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

    def _get_result_string_test(self, loss_sens, weight, sens_mse_unweighted=None):
        temp_str = 'Sensitivity loss weighted: {0:^' + str(self.result_space) + \
                   'f} Sensitivity loss {1:^' + str(self.result_space) + \
                   'f} Group weight: {2:^' + str(self.result_space) + 'f}'
        if sens_mse_unweighted is None:
            return temp_str.format(loss_sens * weight, loss_sens, weight)
        else:
            return temp_str.format(loss_sens, sens_mse_unweighted, weight)

    def _get_layer_log_string(self, action):
        if 'edgeconv' in action[0]:
            self.log_line(self.get_header(action[0]) + self._get_edgeconv_layer_string(action[1].k,
                                                                                       action[1].conv_data.bias,
                                                                                       action[1].conv_data.bnorm,
                                                                                       action[1].conv_data.drate,
                                                                                       action[1].conv_data.act,
                                                                                       action[1].aggregation))
        elif 'conv1d' in action[0]:
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
        elif 'adapool1d' in action[0]:
            self.log_line(self.get_header(action[0]) + self._get_adapool_layer_string(action[1].out_size))
        elif 'pool1d' in action[0]:
            self.log_line(self.get_header(action[0]) + self._get_pool_layer_string(action[1].kernel,  # kernel
                                                                                   0,
                                                                                   0))
        elif 'linear' in action[0]:
            self.log_line(self.get_header(action[0]) + self._get_linear_layer_string(action[1].in_neurons,
                                                                                     action[1].out_neurons,
                                                                                     action[1].bias,
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

    def log_epoch_results_test(self, header, loss_sens, weight, sens_mse_unweighted=None):
        self.log_line(self.get_header(header) + self._get_result_string_test(loss_sens, weight, sens_mse_unweighted))
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
        try:
            self.log_line('Concatenating the edgeconv outputs: ' + str(model.concat_edge))
            self.log_line('Adaptive pooling avg/max: ' + str(model.flatten_type))
        except:
            pass
        # ----------------------------------------------------------------------------------------------------------
        # Logging layers
        # ----------------------------------------------------------------------------------------------------------====
        for action in model.topology:
            self._get_layer_log_string(action)
