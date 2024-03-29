import os
from ConfigCNN import *
from LoggerGeneric import LoggerGeneric
from database_functions import ModelManipulationFunctions
from auxiliary_functions import compute_output_dim


class LoggerCNN(LoggerGeneric):
    def __init__(self, logdir=None, filename=None, write_to_file=True):
        super().__init__(logdir=logdir, filename=filename, write_to_file=write_to_file)

    # ==================================================================================================================
    # Basic help functions, internal
    # ==================================================================================================================
    def _get_result_string_train(self, sens_mse_loss_wn, sens_mse_loss_w, sens_mse_loss):
        temp_str = 'Sensitivity_MSE_WN: {0:^' + str(self.result_space) + \
                   'f} Sensitivity_MSE_W: {1:^' + str(self.result_space) + \
                   'f} Sensitivity_MSE: {2:^' + str(self.result_space) + 'f}'
        return temp_str.format(sens_mse_loss_wn, sens_mse_loss_w, sens_mse_loss)

    def _get_result_string_test(self, sens_mse_loss_wn, sens_mse_loss_w, sens_mse_loss, weight=0):
        temp_str = 'Sensitivity_MSE_WN: {0:^' + str(self.result_space) + \
                   'f} Sensitivity_MSE_W: {1:^' + str(self.result_space) + \
                   'f} Sensitivity_MSE: {2:^' + str(self.result_space) + \
                   'f} Group_weight: {3:^' + str(self.result_space) + 'f}'
        return temp_str.format(sens_mse_loss_wn, sens_mse_loss_w, sens_mse_loss, weight)

    def _get_layer_log_string(self, x_dim_size, y_dim_size, channels, action):
            if 'convTrans' in action[0]:
                self.log_line(self.get_header(action[0]) + self._get_conv_transpose_layer_string(action[1].in_channels,
                                                                                                 action[1].out_channels,
                                                                                                 action[1].kernel,
                                                                                                 action[1].stride,
                                                                                                 action[1].padding,
                                                                                                 action[
                                                                                                     1].output_padding,
                                                                                                 action[1].bnorm,
                                                                                                 action[1].drate,
                                                                                                 action[1].act,
                                                                                                 x_dim_size,
                                                                                                 y_dim_size))
            elif 'modedgeconv' in action[0]:
                self.log_line(self.get_header(action[0]) + self._get_edgeconv_layer_string(action[1].k,
                                                                                           action[1].conv_data.bias,
                                                                                           action[1].conv_data.bnorm,
                                                                                           action[1].conv_data.drate,
                                                                                           action[1].conv_data.act,
                                                                                           action[1].aggregation))
            elif 'res-conv' in action[0]:
                self.log_line(self.get_header(action[0]) + self._get_residual_conv_layer_string(action[1].in_channels,
                                                                                                action[1].out_channels,
                                                                                                action[1].layers,
                                                                                                action[1].kernel,
                                                                                                action[1].stride,
                                                                                                action[1].padding,
                                                                                                action[1].bnorm,
                                                                                                action[1].drate,
                                                                                                action[1].act,
                                                                                                x_dim_size,
                                                                                                y_dim_size))
            elif 'dense' in action[0]:
                self.log_line(self.get_header(action[0]) + self._get_dense_layer_string(channels,
                                                                                        action[1].depth,
                                                                                        action[1].growth,
                                                                                        action[1].kernel,
                                                                                        action[1].stride,
                                                                                        action[1].padding,
                                                                                        action[1].bnorm,
                                                                                        action[1].drate,
                                                                                        action[1].act,
                                                                                        x_dim_size,
                                                                                        y_dim_size))
            elif 'transition' in action[0]:
                self.log_line(self.get_header(action[0]) + self._get_transition_layer_string(channels,
                                                                                             action[1].reduction_rate,
                                                                                             action[1].kernel,
                                                                                             action[1].stride,
                                                                                             action[1].padding,
                                                                                             action[1].bnorm,
                                                                                             action[1].drate,
                                                                                             action[1].act,
                                                                                             action[1].pool_type,
                                                                                             action[1].pool_size,
                                                                                             x_dim_size,
                                                                                             y_dim_size))
            elif 'res-linear' in action[0]:
                self.log_line(self.get_header(action[0]) + self._get_residual_linear_layer_string(action[1].in_neurons,
                                                                                                  action[1].out_neurons,
                                                                                                  action[1].layers,
                                                                                                  action[1].bnorm,
                                                                                                  action[1].drate,
                                                                                                  action[1].act))
            elif 'linear' in action[0]:
                self.log_line(self.get_header(action[0]) + self._get_linear_layer_string(action[1].in_neurons,
                                                                                         action[1].out_neurons,
                                                                                         action[1].bias,
                                                                                         action[1].bnorm,
                                                                                         action[1].drate,
                                                                                         action[1].act))
            elif 'conv' in action[0]:
                self.log_line(self.get_header(action[0]) + self._get_conv_layer_string(action[1].in_channels,
                                                                                       action[1].out_channels,
                                                                                       action[1].kernel,
                                                                                       action[1].stride,
                                                                                       action[1].padding,
                                                                                       action[1].bnorm,
                                                                                       action[1].drate,
                                                                                       action[1].act,
                                                                                       x_dim_size,
                                                                                       y_dim_size))
            elif 'pool' in action[0]:
                self.log_line(self.get_header(action[0]) + self._get_pool_layer_string(action[1].kernel,  # kernel
                                                                                       x_dim_size,
                                                                                       y_dim_size))
            elif 'transformer' in action[0]:
                self.log_line(self.get_header(action[0]) + self._get_transformer_layer_string(action[1].patch_size_x,
                                                                                              action[1].patch_size_y,
                                                                                              action[1].embed_size))

    # ==================================================================================================================
    # Logging functions
    # ==================================================================================================================
    def log_epoch_results_train(self, header, sens_mse_loss_wn, sens_mse_loss_w, sens_mse_loss):
        self.log_line(self.get_header(header) + self._get_result_string_train(sens_mse_loss_wn, sens_mse_loss_w, sens_mse_loss))
        self.end_log()
        if self.write_to_file:
            full_path = os.path.join(self.logdir, self.filename)
            self.fileID = open(full_path, 'a')

    def log_epoch_results_test(self, header, sens_mse_loss_wn, sens_mse_loss_w, sens_mse_loss, weight=0):
        self.log_line(self.get_header(header) + self._get_result_string_test(sens_mse_loss_wn, sens_mse_loss_w, sens_mse_loss, weight))
        self.end_log()
        if self.write_to_file:
            full_path = os.path.join(self.logdir, self.filename)
            self.fileID = open(full_path, 'a')

    def log_model_arch(self, model):
        """
        :param model:Trained model
        :return: function logs the VAE architecture
        """
        mmf = ModelManipulationFunctions()
        # ==============================================================================================================
        # Init variables
        # ==============================================================================================================
        x_dim_size = XQUANTIZE
        y_dim_size = YQUANTIZE
        out_channel = IMG_CHANNELS
        conv_idx = 0
        maxpool_idx = 0
        linear_idx = 0
        # ==============================================================================================================
        # Encoder log
        # ==============================================================================================================
        self.log_title('CNN architecture')
        self.log_line('Number of parameters: ' + str(mmf.get_nof_params(model)))
        self.log_line('Input size: {}X{}'.format(x_dim_size, y_dim_size))
        for action in model.topology:
            # ------------------------------------------------------------------------------------------------------
            # For each action, computing the output size and logging
            # ------------------------------------------------------------------------------------------------------
            x_dim_size, y_dim_size, channels_temp = compute_output_dim(x_dim_size, y_dim_size, out_channel, action)
            self._get_layer_log_string(x_dim_size, y_dim_size, out_channel, action)
            out_channel = channels_temp
