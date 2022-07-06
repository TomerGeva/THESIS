import os
import math
from ConfigVAE import *
from LoggerGeneric import LoggerGeneric
from database_functions import ModelManipulationFunctions
from auxiliary_functions import compute_output_dim


class LoggerVAE(LoggerGeneric):
    """
    This class holds the logger for the Variational auto-encoder
    """
    def __init__(self, logdir=None, filename=None, write_to_file=True):
        super().__init__(logdir=logdir, filename=filename, write_to_file=write_to_file)

    # ==================================================================================================================
    # Basic help functions, internal
    # ==================================================================================================================
    def _get_result_string_train(self, sens_wmse_loss, d_kl, grid_mse_loss, cost):
        temp_str = 'Sensitivity W_MSE: {0:^' + str(self.result_space) +\
                   'f} D_kl: {1:^' + str(self.result_space) + \
                   'f} Grid MSE : {2:^' + str(self.result_space) + \
                   'f} Total cost: {3:^' + str(self.result_space) + 'f}'
        return temp_str.format(sens_wmse_loss, d_kl, grid_mse_loss, cost)

    def _get_result_string_test(self, sens_mse, grid_mse, weight):
        temp_str = 'Sensitivity W_MSE: {0:^' + str(self.result_space) +\
                   'f} Sensitivity MSE {1:^' + str(self.result_space) + \
                   'f} Grid MSE: {2:^' + str(self.result_space) + \
                   'f} Group weight: {3:^' + str(self.result_space) + 'f}'
        return temp_str.format(sens_mse*weight, sens_mse, grid_mse, weight)

    # ==================================================================================================================
    # Logging functions
    # ==================================================================================================================
    def log_epoch_results_train(self, header, sens_wmse, d_kl, grid_mse, cost):
        self.log_line(self.get_header(header) + self._get_result_string_train(sens_wmse, d_kl, grid_mse, cost))
        self.end_log()
        if self.write_to_file:
            full_path = os.path.join(self.logdir, self.filename)
            self.fileID = open(full_path, 'a')

    def log_epoch_results_test(self, header, sens_mse, grid_mse, weight):
        self.log_line(self.get_header(header) + self._get_result_string_test(sens_mse, grid_mse, weight))
        self.end_log()
        if self.write_to_file:
            full_path = os.path.join(self.logdir, self.filename)
            self.fileID = open(full_path, 'a')

    def log_model_arch(self, mod_vae):
        """
        :param mod_vae:Trained model
        :return: function logs the VAE architecture
        """
        mmf = ModelManipulationFunctions()
        # ==============================================================================================================
        # Init variables
        # ==============================================================================================================
        x_dim_size  = XQUANTIZE
        y_dim_size  = YQUANTIZE
        out_channel = IMG_CHANNELS
        conv_idx    = 0
        maxpool_idx = 0
        linear_idx  = 0
        # ==============================================================================================================
        # Encoder log
        # ==============================================================================================================
        self.log_title('VAE Encoder architecture')
        self.log_line('Number of parameters: ' + str(mmf.get_nof_params(mod_vae)))
        self.log_line('Input size: {}X{}' .format(x_dim_size, y_dim_size))
        for action in mod_vae.encoder.topology:
            # ------------------------------------------------------------------------------------------------------
            # For each action, computing the output size and logging
            # ------------------------------------------------------------------------------------------------------
            x_dim_size, y_dim_size, channels_temp = compute_output_dim(x_dim_size, y_dim_size, out_channel, action)
            self._get_layer_log_string(x_dim_size, y_dim_size, out_channel, action)
            out_channel = channels_temp
        # ==============================================================================================================
        # Decoder log
        # ==============================================================================================================
        self.log_title('VAE Decoder architecture')
        self.log_line('Input size: {}'.format(mod_vae.latent_dim))
        for action in mod_vae.decoder.topology:
            # ------------------------------------------------------------------------------------------------------
            # For each action, computing the output size and logging
            # ------------------------------------------------------------------------------------------------------
            x_dim_size, y_dim_size, channels_temp = compute_output_dim(x_dim_size, y_dim_size, out_channel, action)
            self._get_layer_log_string(x_dim_size, y_dim_size, out_channel, action)
            out_channel = channels_temp

    def log_dense_model_arch(self, mod_vae):
        """
        :param mod_vae:Trained model
        :return: function logs the VAE architecture
        """
        # ==============================================================================================================
        # Init variables
        # ==============================================================================================================
        x_dim_size  = XQUANTIZE
        y_dim_size  = YQUANTIZE
        channels    = IMG_CHANNELS

        # ==============================================================================================================
        # Encoder log
        # ==============================================================================================================
        self.log_title('Mode is {} ; Model output is {}' .format(mod_vae.mode.name, mod_vae.model_out.name))
        self.log_title('VAE Dense Encoder architecture')
        self.log_line('Input size: {}X{}X{}'.format(channels, x_dim_size, y_dim_size))
        for action in mod_vae.encoder.topology:
            # ------------------------------------------------------------------------------------------------------
            # For each action, computing the output size and logging
            # ------------------------------------------------------------------------------------------------------
            x_dim_size, y_dim_size, channels_temp = compute_output_dim(x_dim_size, y_dim_size, channels, action)
            self._get_layer_log_string(x_dim_size, y_dim_size, channels, action)
            channels = channels_temp
        # ==============================================================================================================
        # Decoder log
        # ==============================================================================================================
        self.log_title('VAE Decoder architecture')
        self.log_line('Input size: {}'.format(mod_vae.latent_dim))
        for action in mod_vae.decoder.topology:
            # ------------------------------------------------------------------------------------------------------
            # For each action, computing the output size and logging
            # ------------------------------------------------------------------------------------------------------
            x_dim_size, y_dim_size, channels_temp = compute_output_dim(x_dim_size, y_dim_size, channels, action)
            self._get_layer_log_string(x_dim_size, y_dim_size, channels, action)
            channels = channels_temp
