import os
import math
from ConfigVAE import *
from LoggerGeneric import LoggerGeneric


class LoggerVAE(LoggerGeneric):
    """
    This class holds the logger for the Variational auto-encoder
    """
    def __init__(self, logdir=None, filename=None, write_to_file=True):
        super().__init__(logdir=logdir, filename=filename, write_to_file=write_to_file)

    # ==================================================================================================================
    # Basic help functions, internal
    # ==================================================================================================================
    def _get_result_string_train(self, wmse_loss, d_kl, cost):
        temp_str = 'W_MSE loss: {0:^' + str(self.result_space) +\
                   'f} D_kl: {1:^' + str(self.result_space) +\
                   'f} Total cost: {2:^' + str(self.result_space) + 'f}'
        return temp_str.format(wmse_loss, d_kl, cost)

    def _get_result_string_test(self, wmse_loss, weight):
        temp_str = 'W_MSE loss: {0:^' + str(self.result_space) +\
                   'f} MSE loss {1:^' + str(self.result_space) +\
                   'f} Group weight: {2:^' + str(self.result_space) + 'f}'
        return temp_str.format(wmse_loss*weight, wmse_loss, weight)

    # ==================================================================================================================
    # Regular VAE log functions, used to log the layer architecture from the description
    # ==================================================================================================================
    def _get_conv_layer_string(self, in_ch, out_ch, ktilde, stride, pad, bnorm, drate, active, x_dim, y_dim):
        temp_str = 'In channels:    {0:^' + str(self.desc_space) +\
                   'd} Out channels:{1:^' + str(self.desc_space) +\
                   'd} Kernel:      {2:^' + str(self.desc_space) + \
                   'd} Stride:      {3:^' + str(self.desc_space) + \
                   'd} Padding:     {4:^' + str(self.desc_space) + \
                   'd} batch_norm:  {5:^' + str(self.desc_space) + \
                   's} drop_rate:   {6:^' + str(self.desc_space) + \
                   'd} activation:  {7:^' + str(self.desc_space) + \
                   's} Output size: {8:^' + str(self.desc_space) + '}X{9:^' + str(self.desc_space) + '}'
        return temp_str.format(in_ch, out_ch, ktilde, stride, pad, str(bnorm), drate, active.name, x_dim, y_dim)

    def _get_pool_layer_string(self, ktilde, x_dim, y_dim):
        temp_str = 'K^tilde: {0:1d} Output size: {1:^' + str(self.desc_space) + '}X{2:^' + str(self.desc_space) + '}'
        return temp_str.format(ktilde, x_dim, y_dim)

    def _get_linear_layer_string(self, num_in, num_out, bnorm, drate, active):
        temp_str = 'Input size:     {0:^' + str(self.desc_space) +\
                   'd} Output size: {1:^' + str(self.desc_space) + \
                   'd} batch_norm:  {2:^' + str(self.desc_space) + \
                   's} drop_rate:   {3:^' + str(self.desc_space) + \
                   'd} activation:  {4:^' + str(self.desc_space) + 's}'
        return temp_str.format(num_in, num_out, str(bnorm), drate, active.name)

    # ==================================================================================================================
    # Dense VAE log functions, used to log the layer architecture
    # ==================================================================================================================
    def _get_dense_layer_string(self, in_ch, depth, growth, ktilde, stride, pad, bnorm, drate, active, x_dim, y_dim):
        temp_str = 'In channels:    {0:^' + str(self.desc_space) + \
                   'd} Depth:       {1:^' + str(self.desc_space) + \
                   'd} Growth rate: {2:^' + str(self.desc_space) + \
                   'd} Kernel:      {3:^' + str(self.desc_space) + \
                   'd} Stride:      {4:^' + str(self.desc_space) + \
                   'd} Padding:     {5:^' + str(self.desc_space) + \
                   'd} batch_norm:  {6:^' + str(self.desc_space) + \
                   's} drop_rate:   {7:^' + str(self.desc_space) + \
                   'd} activation:  {8:^' + str(self.desc_space) + \
                   's} Output size: {9:^' + str(self.desc_space) + \
                   '}X{10:^' + str(self.desc_space) +\
                   '}X{11:^' + str(self.desc_space) + '}'
        return temp_str.format(in_ch, depth, growth, ktilde, stride, pad, str(bnorm), drate, active.name,
                               in_ch+depth*growth, x_dim, y_dim)

    def _get_transition_layer_string(self, in_ch, reduction, ktilde, stride, pad, bnorm, drate, active, pool_type,
                                     pool_size, x_dim, y_dim):
        temp_str = 'In channels:    {0:^' + str(self.desc_space) + \
                   'd} Reduction:   {1:^' + str(self.desc_space) + \
                   '.1f} Kernel:      {2:^' + str(self.desc_space) + \
                   'd} Stride:      {3:^' + str(self.desc_space) + \
                   'd} Padding:     {4:^' + str(self.desc_space) + \
                   'd} batch_norm:  {5:^' + str(self.desc_space) + \
                   's} drop_rate:   {6:^' + str(self.desc_space) + \
                   'd} activation:  {7:^' + str(self.desc_space) + \
                   's} Pool type:   {8:^' + str(self.desc_space) + \
                   's} Pool size:   {9:^' + str(self.desc_space) + \
                   'd} Output size: {10:^' + str(self.desc_space) + \
                   '}X{11:^' + str(self.desc_space) +\
                   '}X{12:^' + str(self.desc_space) + '}'
        return temp_str.format(in_ch, reduction, ktilde, stride, pad, str(bnorm), drate, active.name, pool_type.name,
                               pool_size, math.floor(in_ch*reduction), x_dim, y_dim)

    # ==================================================================================================================
    # Logging functions
    # ==================================================================================================================
    def log_epoch(self, epoch_num):
        self.log_line('Epoch: {0:5d}'.format(epoch_num))

    def log_epoch_results_train(self, header, wmse_loss, d_kl, cost):
        self.log_line(self.get_header(header) + self._get_result_string_train(wmse_loss, d_kl, cost))
        self.end_log()
        if self.write_to_file:
            full_path = os.path.join(self.logdir, self.filename)
            self.fileID = open(full_path, 'a')

    def log_epoch_results_test(self, header, mse_loss, weight):
        self.log_line(self.get_header(header) + self._get_result_string_test(mse_loss, weight))
        self.end_log()
        if self.write_to_file:
            full_path = os.path.join(self.logdir, self.filename)
            self.fileID = open(full_path, 'a')

    def log_model_arch(self, mod_vae):
        """
        :param mod_vae:Trained model
        :return: function logs the VAE architecture
        """
        # ==============================================================================================================
        # Init variables
        # ==============================================================================================================
        x_dim_size  = XQUANTIZE
        y_dim_size  = YQUANTIZE
        conv_idx    = 0
        maxpool_idx = 0
        linear_idx  = 0
        out_channel = 1
        # ==============================================================================================================
        # Encoder log
        # ==============================================================================================================
        self.log_title('VAE Encoder architecture')
        self.log_line('Input size: {}X{}' .format(x_dim_size, y_dim_size))

        for ii in range(len(mod_vae.encoder.topology)):
            # ------------------------------------------------------------------------------------------------------
            # For each action, computing the output size and logging
            # ------------------------------------------------------------------------------------------------------
            action = mod_vae.encoder.topology[ii]
            if 'conv' in action[0]:
                x_dim_size  = int((x_dim_size - (action[3] - action[4]) + 2 * action[5]) / action[4])
                y_dim_size  = int((y_dim_size - (action[3] - action[4]) + 2 * action[5]) / action[4])
                out_channel = action[2]
                self.log_line(self.get_header(action[0]) + self._get_conv_layer_string(action[1],  # in channels
                                                                                       action[2],  # out channels
                                                                                       action[3],  # kernel
                                                                                       action[4],  # stride
                                                                                       action[5],  # padding
                                                                                       x_dim_size,
                                                                                       y_dim_size))
                conv_idx += 1
            elif 'pool' in action[0]:
                x_dim_size = int(x_dim_size / action[1])
                y_dim_size = int(y_dim_size / action[1])
                self.log_line(self.get_header(action[0]) + self._get_pool_layer_string(action[1],  # kernel
                                                                                       x_dim_size,
                                                                                       y_dim_size))
                maxpool_idx += 1
            elif 'linear' in action[0]:
                if linear_idx == 0:
                    self.log_line(self.get_header(action[0]) + self._get_linear_layer_string(x_dim_size * y_dim_size * out_channel, action[1]))
                else:
                    action_last = mod_vae.encoder.topology[ii-1]
                    self.log_line(self.get_header(action[0]) + self._get_linear_layer_string(action_last[1], action[1]))
                linear_idx += 1

        # ==============================================================================================================
        # Decoder log
        # ==============================================================================================================
        self.log_title('VAE Decoder architecture')
        self.log_line('Input size: {}'.format(LATENT_SPACE_DIM))
        linear_idx = 0
        for ii in range(len(mod_vae.decoder.topology)):
            # ------------------------------------------------------------------------------------------------------
            # For each action, computing the output size and logging
            # ------------------------------------------------------------------------------------------------------
            action = mod_vae.decoder.topology[ii]
            if 'linear' in action[0]:
                if linear_idx == 0:
                    self.log_line(self.get_header(action[0]) + self._get_linear_layer_string(LATENT_SPACE_DIM, action[1]))
                else:
                    action_last = mod_vae.decoder.topology[ii - 1]
                    self.log_line(self.get_header(action[0]) + self._get_linear_layer_string(action_last[1], action[1]))
                linear_idx += 1

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
        self.log_title('VAE Dense Encoder architecture')
        self.log_line('Input size: {}X{}X{}'.format(channels, x_dim_size, y_dim_size))
        for ii in range(len(mod_vae.encoder.topology)):
            # ------------------------------------------------------------------------------------------------------
            # For each action, computing the output size and logging
            # ------------------------------------------------------------------------------------------------------
            action = mod_vae.encoder.topology[ii]
            if 'conv' in action[0]:
                x_dim_size = int((x_dim_size - (action[1].kernel - action[1].stride) + 2 * action[1].padding) / action[1].stride)
                y_dim_size = int((y_dim_size - (action[1].kernel - action[1].stride) + 2 * action[1].padding) / action[1].stride)

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
                channels = action[1].out_channels
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
                channels  += action[1].depth * action[1].growth
            elif 'transition' in action[0]:
                x_conv_size = int((x_dim_size - (action[1].kernel - action[1].stride) + 2 * action[1].padding) / action[1].stride)
                y_conv_size = int((y_dim_size - (action[1].kernel - action[1].stride) + 2 * action[1].padding) / action[1].stride)
                if type(action[1].pool_padding) is not tuple:
                    x_dim_size = int((x_conv_size + 2 * action[1].pool_padding) / action[1].pool_size)
                    y_dim_size = int((y_conv_size + 2 * action[1].pool_padding) / action[1].pool_size)
                else:
                    x_dim_size = int((x_conv_size + 2 * action[1].pool_padding[0] + 2 * action[1].pool_padding[1]) / action[1].pool_size)
                    y_dim_size = int((y_conv_size + 2 * action[1].pool_padding[2] + 2 * action[1].pool_padding[3]) / action[1].pool_size)
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
                channels = action[1].out_channels
            elif 'linear' in action[0]:
                self.log_line(self.get_header(action[0]) + self._get_linear_layer_string(action[1].in_neurons,
                                                                                         action[1].out_neurons,
                                                                                         action[1].bnorm,
                                                                                         action[1].drate,
                                                                                         action[1].act))

        # ==============================================================================================================
        # Decoder log
        # ==============================================================================================================
        self.log_title('VAE Decoder architecture')
        self.log_line('Input size: {}'.format(mod_vae.latent_dim))
        for ii in range(len(mod_vae.decoder.topology)):
            # ------------------------------------------------------------------------------------------------------
            # For each action, computing the output size and logging
            # ------------------------------------------------------------------------------------------------------
            action = mod_vae.decoder.topology[ii]
            if 'linear' in action[0]:
                self.log_line(self.get_header(action[0]) + self._get_linear_layer_string(action[1].in_neurons,
                                                                                         action[1].out_neurons,
                                                                                         action[1].bnorm,
                                                                                         action[1].drate,
                                                                                         action[1].act))