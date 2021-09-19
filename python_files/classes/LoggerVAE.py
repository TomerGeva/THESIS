import os
import math
import datetime
from ConfigVAE import *


class LoggerVAE:
    """
    This class holds the logger for the Variational auto-encoder
    """
    def __init__(self, logdir=None, filename=None, write_to_file=True):
        self.logdir         = logdir
        self.filename       = filename
        self.write_to_file  = write_to_file
        self.verbose        = 'INFO'
        self.header_space   = 16
        self.result_space   = 15.6
        self.desc_space     = 6
        self.fileID         = None

    # ==================================================================================================================
    # Basic help functions, internal
    # ==================================================================================================================
    def _get_header(self):
        temp_str = '|{0:^' + str(self.header_space) + '}| '
        return temp_str.format(self.verbose)

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
                   'd} K^tilde:     {2:^' + str(self.desc_space) + \
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
                   'd} K^tilde:     {3:^' + str(self.desc_space) + \
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

    def _get_transition_layer_string(self, in_ch, reduction, ktilde, stride, pad, bnorm, drate, active, pool_size,
                                     x_dim, y_dim):
        temp_str = 'In channels:    {0:^' + str(self.desc_space) + \
                   'd} Reduction:   {1:^' + str(self.desc_space) + \
                   '.1f} K^tilde:   {2:^' + str(self.desc_space) + \
                   'd} Stride:      {3:^' + str(self.desc_space) + \
                   'd} Padding:     {4:^' + str(self.desc_space) + \
                   'd} batch_norm:  {5:^' + str(self.desc_space) + \
                   's} drop_rate:   {6:^' + str(self.desc_space) + \
                   'd} activation:  {7:^' + str(self.desc_space) + \
                   's} Pool size:   {8:^' + str(self.desc_space) + \
                   'd} Output size: {9:^' + str(self.desc_space) + \
                   '}X{10:^' + str(self.desc_space) +\
                   '}X{11:^' + str(self.desc_space) + '}'
        return temp_str.format(in_ch, reduction, ktilde, stride, pad, str(bnorm), drate, active.name, pool_size,
                               math.floor(in_ch*reduction), x_dim, y_dim)

    # ==================================================================================================================
    # Logging functions
    # ==================================================================================================================
    def start_log(self):
        # ============================================================
        # Starting the logging
        # ============================================================
        if self.filename is None:
            # -------------------------------------
            # Creating the filename
            # -------------------------------------
            self.filename = 'logger.txt'

        time_data = datetime.datetime.now()
        time_list = [time_data.day, time_data.month, time_data.year, time_data.hour, time_data.minute]
        time_string = '_'.join([str(ii) for ii in time_list])
        del time_data, time_list
        if self.logdir is None:
            self.logdir = os.path.join(os.getcwd(), time_string)
        else:
            self.logdir = os.path.join(self.logdir, time_string)

        if self.write_to_file:
            # -------------------------------------
            # Creating the directory
            # -------------------------------------
            try:
                os.makedirs(self.logdir)
                print(self._get_header() +
                      '{0:s} {1:s}'.format(' Created new directory ', self.logdir))
            except OSError:
                pass
            # -------------------------------------
            # Creating the text file log
            # -------------------------------------
            full_path = os.path.join(self.logdir, self.filename)
            self.fileID = open(full_path, 'w')

        elif self.verbose == 'INFO':
            print(self._get_header() +
                  '{0:s}'.format(' Attribute write_to_file is set to False, not writing log to file ! ! !'))

        self.log_line('==============================================================================================')
        self.log_line('==============================================================================================')
        self.log_line('=====                                 STATING THE LOG                                    =====')
        self.log_line('==============================================================================================')
        self.log_line('==============================================================================================')

    def end_log(self):
        if self.write_to_file:
            self.fileID.close()

    def get_header(self, header):
        temp_str = '{0:^' + str(self.header_space) + '}| '
        return temp_str.format(header)

    def log_line(self, line):
        print(self._get_header() + line)
        if self.write_to_file:
            self.fileID.write(self._get_header() + line + '\n')

    def log_title(self, title):
        self.log_line('-------------------------------------------------------------------------------------------')
        self.log_line('{0:^86}' .format(title))
        self.log_line('-------------------------------------------------------------------------------------------')

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
        channels    = 0
        action_prev = None

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
                x_dim_size = int((x_dim_size - (action[3] - action[4]) + 2 * action[5]) / action[4])
                y_dim_size = int((y_dim_size - (action[3] - action[4]) + 2 * action[5]) / action[4])

                self.log_line(self.get_header(action[0]) + self._get_conv_layer_string(action[1],
                                                                                       action[2],
                                                                                       action[3],
                                                                                       action[4],
                                                                                       action[5],
                                                                                       action[6],
                                                                                       action[7],
                                                                                       action[8],
                                                                                       x_dim_size,
                                                                                       y_dim_size))
                channels = action[2]
            elif 'dense' in action[0]:
                self.log_line(self.get_header(action[0]) + self._get_dense_layer_string(channels,
                                                                                        action[2],
                                                                                        action[1],
                                                                                        action[3],
                                                                                        action[4],
                                                                                        action[5],
                                                                                        action[6],
                                                                                        action[7],
                                                                                        action[8],
                                                                                        x_dim_size,
                                                                                        y_dim_size))
                channels  += action[1] * action[2]
            elif 'transition' in action[0]:
                if type(action[6]) is not tuple:
                    x_dim_size = int((x_dim_size - (action[2] - action[3]) + 2 * action[4]) / action[3] / action[5])
                    y_dim_size = int((y_dim_size - (action[2] - action[3]) + 2 * action[4]) / action[3] / action[5])
                else:
                    x_conv_size = int((x_dim_size - (action[2] - action[3]) + 2 * action[4]) / action[3])
                    y_conv_size = int((y_dim_size - (action[2] - action[3]) + 2 * action[4]) / action[3])
                    x_dim_size  = int((x_conv_size + action[6][0] + action[6][1]) / action[5])
                    y_dim_size  = int((y_conv_size + action[6][2] + action[6][3]) / action[5])
                self.log_line(self.get_header(action[0]) + self._get_transition_layer_string(channels,
                                                                                             action[1],
                                                                                             action[2],
                                                                                             action[3],
                                                                                             action[4],
                                                                                             action[5],
                                                                                             action[6],
                                                                                             action[7],
                                                                                             action[9],
                                                                                             x_dim_size,
                                                                                             y_dim_size))
                channels = math.floor(channels * action[1])
            elif 'linear' in action[0]:
                if action_prev is None:
                    action_prev = action
                    self.log_line(self.get_header(action[0]) + self._get_linear_layer_string(x_dim_size * y_dim_size * channels,
                                                                                             action[1],
                                                                                             action[2],
                                                                                             action[3],
                                                                                             action[4]
                                                                                             )
                                  )
                else:
                    self.log_line(self.get_header(action[0]) + self._get_linear_layer_string(action_prev[1],
                                                                                             action[1],
                                                                                             action[2],
                                                                                             action[3],
                                                                                             action[4]
                                                                                             )
                                  )
                    action_prev = action

        # ==============================================================================================================
        # Decoder log
        # ==============================================================================================================
        self.log_title('VAE Decoder architecture')
        self.log_line('Input size: {}'.format(mod_vae.latent_dim))
        action_prev = None
        for ii in range(len(mod_vae.decoder.topology)):
            # ------------------------------------------------------------------------------------------------------
            # For each action, computing the output size and logging
            # ------------------------------------------------------------------------------------------------------
            action = mod_vae.decoder.topology[ii]
            if 'linear' in action[0]:
                if action_prev is None:
                    self.log_line(self.get_header(action[0]) + self._get_linear_layer_string(mod_vae.latent_dim,
                                                                                             action[1],
                                                                                             action[2],
                                                                                             action[3],
                                                                                             action[4]
                                                                                             )
                                  )
                else:
                    self.log_line(self.get_header(action[0]) + self._get_linear_layer_string(action_prev[1],
                                                                                             action[1],
                                                                                             action[2],
                                                                                             action[3],
                                                                                             action[4]
                                                                                             )
                                  )
                action_prev = action
