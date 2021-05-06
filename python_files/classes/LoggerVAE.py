import os
import datetime
from ConfigVAE import XQUANTIZE, YQUANTIZE, ENCODER_KERNEL_SIZE, ENCODER_STRIDES, ENCODER_PADDING, ENCODER_FC_LAYERS,\
    ENCODER_MAX_POOL_SIZE, ENCODER_FILTER_NUM, LATENT_SPACE_DIM, DECODER_FC_LAYERS


class LoggerVAE:
    """
    This class holds the logger for the Variational auto-encoder
    """
    def __init__(self, logdir=None, filename=None):
        self.logdir         = logdir
        self.filename       = filename
        self.write_to_file  = True
        self.verbose        = 'INFO'
        self.header_space   = 12
        self.result_space   = 15.6
        self.desc_space     = 6
        self.fileID         = None

    # ==================================================================================================================
    # Basic help functions, internal
    # ==================================================================================================================
    def _get_header(self):
        temp_str = '|{0:^' + str(self.header_space) + '}| '
        return temp_str.format(self.verbose)

    def _get_result_string(self, mse_loss, d_kl, cost):
        temp_str = 'MSE loss: {0:^' + str(self.result_space) +\
                   'f} D_kl: {1:^' + str(self.result_space) +\
                   'f} Total cost: {2:^' + str(self.result_space) + 'f}'
        return temp_str.format(mse_loss, d_kl, cost)

    # ==================================================================================================================
    # Regular VAE log functions, used to log the layer architecture from the description
    # ==================================================================================================================
    def _get_conv_layer_string(self, in_ch, out_ch, ktilde, stride, pad, x_dim, y_dim):
        temp_str = 'In channels: {0:^' + str(self.desc_space) +\
                   'd} Out channels: {1:^' + str(self.desc_space) +\
                   'd} K^tilde: {2:^' + str(self.desc_space) + \
                   'd} Stride: {3:^' + str(self.desc_space) + \
                   'd} Padding: {4:^' + str(self.desc_space) + \
                   'd} Output size: {5:^' + str(self.desc_space) + '}X{6:^' + str(self.desc_space) + '}'
        return temp_str.format(in_ch, out_ch, ktilde, stride, pad, x_dim, y_dim)

    def _get_pool_layer_string(self, ktilde, x_dim, y_dim):
        temp_str = 'K^tilde: {0:1d} Output size: {1:^' + str(self.desc_space) + '}X{2:^' + str(self.desc_space) + '}'
        return temp_str.format(ktilde, x_dim, y_dim)

    def _get_linear_layer_string(self, num_in, num_out):
        temp_str = 'Input size: {0:^' + str(self.desc_space) + '} Output size: {1:^' + str(self.desc_space) + '}'
        return temp_str.format(num_in, num_out)

    # ==================================================================================================================
    # Dense VAE log functions, used to log the layer architecture
    # ==================================================================================================================

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

        if self.logdir is None:
            time_data = datetime.datetime.now()
            time_list = [time_data.day, time_data.month, time_data.year, time_data.hour, time_data.minute]
            time_string = '_'.join([str(ii) for ii in time_list])
            del time_data, time_list
            self.logdir = os.path.join(os.getcwd(), time_string)

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

    def log_epoch_results(self, epoch_num, train_mse_loss, train_d_kl, train_cost, test_mse_loss, test_d_kl, test_cost):
        self.log_line('Epoch: {0:5d}' .format(epoch_num))
        self.log_line(self.get_header('Train') + self._get_result_string(train_mse_loss, train_d_kl, train_cost))
        self.log_line(self.get_header('Test') + self._get_result_string(test_mse_loss, test_d_kl, test_cost))

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
        # ==============================================================================================================
        # Encoder log
        # ==============================================================================================================
        self.log_title('VAE Encoder architecture')
        self.log_line('Input size: {}X{}' .format(x_dim_size, y_dim_size))

        for ii in range(len(mod_vae.encoder.description)):
            # ------------------------------------------------------------------------------------------------------
            # For each action, computing the output size and logging
            # ------------------------------------------------------------------------------------------------------
            action = mod_vae.encoder.description[ii]
            if 'conv' in action:
                x_dim_size = int((x_dim_size - (ENCODER_KERNEL_SIZE[conv_idx] - ENCODER_STRIDES[conv_idx]) + 2 *
                                  ENCODER_PADDING[conv_idx]) / ENCODER_STRIDES[conv_idx])
                y_dim_size = int((y_dim_size - (ENCODER_KERNEL_SIZE[conv_idx] - ENCODER_STRIDES[conv_idx]) + 2 *
                                  ENCODER_PADDING[conv_idx]) / ENCODER_STRIDES[conv_idx])

                self.log_line(self.get_header(action) + self._get_conv_layer_string(ENCODER_FILTER_NUM[conv_idx],
                                                                                    ENCODER_FILTER_NUM[conv_idx+1],
                                                                                    ENCODER_KERNEL_SIZE[conv_idx],
                                                                                    ENCODER_STRIDES[conv_idx],
                                                                                    ENCODER_PADDING[conv_idx],
                                                                                    x_dim_size,
                                                                                    y_dim_size))
                conv_idx += 1
            elif 'pool' in action:
                x_dim_size = int(x_dim_size / ENCODER_MAX_POOL_SIZE[maxpool_idx])
                y_dim_size = int(y_dim_size / ENCODER_MAX_POOL_SIZE[maxpool_idx])

                self.log_line(self.get_header(action) + self._get_pool_layer_string(ENCODER_MAX_POOL_SIZE[maxpool_idx],
                                                                                    x_dim_size,
                                                                                    y_dim_size))
                maxpool_idx += 1
            elif 'linear' in action:
                if linear_idx == 0:
                    self.log_line(self.get_header(action) + self._get_linear_layer_string(x_dim_size * y_dim_size * ENCODER_FILTER_NUM[-1], ENCODER_FC_LAYERS[linear_idx]))
                else:
                    self.log_line(self.get_header(action) + self._get_linear_layer_string(ENCODER_FC_LAYERS[linear_idx-1], ENCODER_FC_LAYERS[linear_idx]))
                linear_idx += 1

        # ==============================================================================================================
        # Decoder log
        # ==============================================================================================================
        self.log_title('VAE Decoder architecture')
        self.log_line('Input size: {}'.format(LATENT_SPACE_DIM))
        linear_idx = 0
        for ii in range(len(mod_vae.decoder.description)):
            # ------------------------------------------------------------------------------------------------------
            # For each action, computing the output size and logging
            # ------------------------------------------------------------------------------------------------------
            action = mod_vae.decoder.description[ii]
            if 'linear' in action:
                if linear_idx == 0:
                    self.log_line(self.get_header(action) + self._get_linear_layer_string(LATENT_SPACE_DIM, DECODER_FC_LAYERS[linear_idx]))
                else:
                    self.log_line(self.get_header(action) + self._get_linear_layer_string(DECODER_FC_LAYERS[linear_idx - 1],
                                                                                          DECODER_FC_LAYERS[linear_idx]))
                linear_idx += 1
