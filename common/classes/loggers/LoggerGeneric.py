import os
import math
import datetime
from ConfigVAE import *
from time import time


class LoggerGeneric:
    """
    This class holds the generic logger.
    From this logger, the loggers for all the following phases will be built
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

    # ==================================================================================================================
    # Regular layer logging functions
    # ==================================================================================================================
    def _get_conv_layer_string(self, in_ch, out_ch, ktilde, stride, pad, bnorm, drate, active, x_dim, y_dim):
        temp_str = 'In channels:    {0:^' + str(self.desc_space) + \
                   'd} Out channels:{1:^' + str(self.desc_space) + \
                   'd} Kernel:      {2:^' + str(self.desc_space) + \
                   'd} Stride:      {3:^' + str(self.desc_space) + \
                   'd} Padding:     {4:^' + str(self.desc_space) + \
                   'd} batch_norm:  {5:^' + str(self.desc_space) + \
                   's} drop_rate:   {6:^' + str(self.desc_space) + \
                   'd} activation:  {7:^' + str(self.desc_space) + \
                   's} Output size: {8:^' + str(self.desc_space) + '}X{9:^' + str(self.desc_space) + '}'
        return temp_str.format(in_ch, out_ch, ktilde, stride, pad, str(bnorm), drate, active.name, x_dim, y_dim)

    def _get_residual_conv_layer_string(self, in_ch, out_ch, layers, ktilde, stride, pad, bnorm, drate, active,
                                        x_dim, y_dim):
        temp_str = 'In channels:    {0:^' + str(self.desc_space) + \
                   'd} Out channels:{1:^' + str(self.desc_space) + \
                   'd} Layers:      {2:^' + str(self.desc_space) + \
                   'd} Kernel:      {3:^' + str(self.desc_space) + \
                   'd} Stride:      {4:^' + str(self.desc_space) + \
                   'd} Padding:     {5:^' + str(self.desc_space) + \
                   'd} batch_norm:  {6:^' + str(self.desc_space) + \
                   's} drop_rate:   {7:^' + str(self.desc_space) + \
                   'd} activation:  {8:^' + str(self.desc_space) + \
                   's} Output size: {9:^' + str(self.desc_space) + '}X{10:^' + str(self.desc_space) + '}'
        return temp_str.format(in_ch, out_ch, layers, ktilde, stride, pad, str(bnorm), drate, active.name, x_dim,
                               y_dim)

    def _get_conv_transpose_layer_string(self, in_ch, out_ch, ktilde, stride, pad, out_pad, bnorm, drate, active,
                                         x_dim, y_dim):
        temp_str = 'In channels:    {0:^' + str(self.desc_space) + \
                   'd} Out channels:{1:^' + str(self.desc_space) + \
                   'd} Kernel:      {2:^' + str(self.desc_space) + \
                   'd} Stride:      {3:^' + str(self.desc_space) + \
                   'd} Padding:     {4:^' + str(self.desc_space) + \
                   'd} Out Padding: {5:^' + str(self.desc_space) + \
                   'd} batch_norm:  {6:^' + str(self.desc_space) + \
                   's} drop_rate:   {7:^' + str(self.desc_space) + \
                   'd} activation:  {8:^' + str(self.desc_space) + \
                   's} Output size: {9:^' + str(self.desc_space) + '}X{10:^' + str(self.desc_space) + '}'
        return temp_str.format(in_ch, out_ch, ktilde, stride, pad, out_pad, str(bnorm), drate, active.name, x_dim,
                               y_dim)

    def _get_pool_layer_string(self, ktilde, x_dim, y_dim):
        temp_str = 'K^tilde: {0:1d} Output size: {1:^' + str(self.desc_space) + '}X{2:^' + str(
            self.desc_space) + '}'
        return temp_str.format(ktilde, x_dim, y_dim)

    def _get_linear_layer_string(self, num_in, num_out, bias, bnorm, drate, active):
        temp_str = 'Input size:     {0:^' + str(self.desc_space) + \
                   'd} Output size: {1:^' + str(self.desc_space) + \
                   'd} bias:        {2:^' + str(self.desc_space) + \
                   'd} batch_norm:  {3:^' + str(self.desc_space) + \
                   's} drop_rate:   {4:^' + str(self.desc_space) + \
                   'd} activation:  {5:^' + str(self.desc_space) + 's}'
        return temp_str.format(num_in, num_out, bias, str(bnorm), drate, active.name)

    def _get_residual_linear_layer_string(self, num_in, num_out, layers, bnorm, drate, active):
            temp_str = 'Input size:     {0:^' + str(self.desc_space) + \
                       'd} Output size: {1:^' + str(self.desc_space) + \
                       'd} layers:      {2:^' + str(self.desc_space) + \
                       'd} batch_norm:  {3:^' + str(self.desc_space) + \
                       's} drop_rate:   {4:^' + str(self.desc_space) + \
                       'd} activation:  {5:^' + str(self.desc_space) + 's}'
            return temp_str.format(num_in, num_out, layers, str(bnorm), drate, active.name)

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

    def _get_transformer_layer_string(self, patch_size_x, patch_size_y, embed_size):
        temp_str = 'patch_size_x: {0:^1' + str(self.desc_space) + \
                   '} patch_size_y: {1:^' + str(self.desc_space) + \
                   '} embed_size: {2:^' + str(self.desc_space) + '}'
        return temp_str.format(patch_size_x, patch_size_y, embed_size)

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
            self.logdir = os.getcwd()

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
        self.log_line('{0:^86}'.format(title))
        self.log_line('-------------------------------------------------------------------------------------------')

    def log_epoch(self, epoch_num, t):
        self.log_line('Epoch: {0:5d}, training time: {1:5.2f}'.format(epoch_num, time() - t))

