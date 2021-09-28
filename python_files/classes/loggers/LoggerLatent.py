import os
import math
from ConfigVAE import *
from LoggerGeneric import LoggerGeneric


class LoggerLatent(LoggerGeneric):
    """
    This class hold the logger for the latent space optimization
    """
    def __init__(self, logdir=None, filename=None, write_to_file=True):
        super().__init__(logdir=logdir, filename=filename, write_to_file=write_to_file)

    # ==================================================================================================================
    # Logging functions
    # ==================================================================================================================
    def log_start_setup(self, latent_dim):
        """
        :param latent_dim: latent space size
        :return: function logs the starting point of the optimization
        """
        self.log_title('Latent space optimization')
        self.log_line('Latent space size: {}'.format(latent_dim))

    def log_step(self, step_num, sensitivity):
        self.log_line('Step: {0:5d}  Sensitivity: {1:8.3f'.format(step_num, sensitivity))
        if self.write_to_file:
            self.end_log()
            full_path = os.path.join(self.logdir, self.filename)
            self.fileID = open(full_path, 'a')

