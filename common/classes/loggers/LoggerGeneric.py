import os
import math
import datetime
from ConfigVAE import *


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

