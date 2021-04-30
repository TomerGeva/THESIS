import os
import datetime


class LoggerNew:
    """
    This class holds the logger for the Variational auto-encoder
    """
    def __init__(self, logdir=None, filename=None):
        self.logdir         = logdir
        self.filename       = filename
        self.write_to_file  = True
        self.verbose        = 'INFO'
        self.header_space   = 10
        self.fileID         = None

    def _get_header(self):
        temp_str = '|{0:^' + str(self.header_space) + '}| '
        return temp_str.format(self.verbose)

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

    def log_line(self, line):
        print(self._get_header() + line)
        if self.write_to_file:
            self.fileID.write(self._get_header() + line + '\n')

    def log_epoch_results(self, epoch_num, recon_loss, d_kl, beta):
        loss = recon_loss + beta * d_kl
        self.log_line('Epoch: ' + str(epoch_num) + ' Reconstruction loss: ' + str(recon_loss) + ' D_kl: ' + str(d_kl) +
                      ' Total loss: ' + str(loss))


if __name__ == '__main__':
    logger = LoggerNew()
    logger.write_to_file = True
    logger.start_log()
    logger.log_line('Hi, this is Jerry')
    logger.log_epoch_results(1, 4, 8, 2)
    logger.end_log()
    print('hi')