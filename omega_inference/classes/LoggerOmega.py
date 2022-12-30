import os
from LoggerGeneric import LoggerGeneric
from database_functions import ModelManipulationFunctions


class LoggerOmega(LoggerGeneric):
    def __init__(self, logdir=None, filename=None, write_to_file=True):
        super().__init__(logdir=logdir, filename=filename, write_to_file=write_to_file)

    # ==================================================================================================================
    # Basic help functions, internal
    # ==================================================================================================================
    def _get_result_string(self, loss):
        temp_str = 'Omega_MSE: {0:^' + str(self.result_space) + 'f}'
        return temp_str.format(loss)

    def _get_layer_log_string(self, action):
        if 'res-linear' in action[0]:
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

    # ==================================================================================================================
    # Logging functions
    # ==================================================================================================================
    def log_epoch_results(self, header, loss):
        self.log_line(self.get_header(header) + self._get_result_string(loss))
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
        # Encoder log
        # ==============================================================================================================
        self.log_title('Omega Inference Architecture')
        self.log_line('Number of parameters: ' + str(mmf.get_nof_params(model)))
        for action in model.topology:
            # ------------------------------------------------------------------------------------------------------
            # For each action, computing the output size and logging
            # ------------------------------------------------------------------------------------------------------
            self._get_layer_log_string(action)
