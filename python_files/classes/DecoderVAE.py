from ConfigVAE import *
import torch.nn as nn
from torch import zeros as t_zeros
from neural_network_block_classes import FullyConnectedBlock, ConvTransposeBlock


class DecoderVAE(nn.Module):
    """
    This class holds the Variational auto-encoder Decoder part
    IMPORTANT:
        - If the model output is sensitivity, the decoder should have ONLY linear layers, where the last layer has
          the size of 1
    """
    def __init__(self, device, topology, latent_dim, model_out):
        super(DecoderVAE, self).__init__()
        self.device         = device
        self.topology       = topology
        self.layers         = nn.ModuleList()
        self.model_out      = model_out
        # ---------------------------------------------------------
        # Creating the Blocks according to the description
        # ---------------------------------------------------------
        linear_idx      = 0
        conv_trans_idx  = 0
        action_prev     = None
        sens_in_neurons = None
        for ii in range(len(self.topology)):
            action = self.topology[ii]
            if 'linear' in action[0]:
                linear_idx += 1
                if action_prev is None:  # First linear layer
                    action[1].in_neurons = latent_dim
                else:
                    action[1].in_neurons = action_prev[1].out_neurons
                self.layers.append(FullyConnectedBlock(action[1]))
                action_prev = action
                if 'last' in action[0]:
                    sens_in_neurons = action[1].out_neurons
            elif 'convTrans' in action[0]:
                conv_trans_idx += 1
                in_channels = action[1].out_channels
                self.layers.append(ConvTransposeBlock(action[1]))
        # ---------------------------------------------------------
        # Adding additional layer for the sensitivity output
        # ---------------------------------------------------------
        if self.model_out == model_output_e.BOTH:
            # self.sens_out_layer = FullyConnectedBlock(FCBlockData(1, in_neurons=sens_in_neurons, batch_norm=True, dropout_rate=0, activation=activation_type_e.null)),
            self.sens_out_layer = nn.Sequential(
                FullyConnectedBlock(FCBlockData(25, in_neurons=sens_in_neurons, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)),
                FullyConnectedBlock(FCBlockData(1, in_neurons=25, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)),
            )
        else:
            self.sens_out_layer = None

        self.fc_len         = linear_idx
        self.convTrans_len  = conv_trans_idx

    def forward(self, x):
        # ---------------------------------------------------------
        # passing through the fully connected blocks
        # ---------------------------------------------------------
        for ii in range(self.fc_len):
            layer = self.layers[ii]
            x = layer(x)
        if self.model_out is model_output_e.SENS:
            return t_zeros(1).to(self.device), x
        # ---------------------------------------------------------
        # Extracting the sensitivity
        # ---------------------------------------------------------
        if self.model_out is model_output_e.BOTH:
            sensitivity = self.sens_out_layer(x)
        else:
            sensitivity = t_zeros(1).to(self.device)
        # ---------------------------------------------------------
        # Restoring the grid
        # ---------------------------------------------------------
        x = x.view(-1, x.size(1), 1, 1)
        for ii in range(self.convTrans_len):
            layer = self.layers[self.fc_len + ii]
            x = layer(x)

        return x, sensitivity
