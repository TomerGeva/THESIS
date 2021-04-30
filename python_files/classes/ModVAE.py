from ConfigVAE import *
import torch.nn            as nn
from EncoderVAE import EncoderVAE
from DecoderVAE import DecoderVAE


class ModVAE(nn.Module):
    """
    This class holds the modified Variational auto-encoder
    """
    def __init__(self, device):
        super(ModVAE, self).__init__()
        self.device     = device
        self.encoder    = EncoderVAE(device=device)
        self.decoder    = DecoderVAE(device=device)

    def forward(self, x):
        # -------------------------------------------------------------------------
        # Encoding, outputs 100 points, 50 expectations and 50 standard deviations
        # -------------------------------------------------------------------------
        encoder_out         = self.encoder(x)
        encoder_out_reshape = encoder_out.view(-1, 2, LATENT_SPACE_DIM)
        mu                  = encoder_out_reshape[:, 0, :]
        logvar              = encoder_out_reshape[:, 1, :]

        # -------------------------------------------------------------------------
        # Generating gaussian samples, and performing the re-parameterization trick
        # -------------------------------------------------------------------------
        if self.training:
            std = logvar.mul(0.5).exp_()
            zeta = std.data.new(std.size()).normal_()
            sampled_latent = zeta.mul(std).add_(mu)
        else:
            sampled_latent = mu
            # __________Reshaping to enter the decoder__________
            sampled_latent = sampled_latent.view(-1, LATENT_SPACE_DIM)

        # -------------------------------------------------------------------------
        # Decoding, outputs a 28 X 28 1 channel picture
        # -------------------------------------------------------------------------
        decoder_out = self.decoder(sampled_latent)

        return decoder_out, mu, logvar
