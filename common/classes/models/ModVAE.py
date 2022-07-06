import torch.nn         as nn
from models.EncoderVAE import EncoderVAE
from models.DecoderVAE import DecoderVAE
from global_const       import encoder_type_e, mode_e, model_output_e


class ModVAE(nn.Module):
    """
    This class holds the modified Variational auto-encoder
    """
    def __init__(self, device, encoder_topology, decoder_topology, latent_space_dim, encoder_type=encoder_type_e.DENSE, mode=mode_e.VAE, model_out=model_output_e.BOTH):
        super(ModVAE, self).__init__()
        self.device         = device
        self.encoder_type   = encoder_type
        self.mode           = mode
        self.model_out      = model_out

        self.encoder    = EncoderVAE(device=device, topology=encoder_topology)
        self.decoder    = DecoderVAE(device=device, topology=decoder_topology, latent_dim=latent_space_dim, model_out=model_out)
        self.latent_dim = latent_space_dim

    def forward(self, x):
        # -------------------------------------------------------------------------
        # Encoding, outputs 100 points, 50 expectations and 50 standard deviations
        # -------------------------------------------------------------------------
        encoder_out         = self.encoder(x)
        encoder_out_reshape = encoder_out.view(-1, 2, self.latent_dim)
        mu                  = encoder_out_reshape[:, 0, :]
        logvar              = encoder_out_reshape[:, 1, :]

        # -------------------------------------------------------------------------
        # Generating gaussian samples, and performing the re-parameterization trick
        # -------------------------------------------------------------------------
        if self.training and self.mode is mode_e.VAE:
            std = logvar.mul(0.5).exp_()
            zeta = std.data.new(std.size()).normal_()
            sampled_latent = zeta.mul(std).add_(mu)
        else:
            sampled_latent = mu

        # -------------------------------------------------------------------------
        # Decoding, outputs a 2500 X 2500 X 1 channel grid and a sensitivity value
        # -------------------------------------------------------------------------
        # __________Reshaping to enter the decoder__________
        sampled_latent = sampled_latent.view(-1, self.latent_dim)
        grid_out, sens_out = self.decoder(sampled_latent)

        return grid_out, sens_out, mu, logvar
