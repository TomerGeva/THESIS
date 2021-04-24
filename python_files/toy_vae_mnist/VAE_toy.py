import torch
import torch.nn            as nn
import torch.distributions as dist
import torch.nn.functional as F
from Encoder_toy import EncoderToy
from Decoder_toy import DecoderToy


class VaeToy(nn.Module):
    def __init__(self):
        super(VaeToy, self).__init__()

        self.encoder = EncoderToy()
        self.decoder = DecoderToy()

    def forward(self, x):
        # -------------------------------------------------------------------------
        # Encoding, outputs 100 points, 50 expectations and 50 standard deviations
        # -------------------------------------------------------------------------
        encoder_out         = self.encoder(x)
        encoder_out_reshape = encoder_out.view(-1, 2, 50)
        mu                  = encoder_out_reshape[:, 0, :]
        std                 = encoder_out_reshape[:, 1, :].abs()

        # -------------------------------------------------------------------------
        # Generating gaussian samples, and performing the re-parameterization trick
        # -------------------------------------------------------------------------
        normal_dist = dist.Normal(0, 1)
        zeta = normal_dist.sample(mu.size())
        if self.training:
            sampled_latent = zeta.mul(std).add_(mu)
        else:
            sampled_latent = mu

        # -------------------------------------------------------------------------
        # Decoding, outputs a 28 X 28 1 channel picture
        # -------------------------------------------------------------------------
        decoder_out = self.decoder(sampled_latent)

        return decoder_out, mu, std
