import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence
from .blocks.conv import *


class _Encoder(nn.Module):
    def __init__(self, image_channels=3, ds_start_features=8, num_skip_downsamplings=5, skip_features_start=2):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(image_channels, ds_start_features, kernel_size=3, padding=1),
            nn.ELU(),
            ResidualBlock(ds_start_features),
            nn.Conv2d(ds_start_features, ds_start_features, kernel_size=3, padding=1),
            nn.ELU()
        )
        self.skip_downsamplings = nn.ModuleList(
            [UnetDownsamplingConvBlock(ds_start_features * (2**i), ds_start_features * (2**(i+1)),
                                       skip_features_start * (2**i))
              for i in range(num_skip_downsamplings)]
        )
        self.distrib_modules = nn.ModuleList(
            [VaeTransformBlock(skip_features_start * (2**i),
                               skip_features_start * (2**i))
              for i in range(num_skip_downsamplings)]
        )

        self.final_downsampling = nn.Sequential(
            DownsamplingConvBlock(ds_start_features * (2**num_skip_downsamplings),
                                  ds_start_features * (2**num_skip_downsamplings)),
            ResidualBlock(ds_start_features * (2**num_skip_downsamplings)),
        )

        self.distrib_modules.append(VaeTransformBlock(ds_start_features * (2**num_skip_downsamplings),
                                                      skip_features_start * (2**num_skip_downsamplings)))

    def forward(self, x):
        x = self.preprocess(x)

        hiddens = []
        for m in self.skip_downsamplings:
            x, h = m(x)
            hiddens.append(h)
        hiddens.append(self.final_downsampling(x))

        prior = Normal(0, 1)
        mu_sigmas = [m(h) for h, m in zip(hiddens, self.distrib_modules)]
        z = [mu + torch.randn_like(sigma) * sigma for mu, sigma in mu_sigmas]
        kld = torch.stack([kl_divergence(Normal(mu, sigma), prior).sum((-1, -2, -3)) for mu, sigma in mu_sigmas]).sum(-1)
        return z, kld


class _Decoder(nn.Module):
    def __init__(self, image_channels=3, us_end_features=16, num_skip_downsamplings=5, skip_features_start=2):
        super().__init__()
        self.postprocess = nn.Sequential(
            nn.Conv2d(us_end_features, us_end_features, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(us_end_features, image_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.start_upsampling = nn.Sequential(
            nn.Conv2d(skip_features_start * (2 ** num_skip_downsamplings),
                      us_end_features * (2 ** num_skip_downsamplings), kernel_size=3, padding=1),
            nn.ELU(),
            UpsamplingConvBlock(us_end_features * (2 ** num_skip_downsamplings),
                                us_end_features * (2 ** num_skip_downsamplings))
        )
        self.skip_upsamplings = nn.ModuleList(
            [UnetUpsamplingConvBlock(us_end_features * (2 ** (i + 1)), us_end_features * (2 ** i), skip_features_start * (2 ** i))
             for i in reversed(range(num_skip_downsamplings))]
        )

    def forward(self, zs):
        x = self.start_upsampling(zs[-1])
        for z, m in zip(reversed(zs[:-1]), self.skip_upsamplings):
            x = m(x, z)
        return self.postprocess(x)


class VAE(nn.Module):
    def __init__(self, image_channels=3, ds_start_features=8, us_end_features=16, num_skip_downsamplings=5,
                 skip_features_start=2):
        super().__init__()

        self.encoder = _Encoder(image_channels, ds_start_features, num_skip_downsamplings, skip_features_start)
        self.decoder = _Decoder(image_channels, us_end_features, num_skip_downsamplings, skip_features_start)

    def forward(self, x):
        z, kld = self.encoder(x)
        x = self.decoder(z)
        return x, z, kld

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, z):
        return self.decoder(z)
