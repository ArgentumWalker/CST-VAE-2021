import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from models.blocks.conv import ResidualBlock, UnetDownsamplingConvBlock, VaeTransformBlock, DownsamplingConvBlock
from models.blocks.conv import ProductConditionTransform, UnetUpsamplingConvBlock, ProductUpsamplingConvBlock, \
    UpsamplingConvBlock, ConcatenateConditionTransform
from models.blocks.style import StyledSkipDownsampling, StyledConv


class SelfConditionalEncoder(nn.Module):
    def __init__(self, image_channels=3, ds_start_features=8, num_skip_downsamplings=5,
                 skip_features_start=2, sc_features=4, product_sc=False, product_sc_up=False):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(image_channels, ds_start_features, kernel_size=3, padding=1),
            nn.ELU(),
            ResidualBlock(ds_start_features),
            nn.Conv2d(ds_start_features, ds_start_features, kernel_size=3, padding=1),
            nn.ELU()
        )
        self.skip_downsamplings = nn.ModuleList(
            [UnetDownsamplingConvBlock(ds_start_features * (2 ** i), ds_start_features * (2 ** (i + 1)),
                                       skip_features_start * (2 ** i))
             for i in range(num_skip_downsamplings)]
        )
        self.final_downsampling = nn.Sequential(
            DownsamplingConvBlock(ds_start_features * (2 ** num_skip_downsamplings),
                                  ds_start_features * (2 ** num_skip_downsamplings)),
            # ResidualBlock(ds_start_features * (2**num_skip_downsamplings)),
        )

        if product_sc_up:
            sc_upsamplings = [ProductUpsamplingConvBlock(sc_features, sc_features, skip_features_start * (2 ** i))
                              for i in range(1, num_skip_downsamplings)]
        else:
            sc_upsamplings = [UnetUpsamplingConvBlock(sc_features, sc_features, skip_features_start * (2 ** i))
                              for i in range(1, num_skip_downsamplings)]
        self.sc_upsamplings = nn.ModuleList(sc_upsamplings)
        self.sc_first_upsample = UpsamplingConvBlock(skip_features_start * (2 ** num_skip_downsamplings),
                                                     sc_features)
        if product_sc:
            self.sc_modules = nn.ModuleList(
                ProductConditionTransform(skip_features_start * (2 ** i), sc_features)
                for i in range(num_skip_downsamplings)
            )
        else:
            self.sc_modules = nn.ModuleList(
                ConcatenateConditionTransform(skip_features_start * (2 ** i), sc_features)
                for i in range(num_skip_downsamplings)
            )
        self.distrib_modules = nn.ModuleList(
            [VaeTransformBlock(skip_features_start * (2**i),
                               skip_features_start * (2**i))
              for i in range(num_skip_downsamplings)]
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

        curr_mu, curr_sigma = self.distrib_modules[-1](hiddens[-1])
        mu_sigmas = [(curr_mu, curr_sigma)]
        z = [curr_mu + torch.randn_like(curr_sigma) * curr_sigma]
        curr_sc = self.sc_first_upsample(z[0])
        for i in reversed(range(len(hiddens) - 1)):
            h = hiddens[i]
            h = self.sc_modules[i](h, curr_sc)
            curr_mu, curr_sigma = self.distrib_modules[i](h)
            mu_sigmas.append((curr_mu, curr_sigma))
            z.append(curr_mu + torch.randn_like(curr_sigma) * curr_sigma)
            if i > 0:
                curr_sc = self.sc_upsamplings[i-1](curr_sc, z[-1])
        mu_sigmas = list(reversed(mu_sigmas))
        z = list(reversed(z))

        prior = Normal(0, 1)
        kld = torch.stack([kl_divergence(Normal(mu, sigma), prior).sum((-1, -2, -3)) for mu, sigma in mu_sigmas]).sum(-1)
        return z, kld


class ClassicStyledEncoder(nn.Module):
    def __init__(self, image_channels=3, ds_start_features=8, num_skip_downsamplings=5, skip_features_start=2, style_size=256):
        super().__init__()
        self.style_transform = nn.Sequential(
            nn.Linear(style_size, style_size),
            nn.ReLU(),
            nn.Linear(style_size, style_size),
            nn.ReLU()
        )
        self.img2features = StyledConv(image_channels, ds_start_features, 5, style_size, "relu")
        self.downsamplings = nn.ModuleList()
        for i in range(num_skip_downsamplings):
            self.downsamplings.append(StyledSkipDownsampling(ds_start_features * 2**i, skip_features_start * 2**i, style_size))
        self.vae_transformations = nn.ModuleList()
        for i in range(num_skip_downsamplings):
            self.vae_transformations.append(VaeTransformBlock(skip_features_start * 2**i, skip_features_start * 2**i))
        self.vae_transformations.append(VaeTransformBlock(ds_start_features * 2**num_skip_downsamplings,
                                                          skip_features_start * 2**num_skip_downsamplings))

    def forward(self, x, style):
        style = self.style_transform(style)
        x = self.img2features(x, style)
        hs = []
        for m in self.downsamplings:
            x, h = m(x, style)
            hs.append(h)
        hs.append(x)

        prior = Normal(0, 1)
        mu_sigma = [v(h) for h, v in zip(hs, self.vae_transformations)]
        z = [mu + torch.randn_like(sigma) * sigma for mu, sigma in mu_sigma]
        kld = torch.stack([kl_divergence(Normal(mu, sigma), prior).sum((-1, -2, -3)) for mu, sigma in mu_sigma]).sum(-1)
        return z, kld


class ClassicEncoder(nn.Module):
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
            #ResidualBlock(ds_start_features * (2**num_skip_downsamplings)),
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