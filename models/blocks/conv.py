import torch
from torch import nn
from torch.nn import functional as F


class DownsamplingConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ELU()
        )

    def forward(self, x):
        return self.model(x)


class UnetDownsamplingConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ELU()
        )
        self.skip_model = nn.Sequential(
            nn.Conv2d(out_channels, skip_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(skip_channels),
            nn.ELU()
        )

    def forward(self, x):
        out = self.model(x)
        return out, self.skip_model(out)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.model(x)


class VaeTransformBlock(nn.Module):
    def __init__(self, channels, latent_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 2 * latent_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        mu, sigma = torch.chunk(self.model(x), 2, -3)
        sigma = torch.exp(sigma)
        return mu, sigma


class UpsamplingConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ELU()
        )

    def forward(self, x):
        return self.model(x)


class UnetUpsamplingConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.skip_model = nn.Sequential(
            nn.Conv2d(skip_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.ELU()
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(2 * in_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ELU()
        )

    def forward(self, x, skip):
        x = torch.cat([x, self.skip_model(skip)], -3)
        return self.model(x)