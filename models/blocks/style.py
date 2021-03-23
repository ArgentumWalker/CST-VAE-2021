import torch
from torch import nn
from torch.nn import functional as F


class AdaIN(nn.Module):
    def __init__(self, features, style_size):
        super().__init__()
        self.norm = nn.InstanceNorm2d(features, affine=False, track_running_stats=True)
        self.style_weight = nn.Linear(style_size, features)
        self.style_bias = nn.Linear(style_size, features)

    def forward(self, x, style):
        x = self.norm(x)
        w = self.style_weight(style)[:, :, None, None]
        b = self.style_bias(style)[:, :, None, None]
        return x * w + b


class StyledConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, style_size, activation="none"):
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, 1, (kernel_size - 1)//2)
        self.adain = AdaIN(channels_out, style_size)
        self.activation = nn.Identity()
        if activation == "tanh":
            self.activation = nn.Tanh()
        if activation == "relu":
            self.activation = nn.ReLU()

    def forward(self, x, style):
        return self.activation(self.adain(self.conv(x), style))


class StyledDownsampling(nn.Module):
    def __init__(self, channels, style_size):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 3, 2, 1)
        self.adain = AdaIN(channels * 2, style_size)
        self.activation = nn.ReLU()

    def forward(self, x, style):
        return self.activation(self.adain(self.conv(x), style))


class StyledUpsampling(nn.Module):
    def __init__(self, channels, style_size):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1)
        self.adain = AdaIN(channels // 2, style_size)
        self.activation = nn.ReLU()

    def forward(self, x, style):
        return self.activation(self.adain(self.conv(x), style))


class StyledSkipDownsampling(nn.Module):
    def __init__(self, channels, skip_channels, style_size):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 3, 2, 1)
        self.adain = AdaIN(channels * 2, style_size)
        self.activation = nn.ReLU()
        self.skip_conv = nn.Conv2d(channels * 2, skip_channels, 3, 1, 1)
        self.skip_adain = AdaIN(skip_channels, style_size)

    def forward(self, x, style):
        x = self.activation(self.adain(self.conv(x), style))
        x_skip = self.activation(self.skip_adain(self.skip_conv(x), style))
        return x, x_skip


class StyledSkipUpsampling(nn.Module):
    def __init__(self, channels, skip_channels, style_size):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels * 2, channels // 2, 4, 2, 1)
        self.adain = AdaIN(channels // 2, style_size)
        self.activation = nn.ReLU()
        self.skip_conv = nn.Conv2d(skip_channels, channels, 3, 1, 1)
        self.skip_adain = AdaIN(channels, style_size)

    def forward(self, x, x_skip, style):
        x = torch.cat([x, self.activation(self.skip_adain(self.skip_conv(x_skip), style))], dim=-3)
        x = self.activation(self.adain(self.conv(x), style))
        return x