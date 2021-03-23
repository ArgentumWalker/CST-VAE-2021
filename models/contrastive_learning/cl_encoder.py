from models.blocks.conv import DownsamplingConvBlock, ResidualBlock
import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveLearningEncoder(nn.Module):
    def __init__(self, input_image_size=256, input_image_channels=3, start_channels=8, latent_size=128, hidden_size=256):
        super().__init__()
        modules = [nn.Conv2d(input_image_channels, start_channels, kernel_size=3, padding=1),
                   nn.InstanceNorm2d(start_channels, affine=True),
                   nn.ReLU()]
        output_total_size = input_image_size**2 * start_channels
        output_size = input_image_size
        channels = start_channels
        while output_total_size > 512 and output_size > 4:
            modules.append(nn.Conv2d(channels, channels*2, kernel_size=3, padding=1, stride=2))
            modules.append(nn.InstanceNorm2d(channels*2, affine=True))
            modules.append(nn.ReLU())
            modules.append(ResidualBlock(channels*2))
            channels *= 2
            output_total_size //= 2
            output_size //= 2
            print(channels)

        print("Done")
        modules.append(nn.Flatten(-3))
        modules.append(nn.Linear(output_total_size, hidden_size))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_size, hidden_size))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_size, latent_size))
        modules.append(nn.Tanh())
        self.model = nn.Sequential(*modules)
        self.last = nn.Linear(latent_size, latent_size)

    def encode(self, x):
        return self.model(x)

    def forward(self, x):
        return self.last(self.model(x))