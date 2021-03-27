from torch import nn

from models.blocks.conv import UpsamplingConvBlock, UnetUpsamplingConvBlock, ProductUpsamplingConvBlock
from models.blocks.style import StyledSkipUpsampling, StyledConv


class ClassicDecoder(nn.Module):
    def __init__(self, image_channels=3, us_end_features=16, postprocess_channels=32,
                 num_skip_downsamplings=5, skip_features_start=2, is_product_skip=False):
        super().__init__()
        self.postprocess = nn.Sequential(
            nn.Conv2d(us_end_features, postprocess_channels, kernel_size=7, padding=3),
            nn.ELU(),
            nn.Conv2d(postprocess_channels, image_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.start_upsampling = nn.Sequential(
            nn.Conv2d(skip_features_start * (2 ** num_skip_downsamplings),
                      us_end_features * (2 ** num_skip_downsamplings), kernel_size=3, padding=1),
            nn.ELU(),
            UpsamplingConvBlock(us_end_features * (2 ** num_skip_downsamplings),
                                us_end_features * (2 ** num_skip_downsamplings))
        )
        if is_product_skip:
            self.skip_upsamplings = nn.ModuleList(
                [ProductUpsamplingConvBlock(us_end_features * (2 ** (i + 1)), us_end_features * (2 ** i),
                                            skip_features_start * (2 ** i))
                 for i in reversed(range(num_skip_downsamplings))]
            )
        else:
            self.skip_upsamplings = nn.ModuleList(
                [UnetUpsamplingConvBlock(us_end_features * (2 ** (i + 1)), us_end_features * (2 ** i), skip_features_start * (2 ** i))
                 for i in reversed(range(num_skip_downsamplings))]
            )

    def forward(self, zs):
        x = self.start_upsampling(zs[-1])
        for z, m in zip(reversed(zs[:-1]), self.skip_upsamplings):
            x = m(x, z)
        return self.postprocess(x)


class StyledDecoder(nn.Module):
    def __init__(self, image_channels=3, us_end_features=16, postprocess_channels=32,
                 num_skip_downsamplings=5, skip_features_start=2, style_size=256):
        super().__init__()
        self.postprocess = nn.Sequential(
            nn.Conv2d(us_end_features, postprocess_channels, kernel_size=7, padding=3),
            nn.ELU(),
            nn.Conv2d(postprocess_channels, image_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.start_transform = StyledConv(skip_features_start * 2**num_skip_downsamplings,
                                          us_end_features * 2 ** num_skip_downsamplings,
                                          5, style_size, "relu")
        self.upsamplings = nn.ModuleList()
        for i in reversed(range(num_skip_downsamplings)):
            self.upsamplings.append(StyledSkipUpsampling(us_end_features * 2 ** (i + 1), skip_features_start, style_size))

    def forward(self, zs, style):
        x = self.start_transform(zs[-1], style)
        for us, z in zip(self.upsamplings, reversed(zs[:-1])):
            x = us(x, z, style)
        return self.postprocess(x)