from models.blocks.conv import *
from models.vae.decoders import StyledDecoder
from models.vae.encoders import ClassicStyledEncoder


class CVAE(nn.Module):
    def __init__(self, image_channels=3, style_size=256, ds_start_features=32, us_end_features=32, num_skip_downsamplings=5,
                 skip_features_start=2, postprocess_channels=6):
        super().__init__()
        self.encoder = ClassicStyledEncoder(image_channels, ds_start_features, num_skip_downsamplings,
                                            skip_features_start, style_size)
        self.decoder = StyledDecoder(image_channels, us_end_features, postprocess_channels, num_skip_downsamplings,
                                     skip_features_start, style_size)

    def forward(self, x, style):
        z, kld = self.encoder(x, style)
        return z, kld

    def encode(self, x, style):
        return self.encoder(x, style)[0]

    def decode(self, z, style):
        return self.decoder(z, style)
