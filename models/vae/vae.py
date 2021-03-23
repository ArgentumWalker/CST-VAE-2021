from models.blocks.conv import *
from models.vae.decoders import ClassicDecoder
from models.vae.encoders import ClassicEncoder, SelfConditionalEncoder


class VAE(nn.Module):
    def __init__(self, image_channels=3, ds_start_features=32, us_end_features=32, num_skip_downsamplings=5,
                 skip_features_start=2, postprocess_channels=64, sc_encoder=False, sc_features=4,
                 dec_product_skip=False, sc_encoder_prod_cond=False, sc_encoder_prod_ups=False):
        super().__init__()

        if sc_encoder:
            self.encoder = SelfConditionalEncoder(image_channels, ds_start_features, num_skip_downsamplings,
                                                  skip_features_start, sc_features, sc_encoder_prod_cond,
                                                  sc_encoder_prod_ups)
        else:
            self.encoder = ClassicEncoder(image_channels, ds_start_features, num_skip_downsamplings,
                                          skip_features_start)
        self.decoder = ClassicDecoder(image_channels, us_end_features, postprocess_channels, num_skip_downsamplings,
                                      skip_features_start, dec_product_skip)

    def forward(self, x):
        z, kld = self.encoder(x)
        return z, kld

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, z):
        return self.decoder(z)
