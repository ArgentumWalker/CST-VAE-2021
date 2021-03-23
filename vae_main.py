from trainers.vae import train_vae
from models.vae.vae import VAE
from datasets.images_datasets import RawImagesDataset
from utils.pytorch import init_torch

if __name__ == "__main__":
    init_torch()
    vae = VAE(skip_features_start=2, num_skip_downsamplings=5, sc_encoder=True, sc_features=8,
              sc_encoder_prod_ups=True, dec_product_skip=True).cuda()
    dataset = RawImagesDataset("/home/argentumwalker/Projects/#DATA/deviantart")
    train_vae(vae, dataset, dataloader_workers=6, batch_size=4, lr=5e-5, epochs=500, kld_coef=0.1)
