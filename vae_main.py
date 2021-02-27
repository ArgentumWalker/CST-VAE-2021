from trainers.vae import train_vae
from models.vae import VAE
from datasets.images_datasets import RawImagesDataset
from utils.pytorch import init_torch

if __name__ == "__main__":
    init_torch()
    vae = VAE().cuda()
    dataset = RawImagesDataset("/home/argentumwalker/hdd/CST-VAE/data/deviantart")
    train_vae(vae, dataset, dataloader_workers=2, batch_size=16, lr=1e-4, epochs=400)
