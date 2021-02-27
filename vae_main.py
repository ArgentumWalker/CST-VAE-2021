from trainers.vae import train_vae
from models.vae import VAE
from datasets.images_datasets import RawImagesDataset

if __name__ == "__main__":
    vae = VAE().cuda()
    dataset = RawImagesDataset("/home/argentumwalker/hdd/CST-VAE/data/deviantart")
    train_vae(vae, dataset, dataloader_workers=8, batch_size=24, lr=2e-4, epochs=400)
