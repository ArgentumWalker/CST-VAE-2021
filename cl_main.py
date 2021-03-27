from trainers.cl_trainer import train_contrastive_learning
from models.contrastive_learning.cl_encoder import ContrastiveLearningEncoder
from datasets.images_datasets import RawImagesDataset, PairedImagesDataset, DoubledImagesDataset
from utils.pytorch import init_torch

if __name__ == "__main__":
    init_torch()
    cl_encoder = ContrastiveLearningEncoder(256, 3, start_channels=16, latent_size=256, hidden_size=512).cuda()
    #dataset = DoubledImagesDataset("/home/argentumwalker/Projects/#DATA/deviantart")
    dataset = PairedImagesDataset("/home/argentumwalker/Projects/#DATA/deviantart")
    train_contrastive_learning(cl_encoder, dataset, lr=5e-5, batch_size=32, epochs=500, batches_per_epoch=1000,
                               dataloader_workers=8, log_every=100)
