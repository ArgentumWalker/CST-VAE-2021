from trainers.cvae import train_vae
from models.vae.cvae import CVAE
from models.contrastive_learning.cl_encoder import ContrastiveLearningEncoder
from datasets.images_datasets import StyledImagesDataset
from utils.pytorch import init_torch
import torch

if __name__ == "__main__":
    init_torch()
    cl_encoder = torch.load("style_extractor.pkl")
    vae = CVAE(skip_features_start=2, num_skip_downsamplings=3, style_size=256).cuda()
    dataset = StyledImagesDataset("/home/argentumwalker/Projects/#DATA/deviantart")
    train_vae(vae, cl_encoder, dataset, dataloader_workers=6, batch_size=4, lr=5e-5, epochs=500, cos_loss_coef=0.001)
