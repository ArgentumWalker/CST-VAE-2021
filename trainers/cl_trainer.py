import wandb
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F
import torch
from tqdm import tqdm
import time
import numpy as np


def train_contrastive_learning(cl_encoder, dataset, lr=1e-4, batch_size=16, epochs=100, batches_per_epoch=1000,
                               dataloader_workers=8, log_every=100):
    wandb.init(project="CST-GAN-2021-contrastive")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers,
                            pin_memory=True)
    target = torch.tensor(np.array(range(batch_size)), device="cuda", dtype=torch.long)
    optim = Adam(cl_encoder.parameters(), lr)

    for i in range(epochs):
        data_iter = iter(dataloader)
        for k in tqdm(range(batches_per_epoch)):

            get_batch_time = 0.
            load_batch_time = 0.

            tmp = time.time()
            batch = next(data_iter)
            if batch is None:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            x_left, x_right, cls = batch
            get_batch_time += time.time() - tmp
            tmp = time.time()
            x_left = x_left.cuda()
            x_right = x_right.cuda()
            cls = cls.cuda()
            load_batch_time += time.time() - tmp

            tmp = time.time()
            z_left = cl_encoder(x_left)
            z_right = cl_encoder(x_right)
            z_left = z_left.unsqueeze(0).repeat_interleave(len(x_right), dim=0)
            z_right = z_right.unsqueeze(1).repeat_interleave(len(x_left), dim=1)
            cls_left = cls.unsqueeze(0).repeat_interleave(len(x_right), dim=0)
            cls_right = cls.unsqueeze(1).repeat_interleave(len(x_right), dim=1)
            cls_mask = torch.eq(cls_left, cls_right)
            cls_mask[np.diag_indices(len(cls_mask))] = False

            dst_cos = (z_left * z_right).sum(-1)
            dst_cos[cls_mask] = -1e10  # Just in case there are two or more images of the same class in the batch

            loss = F.cross_entropy(dst_cos, target)
            for param in cl_encoder.parameters():
                param.grad = None
            optim.zero_grad()
            loss.backward()
            optim.step()
            step_time = time.time() - tmp

            wandb.log({"loss": loss.detach().cpu().item(),
                       "get_batch_time": get_batch_time,
                       "load_batch_time": load_batch_time,
                       "step_time": step_time,
                       "step": i * batches_per_epoch + k + 1}, step=i * batches_per_epoch + k + 1)

            if (i * batches_per_epoch + k + 1) % log_every == 0:
                torch.save(cl_encoder, "style_extractor.pkl")
                with torch.no_grad():
                    similarity_matrix = F.softmax(dst_cos, dim=-1).cpu().numpy()
                    wandb.log({
                        "confusion_matrix": wandb.plot.confusion_matrix(similarity_matrix, target.cpu().numpy()),
                        "similarity_matrix": wandb.Image(similarity_matrix)
                    }, step=i * batches_per_epoch + k + 1)