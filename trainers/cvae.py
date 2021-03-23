from models.vae.cvae import CVAE
from models.contrastive_learning.cl_encoder import ContrastiveLearningEncoder
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb
from tqdm import tqdm
import time
import random


def build_images_to_log(vae:CVAE, style_encoder:ContrastiveLearningEncoder, source, style):
    log_dict = {"orig_imgs": [wandb.Image(img.transpose(0, 2).numpy()) for img in source.cpu()]}
    with torch.no_grad():
        z = vae.encode(source, style)
        shifted_styles = [s.unsqueeze(0).expand_as(style) for s in style]
        # reconstructed images
        x_rec = vae.decode(z, style)
        log_dict["rec_imgs"] = [wandb.Image(img.transpose(0, 2).numpy()) for img in x_rec.cpu()]
        # with randomized z
        for i in range(len(z)):
            new_z = [zz for zz in z]
            imgs = []
            for _ in range(4):
                new_z[i] = torch.randn_like(new_z[i])
                x_rec = vae.decode(new_z, style)
                imgs.extend([wandb.Image(img.transpose(0, 2).numpy()) for img in x_rec.cpu()])
            log_dict[f"noised_{i+1}_imgs"] = imgs

        # Style transfer
        for i, s in enumerate(shifted_styles):
            x_rec = vae.decode(z, s)
            log_dict[f"transfer_to_{i}"] = [wandb.Image(img.transpose(0, 2).numpy()) for img in x_rec.cpu()]

        # generated
        new_z = [torch.randn_like(zz) for zz in z]
        x_rec = vae.decode(new_z, style)
        log_dict["gen_imgs"] = [wandb.Image(img.transpose(0, 2).numpy()) for img in x_rec.cpu()]

    return log_dict


def random_crop(x, size=256):
    x_crop = random.randint(0, x.shape[-2]-size)
    y_crop = random.randint(0, x.shape[-1]-size)
    return x[:, :, x_crop:x_crop+size, y_crop:y_crop+size]


def compute_style_shifted_loss(vae:CVAE, style_encoder:ContrastiveLearningEncoder, z, style):
    style = torch.cat([style[1:], style[:1]], dim=0)  # shift styles by one
    x_trans = vae.decode(z, style)
    gen_style = style_encoder(random_crop(x_trans))
    with torch.no_grad():
        style = style_encoder.last(style)
    cos_dst = (gen_style * style).sum(-1)
    return -cos_dst.mean()  # simply maximize cos distance


def train_vae(vae:CVAE, style_encoder:ContrastiveLearningEncoder, dataset, dataloader_workers=8, lr=5e-5, kld_coef=0.1,
              cos_loss_coef=0.1, epochs=400, batches_per_epoch=1000, batch_size=6, log_images_every=10):
    wandb.init(project="CST-GAN-2021-styled")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers, pin_memory=True)
    optim = Adam(vae.parameters(), lr)
    test_images = torch.stack([dataset[10094][0], dataset[1282][0], dataset[25954][0], dataset[25513][0], dataset[7007][0]]).cuda()
    test_styles = torch.stack([dataset[10094][1], dataset[1282][1], dataset[25954][1], dataset[25513][1], dataset[7007][1]]).cuda()
    with torch.no_grad():
        test_styles = style_encoder.encode(test_styles)

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
            x, style = batch
            get_batch_time += time.time() - tmp
            tmp = time.time()
            x = x.cuda()
            style = style.cuda()
            with torch.no_grad():
                style = style_encoder.encode(style)
            load_batch_time += time.time() - tmp
            z, kld = vae(x, style)
            x_rec = vae.decode(z, style)

            #x_rec, x_rec_noised = compute_with_noised(z, vae)

            #noised_losses = [((x - xr)**2).sum((-1, -2, -3)) for xr in x_rec_noised]
            #rec_noised_loss = torch.stack(noised_losses).mean()

            kld_loss = kld.mean()
            rec_loss = ((x - x_rec)**2).sum((-1, -2, -3)).mean()

            scale_factor = x.size(-1) * x.size(-2)
            if cos_loss_coef > 0.:
                style_loss = compute_style_shifted_loss(vae, style_encoder, z, style)
            else:
                style_loss = torch.scalar_tensor(0.).cuda()

            #loss = (rec_loss + kld_coef * kld_loss + noised_coef * rec_noised_loss) / scale_factor
            loss = (rec_loss + kld_coef * kld_loss) / scale_factor + style_loss * cos_loss_coef

            for param in vae.parameters():
                param.grad = None
            optim.zero_grad()
            loss.backward()
            #clip_grad_norm(vae.parameters(), 100)
            optim.step()

            wandb.log({"kld_loss": kld_loss.detach().cpu().item(),
                       "rec_loss": rec_loss.detach().cpu().item(),
                       "style_loss": style_loss.detach().cpu().item(),
                       #"rec_noised_loss": rec_noised_loss.detach().cpu().item(),
                       "loss": loss.detach().cpu().item(),
                       "get_batch_time": get_batch_time,
                       "load_batch_time": load_batch_time,
                       "step": i*batches_per_epoch + k + 1}, step=i*batches_per_epoch + k + 1)

        if i % log_images_every == 0:
            wandb.log(build_images_to_log(vae, style_encoder, test_images, test_styles), step=(i+1) * batches_per_epoch)
