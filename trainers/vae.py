from models.vae import VAE
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb
from tqdm import tqdm


def build_images_to_log(vae:VAE, source):
    log_dict = {"orig_imgs": [wandb.Image(img.transpose(0, 2).numpy()) for img in source.cpu()]}
    with torch.no_grad():
        z = vae.encode(source)
        # reconstructed images
        x_rec = vae.decode(z)
        log_dict["rec_imgs"] = [wandb.Image(img.transpose(0, 2).numpy()) for img in x_rec.cpu()]
        # with randomized z
        for i in range(len(z)):
            new_z = [zz for zz in z]
            imgs = []
            for _ in range(4):
                new_z[i] = torch.randn_like(new_z[i])
                x_rec = vae.decode(new_z)
                imgs.extend([wandb.Image(img.transpose(0, 2).numpy()) for img in x_rec.cpu()])
            log_dict[f"noised_{i+1}_imgs"] = imgs
        # generated
        new_z = [torch.randn_like(zz) for zz in z]
        x_rec = vae.decode(new_z)
        log_dict["gen_imgs"] = [wandb.Image(img.transpose(0, 2).numpy()) for img in x_rec.cpu()]

    return log_dict


def train_vae(vae:VAE, dataset, dataloader_workers=8, lr=5e-5, kld_coef=0.1, noised_coef=0.2, epochs=400,
              batches_per_epoch=500, batch_size=6, log_images_every=5):
    wandb.init(project="CST-GAN-2021")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers, pin_memory=True)
    optim = Adam(vae.parameters(), lr)
    test_images = torch.stack([dataset[1723], dataset[841], dataset[9212], dataset[33443], dataset[24631]]).cuda()

    for i in range(epochs):
        for k in tqdm(range(batches_per_epoch)):
            x = next(iter(dataloader)).cuda()
            x_rec, z, kld = vae(x)
            z_noised = [torch.randn_like(zz) for zz in z]

            noised_losses = []
            for j in reversed(range(len(z))):
                if j == 0:
                    continue
                z_noised[j] = z[j]
                x_rec_noised = vae.decode(z_noised)
                noised_losses.append(((x - x_rec_noised)**2).sum((-1, -2, -3)))
            rec_noised_loss = torch.stack(noised_losses).mean()

            kld_loss = kld.mean()
            rec_loss = ((x - x_rec)**2).sum((-1, -2, -3)).mean()

            scale_factor = x.size(-1) * x.size(-2)
            loss = (rec_loss + kld_coef * kld_loss + noised_coef * rec_noised_loss) / scale_factor
            optim.zero_grad()
            loss.backward()
            optim.step()

            wandb.log({"kld_loss": kld_loss.detach().cpu().item(),
                       "rec_loss": rec_loss.detach().cpu().item(),
                       "rec_noised_loss": rec_noised_loss.detach().cpu().item(),
                       "loss": loss.detach().cpu().item()}, step=i*batches_per_epoch + k + 1)

        if i % log_images_every == 0:
            wandb.log(build_images_to_log(vae, test_images), step=(i+1) * batches_per_epoch)
