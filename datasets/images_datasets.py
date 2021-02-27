import torch
import cv2
import numpy as np
import random
from torch.nn import functional as F
from torch.utils.data import Dataset
import os


class RawImagesDataset(Dataset):
    def __init__(self, root_dir:str, crop_to=512, flip_augmentation=True):
        self.paths = [os.path.join(root_dir, dir_path, f) for dir_path, _, files in os.walk(root_dir) for f in files]
        self.crop_to = crop_to
        self.flip_augmentation = flip_augmentation

    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])[:, :, :3]  # Remove alpha component

        # Random crop
        if img.shape[0] > self.crop_to:
            tmp = random.randint(0, img.shape[0] - self.crop_to)
            img = img[tmp:self.crop_to+tmp]
        if img.shape[1] > self.crop_to:
            tmp = random.randint(0, img.shape[1] - self.crop_to)
            img = img[:, tmp:self.crop_to + tmp]

        img = torch.tensor(img, dtype=torch.float)

        # Flip if needed
        if self.flip_augmentation and random.random() > 0.5:
            img.flip([1])
        return img.flip([2]).transpose(0, 2) / 255.

    def __len__(self):
        return len(self.paths)

