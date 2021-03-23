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


class DoubledImagesDataset(RawImagesDataset):
    def __init__(self, root_dir: str, crop_to=256, flip_augmentation=True):
        super().__init__(root_dir, crop_to, flip_augmentation)

    def __getitem__(self, index):
        return super().__getitem__(index), super().__getitem__(index)


class StyledImagesDataset(RawImagesDataset):
    def __init__(self, root_dir: str, crop_to_left=512, crop_to_right=256, flip_augmentation=True):
        super().__init__(root_dir, crop_to_left, flip_augmentation)
        self.crop_to_left = crop_to_left
        self.crop_to_right = crop_to_right

    def __getitem__(self, index):
        self.crop_to = self.crop_to_left
        left = super().__getitem__(index)
        self.crop_to = self.crop_to_right
        right = super().__getitem__(index)
        return left, right


class PairedImagesDataset(RawImagesDataset):
    def __init__(self, root_dir: str, crop_to=256, flip_augmentation=True, dataset_repeats=256):
        super().__init__(root_dir, crop_to, flip_augmentation)
        self.dataset_repeats = dataset_repeats
        self.id2cls = []
        self.cls2ids = []
        tmp = {}
        for p in self.paths:
            idx = p.rfind("/")
            cls = p[:idx]
            if cls not in tmp:
                tmp[cls] = len(tmp)
                self.cls2ids.append([])
            self.cls2ids[tmp[cls]].append(len(self.id2cls))
            self.id2cls.append(tmp[cls])

    def __getitem__(self, index):
        index = index % len(self.cls2ids)
        id1 = random.choice(self.cls2ids[index])
        id2 = random.choice(self.cls2ids[index])
        return super().__getitem__(id1), super().__getitem__(id2), torch.scalar_tensor(index)

    def __len__(self):
        return len(self.cls2ids) * self.dataset_repeats