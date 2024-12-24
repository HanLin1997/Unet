import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2

dictionary = {"baso": 0, "eosi": 1, "lymp": 2, "mono": 3, "neut": 4}

class MyDataset(Dataset):
    def __init__(self):
        self.items = []
        with open("train.txt") as f:
            lines = f.readlines()
        for line in lines:
            line_split = line.split(',')
            img_path = line_split[0]
            mask_path = line_split[1]
            label = int(line_split[2])
            self.items.append((img_path, mask_path, label))
        
        self.train_transform = Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            OneOf([
                A.HueSaturationValue(),
                A.RandomBrightnessContrast(),
                A.RGBShift(),
                A.RandomGamma(),
                A.Blur()
            ], p=1),
            A.Rotate(limit=45, p=0.5),
            #A.Normalize(),
            ToTensorV2()
        ], is_check_shapes=False)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path, mask_path, label = self.items[index]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f'Path "{img_path}" does not exist.')
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f'Path "{mask_path}" does not exist.')
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_LINEAR)
        mask_monolayer = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_monolayer = cv2.resize(mask_monolayer, (1024, 768), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros((768, 1024, len(dictionary)), dtype=int)
        mask[:, :, label] = mask_monolayer > 127

        augmented = self.train_transform(image=img, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']
        return augmented_image, augmented_mask.permute(2, 0, 1)

def show_mask(mask_label):
    c, h, w = mask_label.shape
    cmp_mask = np.zeros((h, w), dtype=np.int32)
    color_idx = 0
    for i in range(c):
        if mask_label[i].sum() == 0:
            continue
        else:
            color_idx = color_idx + 1
            mask = mask_label[i] > 0
            cmp_mask[mask] = color_idx
    return cmp_mask

if __name__ == "__main__":
    dataset = MyDataset()
    
    fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(20, 8))

    for i in range(8):
        a = dataset[380]
        img = a[0].permute(1, 2, 0).numpy()
        mask = show_mask(a[1].numpy())
        axes[0, i].imshow(img)
        axes[1, i].imshow(mask, cmap=plt.cm.nipy_spectral)
    
    fig.tight_layout()
    plt.show()
    