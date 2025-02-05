import os
import cv2
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2

dictionary = {"background": 0, "baso": 1, "eosi": 2, "lymp": 3, "mono": 4, "neut": 5}

class MyDataset(Dataset):
    def __init__(self, aug=True):
        self.items = []
        self.aug = aug
        with open("train.txt") as f:
            lines = f.readlines()
        for line in lines:
            line_split = line.split(',')
            img_path = line_split[0].strip()
            mask_path = line_split[1].strip()
            #label = int(line_split[2])
            self.items.append((img_path, mask_path))
        
        self.train_transform = Compose([
            #A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            #A.Rotate(limit=45, p=0.5),
            A.Affine(scale=(1.0, 6.0), translate_percent=(-0.4, 0.4), keep_ratio=True, p=1),
            #A.Normalize(),
            ToTensorV2()
        ], is_check_shapes=False)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path, mask_path = self.items[index]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f'Path "{img_path}" does not exist.')
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f'Path "{mask_path}" does not exist.')
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        '''
        mask_monolayer = Image.open(mask_path)
        mask_monolayer = np.array(mask_monolayer.convert("L"))
        mask_monolayer = cv2.resize(mask_monolayer, (1024, 768), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros((768, 1024, len(dictionary)), dtype=int)
        mask[:, :, label] = mask_monolayer > 127
        mask[:, :, 0] = mask_monolayer < 127
        '''
        mask = Image.open(mask_path)
        mask = np.array(mask.convert("L"))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_LINEAR)
        if self.aug:
            augmented = self.train_transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        else:
            img = torch.tensor(img).permute(2, 0, 1)
            mask = torch.tensor(mask).unsqueeze(0)
        return img, mask

def show_mask(mask_label):
    c, h, w = mask_label.shape
    cmp_mask = np.zeros((h, w), dtype=np.int32)
    #color_idx = 0
    for i in range(1, c):
        if mask_label[i].sum() == 0:
            continue
        else:
            color_idx = i
            mask = mask_label[i] > 0
            cmp_mask[mask] = color_idx
    return cmp_mask

if __name__ == "__main__":
    dataset = MyDataset(aug=True)
    
    fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(20, 8))

    for i in range(8):
        a = dataset[1]
        img = a[0].permute(1, 2, 0).numpy()
        #mask = show_mask(a[1].numpy())
        mask = a[1].permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[1, i].imshow(mask, cmap=plt.cm.nipy_spectral)
    
    fig.tight_layout()
    plt.show()
    