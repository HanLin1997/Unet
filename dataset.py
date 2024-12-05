import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf

class MyDataset(Dataset):
    def __init__(self):
        self.items = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        if not os.path.exists(path):
            print(path)
            raise(f'path {path} not exists.')
        
        img = cv2.imread(path)
        return img, label