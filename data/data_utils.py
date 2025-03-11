import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Tuple

class NpToTensor:
    def __call__(self, x):
        return torch.from_numpy(x).contiguous()

class NumpyDataset(Dataset):
    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        transform: transforms.transforms.Compose = None,
        target_transform: transforms.transforms.Compose = None,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.y_data)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x = self.x_data[idx].astype("float32")
        y = self.y_data[idx].astype("float32")

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

def get_dataloader(
    x_data: np.ndarray,
    y_data: np.ndarray,
    transform_data: transforms.transforms.Compose,
    transform_label: transforms.transforms.Compose,
    shuffle: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = NumpyDataset(x_data, y_data, transform_data, transform_label)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    ) 