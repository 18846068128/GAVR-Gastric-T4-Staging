from typing import Tuple
import torch
from torch.utils.data import Dataset


class MultimodalFeatureDataset(Dataset):
    def __init__(
        self,
        radiomics,
        feat2d,
        feat3d,
        clinical,
        labels,
    ):
        self.radiomics = torch.tensor(radiomics, dtype=torch.float32)
        self.feat2d = torch.tensor(feat2d, dtype=torch.float32)
        self.feat3d = torch.tensor(feat3d, dtype=torch.float32)
        self.clinical = torch.tensor(clinical, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (self.radiomics[idx], self.feat2d[idx], self.feat3d[idx], self.clinical[idx]), self.labels[idx]
