import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TextDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.embeddings = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long).unsqueeze(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
