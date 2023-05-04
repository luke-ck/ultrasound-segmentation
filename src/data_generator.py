import torch
from torch.utils.data import Dataset
from src.transforms import Lambda


class MyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.seed = 42
        self.lmbda = Lambda(lambda x: torch.where(x > 0, 1, 0))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]

        if self.transform:
            x, y = self.transform(x, y)
        else:
            x, y = self.lmbda(torch.from_numpy(x), torch.from_numpy(y))
            y = y.long()

        return x, y
