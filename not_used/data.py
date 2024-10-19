import lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Need to add in_len and out_len in the dataset, see data/data_loader.py
class CircuitDataset(Dataset):
    def __init__(self, dataset):
        self.dataset=dataset
    def __len__(self):
        for key in self.dataset:
            return self.dataset[key].size(0)
    def __getitem__(self, idx):
        sample={}
        for key in self.dataset:
            sample[key]=self.dataset[key][idx]
        return sample

class CircuitDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=32, num_workers=4):
        super().__init__()
        CircuitDataset,
        self.train_dataset = train_dataset,
        self.valid_dataset = val_dataset,
        self.batch_size = batch_size,
        self.num_workers = num_workers,

    def setup(self, stage=str) -> None:
        """
        Sets up the datasets for training and validation.
        """
        self.train_dataset =  CircuitDataset(self.train_dataset)
        self.val_dataset = CircuitDataset(self.val_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
