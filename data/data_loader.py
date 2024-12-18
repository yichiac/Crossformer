import os
import numpy as np
import pandas as pd
import lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler

import warnings
warnings.filterwarnings('ignore')


class CircuitDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        for key in self.dataset:
            return self.dataset[key].size(0)

    def __getitem__(self, idx):
        sample = {}
        for key in self.dataset:
            sample[key] = self.dataset[key][idx]
        return sample


class CircuitDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size: int = 16, num_workers: int = 4, device=torch.device('cpu')):
        super().__init__()
        CircuitDataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def setup(self, stage=str) -> None:
        """
        Sets up the datasets for training and validation.
        """
        for key in self.train_dataset:
            self.train_dataset[key]=torch.tensor(self.train_dataset[key], device=self.device)
            self.val_dataset[key]=torch.tensor(self.val_dataset[key], device=self.device)

        self.train_dataset =  CircuitDataset(self.train_dataset)
        self.val_dataset = CircuitDataset(self.val_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True)


# class Dataset_MTS(Dataset):
#     def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None,
#                   data_split = [0.7, 0.1, 0.2], scale=True, scale_statistic=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         self.in_len = size[0]
#         self.out_len = size[1]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train':0, 'val':1, 'test':2}
#         self.set_type = type_map[flag]

#         self.scale = scale
#         #self.inverse = inverse

#         self.root_path = root_path
#         self.data_path = data_path
#         self.data_split = data_split
#         self.scale_statistic = scale_statistic
#         self.__read_data__()

#     def __read_data__(self):
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#         if (self.data_split[0] > 1):
#             train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
#         else:
#             train_num = int(len(df_raw)*self.data_split[0]);
#             test_num = int(len(df_raw)*self.data_split[2])
#             val_num = len(df_raw) - train_num - test_num;
#         border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
#         border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         cols_data = df_raw.columns[1:]
#         df_data = df_raw[cols_data]

#         if self.scale:
#             if self.scale_statistic is None:
#                 self.scaler = StandardScaler()
#                 train_data = df_data[border1s[0]:border2s[0]]
#                 self.scaler.fit(train_data.values)
#             else:
#                 self.scaler = StandardScaler(mean = self.scale_statistic['mean'], std = self.scale_statistic['std'])
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.in_len
#         r_begin = s_end
#         r_end = r_begin + self.out_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]

#         return seq_x, seq_y

#     def __len__(self):
#         return len(self.data_x) - self.in_len- self.out_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
