import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
# from data_process import importdata
import data_process as data
import json


with open("./config.json") as f:
    config=json.load(f)

device = torch.device("cuda" if config['training']['gpu'] and (torch.cuda.is_available()) else "cpu")

datadir='datasets/'+config['circuit']['dev']+'/'+config['circuit']['sel']+'/'
rawdata = data.importdata(path=datadir, sel=config['circuit']['dev'],ni=config['circuit']['ninput'], \
                          no=config['circuit']['noutput'],aging=config['circuit']['aging'], \
                          agestep=config['circuit']['agestep'], unistep=config['circuit']['unistep'], \
                          maxsample=2000,tscale=config['circuit']['hRNN'])

if config['circuit']['unistep']:
    for key in rawdata:
        for i in range(len(rawdata[key])):
            if len(rawdata[key][i])>config['circuit']['nstep']:
                rawdata[key][i]=rawdata[key][i][0:config['circuit']['nstep']]

#Outputs must be normalized if cross-entropy loss is used:
if config['training']['loss']=='CE':
    for i in range(config['circuit']['noutput']):
        if str(i) not in config['model']['normoutputs']:
            config['model']['normoutputs'][str(i)]=[]
    if config['model']['outclip'] is None:
        config['model']['outclip']='sig_cust'
rawdata['inset'],config['model']['norminputs']=data.normalize(rawdata['inset'], config['model']['norminputs'])
rawdata['outset'],config['model']['normoutputs']=data.normalize(rawdata['outset'], config['model']['normoutputs'])

train_dataset, val_dataset= data.split(rawdata,rate=0.8,shuffle=False,aging=config['circuit']['aging'], \
                              agestep=config['circuit']['agestep'])

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


data_module = CircuitDataModule(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=32, num_workers=4)
print('done')