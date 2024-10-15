import pytorch_lightning as pl
from pytorch_lightning import Trainer
from cross_models.cross_former import Crossformer
from data import CircuitDataModule
import torch
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

data_module = CircuitDataModule(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=32, num_workers=4)

crossformer_model = Crossformer(data_dim=3, in_len=500, out_len=500, seg_len=6, win_size=4,
                factor=10, d_model=512, d_ff=1024, n_heads=8, e_layers=3,
                dropout=0.0, baseline = False, device=torch.device('cuda:0'))

trainer = Trainer(
    max_epochs = config['training']['maxepoch'],
    # accelerator = 'auto',
    # accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
    # devices = 1 if torch.cuda.is_available() else None,
    precision= 32,
    # precision=16 if config['training']['mixed_precision'] else 32,
)

trainer.fit(crossformer_model, datamodule=data_module)