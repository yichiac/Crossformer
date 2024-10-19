import lightning as pl
from lightning.pytorch import Trainer
# from cross_models.cross_former import Crossformer
from cross_models.crossformer_pl import CrossformerLightningModule

from data import CircuitDataModule, CircuitDataset
import torch
import models.data_process as data
from torch.utils.data import DataLoader

import json

if __name__ == "__main__":

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

    train_dataset, val_dataset= data.split(rawdata, rate=0.8, shuffle=False, aging=config['circuit']['aging'], \
                                agestep=config['circuit']['agestep'])

    datamodule = CircuitDataModule(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=16, num_workers=4, device=torch.device('cpu'))
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    crossformer_model = CrossformerLightningModule(data_dim=3, in_len=500, out_len=500, seg_len=6, win_size=4,
                    factor=10, d_model=512, d_ff=1024, n_heads=8, e_layers=3,
                    dropout=0.0, baseline = False) #, device=torch.device('cuda:0'))


    # model.to(device)
    NRMSE_list = []

    trainer = Trainer(
        max_epochs = config['training']['maxepoch'],
        precision= 32,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices = 1,
        # devices = 1 if torch.cuda.is_available() else None,
        # precision=16 if config['training']['mixed_precision'] else 32,
    )

    trainer.fit(model=crossformer_model, datamodule=datamodule)
    print("Training finished!")