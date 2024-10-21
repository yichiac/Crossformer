# import lightning as pl
# from lightning.pytorch import Trainer
from cross_models.cross_former import CrossformerCircuit
from utils.tools import EarlyStopping, adjust_learning_rate
# from models.data_process import NRMSE
import models.data_process as data
from data import CircuitDataModule

import numpy as np
# from utils.metrics import metric
# from cross_models.crossformer_pl import CrossformerLightningModule

import torch
from torch import optim
import torch.nn as nn
from torch.nn import DataParallel

import json
import time
import os

# if __name__ == "__main__":

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
# class loss_crossentropy(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, pred, targ):
#         a=torch.mul(-(1-targ),torch.log10(1-pred))
#         b=torch.mul(-targ,torch.log10(pred))
#         return torch.mean(a+b)

if config['training']['loss']=='CE':
    # loss = loss_crossentropy()
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

# for sample in train_loader:
#     input = sample['inset']
#     target = sample['outset']
args = {'data_dim': 5,
        'in_len': 501,
        'out_len': 501,
        'seg_len': 6,
        'win_size': 4,
        'factor': 10,
        'd_model': 256,
        'd_ff': 512,
        'n_heads': 8,
        'e_layers': 3,
        'dropout': 0.0,
        'baseline': False,
        'iter': 1,
        }

# model = CrossformerCircuit(data_dim=5, in_len=501, out_len=501, seg_len=6, win_size=4,
#                 factor=10, d_model=256, d_ff=512, n_heads=8, e_layers=3,
#                 dropout=0.0, baseline = False) #, device=torch.device('cuda:0'))

setting = 'Crossformer_{}_il{}_ol{}_sl{}_win{}_fa{}_dm{}_nh{}_el{}_itr{}'.format('circuit',
            args['in_len'], args['out_len'], args['seg_len'], args['win_size'], args['factor'],
            args['d_model'], args['n_heads'], args['e_layers'], args['iter'])
path = os.path.join('./checkpoints/', setting)
if not os.path.exists(path):
    os.makedirs(path)

model = CrossformerCircuit(data_dim=args['data_dim'],
                           in_len=args['in_len'],
                           out_len=args['out_len'],
                           seg_len=args['seg_len'],
                           win_size=args['win_size'],
                           factor=args['factor'],
                           d_model=args['d_model'],
                           d_ff=args['d_ff'],
                           n_heads=args['n_heads'],
                           e_layers=args['e_layers'],
                           dropout=args['dropout'],
                           baseline=args['baseline']) #, device=torch.device('cuda:0'))


model.to(device)
# NRMSE_list = []

train_steps = len(train_loader)
early_stopping = EarlyStopping(patience=3, verbose=True) #default patience=3 in args
optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-7, eps=0.0001) # might need to change the optimizer later

model_optim = optim.Adam(model.parameters(), lr=1e-4)
# criterion =  nn.MSELoss() #NRMSE
savedir='model/'+config['circuit']['dev']+'_'+config['circuit']['sel']+'_gru'+str(i)

l = val_dataset['outset'].size(1)


def vali(model, vali_loader, l):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for sample in vali_loader:
            # insample = sample['inset'][:,0:l,:].to(device)
            # true = sample['outset'][:,0:l,:].to(device)
            insample = torch.cat([sample['inset'][:,0:l,:], sample['outset'][:,0:l,:]], dim=-1).to(device)
            true = insample
            pred = model(insample)
            loss = NRMSE(pred, true)
            total_loss.append(loss.item())

    total_loss = np.average(total_loss)
    model.train()
    return total_loss

# class trainingloss(nn.Module):
#     def __init__(self,losstype,lds=None,ldsweight=0,device=torch.device('cpu')):
#         super().__init__()
#         self.loss=loss_crossentropy()
#         self.device=device
#     def forward(self,pred,targ,model,otherloss=None):
#         lossvalues=[self.loss(pred,targ)]
#         # if hasattr(self,'ldsreg'):
#         #     lossvalues.append(self.ldsreg(model)*self.ldsweight)
#         # if otherloss:
#         #     lossvalues.append(otherloss)
#         lossvalues.append(otherloss)
#         return (torch.sum(torch.stack(lossvalues)), lossvalues)

def NRMSE(prediction, target):
    # prediction = prediction.detach().numpy()
    # target = target.detach().numpy()

    # error=np.sum(np.power(np.subtract(prediction,target),2),axis=1)
    # norm=np.sum(np.power(target,2),axis=1)
    # NRMSerror=np.mean(np.sqrt(error/norm))
    # return (NRMSerror)
    error = torch.sum((prediction - target) ** 2, dim=1)
    norm = torch.sum(target ** 2, dim=1)
    NRMSerror = torch.mean(torch.sqrt(error / norm))

    return NRMSerror



if __name__ == '__main__':
    for epoch in range(1, config['training']['maxepoch']+1):
    # for epoch in range(3):
        time_now = time.time()
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for sample in train_loader:
            # insample = sample['inset'][:,0:l,:].to(device)
            insample = torch.cat([sample['inset'][:,0:l,:], sample['outset'][:,0:l,:]], dim=-1).to(device)
            # true = sample['outset'][:,0:l,:].to(device)
            true = insample
            optimizer.zero_grad()
            # print('shape of insample before forward pass')
            # print(insample.shape)
            pred = model(insample)
            loss = NRMSE(pred, true)
            # losstotal, losses = loss(pred, true)
            train_loss.append(loss.item())

            if (i+1) % 5==0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time()-time_now)/iter_count
                left_time = speed*((config['training']['maxepoch'] - epoch)*train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            loss.backward()
            optimizer.step()

        print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = vali(model, val_loader, l)
        # test_loss = vali(model, test_loader)
        test_loss = vali(model, val_loader, l)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(model_optim, epoch+1)

    best_model_path = path+'/'+'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    state_dict = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
    torch.save(state_dict, path+'/'+'checkpoint.pth')



# model,history=RNNcustom.train(model,config,config['training']['maxepoch'],dataloader,valid_tensor,optimizer,scheduler,loss,savedir,init,device,freshmodel)
# if 't' in valid_tensor:
#     invalid={'inset':valid_tensor['inset'], 't':valid_tensor['t']}
# else:
#     invalid=valid_tensor['inset']
# with torch.no_grad():
#     outpred=model(invalid)
# NRMSE_list.append(data.NRMSE(outpred.tolist(),valid_tensor['outset'].tolist()))



# trainer = Trainer(
#     max_epochs = config['training']['maxepoch'],
#     precision= 32,
#     accelerator='gpu' if torch.cuda.is_available() else 'cpu',
#     devices = 1,
#     # devices = 1 if torch.cuda.is_available() else None,
#     # precision=16 if config['training']['mixed_precision'] else 32,
# )

# trainer.fit(model=crossformer_model, datamodule=datamodule)
