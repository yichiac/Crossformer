from cross_models.cross_former import CrossformerCircuit
from utils.tools import EarlyStopping, adjust_learning_rate
import models.data_process as data
from data import CircuitDataModule

import torch
from torch import optim
import torch.nn as nn
from torch.nn import DataParallel

import matplotlib.pyplot as plt
import numpy as np
import json
import time
import os


with open("./config.json") as f:
    config=json.load(f)

device = torch.device("cuda" if config['training']['gpu'] and (torch.cuda.is_available()) else "cpu")
print(device)

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

# for plotting
train_tensor,valid_tensor=[{},{}]
for key in train_dataset:
    train_tensor[key]=torch.tensor(train_dataset[key], device=device)
    valid_tensor[key]=torch.tensor(val_dataset[key], device=device)

datamodule = CircuitDataModule(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=16, num_workers=4, device=torch.device('cpu'))
datamodule.setup()
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

args = {'data_dim': 3,
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
        'reproj': 'fc'
        }

# model = CrossformerCircuit(data_dim=5, in_len=501, out_len=501, seg_len=6, win_size=4,
#                 factor=10, d_model=256, d_ff=512, n_heads=8, e_layers=3,
#                 dropout=0.0, baseline = False) #, device=torch.device('cuda:0'))

setting = 'Crossformer_{}_il{}_ol{}_sl{}_win{}_fa{}_dm{}_nh{}_el{}_itr{}_reproj_{}'.format('circuit',
            args['in_len'], args['out_len'], args['seg_len'], args['win_size'], args['factor'],
            args['d_model'], args['n_heads'], args['e_layers'], args['iter'], args['reproj'])
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
                           baseline=args['baseline'])


model.to(device)

train_steps = len(train_loader)

early_stopping = EarlyStopping(patience=3, verbose=True) #default patience=3 in args
optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-7, eps=0.0001) # might need to change the optimizer later
model_optim = optim.Adam(model.parameters(), lr=1e-4)
# savedir='model/'+config['circuit']['dev']+'_'+config['circuit']['sel']+'_gru'+str(i)

l = val_dataset['outset'].size(1)


def vali(model, vali_loader, l):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for sample in vali_loader:
            insample = sample['inset'][:,0:l,:].to(device)
            # insample = torch.cat([sample['inset'][:,0:l,:], sample['outset'][:,0:l,:]], dim=-1).to(device)
            # true = insample
            true = sample['outset'][:,0:l,:].to(device)
            pred = model(insample)
            loss = NRMSE(pred, true)
            total_loss.append(loss.item())

    total_loss = np.average(total_loss)
    model.train()
    return total_loss


def NRMSE(prediction, target):
    error = torch.sum((prediction - target) ** 2, dim=1)
    norm = torch.sum(target ** 2, dim=1)
    NRMSerror = torch.mean(torch.sqrt(error / norm))
    return NRMSerror


if __name__ == '__main__':
    train_loss_total = []
    vali_loss_total = []
    for epoch in range(0, config['training']['maxepoch']+1):
        time_now = time.time()
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for sample in train_loader:
            insample = sample['inset'][:,0:l,:].to(device)
            # insample = torch.cat([sample['inset'][:,0:l,:], sample['outset'][:,0:l,:]], dim=-1).to(device)
            # true = insample
            true = sample['outset'][:,0:l,:].to(device)
            optimizer.zero_grad()
            pred = model(insample)
            loss = NRMSE(pred, true)
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
        train_loss_total.append(train_loss)
        vali_loss = vali(model, val_loader, l)
        vali_loss_total.append(vali_loss)
        # test_loss = vali(model, test_loader, l)
        # test_loss = vali(model, val_loader, l)

        # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
        #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(model_optim, epoch+1)

    best_model_path = path+'/'+'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    state_dict = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
    torch.save(state_dict, path+'/'+'checkpoint.pth')

# plot loss
fig, ax1 = plt.subplots(figsize=(20, 20))
x = np.linspace(0, len(train_loss_total)-1, num=len(train_loss_total))

ax1.plot(x, train_loss_total, label='Training loss')
ax1.plot(x, vali_loss_total, label='Validation loss')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_ylim(1e-4, 1)
ax1.legend(loc='upper left', fancybox=False, framealpha=1, facecolor='white', edgecolor='black')

plt.savefig('figs/loss.png', dpi=300)
plt.show()
plt.close


# outpred = []
# model.eval()
# with torch.no_grad():
#     for sample in val_loader:
#         insample = sample['inset'][:,0:l,:].to(device)
#         pred = model(insample)

# # plot prediction
# t = np.linspace(0,config['circuit']['hRNN']*1e9*(config['circuit']['nstep']-1),config['circuit']['nstep'])
# ns = 17
# plt.figure(1,figsize=(10,10))
# plt.clf()
# for i in range(config['circuit']['ninput']):
#     if str(i) in config['model']['norminputs']:
#         m1,m2=config['model']['norminputs'][str(i)]
#     else:
#         m1,m2=[0,1]
#     plt.subplot(config['circuit']['ninput']+config['circuit']['noutput'],1,i+1)
#     plt.plot(t, valid_tensor['inset'][ns,:,i]*(m2-m1)+m1,linewidth=4)
#     plt.ylabel('Vin%d [V]'%(i+1),fontweight='bold')
#     plt.grid()
# for i in range(config['circuit']['noutput']):
#     if str(i) in config['model']['normoutputs']:
#         m1,m2=config['model']['normoutputs'][str(i)]
#     else:
#         m1,m2=[0,1]
#     plt.subplot(config['circuit']['ninput']+config['circuit']['noutput'],1,config['circuit']['ninput']+i+1)
#     plt.plot(t,(valid_tensor['outset'][ns,:,i]*(m2-m1)+m1),linewidth=4,label='True',zorder=2)
#     plt.scatter(t, (outpred[ns,:,i]*(m2-m1)+m1),s=50,c='darkorange',label='Pred',zorder=1)
#     # plt.plot(t,valid_tensor['outset'][ns*config['circuit']['agestep']+config['circuit']['agestep']-1,:,i],linewidth=4,label='True (10)')
#     # plt.plot(t,outpred[ns*config['circuit']['agestep']+config['circuit']['agestep']-1,:,i],'--',linewidth=4,label='Pred (10)')
#     plt.ylabel('Vout%d [V]'%(i+1),fontweight='bold')
#     plt.grid()
# #plt.ylim([-0.01,1.3])
# plt.legend(loc='best',fancybox=True, framealpha=1, facecolor='white', edgecolor='black',ncol=1,prop={'size': 20})

# plt.xlabel('Time [ns]',fontweight='bold')
