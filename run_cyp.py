from cross_models.cross_former import Crossformer, CrossformerCircuit
from utils.tools import EarlyStopping, adjust_learning_rate
import models.data_process as data
from data import CircuitDataModule

from data_cyp import CropYieldDataset, CropYieldDataModule

import torch
from torch import optim
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F


import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import json
import time
import os
from sklearn.metrics import root_mean_squared_error, r2_score


# with open("./config.json") as f:
#     config=json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

max_epoch = 100

# datadir='datasets/'+config['circuit']['dev']+'/'+config['circuit']['sel']+'/'
# rawdata = data.importdata(path=datadir, sel=config['circuit']['dev'],ni=config['circuit']['ninput'], \
#                         no=config['circuit']['noutput'],aging=config['circuit']['aging'], \
#                         agestep=config['circuit']['agestep'], unistep=config['circuit']['unistep'], \
#                         maxsample=2000,tscale=config['circuit']['hRNN'])

# if config['circuit']['unistep']:
#     for key in rawdata:
#         for i in range(len(rawdata[key])):
#             if len(rawdata[key][i])>config['circuit']['nstep']:
#                 rawdata[key][i]=rawdata[key][i][0:config['circuit']['nstep']]

# #Outputs must be normalized if cross-entropy loss is used:

# if config['training']['loss']=='CE':
#     for i in range(config['circuit']['noutput']):
#         if str(i) not in config['model']['normoutputs']:
#             config['model']['normoutputs'][str(i)]=[]
#     if config['model']['outclip'] is None:
#         config['model']['outclip']='sig_cust'
# rawdata['inset'],config['model']['norminputs']=data.normalize(rawdata['inset'], config['model']['norminputs'])
# rawdata['outset'],config['model']['normoutputs']=data.normalize(rawdata['outset'], config['model']['normoutputs'])

# train_dataset, val_dataset= data.split(rawdata, rate=0.8, shuffle=False, aging=config['circuit']['aging'], \
#                             agestep=config['circuit']['agestep'])

# # for plotting
# train_tensor,valid_tensor=[{},{}]
# for key in train_dataset:
#     train_tensor[key]=torch.tensor(train_dataset[key], device=device)
#     valid_tensor[key]=torch.tensor(val_dataset[key], device=device)

# datamodule = CircuitDataModule(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=16, num_workers=4, device=torch.device('cpu'))
# datamodule.setup()
# train_loader = datamodule.train_dataloader()
# val_loader = datamodule.val_dataloader()


# CYP dataset and dataloader
base_path = '/projects/dali/yichia3'
# base_path = 'datasets'
data_path = os.path.join(base_path, 'brazil_soybeans/Brazil_soybeans_data.pkl')
datamodule = CropYieldDataModule(data_path=data_path)

datamodule.prepare_data()
datamodule.setup()
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
# test_loader = datamodule.test_dataloader()

args = {'data_dim': 7,
        'in_len': 240,
        'out_len': 1,
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

model = Crossformer(data_dim=args['data_dim'],
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

early_stopping = EarlyStopping(patience=10, verbose=True)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-7, eps=0.0001) # might need to change the optimizer later
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# savedir='model/'+config['circuit']['dev']+'_'+config['circuit']['sel']+'_gru'+str(i)

# l = val_dataset['outset'].size(1)
l = args['in_len']

# ================ Need to do the normalization for datasets ================

# def compute_loss(outputs, targets):
#         return F.mse_loss(outputs, targets)

def vali(model, vali_loader, l):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for sample in vali_loader:
            insample = sample[0][:,0:l,:].to(device)
            true = sample[2].to(device)
            pred = model(insample)
            loss = compute_loss(pred, true)
            total_loss.append(loss.item())

    total_loss = np.average(total_loss)
    model.train()
    return total_loss


# def NRMSE(prediction, target):
#     error = torch.sum((prediction - target) ** 2, dim=1)
#     norm = torch.sum(target ** 2, dim=1)
#     NRMSerror = torch.mean(torch.sqrt(error / norm))
#     return NRMSerror

print('start training')

if __name__ == '__main__':
    train_loss_total = []
    vali_loss_total = []
    criterion = nn.MSELoss()

    for epoch in range(0, 500): # set max_epoch
        # time_now = time.time()
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for sample in train_loader:
            insample = sample[0][:,0:l,:].to(device)
            true = sample[2].to(device)
            optimizer.zero_grad()
            pred = model(insample)
            loss = torch.sqrt(criterion(pred, true))
            train_loss.append(loss.item())
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

        adjust_learning_rate(optimizer, epoch+1)

    best_model_path = path+'/'+'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    state_dict = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
    torch.save(state_dict, path+'/'+'checkpoint.pth')

# plot loss
fig, ax1 = plt.subplots()
x = np.linspace(0, len(train_loss_total)-1, num=len(train_loss_total))

ax1.plot(x, train_loss_total, label='Training loss')
ax1.plot(x, vali_loss_total, label='Validation loss')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
# ax1.set_ylim(1e-4, 1)
ax1.legend(loc='upper right', fancybox=False, framealpha=1, facecolor='white', edgecolor='black')

plt.savefig('figs/loss.png', dpi=300)
plt.show()
plt.close


# plot prediction
model.eval()
predictions = []
true_labels = []
train_predictions = []
train_true_labels = []

with torch.no_grad():
    for sample in val_loader:
        insample = sample[0][:,0:l,:].to(device)
        true = sample[2].to(device)
        pred = model(insample)
        predictions.extend(pred.squeeze().tolist())
        true_labels.extend(true.squeeze().tolist())

    for sample in train_loader:
        insample = sample[0][:,0:l,:].to(device)
        true = sample[2].to(device)
        pred = model(insample)
        train_predictions.extend(pred.squeeze().tolist())
        train_true_labels.extend(true.squeeze().tolist())

# rescale to original scale
predictions = np.array(predictions)
true_labels = np.array(true_labels)
# predictions = predictions*(5000-105) + 105 # max=5000, min=105, minmax scaling
# true_labels = true_labels*(5000-105) + 105

train_predictions = np.array(train_predictions)
train_true_labels = np.array(train_true_labels)
# train_predictions = train_predictions*(5000-105) + 105 # max=5000, min=105, minmax scaling
# train_true_labels = train_true_labels*(5000-105) + 105

plt.figure(figsize=(12,5))
ax=plt.subplot(1,2,1)
plt.plot(true_labels, predictions,'.')
plt.plot([0,5000],[0,5000],'r--')
plt.xlabel('Obs')
plt.ylabel('Model Est.')
plt.text(0.05,0.95,f'RMSE={root_mean_squared_error(true_labels, predictions):.2f}\nR2={r2_score(true_labels, predictions):.2f}',
         transform=ax.transAxes, ha='left', va='top')
plt.title('Validation set')

ax=plt.subplot(1,2,2)
plt.plot(train_true_labels, train_predictions,'.')
plt.plot([0,5000],[0,5000],'r--')
plt.xlabel('Obs')
plt.ylabel('Model Est.')
plt.text(0.05,0.95,f'RMSE={root_mean_squared_error(train_true_labels, train_predictions):.2f}\nR2={r2_score(train_true_labels, train_predictions):.2f}',
         transform=ax.transAxes, ha='left', va='top')
plt.title('Train set')
plt.savefig('figs/prediction.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
plt.close()




# t = np.linspace(0,config['circuit']['hRNN']*1e9*(config['circuit']['nstep']-1),config['circuit']['nstep'])
# ns = 17
# plt.figure(1, figsize=(10, 16))
# plt.clf()

# plt.subplots_adjust(hspace=0.5)

# for i in range(config['circuit']['ninput']):
#     if str(i) in config['model']['norminputs']:
#         m1,m2=config['model']['norminputs'][str(i)]
#     else:
#         m1,m2=[0,1]
#     plt.subplot(config['circuit']['ninput']+config['circuit']['noutput'],1,i+1)
#     plt.plot(t, valid_tensor['inset'][ns,:,i].cpu().numpy()*(m2-m1)+m1,linewidth=4)
#     plt.ylabel('Vin%d [V]'%(i+1),fontweight='bold')
#     plt.grid()
#     plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# for i in range(config['circuit']['noutput']):
#     if str(i) in config['model']['normoutputs']:
#         m1,m2=config['model']['normoutputs'][str(i)]
#     else:
#         m1,m2=[0,1]
#     plt.subplot(config['circuit']['ninput']+config['circuit']['noutput'],1,config['circuit']['ninput']+i+1)
#     plt.plot(t,(valid_tensor['outset'][ns,:,i].cpu().numpy()*(m2-m1)+m1),linewidth=4,label='True',zorder=2)
#     plt.scatter(t, (outpred[ns,:,i].cpu().numpy()*(m2-m1)+m1),s=20,c='darkorange',label='Pred',zorder=1)
#     # plt.plot(t,valid_tensor['outset'][ns*config['circuit']['agestep']+config['circuit']['agestep']-1,:,i],linewidth=4,label='True (10)')
#     # plt.plot(t,outpred[ns*config['circuit']['agestep']+config['circuit']['agestep']-1,:,i],'--',linewidth=4,label='Pred (10)')
#     plt.ylabel('Vout%d [V]'%(i+1),fontweight='bold')
#     plt.grid()
#     plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# plt.legend(bbox_to_anchor=(0.5, -0.4), loc='upper center',
#            fancybox=True, framealpha=1, facecolor='white',
#            edgecolor='black', ncol=2, prop={'size': 12})
# plt.xlabel('Time [ns]',fontweight='bold')
# plt.tight_layout()
# plt.savefig('figs/prediction.png', dpi=300, bbox_inches='tight', pad_inches=0.3)

# plt.ylim([-0.01,1.3])
# plt.legend(loc='best',fancybox=True, framealpha=1, facecolor='white', edgecolor='black',ncol=1,prop={'size': 20})