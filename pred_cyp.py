from cross_models.cross_former import Crossformer
from utils.tools import EarlyStopping, adjust_learning_rate
import models.data_process as data
from data_cyp import CropYieldDataModule

import torch
from torch import optim
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F


import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import time
import os
from sklearn.metrics import root_mean_squared_error, r2_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

max_epoch = 100

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

# l = val_dataset['outset'].size(1)
l = args['in_len']


# plot prediction
checkpoint_path = '/home/yichia3/Crossformer/checkpoints/Crossformer_circuit_il240_ol1_sl6_win4_fa10_dm256_nh8_el3_itr1/checkpoint.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint, strict=True)

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
        print('before squeeze: ', pred.shape)
        print('before squeeze: ', true.shape)
        print('after squeeze: ', len(pred.squeeze().tolist()))
        print('after squeeze: ', len(true.squeeze().tolist()))
        train_predictions.extend(pred.squeeze().tolist())
        train_true_labels.append(true.squeeze().tolist())

# rescale to original scale
predictions = np.array(predictions)
true_labels = np.array(true_labels)
predictions = predictions*(5000-105) + 105 # max=5000, min=105, minmax scaling
true_labels = true_labels*(5000-105) + 105

train_predictions = np.array(train_predictions)
train_true_labels = np.array(train_true_labels)
train_predictions = train_predictions*(5000-105) + 105 # max=5000, min=105, minmax scaling
train_true_labels = train_true_labels*(5000-105) + 105

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