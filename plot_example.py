from cross_models.cross_former import CrossformerCircuit
from utils.tools import EarlyStopping, adjust_learning_rate
import models.data_process as data
from data import CircuitDataModule

import torch
from torch import optim
import torch.nn as nn
from torch.nn import DataParallel

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import json
import time
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Modify KAN parameters via CLI")
    parser.add_argument("--n", type=int, default=5, help="Neurons for KAN")
    parser.add_argument("--grid", type=int, default=5, help="Grid size for KAN")
    parser.add_argument("--k", type=int, default=3, help="Value of k for KAN")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for KAN")
    return parser.parse_args()

cli_args = parse_args()


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

l = val_dataset['outset'].size(1)

t = np.linspace(0,config['circuit']['hRNN']*1e9*(config['circuit']['nstep']-1),config['circuit']['nstep'])
ns = 17
plt.figure(1, figsize=(10, 16))
plt.clf()

plt.subplots_adjust(hspace=0.5)

for i in range(config['circuit']['ninput']):
    if str(i) in config['model']['norminputs']:
        m1,m2=config['model']['norminputs'][str(i)]
    else:
        m1,m2=[0,1]
    plt.subplot(config['circuit']['ninput']+config['circuit']['noutput'],1,i+1)
    plt.plot(t, valid_tensor['inset'][ns,:,i].cpu().numpy()*(m2-m1)+m1,linewidth=4)
    plt.ylabel('Vin%d [V]'%(i+1),fontweight='bold')
    plt.grid()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

for i in range(config['circuit']['noutput']):
    if str(i) in config['model']['normoutputs']:
        m1,m2=config['model']['normoutputs'][str(i)]
    else:
        m1,m2=[0,1]
    plt.subplot(config['circuit']['ninput']+config['circuit']['noutput'],1,config['circuit']['ninput']+i+1)
    plt.plot(t,(valid_tensor['outset'][ns,:,i].cpu().numpy()*(m2-m1)+m1),linewidth=4,label='True',zorder=2)
    # plt.scatter(t, (outpred[ns,:,i].cpu().numpy()*(m2-m1)+m1),s=20,c='darkorange',label='Pred',zorder=1)
    # plt.plot(t,valid_tensor['outset'][ns*config['circuit']['agestep']+config['circuit']['agestep']-1,:,i],linewidth=4,label='True (10)')
    # plt.plot(t,outpred[ns*config['circuit']['agestep']+config['circuit']['agestep']-1,:,i],'--',linewidth=4,label='Pred (10)')
    plt.ylabel('Vout%d [V]'%(i+1),fontweight='bold')
    plt.grid()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.legend(bbox_to_anchor=(0.5, -0.4), loc='upper center',
           fancybox=True, framealpha=1, facecolor='white',
           edgecolor='black', ncol=2, prop={'size': 12})
plt.xlabel('Time [ns]',fontweight='bold')
plt.tight_layout()
plt.savefig(f'figs/example.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
# plt.savefig('figs/prediction_kan.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
# plt.savefig(f'figs/prediction_kan_n{cli_args.n}_grid{cli_args.grid}_k{cli_args.k}.png', dpi=300, bbox_inches='tight', pad_inches=0.3)


# plt.ylim([-0.01,1.3])
# plt.legend(loc='best',fancybox=True, framealpha=1, facecolor='white', edgecolor='black',ncol=1,prop={'size': 20})
