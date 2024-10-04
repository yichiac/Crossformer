#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:14:44 2021
Top level test bench
@author: jiex2
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import data_process as data
import RNNcustom
import json
import os


with open("./config.json") as f:
    config=json.load(f)

train,load=[False,True]
#train,load=[True,False]

savedir='model/'+config['circuit']['dev']+'_best'
#savedir='model/'+config['circuit']['dev']+'_'+config['circuit']['sel']+'0'
if load:
    model,config,history=RNNcustom.load(savedir,'best')

#freshmodel='model/'+config['circuit']['trans_model']    #None if does not load fresh model
freshmodel=None
device = torch.device("cuda" if config['training']['gpu'] and (torch.cuda.is_available()) else "cpu")

datadir='data/'+config['circuit']['dev']+'/'+config['circuit']['sel']+'/'
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

trainset,validset= data.split(rawdata,rate=0.8,shuffle=False,aging=config['circuit']['aging'], \
                              agestep=config['circuit']['agestep'])
#validset,trainset=[rawdata,rawdata]

train_tensor,valid_tensor=[{},{}]
for key in trainset:
    train_tensor[key]=torch.tensor(trainset[key],device=device)
    valid_tensor[key]=torch.tensor(validset[key],device=device)
if not load:
    model=RNNcustom.model_sel(config)
model.to(device)
print(model)
NRMSE_list = []
#%%
if train:
    i = 0
    while i < 5:
        init = not load
        loss=RNNcustom.trainingloss(config['training']['loss'], config['training']['reg'], \
                                    config['training']['regweight'])
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-7, eps=0.0001)
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.3,patience=15,cooldown=15)
        train_dataset=RNNcustom.mydataset(train_tensor)
        dataloader = DataLoader(train_dataset, batch_size=config['training']['nbatch'],shuffle=True,drop_last=True)
        try:
            savedir='model/'+config['circuit']['dev']+'_'+config['circuit']['sel']+'_gru'+str(i)
            model,history=RNNcustom.train(model,config,config['training']['maxepoch'],dataloader,valid_tensor,optimizer,scheduler,loss,savedir,init,device,freshmodel)
            if 't' in valid_tensor:
                invalid={'inset':valid_tensor['inset'], 't':valid_tensor['t']}

            else:
                invalid=valid_tensor['inset']
            with torch.no_grad():
                outpred=model(invalid)
            #%%
            NRMSE_list.append(data.NRMSE(outpred.tolist(),valid_tensor['outset'].tolist()))
        except RuntimeError:
            i = i -1
            if os.path.exists(savedir):
                for f_sub0 in os.listdir(savedir):
                    subdir = os.path.join(savedir, f_sub0)
                    for f in os.listdir(subdir):
                        os.remove(os.path.join(subdir, f))
                    os.rmdir(subdir)
                os.rmdir(savedir)
        i = i + 1
#%%
plt.figure(0,figsize=(20,20))
x=np.linspace(0, len(history['loss_train'])-1, num=len(history['loss_train']))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.semilogy(x, history['loss_train'],label='Training loss')
ax1.semilogy(x, history['loss_valid'],label='Validation loss')
plt.legend(loc='upper right',fancybox=False, framealpha=1, facecolor='white', edgecolor='black')
ax2.plot(x, np.array(history['lr']),'g',label='Learning rate')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_ylim(0.0001,0.5)
ax2.set_ylabel('Learning rate')
ax1.legend(loc='upper left',fancybox=False, framealpha=1, facecolor='white', edgecolor='black')
ax2.legend(loc='upper right',fancybox=False, framealpha=1, facecolor='white', edgecolor='black')
plt.show()
#%%
# error=[]
# #model=RNNcustom.RNNmodel(config)

# #model.load('./RNNt_MSE_spectral2_64(log20)')
# output_tensor,hidden_tensor,temp = model(invalid_tensor,invalid_tensor[:,:,0:config['input_pre']])
# output=output_tensor.tolist()
# if norm:
#     output=np.multiply(output,(maxim-minim))+minim
#     outvalid=np.multiply(outvalid,(maxim-minim))+minim
# #    error.append(data.RMSE(output,outvalid))
# error.append(data.NRMSE(output,outvalid))
# #    error.append(data.Russell(output,outvalid))e
# #    error.append(data.SG(output,outvalid))

# errorage=np.zeros([2,agestep+1])
# for i in range(len(output)):
#     errorage[0,int(invalid_tensor[i,0,-1]*agestep)]+=data.NRMSE(output[i],outvalid[i])
#     errorage[1,int(invalid_tensor[i,0,-1]*agestep)]+=1

# errorage[0]=errorage[0]/errorage[1]
# #%%
# import matplotlib.pylab as pl
# ns=3

# hRNN=0.02e-9
# t=np.linspace(0, (len(invalid[0])-1)*hRNN, num=len(invalid[0]))*1e9-0.1
font = {
        'weight' : 'bold',
        'size'   : 10}
plt.rc('font', **font)
savefig=False
if 't' in valid_tensor:
    invalid={'inset':valid_tensor['inset'], 't':valid_tensor['t']}

else:
    invalid=valid_tensor['inset']
with torch.no_grad():
    outpred=model(invalid)
#%%
NRMSE=data.NRMSE(outpred.tolist(),valid_tensor['outset'].tolist())
print(outpred.shape)
print(NRMSE)
ns=17
if 't' in valid_tensor:
    t=valid_tensor['t'][ns]
else:
    t=np.linspace(0,config['circuit']['hRNN']*1e9*(config['circuit']['nstep']-1),config['circuit']['nstep'])
plt.figure(1,figsize=(10,10))
plt.clf()
for i in range(config['circuit']['ninput']):
    if str(i) in config['model']['norminputs']:
        m1,m2=config['model']['norminputs'][str(i)]
    else:
        m1,m2=[0,1]
    plt.subplot(config['circuit']['ninput']+config['circuit']['noutput'],1,i+1)
    plt.plot(t,valid_tensor['inset'][ns,:,i]*(m2-m1)+m1,linewidth=4)
    plt.ylabel('Vin%d [V]'%(i+1),fontweight='bold')
    plt.grid()
for i in range(config['circuit']['noutput']):
    if str(i) in config['model']['normoutputs']:
        m1,m2=config['model']['normoutputs'][str(i)]
    else:
        m1,m2=[0,1]
    plt.subplot(config['circuit']['ninput']+config['circuit']['noutput'],1,config['circuit']['ninput']+i+1)
    plt.plot(t,(valid_tensor['outset'][ns,:,i]*(m2-m1)+m1),linewidth=4,label='True',zorder=2)
    plt.scatter(t,(outpred[ns,:,i]*(m2-m1)+m1),s=50,c='darkorange',label='Pred',zorder=1)
    # plt.plot(t,valid_tensor['outset'][ns*config['circuit']['agestep']+config['circuit']['agestep']-1,:,i],linewidth=4,label='True (10)')
    # plt.plot(t,outpred[ns*config['circuit']['agestep']+config['circuit']['agestep']-1,:,i],'--',linewidth=4,label='Pred (10)')
    plt.ylabel('Vout%d [V]'%(i+1),fontweight='bold')
    plt.grid()
#plt.ylim([-0.01,1.3])
plt.legend(loc='best',fancybox=True, framealpha=1, facecolor='white', edgecolor='black',ncol=1,prop={'size': 20})

plt.xlabel('Time [ns]',fontweight='bold')
#%%
