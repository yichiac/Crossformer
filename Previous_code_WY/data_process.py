#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import csv


def readfile(file,ni,no,unistep,tscale):
    f = open(file,'r')
    nline=ni+no
    if not unistep:
        t = []
        nline+=1
    inset,outset = [[],[]]
    csvr = csv.reader(f)
    csvcount = 0
    datum=[]
    nanflag = False
    for row in csvr:
        csvcount += 1
        sl = np.array([float(x) for x in row])
        datum.append(sl)
        nanflag = nanflag or np.any(np.isnan(sl))
        if csvcount%nline==0:       #Read in each sample
            if nanflag==True:
                print(file+', line '+str(csvcount)+': Invalid sample.\n')
                nanflag = False
            else:
                if not unistep:
                    t.append((datum[0]/tscale).tolist()) 
                    del datum[0]
                inarray,outarray=np.split(np.array(datum),[ni])
                if ni==0:
                    inarray=np.zeros((1,outarray.shape[-1]))
                inset.append(inarray.transpose().tolist())
                outset.append(outarray.transpose().tolist())                                        
            datum.clear()
    f.close()
    if unistep:
        return {'inset':inset,'outset':outset}
    else:
        return {'t':t, 'inset':inset, 'outset':outset}

def importdata(path,sel, ni, no, aging, agestep, unistep,maxsample,tscale=1):
    if not aging:
        fileID=path+sel+'.csv'
        dataset=readfile(fileID,ni,no,unistep,tscale)
        if len(dataset['inset'])>maxsample:
            for key in dataset:
                dataset[key]=dataset[key][-maxsample:]
    else:
        dataset={'inset':[],'outset':[]}
        if not unistep:
            dataset['t']=[]
        for age in range(agestep):
            agestr=str(age)
            fileID=path+sel+agestr+'.csv'
            data=readfile(fileID,ni,no,unistep,tscale)
            if len(data['inset'])>maxsample:
                for key in data:
                    data[key]=data[key][0:maxsample]
            for i in range(len(data['inset'])):
                for j in range(len(data['inset'][i])):
                    data['inset'][i][j].append(age/(agestep-1))
            dataset['inset']+=data['inset']
            dataset['outset']+=data['outset']
            if not unistep:
                dataset['t']+=data['t']
    if unistep:
        return dataset
    else:
        for i in range(len(dataset['t'])):  #make sure t is strictly inceasing
            l=0
            while l<len(dataset['t'][i])-1:
                if dataset['t'][i][l]>=dataset['t'][i][l+1]:
                    for key in dataset:
                        del dataset[key][i][l+1]
                else:
                    l+=1
        return datapadding(dataset)
    
def datapadding(dataset):
    maxlength=max(len(a) for a in dataset['t'])
    for i in range(len(dataset['t'])):
        padlength=maxlength-len(dataset['t'][i])
        dataset['t'][i]=np.pad(dataset['t'][i],(0,padlength),constant_values=0).tolist()
        dataset['inset'][i]=np.pad(dataset['inset'][i],((0,padlength),(0,0)),constant_values=0).tolist()
        dataset['outset'][i]=np.pad(dataset['outset'][i],((0,padlength),(0,0)),constant_values=0).tolist()
    return dataset
    
def normalize(data,normports):
    for key in normports:
        index=int(key)
        if not normports[key]:
            ma,mi=[data[0][0][index],data[0][0][index]]
            for i in range(len(data)):
                for j in range(len(data[i])):
                    ma=max(ma,data[i][j][index])
                    mi=min(mi,data[i][j][index])
            normports[key]=[mi,ma]
        else:
            mi,ma=normports[key]
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j][index]=(data[i][j][index]-mi)/(ma-mi)
    return data,normports

def split(dataset, rate, shuffle,aging,agestep):
    if not aging:
        agestep=1
    trainset,validset=[{},{}]
    for key in dataset:
        trainset[key],validset[key]=[[],[]]
    Nsample=len(dataset['inset'])//agestep
    Ntrain=int(rate*Nsample)
    if shuffle:
        indices = np.random.permutation(Nsample)
    else:
        indices = np.arange(Nsample)
    for i in range(Ntrain):
        for age in range(agestep):
            for key in dataset:
                trainset[key].append(dataset[key][age*Nsample+indices[i]])
    for i in range(Ntrain,Nsample):
        for age in range(agestep):
            for key in dataset:
                validset[key].append(dataset[key][age*Nsample+indices[i]])
    return (trainset,validset)

# def importcsv_pv(dev, ni=1, no=1):
#     nline=ni+no
#     inset = []
#     outset =[]
#     fileID='./data/'+dev+'/'+dev+'.csv'
#     f = open(fileID,'r')     
#     datum=[]
#     csvr = csv.reader(f)
#     csvcount = 0
#     nanflag = False
#     for row in csvr:
#         csvcount += 1
#         sl = np.array([float(x) for x in row])
#         datum.append(sl)
#         nanflag = nanflag or np.any(np.isnan(sl))
#         if csvcount%nline==0: #last line in each i-v-w set
#             if nanflag==True:
#                 print(fileID, csvcount/nline)
#                 nanflag = False
#             else:
#                 nstep = np.size(datum[0])   
#                 for n in range(nstep):
#                     temp0=[]
#                     for i in range(ni):
#                         temp0.append(datum[i][n])
#                     inset.append(temp0)
#                     temp1=[]
#                     for i in range(no):
#                         temp1.append(datum[ni+i][n])
#                     outset.append(temp1)                                         
#             datum.clear()
#     f.close()
#     inset=np.reshape(inset,(-1,nstep,ni)).tolist()
#     outset=np.reshape(outset,(-1,nstep,no)).tolist()
#     return (inset, outset)


def AE(prediction, target):
    error=np.absolute(np.subtract(prediction,target))
    return (error)
    
def RMSE(prediction, target):
    error=np.power(np.subtract(prediction,target),2)
    RMSerror=np.sqrt(np.mean(error))
    return (RMSerror)

def MAE(prediction, target):
    error=np.absolute(np.subtract(prediction,target))
    meanerror=np.mean(error)
    return (meanerror)

def NRMSE(prediction, target):
    error=np.sum(np.power(np.subtract(prediction,target),2),axis=1)
    norm=np.sum(np.power(target,2),axis=1)
    NRMSerror=np.mean(np.sqrt(error/norm))
    return (NRMSerror)

def NRMSE_t(prediction, target):
    NRMSerror=[]
    for i in range(len(prediction)):
        error=(np.sum(np.power(np.subtract(prediction[i],target[i]),2),axis=1))
        norm=np.sum(np.power(target[i],2),axis=1)
        NRMSerror.append(np.sqrt(error/norm))
    return np.mean(NRMSerror)

def Russell(prediction, target):
    sump=np.sum(np.power(prediction,2),axis=1)
    sumt=np.sum(np.power(target,2),axis=1)
    sumpt=np.sum(np.multiply(prediction,target),axis=1)
    rme=(sump-sumt)/np.sqrt(sump*sumt)
    if rme>0: sign=1 
    else: sign=-1
    Mr=sign*np.log10(1+np.abs(rme))
    Pr=1/np.pi*np.arccos(sumpt/np.sqrt((sump*sumt)))
    Cr=np.sqrt(np.pi/4*((Mr**2)+(Pr**2)))
    return (np.mean(np.abs(Mr)),np.mean(np.abs(Pr)),np.mean(Cr))
    
def SG(prediction, target):
    sump=np.sum(np.power(prediction,2),axis=1)
    sumt=np.sum(np.power(target,2),axis=1)
    sumpt=np.sum(np.multiply(prediction,target),axis=1)
    Msg=np.sqrt(sump/sumt)-1
    Psg=1/np.pi*np.arccos(sumpt/np.sqrt(sump*sumt))
    Csg=np.sqrt((Msg**2)+(Psg**2))
    return (np.mean(np.abs(Msg)),np.mean(np.abs(Psg)),np.mean(Csg))
