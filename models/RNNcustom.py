
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:07:24 2019

@author: jiex2
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import time
import rnn_v1
#import rnn_v2 #
from scipy.optimize import fsolve
import data_process as data
from operator import attrgetter
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
#from TorchDiffEqPack import odesolve_adjoint as odesolve
#from TorchDiffEqPack import odesolve


def model_sel(config,pv=False,corners=None):
    device = torch.device("cuda" if config['training']['gpu'] and (torch.cuda.is_available()) else "cpu")
    nstep=config['circuit']['nstep']
    if config['model']['outclip']=='sig':
        outclip=torch.nn.Sigmoid()
    elif config['model']['outclip']=='sig_cust':
        outclip=sig_cust(eps=1e-7)
    else:
        outclip=None
    variablet=not config['circuit']['unistep']
    if pv:
        return srnn(corners,config['circuit']['nvar'],outclip)
    if config['circuit']['aging']:
        if config['model']['modeltype']=='gru':
            if config['model']['ODEmethod'] is None:
                return gruaging_gru(config['circuit']['ninput'],config['model']['nhidden_age'], \
                                    config['model']['nhidden'],config['circuit']['noutput'],outclip)
        elif config['model']['modeltype']=='rnnosc3ode':
            return gruaging_rnnosc3ode(config['circuit']['ninput'],config['circuit']['noutput'], \
                                        config['model']['nhidden_in'],config['model']['nhidden_age'], \
                                        config['model']['nhidden_out'],config['model']['ODEmethod'], \
                                        variablet,config['circuit']['hRNN'],nstep,outclip,device)
        elif config['model']['modeltype']=='rnnode':
            if config['training']['reg']=='ldsp2':
                return gruaging_rnnode_lyapunov(config['circuit']['ninput'],config['model']['nhidden_age'], \
                           config['model']['nhidden'],config['circuit']['noutput'],config['model']['ODEmethod'], \
                           variablet,config['circuit']['hRNN'],nstep,outclip)
            else:
                return gruaging_rnnode(config['circuit']['ninput'],config['model']['nhidden_age'], \
                           config['model']['nhidden'],config['circuit']['noutput'],config['model']['ODEmethod'], \
                           variablet,config['circuit']['hRNN'],nstep,outclip)
        elif config['model']['modeltype']=='gruode':
            return gruaging_gruode(config['circuit']['ninput'],config['model']['nhidden_age'], \
                       config['model']['nhidden'],config['circuit']['noutput'],config['model']['ODEmethod'], \
                       variablet,config['circuit']['hRNN'],nstep,outclip)
        else:
            return gruaging_rnn(config['circuit']['ninput'],config['model']['nhidden_age'], \
                            config['model']['nhidden'],config['circuit']['noutput'],outclip)
    else:
        if config['model']['modeltype']=='gru':
            return gru(config['circuit']['ninput'],config['model']['nhidden'],config['circuit']['noutput'],outclip)
        elif config['model']['modeltype']=='transfer_rnnode':
            source_model_dir='model/'+config['circuit']['trans_model']
            source_model,_,_ = load(source_model_dir, 'best')
            return transfer_rnnode(config['circuit']['ninput'],config['model']['nhidden'],config['circuit']['noutput'], \
                          config['model']['ODEmethod'],source_model,variablet,config['circuit']['hRNN'],nstep,outclip)
        elif config['model']['modeltype']=='gruode':
            return gruode(config['circuit']['ninput'],config['model']['nhidden'],config['circuit']['noutput'], \
                          config['model']['ODEmethod'],variablet,config['circuit']['hRNN'],nstep,outclip)
        elif config['model']['modeltype']=='rnnosc3ode':
            return rnnosc3ode(config['circuit']['ninput'],config['circuit']['noutput'], \
                            config['model']['nhidden_in'],config['model']['nhidden_out'], \
                            config['model']['ODEmethod'],variablet,config['circuit']['hRNN'],nstep, \
                            outclip,device)
        elif config['model']['modeltype']=='rnnode':
            if config['training']['reg']=='ldsp2':
                return rnnode_lyapunov(config['circuit']['ninput'],config['model']['nhidden'],config['circuit']['noutput'], \
                          config['model']['ODEmethod'],variablet,config['circuit']['hRNN'],nstep,outclip)
            else:
                return rnnode(config['circuit']['ninput'],config['model']['nhidden'],config['circuit']['noutput'], \
                          config['model']['ODEmethod'],variablet,config['circuit']['hRNN'],nstep,outclip)
        else:
            if config['training']['reg']=='ldsp2':
                return rnn_lyapunov(config['circuit']['ninput'],config['model']['nhidden'],config['circuit']['noutput'],outclip)
            else:
                return rnn(config['circuit']['ninput'],config['model']['nhidden'],config['circuit']['noutput'],outclip)

class rnn(nn.Module):
    def __init__(self,ninput,nhidden,noutput,outclip=None):
        super().__init__()
        self.rnn = rnn_v1.RNN(input_size=ninput, hidden_size=nhidden, num_layers=1, bias=True, batch_first=True)
        self.linear=nn.Linear(in_features=nhidden,out_features=noutput,bias=True)
        self.outclip=outclip
        self.ninput=ninput
        self.nhidden=nhidden
    def forward(self, u,device=torch.device("cpu")):
        hidden=rnnzero(u[:,0,:self.ninput],self.rnn.weight_hh_l0.data,self.rnn.weight_ih_l0.data, \
                       self.rnn.bias_hh_l0.data+self.rnn.bias_ih_l0.data)
        hidden=torch.unsqueeze(hidden,0)
        rnnout,hidden=self.rnn(u[:,:,:self.ninput],hidden)
        output=self.linear(rnnout)
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)

class rnn_lyapunov(nn.Module):
    def __init__(self,ninput,nhidden,noutput,outclip=None):
        super().__init__()
        self.rnn = rnn_v1.RNN(input_size=ninput, hidden_size=nhidden, num_layers=1, bias=True, batch_first=True)
        self.linear=nn.Linear(in_features=nhidden,out_features=noutput,bias=True)
        self.outclip=outclip
        self.ninput=ninput
        self.nhidden=nhidden
        self.omega=nn.Parameter(torch.Tensor(nhidden))
        self.rnn.A=torch.matmul(self.rnn.weight_hh_l0,torch.diag(self.omega**2))
    def forward(self, u,device=torch.device("cpu")):
        if not self.training:
            self.rnn.A=torch.matmul(self.rnn.weight_hh_l0,torch.diag(self.omega**2))
        hidden=rnnzero(u[:,0,:self.ninput],self.rnn.A.data,self.rnn.weight_ih_l0.data, \
                       self.rnn.bias_hh_l0.data+self.rnn.bias_ih_l0.data)
        hidden=torch.unsqueeze(hidden,0)
        rnnout,hidden=self.rnn(u[:,:,:self.ninput],hidden)
        output=self.linear(rnnout)
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)
    def rho_update(self):
        self.rho=lds_projection(self.rnn.weight_hh_l0,torch.eye(self.nhidden), \
                           torch.diag(torch.abs(self.omega)),1e-9)
        self.rnn.A=torch.matmul(self.rnn.weight_hh_l0,torch.diag(self.omega**2))/self.rho
    def wr_update(self):
        self.rnn.weight_hh_l0.data/=self.rho.data

class gru(nn.Module):
    def __init__(self,ninput,nhidden,noutput,outclip=None):
        super().__init__()
        self.rnn = nn.GRU(input_size=ninput, hidden_size=nhidden, num_layers=1, bias=True, batch_first=True)
        self.linear=nn.Linear(in_features=nhidden,out_features=noutput,bias=True)
        self.outclip=outclip
        self.ninput=ninput
        self.nhidden=nhidden
    def forward(self, u,device=torch.device("cpu")):
        hidden=gruzero(u[:,0,:self.ninput],self.rnn.weight_hh_l0,self.rnn.weight_ih_l0,self.rnn.bias_ih_l0,self.rnn.bias_hh_l0)
        hidden=torch.unsqueeze(hidden,0)
        rnnout,hidden=self.rnn(u[:,:,:self.ninput],hidden)
        output=self.linear(rnnout)
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)

class rnnode(nn.Module):
    def __init__(self,ninput,nhidden,noutput,method,variablet=False,hRNN=0.02e-9,nstep=501,outclip=None, \
                 adjoint=''):
        super().__init__()
        self.rnn=odefunc(ninput,nhidden,'rnn')
        self.linear=nn.Linear(nhidden,noutput,bias=True)
        self.method=method
        self.variablet=variablet
        self.outclip=outclip
        self.hRNN=hRNN
        self.ninput=ninput
        self.nhidden=nhidden
        self.adjoint=adjoint
        if adjoint=='aca':     # obsolete
            self.options = {}
            self.options.update({'method': method})
            #‘euler’,’rk2’,'rk4',’rk12’,’rk23’,’dopri5’, ‘ode23s’,'sym12async','fixedstep_sym12async'
            self.options.update({'h': hRNN})
            self.options.update({'t0': 0.0})
            self.options.update({'rtol': 1e-7})
            self.options.update({'atol': 1e-8})
            self.options.update({'print_neval': False})
            self.options.update({'neval_max': 5e5})
            self.options.update({'interpolation_method':'cubic'})
            self.options.update({'regenerate_graph':False})
    def forward(self,inputdata,device=torch.device("cpu")):
        if self.variablet:
            outrnn=[]
            length=inputdata['t'].size(1)
            for i in range(len(inputdata['inset'])):
                l=len(inputdata['t'][i])    #unpadding
                while inputdata['t'][i,l-1]==0:
                    l-=1
                self.rnn.u=inputdata['inset'][i:i+1,0:l,:self.ninput]
                self.rnn.t=inputdata['t'][i,0:l]
                x0=rnnzero(self.rnn.u[:,0,:],self.rnn.rnncell.weight_hh.data,self.rnn.rnncell.weight_ih.data, \
                   self.rnn.rnncell.bias_ih.data+self.rnn.rnncell.bias_hh.data,self.rnn.tau.data)
                if self.adjoint=='aca':
                    self.options.update({'t1': self.rnn.t[-1]})
                    self.options.update({'t_eval':self.rnn.t})
                    #odeout = torch.transpose(odesolve(self.rnn, x0, options = self.options),0,1)[0]
                else:
                    odeout=torch.transpose(odeint(self.rnn,x0,self.rnn.t,rtol=1e-7,atol=1e-9, \
                                                  method=self.method),0,1)[0]
                outrnn.append(nn.functional.pad(odeout,(0,0,0,length-l),mode='constant', value=0)) #padding
            outrnn=torch.stack(outrnn,0)
        else:
            self.rnn.u=inputdata[:,:,:self.ninput]
            nstep=self.rnn.u.size(1)
            #self.rnn.t=torch.linspace(0., self.hRNN*(nstep-1)*1e9, nstep)
            self.rnn.t=torch.linspace(0., nstep-1, nstep)
            x0=rnnzero(self.rnn.u[:,0,:],self.rnn.rnncell.weight_hh.data,self.rnn.rnncell.weight_ih.data, \
                   self.rnn.rnncell.bias_ih.data+self.rnn.rnncell.bias_hh.data,self.rnn.tau.data)
            if self.adjoint=='aca':
                self.options.update({'t1': self.rnn.t[-1]})
                self.options.update({'t_eval':self.rnn.t})
                #outrnn = torch.transpose(odesolve(self.rnn, x0, options = self.options),0,1)
            else:
                outrnn = torch.transpose(odeint(self.rnn, x0, self.rnn.t[:nstep], rtol=1e-7, atol=1e-9, \
                                                method=self.method),0,1)
        output=self.linear(outrnn)
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)
    def rho_update(self):
        rho=lds_projection(self.rnn.rnncell.weight_hh.data,torch.diag(torch.exp(self.rnn.tau)), \
                           torch.eye(self.nhidden),1e-9)
        self.rnn.rnncell.weight_hh.data=self.rnn.rnncell.weight_hh.data/rho

class transfer_rnnode(nn.Module):
    def __init__(self,ninput,nhidden,noutput,method,trans_model,variablet=False,hRNN=0.02e-9,nstep=501,outclip=None, \
                 adjoint=''):
        super().__init__()
        self.rnn=odefunc(ninput,nhidden,'rnn')
        self.linear=nn.Linear(nhidden,noutput,bias=True)
        self.method=method
        self.variablet=variablet
        self.outclip=outclip
        self.hRNN=hRNN
        self.ninput=ninput
        self.nhidden=nhidden
        self.adjoint=adjoint
        self.init_param = torch.cat([torch.flatten(trans_model.rnn.rnncell.weight_ih), torch.flatten(trans_model.rnn.rnncell.weight_hh), trans_model.rnn.rnncell.bias_ih+trans_model.rnn.rnncell.bias_hh])
        self.train_param = torch.cat([torch.flatten(self.rnn.rnncell.weight_ih), torch.flatten(self.rnn.rnncell.weight_hh), self.rnn.rnncell.bias_ih+self.rnn.rnncell.bias_hh])
        if adjoint=='aca':     # obsolete
            self.options = {}
            self.options.update({'method': method})
            #‘euler’,’rk2’,'rk4',’rk12’,’rk23’,’dopri5’, ‘ode23s’,'sym12async','fixedstep_sym12async'
            self.options.update({'h': hRNN})
            self.options.update({'t0': 0.0})
            self.options.update({'rtol': 1e-7})
            self.options.update({'atol': 1e-8})
            self.options.update({'print_neval': False})
            self.options.update({'neval_max': 5e5})
            self.options.update({'interpolation_method':'cubic'})
            self.options.update({'regenerate_graph':False})
    def forward(self,inputdata,device=torch.device("cpu")):
        if self.variablet:
            outrnn=[]
            length=inputdata['t'].size(1)
            for i in range(len(inputdata['inset'])):
                l=len(inputdata['t'][i])    #unpadding
                while inputdata['t'][i,l-1]==0:
                    l-=1
                self.rnn.u=inputdata['inset'][i:i+1,0:l,:self.ninput]
                self.rnn.t=inputdata['t'][i,0:l]
                x0=rnnzero(self.rnn.u[:,0,:],self.rnn.rnncell.weight_hh.data,self.rnn.rnncell.weight_ih.data, \
                   self.rnn.rnncell.bias_ih.data+self.rnn.rnncell.bias_hh.data,self.rnn.tau.data)
                if self.adjoint=='aca':
                    self.options.update({'t1': self.rnn.t[-1]})
                    self.options.update({'t_eval':self.rnn.t})
                    #odeout = torch.transpose(odesolve(self.rnn, x0, options = self.options),0,1)[0]
                else:
                    odeout=torch.transpose(odeint(self.rnn,x0,self.rnn.t,rtol=1e-7,atol=1e-9, \
                                                  method=self.method),0,1)[0]
                outrnn.append(nn.functional.pad(odeout,(0,0,0,length-l),mode='constant', value=0)) #padding
            outrnn=torch.stack(outrnn,0)
        else:
            self.rnn.u=inputdata[:,:,:self.ninput]
            nstep=self.rnn.u.size(1)
            #self.rnn.t=torch.linspace(0., self.hRNN*(nstep-1)*1e9, nstep)
            self.rnn.t=torch.linspace(0., nstep-1, nstep)
            x0=rnnzero(self.rnn.u[:,0,:],self.rnn.rnncell.weight_hh.data,self.rnn.rnncell.weight_ih.data, \
                   self.rnn.rnncell.bias_ih.data+self.rnn.rnncell.bias_hh.data,self.rnn.tau.data)
            if self.adjoint=='aca':
                self.options.update({'t1': self.rnn.t[-1]})
                self.options.update({'t_eval':self.rnn.t})
                #outrnn = torch.transpose(odesolve(self.rnn, x0, options = self.options),0,1)
            else:
                outrnn = torch.transpose(odeint(self.rnn, x0, self.rnn.t[:nstep], rtol=1e-7, atol=1e-9, \
                                                method=self.method),0,1)
        output=self.linear(outrnn)
        self.train_param = torch.cat([torch.flatten(self.rnn.rnncell.weight_ih), torch.flatten(self.rnn.rnncell.weight_hh), self.rnn.rnncell.bias_ih+self.rnn.rnncell.bias_hh])
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)
    def rho_update(self):
        rho=lds_projection(self.rnn.rnncell.weight_hh.data,torch.diag(torch.exp(self.rnn.tau)), \
                           torch.eye(self.nhidden),1e-9)
        self.rnn.rnncell.weight_hh.data=self.rnn.rnncell.weight_hh.data/rho
    def dparamloss(self):
        criterion = nn.MSELoss()
        return 0.05*criterion(self.init_param, self.train_param)

class rnnode_lyapunov(nn.Module):
    def __init__(self,ninput,nhidden,noutput,method,variablet=False,hRNN=0.02e-9,nstep=501,outclip=None, \
                 adjoint=''):
        super().__init__()
        self.rnn=odefunc(ninput,nhidden,'rnn')
        self.linear=nn.Linear(nhidden,noutput,bias=True)
        self.method=method
        self.variablet=variablet
        self.outclip=outclip
        self.hRNN=hRNN
        self.ninput=ninput
        self.nhidden=nhidden
        self.adjoint=adjoint
        self.rnn.omega=nn.Parameter(torch.Tensor(nhidden))
    def forward(self,inputdata,device=torch.device("cpu")):
        if not self.training:
            self.rnn.rnncell.A=torch.matmul(self.rnn.rnncell.weight_hh,torch.diag(self.rnn.omega**2))
        if self.variablet:
            outrnn=[]
            length=inputdata['t'].size(1)
            for i in range(len(inputdata['inset'])):
                l=len(inputdata['t'][i])    #unpadding
                while inputdata['t'][i,l-1]==0:
                    l-=1
                self.rnn.u=inputdata['inset'][i:i+1,0:l,:self.ninput]
                self.rnn.t=inputdata['t'][i,0:l]
                x0=rnnzero(self.rnn.u[:,0,:],self.rnn.rnncell.A.data,self.rnn.rnncell.weight_ih.data, \
                   self.rnn.rnncell.bias_ih.data+self.rnn.rnncell.bias_hh.data,self.rnn.tau.data)
                odeout=torch.transpose(odeint(self.rnn,x0,self.rnn.t,rtol=1e-7,atol=1e-9, \
                                                  method=self.method),0,1)[0]
                outrnn.append(nn.functional.pad(odeout,(0,0,0,length-l),mode='constant', value=0)) #padding
            outrnn=torch.stack(outrnn,0)
        else:
            self.rnn.u=inputdata[:,:,:self.ninput]
            nstep=self.rnn.u.size(1)
            #self.rnn.t=torch.linspace(0., self.hRNN*(nstep-1)*1e9, nstep)
            self.rnn.t=torch.linspace(0., nstep-1, nstep)
            x0=rnnzero(self.rnn.u[:,0,:],self.rnn.rnncell.A.data,self.rnn.rnncell.weight_ih.data, \
                   self.rnn.rnncell.bias_ih.data+self.rnn.rnncell.bias_hh.data,self.rnn.tau.data)
            outrnn = torch.transpose(odeint(self.rnn, x0, self.rnn.t[:nstep], rtol=1e-7, atol=1e-9, \
                                                method=self.method),0,1)
        output=self.linear(outrnn)
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)
    def rho_update(self):
        self.rho=lds_projection(self.rnn.rnncell.weight_hh,torch.diag(torch.exp(self.rnn.tau)), \
                           torch.diag(torch.abs(self.rnn.omega)),1e-9)
        self.rnn.rnncell.A=torch.matmul(self.rnn.rnncell.weight_hh,torch.diag(self.rnn.omega**2))/self.rho
    def wr_update(self):
        self.rnn.rnncell.weight_hh.data/=self.rho.data

class gruode(nn.Module):
    def __init__(self,ninput,nhidden,noutput,method,variablet=False,hRNN=0.02e-9,nstep=501,outclip=None):
        super().__init__()
        self.rnn=odefunc(ninput,nhidden,'gru')
        self.linear=nn.Linear(nhidden,noutput,bias=True)
        self.method=method
        self.variablet=variablet
        self.outclip=outclip
        self.hRNN=hRNN
        self.ninput=ninput
        self.nhidden=nhidden
    def forward(self,inputdata,device=torch.device("cpu")):
        if self.variablet:
            outrnn=[]
            length=inputdata['t'].size(1)
            for i in range(len(inputdata['inset'])):
                l=len(inputdata['t'][i])    #unpadding
                while inputdata['t'][i,l-1]==0:
                    l-=1
                self.rnn.u=inputdata['inset'][i:i+1,0:l,:self.ninput]
                self.rnn.t=inputdata['t'][i,0:l]
                x0=gruzero(self.rnn.u[:,0,:],self.rnn.rnncell.weight_hh,self.rnn.rnncell.weight_ih, \
                           self.rnn.rnncell.bias_ih,self.rnn.rnncell.bias_hh,self.rnn.tau)
                odeout=torch.transpose(odeint(self.rnn,x0,self.rnn.t,rtol=1e-7,atol=1e-9, \
                                              method=self.method),0,1)[0]
                outrnn.append(nn.functional.pad(odeout,(0,0,0,length-l),mode='constant', value=0))
                #padding
            outrnn=torch.stack(outrnn,0)
        else:
            self.rnn.u=inputdata[:,:,:self.ninput]
            nstep=self.rnn.u.size(1)
            self.rnn.t=torch.linspace(0., nstep-1, nstep)
            x0=gruzero(self.rnn.u[:,0,:],self.rnn.rnncell.weight_hh,self.rnn.rnncell.weight_ih, \
                           self.rnn.rnncell.bias_ih,self.rnn.rnncell.bias_hh,self.rnn.tau)
            outrnn = torch.transpose(odeint(self.rnn, x0, self.rnn.t[:nstep], rtol=1e-7, atol=1e-9, \
                                            method=self.method),0,1)
        output=self.linear(outrnn)
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)

class gruaging_rnn(nn.Module):
    def __init__(self,ninput,nhidden_age,nhidden,noutput,outclip=None):
        super().__init__()
        self.rnnage = nn.GRU(input_size=ninput+1, hidden_size=nhidden_age, num_layers=1, bias=True, \
                             batch_first=True)
        self.linearage = nn.Linear(nhidden_age,ninput*nhidden+nhidden**2+nhidden)
        self.rnn = rnn_v1.RNN(input_size=ninput, hidden_size=nhidden, num_layers=1, bias=True, \
                              batch_first=True)
        self.linear = nn.Linear(in_features=nhidden,out_features=noutput,bias=True)
        self.outclip=outclip
        self.ninput=ninput
        self.nhidden=nhidden
    def forward(self, u, device=torch.device("cpu")):
        unoage=u[:,:,0:self.ninput]
        unoage0=unoage[:,0,:]
        hiddenage=torch.zeros((1,u.size(0),self.rnnage.hidden_size),device=device)
        rnnageout, hiddenage = self.rnnage(u[:,:,-self.ninput-1:], hiddenage)
        self.dparams = self.linearage(hiddenage[0])
        outrnn = []
        for i in range(len(self.dparams)):
            self.rnn.dparams=self.dparams[i]
            self.A=self.rnn.all_weights()[0][1].data
            self.B=self.rnn.all_weights()[0][0].data
            self.C=self.rnn.all_weights()[0][2].data+self.rnn.all_weights()[0][3].data
            hidden=rnnzero(unoage0[i:i+1,:],self.A,self.B,self.C)
            hidden=torch.unsqueeze(hidden,0)
            out,hidden=self.rnn(unoage[i:i+1],hidden)
            outrnn.append(out[0])
        outrnn = torch.stack(outrnn,0)
        output=self.linear(outrnn)
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)
    def dparamloss(self):
        return 0.1*torch.mean(self.dparams**2)

class gruaging_gru(nn.Module):
    def __init__(self,ninput,nhidden_age,nhidden,noutput,outclip=None):
        super().__init__()
        self.rnnage = nn.GRU(input_size=ninput+1,hidden_size=nhidden_age,num_layers=1,bias=True, \
                             batch_first=True)
        self.linearage = nn.Linear(nhidden_age,ninput*nhidden+nhidden**2+nhidden*2)
        self.rnn = rnn_v1.GRU(input_size=ninput,hidden_size=nhidden,num_layers=1,bias=True, \
                              batch_first=True)
        self.linear = nn.Linear(in_features=nhidden,out_features=noutput,bias=True)
        self.outclip=outclip
        self.ninput=ninput
        self.nhidden=nhidden
    def forward(self, u, device=torch.device("cpu")):
        unoage=u[:,:,0:self.ninput]
        unoage0=unoage[:,0,:]
        hiddenage=torch.zeros((1,u.size(0),self.rnnage.hidden_size),device=device)
        rnnageout, hiddenage = self.rnnage(u[:,:,-self.ninput-1:], hiddenage)
        self.dparams = self.linearage(hiddenage[0])
        outrnn = []
        for i in range(len(self.dparams)):
            self.rnn.dparams=self.dparams[i]
            self.A=self.rnn.all_weights()[0][1].data
            self.B=self.rnn.all_weights()[0][0].data
            self.C=self.rnn.all_weights()[0][2].data
            self.D=self.rnn.all_weights()[0][3].data
            hidden=gruzero(unoage0[i:i+1,:],self.A,self.B,self.C,self.D)
            self.x0=hidden
            hidden=torch.unsqueeze(hidden,0)
            out,hidden=self.rnn(unoage[i:i+1],hidden)
            outrnn.append(out[0])
        outrnn = torch.stack(outrnn,0)
        output=self.linear(outrnn)
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)
    def dparamloss(self):
        return 1*torch.mean(self.dparams**2)

class gruaging_rnnode(nn.Module):

    def __init__(self,ninput,nhidden_age,nhidden,noutput,method,variablet=False,hRNN=0.02e-9,nstep=501, \
                 outclip=None):
        super().__init__()
        if variablet:
            raise RuntimeError('Variable time step is not supported for aging model.')
        self.rnnage = nn.GRU(input_size=ninput+1, hidden_size=nhidden_age, num_layers=1, bias=True, \
                             batch_first=True)
        self.linearage = nn.Linear(nhidden_age,ninput*nhidden+nhidden**2+nhidden)
        self.rnn=odefunc(ninput,nhidden,'rnn')
        self.linear=nn.Linear(nhidden,noutput,bias=True)
        self.method=method
        self.variablet=variablet
        self.outclip=outclip
        self.hRNN=hRNN
        self.ninput=ninput
        self.nhidden=nhidden
    def forward(self,u,device=torch.device("cpu")):
        nstep=u.size(1)
        self.rnn.t=torch.linspace(0., nstep-1, nstep)
        unoage=u[:,:,0:self.ninput]
        unoage0=unoage[:,0,:]
        hiddenage=torch.zeros((1,u.size(0),self.rnnage.hidden_size),device=device)
        rnnageout, hiddenage = self.rnnage(u[:,:,-self.ninput-1:], hiddenage)
        self.dparams = self.linearage(hiddenage[0])
        outrnn = []
        for i in range(len(self.dparams)):
            self.rnn.param_update(self.dparams[i])
            x0=rnnzero(unoage0[i:i+1,:],self.rnn.rnncell.A,self.rnn.rnncell.B,self.rnn.rnncell.C, \
                       self.rnn.tau)
            self.rnn.u=unoage[i:i+1]
            odeout=torch.transpose(odeint(self.rnn,x0,self.rnn.t[:nstep],rtol=1e-7,atol=1e-9, \
                                          method=self.method),0,1)[0]
            outrnn.append(odeout)
        outrnn = torch.stack(outrnn,0)
        output=self.linear(outrnn)
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)
    def dparamloss(self):
        return 0.1*torch.mean(self.dparams**2)



class gruaging_rnnode_lyapunov(nn.Module):
    def __init__(self,ninput,nhidden_age,nhidden,noutput,method,variablet=False,hRNN=0.02e-9,nstep=501, \
                 outclip=None):
        super().__init__()
        if variablet:
            raise RuntimeError('Variable time step is not supported for aging model.')
        self.rnnage = nn.GRU(input_size=ninput+1, hidden_size=nhidden_age, num_layers=1, bias=True, \
                             batch_first=True)
        self.linearage = nn.Linear(nhidden_age,ninput*nhidden+nhidden**2+nhidden)
        self.rnn=odefunc(ninput,nhidden,'rnn',True)
        self.linear=nn.Linear(nhidden,noutput,bias=True)
        self.method=method
        self.variablet=variablet
        self.outclip=outclip
        self.hRNN=hRNN
        self.ninput=ninput
        self.nhidden=nhidden
        self.rnn.omega=nn.Parameter(torch.Tensor(nhidden),requires_grad=False)
    def forward(self,u,device=torch.device("cpu")):
        nstep=u.size(1)
        self.rnn.t=torch.linspace(0., nstep-1, nstep)
        unoage=u[:,:,0:self.ninput]
        unoage0=unoage[:,0,:]
        hiddenage=torch.zeros((1,u.size(0),self.rnnage.hidden_size),device=device)
        rnnageout, hiddenage = self.rnnage(u[:,:,-self.ninput-1:], hiddenage)
        self.dparams = self.linearage(hiddenage[0])
        outrnn = []
        for i in range(len(self.dparams)):
            self.rnn.param_update(self.dparams[i],lds=True)
            x0=rnnzero(unoage0[i:i+1,:],self.rnn.rnncell.A,self.rnn.rnncell.B,self.rnn.rnncell.C, \
                       self.rnn.tau)
            self.rnn.u=unoage[i:i+1]
            odeout=torch.transpose(odeint(self.rnn,x0,self.rnn.t[:nstep],rtol=1e-7,atol=1e-9, \
                                          method=self.method),0,1)[0]
            outrnn.append(odeout)
        outrnn = torch.stack(outrnn,0)
        output=self.linear(outrnn)
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)
    def dparamloss(self):
        return 0.1*torch.mean(self.dparams**2)

class gruaging_gruode(nn.Module):
    def __init__(self,ninput,nhidden_age,nhidden,noutput,method,variablet=False,hRNN=0.02e-9,nstep=501, \
                 outclip=None):
        super().__init__()
        if variablet:
            raise RuntimeError('Variable time step is not supported for aging model.')
        self.rnnage = nn.GRU(input_size=ninput+1, hidden_size=nhidden_age, num_layers=1, bias=True, \
                             batch_first=True)
        self.linearage = nn.Linear(nhidden_age,ninput*nhidden+nhidden**2+nhidden*2)
        self.rnn=odefunc(ninput,nhidden,'gru')
        self.linear=nn.Linear(nhidden,noutput,bias=True)
        self.method=method
        self.variablet=variablet
        self.outclip=outclip
        self.ninput=ninput
        self.nhidden=nhidden
    def forward(self,u,device=torch.device("cpu")):
        nstep=u.size(1)
        self.rnn.t=torch.linspace(0., nstep-1, nstep)
        unoage=u[:,:,0:self.ninput]
        unoage0=unoage[:,0,:]
        hiddenage=torch.zeros((1,u.size(0),self.rnnage.hidden_size),device=device)
        rnnageout, hiddenage = self.rnnage(u[:,:,-self.ninput-1:], hiddenage)
        self.dparams = self.linearage(hiddenage[0])
        outrnn = []
        for i in range(len(self.dparams)):
            self.rnn.param_update(self.dparams[i])
            self.x0=gruzero(unoage0[i:i+1,:],self.rnn.rnncell.A,self.rnn.rnncell.B, \
                       self.rnn.rnncell.C,self.rnn.rnncell.D,self.rnn.tau)
            self.rnn.u=unoage[i:i+1]
            odeout=torch.transpose(odeint(self.rnn,self.x0,self.rnn.t[:nstep],rtol=1e-7,atol=1e-9, \
                                          method=self.method),0,1)[0]
            outrnn.append(odeout)
        outrnn = torch.stack(outrnn,0)
        output=self.linear(outrnn)
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)
    def dparamloss(self):
        return 0.01*torch.mean(self.dparams**2)

def rnnzero(u0,A,B,C,tau=None,omega=None,device=torch.device("cpu")):
    u0,A,B,C=[u0.cpu().numpy(),A.data.cpu().numpy(),B.data.cpu().numpy(),C.data.cpu().numpy()]
    nbatch,nhidden=[u0.shape[0],A.shape[1]]
    if tau is None:
        tau=np.zeros(nhidden)
    else:
        tau=tau.data.cpu().numpy()
    if omega is None:
        omega=np.eye(nhidden)
    else:
        omega=omega.data.cpu().numpy()
    def func(x):
        return np.matmul(omega,np.tanh(np.matmul(A,x)+np.matmul(B,u0[i])+C))-x/np.exp(tau)
    x0=np.zeros([nbatch,nhidden])
    xinit=np.zeros(nhidden)
    for i in range(nbatch):
        x0[i,:]=fsolve(func,xinit)
    if np.isnan(x0).any():
        x0=np.zeros(x0.shape)
        print('Cannot find equilibrium. Initial state set to zero.')
    return torch.tensor(x0,dtype=torch.float32,device=device)

def gruzero(u0,A,B,C,D,tau=None,device=torch.device("cpu")):
    u0,A,B,C,D=[u0.cpu().numpy(),A.data.cpu().numpy(),B.data.cpu().numpy(),C.data.cpu().numpy(), \
                D.data.cpu().numpy()]
    nbatch,nhidden=[u0.shape[0],A.shape[1]]
    Ax,Bx,Cx,Dx=[A[-nhidden:],B[-nhidden:],C[-nhidden:],D[-nhidden:]]
    Ar,Br,Cr=[A[:nhidden],B[:nhidden],C[:nhidden]+D[:nhidden]]
    Az,Bz,Cz=[A[nhidden:2*nhidden],B[nhidden:2*nhidden],C[nhidden:2*nhidden]+D[nhidden:2*nhidden]]
    if tau is None:
        tau=np.zeros(nhidden)
    else:
        tau=tau.data.cpu().numpy()
    def func(x):
        r=1/(1+np.exp(-(np.matmul(Ar,x)+np.matmul(Br,u0[i])+Cr)))
        z=1/(1+np.exp(-(np.matmul(Az,x)+np.matmul(Bz,u0[i])+Cz)))
        return (1-z)*np.tanh(r*(np.matmul(Ax,x)+Dx)+np.matmul(Bx,u0[i])+Cx)+z*x-x/np.exp(tau)
    x0=np.zeros([nbatch,nhidden])
    xinit=np.zeros(nhidden)
    for i in range(nbatch):
        x0[i,:]=fsolve(func,xinit)
    if np.isnan(x0).any():
        x0=np.zeros(x0.shape)
        print('Cannot find equilibrium. Initial state set to zero.')
    return torch.tensor(x0,dtype=torch.float32,device=device)

class odefunc(nn.Module):
    def __init__(self,ninput,nhidden,odetype='rnn',bias=True):
        super().__init__()
        self.input_size=ninput
        self.hidden_size=nhidden
        self.odetype=odetype
        if odetype=='gru':
            self.rnncell = rnn_v1.GRUCell(input_size=ninput, hidden_size=nhidden,bias=True)
        else:
            self.rnncell = rnn_v1.RNNCell(input_size=ninput, hidden_size=nhidden,bias=bias)
        self.tau=nn.Parameter(torch.Tensor(nhidden))
    def forward(self,t,x):
        ut=self.lin_interp(t.cpu().item())
        return self.rnncell(ut,x)-x/torch.exp(self.tau)
    def lin_interp(self,t):
        if t<=self.t[0]:
            return self.u[:,0,:].float()
        elif t>=self.t[-1]:
            return self.u[:,-1,:].float()
        else:
            ind=torch.bucketize(t,self.t)
            u_low,u_high,t_low,t_high=[self.u[:,ind-1,:],self.u[:,ind,:],self.t[ind-1],self.t[ind]]
            return (u_low+(u_high-u_low)*(t-t_low)/(t_high-t_low)).float()
    def param_update(self,dparams,lds=False):
        if self.odetype=='gru':
            Dwu,Dwr,Dbu,Dbh= torch.split(dparams,[self.input_size*self.hidden_size,self.hidden_size**2, \
                                              self.hidden_size,self.hidden_size])
            self.Dwu=Dwu.reshape(self.hidden_size,self.input_size)
            self.Dwr=Dwr.reshape(self.hidden_size,self.hidden_size)
            self.Dbu=Dbu.reshape(self.hidden_size)
            self.Dbh=Dbh.reshape(self.hidden_size)
            self.rnncell.B=self.rnncell.weight_ih+torch.cat((torch.zeros(self.Dwu.size()),torch.zeros(self.Dwu.size()),self.Dwu),0)
            self.rnncell.A=self.rnncell.weight_hh+torch.cat((torch.zeros(self.Dwr.size()),torch.zeros(self.Dwr.size()),self.Dwr),0)
            self.rnncell.C=self.rnncell.bias_ih+torch.cat((torch.zeros(self.Dbu.size()),torch.zeros(self.Dbu.size()),self.Dbu),0)
            self.rnncell.D=self.rnncell.bias_hh+torch.cat((torch.zeros(self.Dbh.size()),torch.zeros(self.Dbh.size()),self.Dbh),0)
        else:
            Dwu,Dwr,Dbu= torch.split(dparams,[self.input_size*self.hidden_size,self.hidden_size**2, \
                                              self.hidden_size])
            self.Dwu=Dwu.reshape(self.hidden_size,self.input_size)
            self.Dwr=Dwr.reshape(self.hidden_size,self.hidden_size)
            self.Dbu=Dbu.reshape(self.hidden_size)
            self.rnncell.B=self.rnncell.weight_ih+self.Dwu
            if lds:
                self.rnncell.A=self.rnncell.weight_hh+self.Dwr
                self.rnncell.A=torch.matmul(self.rnncell.A,torch.diag(self.omega**2)) \
                    /lds_projection(self.rnncell.A,torch.diag(torch.exp(self.tau)), \
                           torch.diag(torch.abs(self.omega)),1e-9)
            else:
                self.rnncell.A=self.rnncell.weight_hh+self.Dwr
            self.rnncell.C=self.rnncell.bias_ih+self.rnncell.bias_hh+self.Dbu



class rnnosc3ode(nn.Module):
    def __init__(self,ninput,noutput,nhidden_in,nhidden_out,method,variablet=False,hRNN=0.02e-9, \
                 nstep=501,outclip=None,device=torch.device('cpu')):
        super().__init__()
        self.odefunc=oscodefunc(ninput,nhidden_in,device)
        self.nonlinear=nn.Linear(3,nhidden_out,bias=True)
        self.linear=nn.Linear(nhidden_out,noutput,bias=True)
        self.method=method
        self.variablet=variablet
        self.outclip=outclip
        self.hRNN=hRNN
        self.ninput=ninput
    def forward(self,inputdata,device=torch.device("cpu"),nstep=None):
        if self.variablet:
            outrnn=[]
            length=inputdata['t'].size(1)
            for i in range(len(inputdata['inset'])):
                l=len(inputdata['t'][i])    #unpadding
                while inputdata['t'][i,l-1]==0:
                    l-=1
                self.odefunc.u=inputdata['inset'][i:i+1,0:l,:self.ninput]
                self.odefunc.t=inputdata['t'][i,0:l]
                x0=torch.stack([self.odefunc.x0],dim=0)
                odeout=torch.transpose(odeint(self.odefunc, x0, self.odefunc.t, rtol=1e-7, atol=1e-9, \
                                                     method=self.method),0,1)[0]
                outrnn.append(nn.functional.pad(odeout,(0,0,0,length-l),mode='constant', value=0))
            outrnn=torch.stack(outrnn,0)
        else:
            if nstep is not None:
                self.odefunc.t=torch.linspace(0., self.hRNN*(nstep-1)*1e9, nstep)
            else:
                self.odefunc.t=torch.linspace(0., self.hRNN*(inputdata.size(1)-1)*1e9, inputdata.size(1))
            self.odefunc.u=inputdata[:,:,:self.ninput]
            x0=torch.stack(len(inputdata)*[self.odefunc.x0],dim=0)
            outrnn = torch.transpose(odeint(self.odefunc, x0, self.odefunc.t, rtol=1e-7, atol=1e-9, \
                                            method=self.method),0,1)
        outnonlinear=torch.sigmoid(self.nonlinear(outrnn))
        output=self.linear(outnonlinear)
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)
    def dparamloss(self):
        return 0.005*torch.mean(self.odefunc.dparam_total**2)

class gruaging_rnnosc3ode(nn.Module):
    def __init__(self,ninput,noutput,nhidden_in,nhidden_age,nhidden_out,method,variablet=False, \
                 hRNN=0.02e-9,nstep=501,outclip=None,device=torch.device('cpu')):
        super().__init__()
        self.odefunc=oscodefunc(ninput,nhidden_in,device)
        self.rnnage = nn.GRU(input_size=1+ninput,hidden_size=nhidden_age,num_layers=1,bias=True, \
                             batch_first=True)
        self.linearage = nn.Linear(nhidden_age,1)
        self.nonlinear=nn.Linear(3,nhidden_out,bias=True)
        self.linear=nn.Linear(nhidden_out,noutput,bias=True)
        self.method=method
        self.variablet=variablet
        self.outclip=outclip
        self.hRNN=hRNN
        self.ninput=ninput
    def forward(self,inputdata,device=torch.device("cpu"),nstep=None):
        if self.variablet:
            raise RuntimeError('Model only accepts uniform time step data.')
        else:
            if nstep is None:
                self.odefunc.t=torch.linspace(0., self.hRNN*(inputdata.size(1)-1)*1e9, \
                                              inputdata.size(1),device=device)
            else:
                self.odefunc.t=torch.linspace(0., self.hRNN*(nstep-1)*1e9, nstep,device=device)
            self.uage=inputdata[:,:,-self.ninput-1:]
            hiddenage=torch.zeros((1,self.uage.size(0),self.rnnage.hidden_size),device=device)
            rnnageout, hiddenage = self.rnnage(self.uage, hiddenage)
            self.dparams = self.linearage(hiddenage[0])
            outrnn = []
            for i in range(len(self.dparams)):
                self.odefunc.u=inputdata[i:i+1,:,0:self.ninput]
                self.odefunc.dparam=self.dparams[i:i+1]
                x0=torch.stack(1*[self.odefunc.x0],dim=0)
                out = torch.transpose(odeint(self.odefunc, x0, self.odefunc.t, rtol=1e-7, atol=1e-9, \
                                                    method=self.method),0,1)
                outrnn.append(out[0])
            outrnn = torch.stack(outrnn,0)
        outnonlinear=torch.sigmoid(self.nonlinear(outrnn))
        output=self.linear(outnonlinear)
        if self.outclip is None:
            return output
        else:
            return self.outclip(output)
    def dparamloss(self):
        return 0.01*torch.mean(self.dparams**2)

class oscodefunc(nn.Module):
    def __init__(self,ninput,nhidden_in,device):
        super().__init__()
        self.ninput=ninput
        if ninput>0:
            self.nonlinearinput = nn.Linear(ninput,nhidden_in,True)
            self.nonlinearact = nn.Sigmoid()
            self.linearinput = nn.Linear(nhidden_in,1,True)
        self.tau=nn.Parameter(torch.Tensor(1))
        self.w=torch.tensor([-20.,10.],device=device)
        self.x0=torch.tensor([0.4736,0.8745,1.8497],device=device)
        self.device=device
    def lin_interp(self,t):
        if t<=self.t[0]:
            return self.u[:,0,:].float()
        elif t>=self.t[-1]:
            return self.u[:,-1,:].float()
        else:
            ind=torch.bucketize(t,self.t).data
            u_low,u_high,t_low,t_high=[self.u[:,ind-1,:],self.u[:,ind,:],self.t[ind-1],self.t[ind]]
            return (u_low+(u_high-u_low)*(t-t_low)/(t_high-t_low)).float()
    def forward(self,t,x):
        if hasattr(self,'dparam') and self.ninput>0:
            ut=self.lin_interp(t)
            self.dparam_total=self.dparam+self.linearinput(self.nonlinearact(self.nonlinearinput(ut)))
        elif hasattr(self,'dparam'):
            self.dparam_total=self.dparam
        elif self.ninput>0:
            ut=self.lin_interp(t)
            self.dparam_total=self.linearinput(self.nonlinearact(self.nonlinearinput(ut)))
        else:
            self.dparam_total=0.

#        print(self.dparam.data,self.linearinput(self.nonlinearact(self.nonlinearinput(ut))).data)
        dx1=-x[:,0]+torch.tanh(x[:,1])
        dx2=-x[:,1]+torch.tanh(x[:,2])
        dx3=-x[:,2]+self.w[0]*torch.tanh(x[:,0])+self.w[1]*torch.tanh(x[:,1])
        return torch.stack((dx1,dx2,dx3),dim=1)/torch.exp(self.tau+self.dparam_total)

class srnn(nn.Module):
    def __init__(self,model_corners,nvar,outclip=None):
        super(srnn,self).__init__()
        self.tt=model_corners[0]
        self.corners=model_corners[1:]
        self.pvparams=nn.Parameter(torch.tril(torch.Tensor(nvar,nvar)))
        self.outclip=outclip
        for p in self.parameters():
            nn.init.uniform_(p.data,0,1)
#     def equil(self,u0):
#         u0=u0.cpu().numpy()
#         xinit=np.zeros([self.config['hidden_size']])
#         x0=np.zeros([1,len(u0),self.config['hidden_size']])
#         A=self.A.detach().numpy()
#         B=self.B.detach().numpy()
#         C=self.C[:,0].detach().numpy()
#         for i in range(len(u0)):
#             x0[0,i,:]=fsolve(lambda x:(np.tanh(np.matmul(A,x)+np.matmul(B,u0[i])+C)-x),xinit)
# #        x0=x0.reshape([1,len(u0),self.config['hidden_size']])
#         x0=torch.tensor(x0,dtype=torch.float32,device=device)
#         self.x0=x0
#         return x0
    def forward(self, u,model_tt,model_corners):
        self.param_sample_corr(model_tt,model_corners)
        hidden=rnnzero(u[:,0,:],self.A,self.B,self.C)
        temp=[]
        for i in range(u.size(1)):
            hidden = torch.tanh(torch.matmul(self.A,hidden)+torch.matmul(self.B,u[:,i,:].reshape(u.size(0),u.size(-1),1))+self.C)
            temp.append(hidden)
        outseq=torch.stack(temp,0).transpose(0,1)

        if self.outclip:
            output = torch.squeeze(self.outclip(torch.matmul(self.D,outseq)+self.E),-1)
        else:
            output = torch.squeeze(torch.matmul(self.D,outseq)+self.E,-1)
        return output

    def param_sample_corr(self,model_tt,model_corners):
        randr0=torch.normal(torch.zeros(self.config['ncor']),torch.ones(self.config['ncor']))
        randr=torch.matmul(torch.tril(self.pvparams),randr0)
#        randp=torch.rand(1)*2*np.pi
        if len(model_corners)/2==1:
            corner1=randr
            model_cornerA=model_corners[0].rnn.weight_hh_l0.data if corner1[0]>0 else model_corners[1].rnn.weight_hh_l0.data
            model_cornerB=model_corners[0].rnn.weight_ih_l0.data if corner1[1]>0 else model_corners[1].rnn.weight_ih_l0.data
            model_cornerC=(model_corners[0].rnn.bias_hh_l0.data+model_corners[0].rnn.bias_ih_l0.data) if corner1[2]>0 else (model_corners[1].rnn.bias_hh_l0.data+model_corners[1].rnn.bias_ih_l0.data)
            model_cornerD=model_corners[0].linear.weight.data if corner1[3]>0 else model_corners[1].linear.weight.data
            model_cornerE=model_corners[0].linear.bias.data if corner1[4]>0 else model_corners[1].linear.bias.data

            self.A=model_tt.rnn.weight_hh_l0.data+corner1[0]*(model_cornerA-model_tt.rnn.weight_hh_l0.data)
            self.B=model_tt.rnn.weight_ih_l0.data+corner1[1]*(model_cornerB-model_tt.rnn.weight_ih_l0.data)
            self.C=model_tt.rnn.bias_hh_l0.data+model_tt.rnn.bias_ih_l0.data+corner1[2]*(model_cornerC-model_tt.rnn.bias_hh_l0.data-model_tt.rnn.bias_ih_l0.data)
            self.C=self.C.reshape(self.config['hidden_size'],1)
            self.D=model_tt.linear.weight.data+corner1[3]*(model_cornerD-model_tt.linear.weight.data)
            self.E=model_tt.linear.bias.data+corner1[4]*(model_cornerE-model_tt.linear.bias.data)

class sig_cust(nn.Module):
    def __init__(self,eps=1e-7):
        super().__init__()
        self.eps=eps
    def forward(self,input):
        return torch.sigmoid(input)*(1-2*self.eps)+self.eps

class trainingloss(nn.Module):
    def __init__(self,losstype,lds=None,ldsweight=0,device=torch.device('cpu')):
        super().__init__()
        if losstype=='MSE':
            self.loss=nn.MSELoss()
        elif losstype=='CE':
            self.loss=loss_crossentropy()
        elif losstype=='SG':
            self.loss=loss_SG()
        else:
            raise ValueError('Invalid loss function.')
        if lds=='lds':
            self.ldsreg=check_lds()
        self.ldsweight=ldsweight
        self.device=device
    def forward(self,pred,targ,model,otherloss=None):
        lossvalues=[self.loss(pred,targ)]
        if hasattr(self,'ldsreg'):
            lossvalues.append(self.ldsreg(model)*self.ldsweight)
        if otherloss:
            lossvalues.append(otherloss)
        return (torch.sum(torch.stack(lossvalues)), lossvalues)

def train(model,config,nepoch,traindataloader,validset,optimizer,lrscheduler,loss,savedir,init=True, \
          device=torch.device("cpu"),freshmodel=None):
    if os.path.isdir(savedir):
        raise RuntimeError('Save directory exist!')
    history={'loss_train':[],
      'loss_valid':[],
      'lr':[]}
    if init:
        for p in model.parameters():
            nn.init.uniform_(p.data,-1,1)
        if hasattr(model.rnn,'tau'):
            model.rnn.tau.data=model.rnn.tau.data*0 #new added
        if hasattr(model,'rnnage'):
            for p in model.rnnage.parameters():
                p.data=p.data/100
            for p in model.linearage.parameters():
                p.data=p.data/100
        if hasattr(model,'odefunc'):
            if hasattr(model.odefunc,'nonlinearinput'):
                for p in model.odefunc.nonlinearinput.parameters():
                    p.data=p.data/10
                for p in model.odefunc.linearinput.parameters():
                    p.data=p.data/10
            if hasattr(model.odefunc,'tau'):
                model.odefunc.tau.data=model.odefunc.tau.data-3
    if freshmodel is not None:
        try:
            model=load_rnn2ode(freshmodel,model,sel='last')
            print('Load rnnode from rnn.')
        except:
            model=load_partial(freshmodel,model,sel='best',grad=True)
            print('Load part of the parameters.')
    if not config['circuit']['aging'] and config['training']['reg'].startswith('ldsp'):
        model.rho_update()
    outvalid=validset['outset'].to(device)
    if 't' in validset:
        invalid={'inset':validset['inset'].to(device), 't':validset['t'].to(device)}
    else: invalid=validset['inset'].to(device)
    start_time=time.time()
#    with torch.cuda.device(device if device.type=='cuda' else None):
    minloss=100000000
    for epoch in range(nepoch):
        if epoch<15:
            l=outvalid.size(1)
        elif epoch<30:
            l=outvalid.size(1)
        else:
            l=outvalid.size(1)
        for sample in traindataloader:
            outsample=sample['outset'][:,0:l,:].to(device)
            if 't' in sample:
                insample={'inset':sample['inset'][:,0:l,:].to(device), 't':sample['t'][:,0:l].to(device)}
            else:
                insample=sample['inset'][:,0:l,:].to(device)
            optimizer.zero_grad()
            outpred=model(insample,device)
            losstotal,losses=loss(outpred,outsample,model, \
                                  model.dparamloss() if hasattr(model,'dparamloss') else None)
            if torch.isnan(losstotal):
                path='Debug'
                if not os.path.isdir(path):
                    os.mkdir(path)
                save(path,model,history,config)
                torch.save(insample,path+'/insample_tensor.pt')
                torch.save(outsample,path+'/outsample_tensor.pt')
                print(losstotal,losses)
                raise RuntimeError('Loss value is Nan!')
            losstotal.backward()
            if config['training']['gradclip']=='abs':
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optimizer.step()
            if not config['circuit']['aging'] and config['training']['reg'].startswith('ldsp'):
                model.rho_update()

        history['loss_train'].append(losstotal.item())
        with torch.no_grad():
            outpred=model(invalid,device=device)
            losstotal,losses=loss(outpred,outvalid,model, \
                                  model.dparamloss() if hasattr(model,'dparamloss') else None)
        lrscheduler.step(losstotal.item())
        history['loss_valid'].append(losses[0].item())
        history['lr'].append(optimizer.param_groups[0]['lr'])
        print('Epoch:%d, loss:%.4e, lr:%.6f'%(epoch + 1,history['loss_valid'][-1],history['lr'][-1]))
        print([i.item() for i in losses])
        e=int(time.time()-start_time)
        print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
        if (minloss>history['loss_valid'][-1]):
            if not config['circuit']['aging'] and (config['training']['reg']=='ldsp2'):
                model.wr_update()
            minloss=history['loss_valid'][-1]
            save(savedir+'/best',model,history,config)
        if history['loss_valid'][-1]<0.0005 or history['lr'][-1]<0.000001:
            break
    history['NRMSE']=data.NRMSE(outpred.tolist(),outvalid.tolist())
    if not config['circuit']['aging'] and (config['training']['reg']=='ldsp2'):
        model.wr_update()
    save(savedir+'/last',model,history,config)
    print('Finished Training')
    return model,history

def train_pv(model,config,nepoch,traindataloader,validset,optimizer,lrscheduler,loss,savedir,init=True, \
          device=torch.device("cpu"),freshmodel=None):
    if os.path.isdir(savedir):
        raise RuntimeError('Save directory exist!')
    history={'loss_train':[],
      'loss_valid':[],
      'lr':[]}
    if init:
        for p in model.parameters():
            nn.init.uniform_(p.data,-1,1)
    outvalid=validset['outset'].to(device)
    if 't' in validset:
        invalid={'inset':validset['inset'].to(device), 't':validset['t'].to(device)}
    else: invalid=validset['inset'].to(device)
    start_time=time.time()
#    with torch.cuda.device(device if device.type=='cuda' else None):
    for epoch in range(nepoch):
        for sample in traindataloader:
            outsample=sample['outset'].to(device)
            insample=sample['inset'].to(device)
            optimizer.zero_grad()
            outpred=[]
            for i in len(insample):
                outpred.append(model(insample[i:i+1],device)[0])
            outpred=torch.stack(outpred,0)
            losstotal=loss(outpred,outsample)
            if torch.isnan(losstotal):
                path='Debug'
                if not os.path.isdir(path):
                    os.mkdir(path)
                save(path,model,history,config)
                torch.save(insample,path+'/insample_tensor.pt')
                torch.save(outsample,path+'/outsample_tensor.pt')
                print(losstotal)
                raise RuntimeError('Loss value is Nan!')
            losstotal.backward()
            if config['training']['gradclip']=='abs':
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optimizer.step()

        history['loss_train'].append(losstotal.item())

        with torch.no_grad():
            outpred=model(invalid,device=device)
            losstotal=loss(outpred,outvalid)
        lrscheduler.step(losstotal.item())
        history['loss_valid'].append(losstotal.item())
        history['lr'].append(optimizer.param_groups[0]['lr'])
        print('Epoch:%d, loss:%.4e, lr:%.6f'%(epoch + 1,history['loss_valid'][-1],history['lr'][-1]))
        print(losstotal.item())
        e=int(time.time()-start_time)
        print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
        if (not 'minloss' in locals()):
            minloss=history['loss_valid'][-1]
            save(savedir+'/best',model,history,config)
        elif minloss>history['loss_valid'][-1]:
            minloss=history['loss_valid'][-1]
            save(savedir+'/best',model,history,config)
        if history['loss_valid'][-1]<0.0001 or history['lr'][-1]<0.000001:
            break
    history['NRMSE']=data.NRMSE(outpred.tolist(),outvalid.tolist())
    save(savedir+'/last',model,history,config)
    print('Finished Training')
    return model,history

class check_lds(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, model,device=torch.device("cpu")):
        try:
            p=model.rnn.weight_hh_l0
        except:
            p=model.rnn.rnncell.weight_hh
        if model.nhidden<p.shape[0]:
            p=p.split(model.rnn.nhidden)[-1]
        if hasattr(model,'tau'):
            tau=torch.exp(model.tau)
        else:
            tau=torch.ones(model.nhidden,device=device)
        return torch.sum(torch.log(torch.exp(torch.symeig(p+torch.t(p)-2*tau,eigenvectors=True)[0])+1))

def lds(p,tau):
    return torch.max(torch.symeig(p+torch.t(p)-2*tau,eigenvectors=True)[0])

def lds_projection(p,tau,omega,eps=1e-9,device=torch.device('cpu')):
    pp=torch.matmul(tau/2,torch.matmul(torch.matmul(omega,p+torch.t(p)),omega))
    lmax=torch.max(torch.symeig(pp,eigenvectors=True)[0])-(1-eps)
    return torch.relu(lmax)+1

class mydataset(Dataset):
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

def save(savedir, model, history, config):
    # if type(history['lr'][0]) is np.ndarray or type(history['lr'][0]) is np.float32:
    #     history['lr'] = [np.asscalar(a) for a in history['lr']]
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    with open(savedir+'/config.json', 'w') as f:
        json.dump(config, f, indent="\n")
    with open(savedir+'/history.txt','w') as f:
        json.dump(history, f, indent="\n")
    model.to(torch.device('cpu'))
    torch.save(model.state_dict(), savedir+'/parameters.pt')
    return savedir

def load(filedir,sel='last',pv=False,corners=None):
    filedir=filedir+'/'+sel
    with open(filedir+'/config.json','r') as f:
        config = json.load(f)
    with open(filedir+'/history.txt','r') as f:
        history = json.load(f)
    model=model_sel(config,pv,corners)
    model.load_state_dict(torch.load(filedir+'/parameters.pt'))#map_location=torch.device("cpu")
    return (model,config,history)


def load_partial(filedir,model,sel='last',grad=True):
    filedir=filedir+'/'+sel
    with open(filedir+'/config.json','r') as f:
        config = json.load(f)
    freshmodel=model_sel(config)
    freshmodel.load_state_dict(torch.load(filedir+'/parameters.pt'))#map_location=torch.device("cpu")
    model_dict=model.state_dict()
    fresh_dict={k:v for k,v in freshmodel.state_dict().items() if k in model_dict}
    model_dict.update(fresh_dict)
    model.load_state_dict(model_dict)
    for k in fresh_dict:
        attrgetter(k)(model).requires_grad=grad
    return model

def load_rnn2ode(filedir,model,sel='last'):
    filedir=filedir+'/'+sel
    with open(filedir+'/config.json','r') as f:
        config = json.load(f)
    rnnmodel=model_sel(config)
    rnnmodel.load_state_dict(torch.load(filedir+'/parameters.pt'))
    model.rnn.tau.data=torch.zeros(model.rnn.tau.data.shape)
    model.rnn.rnncell.weight_hh.data=rnnmodel.rnn.weight_hh_l0.data
    model.rnn.rnncell.weight_ih.data=rnnmodel.rnn.weight_ih_l0.data

    if config['training']['reg']=='ldsp2':
        model.rnn.omega.data=rnnmodel.omega.data
    else:
        model.rnn.rnncell.bias_hh.data=rnnmodel.rnn.bias_hh_l0.data
        model.rnn.rnncell.bias_ih.data=rnnmodel.rnn.bias_ih_l0.data
    model.linear.weight.data=rnnmodel.linear.weight.data
    model.linear.bias.data=rnnmodel.linear.bias.data
    if hasattr(model,'rnnage'):
        model.rnnage.weight_hh_l0.data=rnnmodel.rnnage.weight_hh_l0.data
        model.rnnage.weight_ih_l0.data=rnnmodel.rnnage.weight_ih_l0.data
        model.rnnage.bias_hh_l0.data=rnnmodel.rnnage.bias_hh_l0.data
        model.rnnage.bias_ih_l0.data=rnnmodel.rnnage.bias_ih_l0.data
        model.linearage.weight.data=rnnmodel.linearage.weight.data
        model.linearage.bias.data=rnnmodel.linearage.bias.data
    return model

class loss_crossentropy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, targ):
        a=torch.mul(-(1-targ),torch.log10(1-pred))
        b=torch.mul(-targ,torch.log10(pred))
        return torch.mean(a+b)

class loss_smse(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse=nn.MSELoss()
    def forward(self, pred, targ):
        pred_sorted,indices=torch.sort(pred,0)
        targ_sorted,indices=torch.sort(targ,0)
        return self.mse(pred_sorted,targ_sorted)

class loss_SG(nn.Module):
    def __init__(self):
        super(loss_SG, self).__init__()
    def forward(self, pred, targ):
        sump=torch.sum(pred**2,1)
        sumt=torch.sum(targ**2,1)
        sumpt=torch.sum(pred*targ,1)
        Msg=torch.sqrt(sump/sumt)-1
        Psg=1/np.pi*torch.acos(sumpt/torch.sqrt(sump*sumt))
        Csg=torch.mean(torch.sqrt((Msg**2)+(Psg**2)))
        return Csg

def freq(t,seq):
    mid=(max(seq)+min(seq))/2
    i=len(seq)//3
    while (seq[i]>=mid) or (seq[i+1]<mid):
        i+=1
    t1=t[i]+(t[i+1]-t[i])/(seq[i+1]-seq[i])*(mid-seq[i])
    i+=1
    while (seq[i]>=mid) or (seq[i+1]<mid):
        i+=1
    t2=t[i]+(t[i+1]-t[i])/(seq[i+1]-seq[i])*(mid-seq[i])
    return 1/(t2-t1)

def freqerror(tpre,outpre,ttar,outtar):
    if len(tpre.size())==1:
        tpre=torch.stack(len(outpre)*[tpre],dim=0)
    if len(ttar.size())==1:
        ttar=torch.stack(len(outpre)*[ttar],dim=0)
    freq_pre=torch.zeros(outpre.size(0),outpre.size(-1))
    freq_tar=torch.zeros(outtar.size(0),outtar.size(-1))
    for i in range(outpre.size(0)):
        for j in range(outpre.size(-1)):
            freq_pre[i,j]=freq(tpre[i],outpre[i,:,j])
            freq_tar[i,j]=freq(ttar[i],outtar[i,:,j])
    print(freq_pre,freq_tar)
    print(torch.abs(freq_pre-freq_tar)/freq_tar)
    return torch.mean(torch.abs(freq_pre-freq_tar)/freq_tar)
