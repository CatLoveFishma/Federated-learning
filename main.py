from fileinput import filename
import os
import json
import scipy.io
import pandas as pd
from pathlib import Path
from helper import matfile_to_dic
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Nets import MLP,CNN_1D_2L
from Update import LocalUpdate,normalization
import argparse
from Options import args_parser
from torchvision import datasets, transforms 
import copy
from tools import FedAvg, test_model 
import time



args=args_parser()
filepath=Path('12k_DE')
dataDic=matfile_to_dic(filepath)
for dicson in dataDic.values():
    del dicson['__header__']
    del dicson['__version__']
    del dicson['__globals__']
processed_data={0:[],1:[],2:[],3:[]}
#0:B 1:IR 2:No 3:OR
for name,data in dataDic.items():
    if 'B' in name:
        for iname in data.keys():
            if 'DE' in iname:
                processed_data[0].extend(data[iname])
    elif 'IR' in name:
        for iname in data.keys():
            if 'DE' in iname:
                processed_data[1].extend(data[iname])
    elif 'No' in name:
        for iname in data.keys():
            if 'DE' in iname:
                processed_data[2].extend(data[iname])
    elif 'OR' in name:
        for iname in data.keys():
            if 'DE' in iname:
                processed_data[3].extend(data[iname])
splited_data=[]
label=[]

data_length=200
for name,data in processed_data.items():
    n=len(data)
    x=n//data_length
    for i in range(x):
        splited_data.append(data[i*data_length:(i+1)*data_length])
        label.append(name)


data_type='iid'
clients_all=1000
 #每次训练的客户占所有客户的比例

splited_data=torch.tensor(splited_data,dtype=torch.float32)
label=torch.tensor(label,dtype=torch.long)
data_and_label=torch.utils.data.TensorDataset(splited_data,label)

#可以用TensorDataset进行数据的打包
if data_type =='iid':
    num_items=int(len(label)*0.8/clients_all)
    #留下20%的数据作为实验集
    dict_users,all_idxs={},[i for i in range(len(label))]
    test_idxs=[]
    for i in range(clients_all):
        dict_users[i]=set(np.random.choice(all_idxs,num_items,replace=False))
        #随机分配数据
        all_idxs=list(set(all_idxs)-dict_users[i])
        #更新数据
    test_idxs=all_idxs

#datatest=data_and_label[test_idxs]
#datatrain=data[dict_users]



net_glob=CNN_1D_2L(data_length)
print(net_glob)
net_glob.train()
w_glob = net_glob.state_dict()
'''''
class option:
    def __init__(self):
        self.lr=0.001
        self.momentum=0.5
        self.local_ep=10
        self.device='cpu'
        self.verbose=False
        self.local_bs=10
        self.bs=64
        self.betas=(0.99,0.999)
        self.wd=1e-5
        self.precision=4
'''''
args=args_parser()

epochs=12
#经过测试，12时不在收敛
local_epoch=3
loss_train=[]
frac=args.r
#[local_epochs,precision,r]

chanshu=[[20,7,0.1],[20,7,0.3],[20,7,0.5],[20,6,0.5],[20,5,0.5],[20,4,0.5],[17,7,0.5],[14,7,0.5]]
for x in chanshu:
    args.local_ep=x[0]
    frac=x[2]
    args.precision=x[1]
    savemodel='models'+"\saved_models_local_epochs_"+str(x[0])+"_r_"+str(x[2]).replace('.', '_')+"_precision_"+str(x[1])
    os.system('mkdir '+savemodel)
    for iter in range(epochs):
        start=time.perf_counter()
        loss_locals=[]
        w_locals = [w_glob for i in range(clients_all)]
        m=round(frac*clients_all)
        idxs_users=np.random.choice(range(clients_all),m,replace=False)
        #选出来的是客户的编号
        for idx in idxs_users:
            local=LocalUpdate(args=args,dataset=data_and_label,idxs=dict_users[idx])
            w,loss=local.train(net=copy.deepcopy(net_glob))
            w_locals[idx]=copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
        w_glob=FedAvg(w_locals,args.precision)
        net_glob.load_state_dict(w_glob)
        
        torch.save(net_glob,savemodel+"/model_epoch_"+str(iter+1)+".pt")
        loss_avg=sum(loss_locals)/len(loss_locals)
        end=time.perf_counter()
        print('Round {:3d}, Average loss {:.3f},time_used {}'.format(iter, loss_avg,end-start))
        loss_train.append(loss_avg)    

    net_glob.eval()
    #acc_train, loss_train = test_model(net_glob, dataset_train, args)
    acc_test, loss_test = test_model(net_glob, data_and_label,test_idxs, args)
    #print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
'''''
for iter in range(epochs):

    start=time.perf_counter()
    loss_locals=[]
    w_locals = [w_glob for i in range(clients_all)]
    m=round(frac*clients_all)
    idxs_users=np.random.choice(range(clients_all),m,replace=False)
    #选出来的是客户的编号
    for idx in idxs_users:
        local=LocalUpdate(args=args,dataset=data_and_label,idxs=dict_users[idx])
        w,loss=local.train(net=copy.deepcopy(net_glob))
        w_locals[idx]=copy.deepcopy(w)
        loss_locals.append(copy.deepcopy(loss))
    w_glob=FedAvg(w_locals,args.precision)
    net_glob.load_state_dict(w_glob)
    loss_avg=sum(loss_locals)/len(loss_locals)
    end=time.perf_counter()
    print('Round {:3d}, Average loss {:.3f},time_used {}'.format(iter, loss_avg,end-start))
    loss_train.append(loss_avg)    

net_glob.eval()
#acc_train, loss_train = test_model(net_glob, dataset_train, args)
acc_test, loss_test = test_model(net_glob, data_and_label,test_idxs, args)
#print("Training accuracy: {:.2f}".format(acc_train))
print("Testing accuracy: {:.2f}".format(acc_test))
'''''