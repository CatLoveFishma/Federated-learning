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

def dataxxx():
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

    data_length=300
    for name,data in processed_data.items():
        n=len(data)
        x=n//data_length
        for i in range(x):
            splited_data.append(data[i*data_length:(i+1)*data_length])
            label.append(name)
    splited_data=np.array(splited_data)
    label=np.array(label)
    splited_data=torch.tensor(splited_data,dtype=torch.float32)
    label=torch.tensor(label,dtype=torch.long)
    data_index=[i for i in range(len(label))]
    num_items=5000
    select_index1=np.random.choice(data_index,num_items,replace=False)
    select_index2=np.random.choice(data_index,num_items,replace=False)
    return [splited_data[select_index1],label[select_index1],splited_data[select_index2],label[select_index2]]