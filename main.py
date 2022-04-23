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
from Nets import MLP
from Update import LocalUpdate
import argparse
import Options

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
for name,data in processed_data.items():
    n=len(data)
    x=n//500
    for i in range(x):
        splited_data.append(data[i*500:(i+1)*500])
        label.append(name)
