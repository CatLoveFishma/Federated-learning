import os
import json
import scipy.io
import pandas as pd
from pathlib import Path

def matfile_to_dic(folder_path): #返回一个字典，关键字是文件名，值是文件名的数据。而数据本身也是类似字典的形式，内含键值对
    '''
    Read all the matlab files of the CWRU Bearing Dataset and return a 
    dictionary. The key of each item is the filename and the value is the data 
    of one matlab file, which also has key value pairs.
    
    Parameter:
        folder_path: 
            Path (Path object) of the folder which contains the matlab files.
    Return:
        output_dic: 
            Dictionary which contains data of all files in the folder_path.
    '''
    output_dic = {} #定义一个字典
    for _, filepath in enumerate(folder_path.glob('*.mat')):
        key_name = str(filepath).split('\\')[-1]
        output_dic[key_name] = scipy.io.loadmat(filepath, squeeze_me=True)
    
    

    