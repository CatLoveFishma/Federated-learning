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
    return output_dic




def mnist_iid(dataset, num_users):  #mnist是手写数字图像数据集，mnist_iid代表符合独立同分布的mnist
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users) #技算出每个客户分配到的数据大小
    dict_users, all_idxs = {}, [i for i in range(len(dataset))] #定义一个空字典dict_users，和一个0~传入数据的长度len的一个list
    for i in range(num_users): #循环，循环次数为客户机的个数
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        """
        np.random.choice是用来产生一个随机样本
        numpy.random.choice(a, size=None, replace=True, p=None)
        从a中随机抽取数字,组成一个大小为size的随机数组,replace=True代表能够出现重复,false代表不能出现重复
        set用于创建一个无重复数据集
        set(np.random.choice(all_idxs, num_items, replace=False))
        这个命令的意思是从0~数据集个数len 的list中随机抽取数字,生成一个大小为每个客户所分配到的数据个数大小的数据，数据中的
        内容是dataset中数据元素的index,并且replace=false,代表不能重复选取index
        并且用set来再次确保无重复数据的index
        """
        all_idxs = list(set(all_idxs) - dict_users[i])
        #去除已经被选取的数据的指数
    return dict_users #返回num_users个客户的字典,dict={"客户编号":[分配给客户的数据集元素的index是list形式]}
    #由于返回的数组都是从dataset数据集中随机抽取的所有符合独立同分布的要求


    