#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset): #Dataset是标准class
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        #交叉熵损失函数
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        '''''
        PyTorch中数据读取的一个重要接口是torch.utils.data.DataLoader，该接口定义在dataloader.py脚本中，
        只要是用PyTorch来训练模型基本都会用到该接口，该接口主要用来将自定义的数
        据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成
        Tensor，后续只需要再包装成Variable即可作为模型的输入
        '''''
    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        #设置优化器，默认lr=0.01，默认momentum=0.5,momentum是带动量的随机梯度下降
        epoch_loss = [] #每一轮的损失
        for iter in range(self.args.local_ep): #循环epoch次
            batch_loss = [] #每个batch的损失，batch_size默认为10
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad() #进行梯度清零
                '''
                调用backward()函数之前都要将梯度清零，因为如果梯度不清零，pytorch中
                会将上次计算的梯度和本次计算的梯度累加。这样逻辑的好处是，
                当我们的硬件限制不能使用更大的bachsize时，使用多次计算较
                小的bachsize的梯度平均值来代替，更方便，坏处当然是每次都要清零梯度
                '''
                log_probs = net(images) #这里的images应该是经过处理的数据
                loss = self.loss_func(log_probs, labels) #计算交叉熵损失
                loss.backward() #调用回调函数
                optimizer.step() #进行单次优化
                if self.args.verbose and batch_idx % 10 == 0: #verbose默认false
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        #state_dict包含了权重，用于聚合

def normalization(data):
    for i,j in enumerate(data):
        x=(j-0)/3
        data[i]=x
    return data