import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        #super()是用于继承父类的函数，继承nn.Module类
        #这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        #nn.Module是nn中十分重要的类，包含网络各层的定义及forward方法。通过直接继承可以省去自己创建对象的功夫
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        #nn.Linear()一般用于设置全连接层，dim_in输入神经元个数 dim_hidden输出神经元个数，默认包含偏置
        self.relu = nn.ReLU()
        #设置relu激活函数
        self.dropout = nn.Dropout()
        #随机舍弃某些神经元，防止过拟合等
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        #定义一个全连接层

    def forward(self, x):
        #x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        #view()函数通常用于数据维度的变化,在这里的作用是行数自动变化，但是列数为
        x = self.layer_input(x)
        #输入处理后的数据
        x = self.dropout(x)
        #随机舍弃
        x = self.relu(x)
        #激活函数
        x = self.layer_hidden(x)
        return x