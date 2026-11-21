import torch
import numpy as np
from torch.utils import data
from torch import nn


def synthetic_data(w,b,num_example):
    X=torch.normal(0,1,(num_example,len(w)))
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape(-1,1)


true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)
"""读取数据集"""
def load_array (data_arrays,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_arrays) 
    """*dataarray表示解包元组  TensorDataset表示把两个东西打包起来feature和label一一对应"""
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
    """DataLoader数据迭代器 投入打包数据 一次训练的数据量 是否在每次训练后打乱"""

batch_size=10
data_iter=load_array((features,labels),batch_size)

"""iter()函数 将可迭代数据(可以被转化(实际是重新产生一个以他为内容的)迭代器 进行依次访问)转化为迭代器"""

net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
"""带_的代表原地操作 不调用新内存 data代表只改变数据 不会产生计算图"""

loss=nn.MSELoss()

trainer=torch.optim.SGD(net.parameters(),lr=0.03)
"""trainer是个优化器 用来  optim优化算法库 SGD小批量随机梯度下降"""

num_epochs=3
for epoch in range (num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l=loss(net(features),labels)
    print("epoch",epoch+1,"loss",l)