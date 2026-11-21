import numpy as np
import random
import torch 
from torch.utils import data
import matplotlib.pyplot as plt

def synthetic_data(w,b,num_examples):    
    X=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))

def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]

true_w=torch.tensor([2,-3.4],dtype=torch.float32)
true_b= 4.2
features,labels=synthetic_data(true_w,true_b,1000)


w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

def linreg(X,w,b):
    return torch.matmul(X,w)+b
def loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2

def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

lr=0.03
num_epochs=3
net = linreg
batch_size=3
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l=loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_1=loss(net(features,w,b),labels)
        print('epoch ',epoch,'loss' ,float(train_1.mean()))




"""
x_data=features.numpy().flatten()
y_data=labels.numpy().flatten()
plt.plot(x_data,y_data)
plt.show()
def synthetic_data(w,b,num_size):
    X=torch.normal(0,1,(num_size,len(w)))
    y=torch.matmul(X,w)+b     #X*w变成一维的
    y+=torch.normal(0,0.1,y.shape)
    return X,y.reshape(-1,1)

def data_iter(features,labels,batch_sizes):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)
    for i in (0,num_examples,batch_sizes):
        batch_indices=indices[i:min(i+batch_sizes,num_examples)]
        yield features[batch_indices],labels[batch_indices]
"""