import time,torch
print(torch.__version__)
t=time.time()
x=torch.ones(10000,784)
w=torch.ones(784,10)
for i in range(1000):
    y=torch.mm(x,w)
print('cost time:',time.time()-t)