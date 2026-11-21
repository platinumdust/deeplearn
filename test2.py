import torch
a=torch.arange(16,dtype=torch.float32).reshape(4,4)
a=a/a.sum(axis=1,keepdim=True)

print(a)
