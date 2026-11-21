import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from IPython import display
d2l.use_svg_display()

class Accumulator :
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def get_fashion_mnist_labels(labels):
    text_labels=['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]



"""
数据读取：
核心函数 DataLoader 它能在训练模型时将数据集包装为可迭代对象 并在每次迭代时从数据集中抽取一个小批量。
DataLoader 的主要参数包括数据集、批量大小、是否打乱数据以及加载数据的进程数。通过设置这些参数，我们可以灵活地控制数据的加载方式，从而提高训练效率。
torchvision.datasets.FashionMNIST 是 PyTorch 提供的一个用于加载 Fashion-MNIST 数据集的类。它可以自动下载数据集，并将其转换为 PyTorch 张量格式，方便后续的模型训练和评估。
需要设置 transform 参数来指定数据预处理方式，例如将图像转换为张量或调整图像大小。通过这些设置，我们可以轻松地获取训练集和测试集的数据迭代器，从而进行模型训练和评估。
具体步骤如下：
1. 定义批量大小、输入和输出的维度。
2. 定义数据加载器的工作进程数。
3. 使用 torchvision.datasets.FashionMNIST 加载训练集和测试集，并应用必要的图像预处理。
4. 使用 DataLoader 将数据集包装为可迭代对象，以便在训练过程中按批次获取数据。
5. 返回训练集和测试集的数据迭代器，供后续模型训练和评估使用。
"""



batch_size=256
num_inputs=784
num_outputs=10

def get_dataloader_workers():
    return 1
def load_data_fashion_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    
    mnist_train=torchvision.datasets.FashionMNIST(
    root="../data",train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.FashionMNIST(
    root="../data",train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test,batch_size,shuffle=False,
                            num_workers=get_dataloader_workers()))

train_iter,test_iter=load_data_fashion_mnist(batch_size)



"""
模型参数初始化：
"""
W=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b=torch.zeros(num_outputs,requires_grad=True)

"""
softmax 函数：
softmax 函数将输入的实数向量转换为概率分布。
"""
def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition

"""
迭代函数
"""
def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W)+b)
"""
交叉熵损失函数：
"""
def cross_entropy(y_hat,y):
    return - torch.log(y_hat[range(len(y_hat)),y])
"""
精度计算：
分两步  第一
1. 如果预测结果是一个二维张量（即每个样本有多个类别的预测概率），则使用 argmax 函数找到每个样本的预测类别。
2. 将预测类别与真实标签进行比较，计算预测正确的样本数量，并返回准确率。

第二 将accracy函数应用于整个数据集 计算模型在测试集上的准确率。具体步骤如下：
1. 将模型设置为评估模式，以确保在评估过程中不会应用 Dropout 等训练时的随机性。
2. 初始化一个累加器，用于统计预测正确的样本数量和总样本数量。
3. 遍历测试集的每个批次，计算每个批次的预测结果，并使用 accuracy 函数计算预测正确的样本数量。
4. 将每个批次的预测正确样本数量和总样本数量累加到累加器中。
5. 最后，计算整个测试集的准确率，即预测正确的样本数量除以总样本数量，并返回该准确率。
"""

def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()     #评估模式，不会有Dropout,准确率稳定
    metric=Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]



"""
更新函数
sgd更新
"""
lr=0.1
num_epochs=10

def updater (batch_size):
    return d2l.sgd([W,b],lr,batch_size)


"""
训练函数：
每一轮中
前馈
计算y_hat
计算损失

清空梯度
反向传播
更新函数

绘图
累加损失函数，训练精度，测试精度

"""
def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric=Accumulator(3)
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics=train_epoch_ch3(net,train_iter,loss,updater)
        test_acc=evaluate_accuracy(net,test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))
    train_loss,train_acc=train_metrics
    assert train_loss<0.5,train_loss
    assert train_acc<=1 and train_acc >0.7,train_acc
    assert test_acc<=1 and test_acc >0.7 ,test_acc


train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater)

"""展示预测结果："""


def predict_ch3(net,test_iter,n=6):
    for X,y in test_iter:
        break
    trues=get_fashion_mnist_labels(y)
    preds=get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
predict_ch3(net, test_iter)