#1.导包定参
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

import matplotlib.pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms

#Hyper Parameters
Epoch = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

#2.数据加载
train_data = dsets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
test_data = dsets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
print(train_data)
print(test_data)





#3.数据预处理
#3.1 数据可视化
plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
plt.show()
#3.2数据归一
test_x = test_data.data.type(torch.FloatTensor)[:2000]/255
test_y = test_data.targets.numpy()[:2000]
#3.3数据分批
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

#了解一下数据特征，图片是28*28个像素
#print(type(train_loader))
dataiter= iter(train_loader)
imgs,labs = next(dataiter)
print(imgs)#imgs.size()=torch.Size([64, 1, 28, 28])
print(labs)#图片的标签，就是类别，mnist一共有10个类别，故其取值分布在[0,9]之间 

#4.定义网络模式
class RNN(nn.Module):
    def __init__(self):
      super(RNN, self).__init__()
      self.rnn = nn.LSTM(
          input_size= INPUT_SIZE,
          hidden_size= 64,
          num_layers=1,
          batch_first=True,
      )

      self.out = nn.Linear(64, 10)
    def forward(self, x):
        # x(batch_size, seq_len, input_size)
        r_out, (h_n, h_c)= self.rnn(x, None)
        # r_out(batch_size, seq_len, hidden_size)
        #print(r_out[:, -1, :].shape)
        out = self.out(r_out[:, -1, :])#r_out(batch_size, seq_len, hidden_size)->(batch_size, hidden_size)->(batch_size, 10)取第二个维度的最后一个
        
        return out


#5.设置使用GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#使用定义好的RNN
#模型和输入数据都需要to device
rnn = RNN().to(device)
print(rnn)



#定义损失函数,分类问题使用交叉信息熵
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


#7.模型训练与测试
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logdir')
for epoch in range(Epoch):
    for step, (b_x, b_y) in enumerate(train_loader):
        #print(b_x.shape, b_y.shape)
        #取出数据及标签
        #b_x为imgs：torch.Size([64, 1, 28, 28]),b_y为lables：torch.size([64])
        b_x = b_x.view(-1, 28, 28)#torch.Size([64, 1, 28, 28])->torch.Size([64,28, 28])
        #数据及标签均送入GPU或CPU
        b_x,b_y = b_x.to(device),b_y.to(device)

        #前向传播
        output = rnn(b_x)#->torch.Size([64,10])
        #计算损失函数
        loss = loss_func(output, b_y)
        #清空上一轮的梯度
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #参数更新
        optimizer.step()
        #测试，计算准确率
        if step % 50 == 0:
            writer.add_scalar("Train/Loss", loss.item(), epoch*len(train_loader)+step)
            test_output = rnn(test_x.cuda())
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
            writer.add_scalar("Accuracy", 100.0*accuracy, epoch*len(train_loader)+step)

test_x =test_x.cuda()
test_output = rnn(test_x[:50].view(-1,28,28))
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
print(pred_y, 'prediction number')
print(test_y[:50], 'real number')