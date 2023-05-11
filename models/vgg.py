import torch.nn as nn
from torch.nn import functional as F

class VGG_block1(nn.Module):
    # 构建一个最小的残差块
    def __init__(self, in_channels, out_channels,  stride=1):
        super(VGG_block1, self).__init__()#初始化固定句式
        ##in_channels输入通道
        ##out_channels输出通道
        ##use_1x1conv是否使用1*1卷积
        ##stride步长说多少
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))

        return Y


class VGG_block2(nn.Module):
    # 构建一个最小的残差块
    def __init__(self, in_channels, out_channels, stride=1):
        super(VGG_block2, self).__init__()  # 初始化固定句式
        ##in_channels输入通道
        ##out_channels输出通道
        ##use_1x1conv是否使用1*1卷积
        ##stride步长说多少
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = F.relu(self.bn3(self.conv2(Y)))

        return Y

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.reshape(x.shape[0], -1)

class VGG(nn.Module):
    def __init__(self,inchannel=1,num_channels=8192,drop_num=0.5):
        super(VGG, self).__init__()

        self.b1 = VGG_block1(in_channels=inchannel,out_channels=64,stride=2)
        self.b2 = VGG_block1(in_channels=64,out_channels=128,stride=2)
        self.b3 = VGG_block2(in_channels=128,out_channels=256,stride=2)
        self.b4 = VGG_block2(in_channels=256,out_channels=512,stride=2)
        self.b5 = VGG_block2(in_channels=512,out_channels=512,stride=2)
        self.FlattenLayer= FlattenLayer()
        self.linear1=nn.Linear(in_features=num_channels,out_features=2048)
        self.linear2 = nn.Linear(in_features=2048, out_features=2048)
        self.linear3 = nn.Linear(in_features=2048, out_features=3)
        self.dropout = nn.Dropout(p=drop_num)
        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):



        y=self.b1(x)
        y = self.b2(y)
        y = self.b3(y)
        y = self.b4(y)
        y = self.b5(y)
        y = self.FlattenLayer(y)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.linear2(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.linear3(y)
        y = self.sigmoid(y)
        return y



