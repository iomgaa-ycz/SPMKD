import torch
import torch.nn as nn

class UpLayer(nn.Module):
    def __init__(self, in_size, out_size, up=True,use_dilation=True):
        super(UpLayer, self).__init__()
        self.up = up
        self.use_dilation = use_dilation
        self.conv1_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_size, track_running_stats=False)
        if use_dilation:
            self.conv1_2 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=3, dilation=3)
            self.conv1_3 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=5, dilation=5)
            self.conv2 = nn.Conv2d(out_size * 3, out_size, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(num_features=out_size, track_running_stats=False)
            self.bn_up2 = nn.BatchNorm2d(num_features=out_size, track_running_stats=False)
            self.bn3 = nn.BatchNorm2d(num_features=out_size, track_running_stats=False)
        if up == True:
            self.up_layer = nn.ConvTranspose2d(in_channels=out_size, out_channels=out_size, kernel_size=2, stride=2)
            if use_dilation:
                self.up_layer = nn.ConvTranspose2d(in_channels=out_size * 3, out_channels=out_size, kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
            self.bn_up = nn.BatchNorm2d(num_features=out_size, track_running_stats=False)
            self.bn_up2 = nn.BatchNorm2d(num_features=out_size, track_running_stats=False)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        x1 = self.conv1_1(x[-1])
        x1 = self.bn1(x1)
        if self.use_dilation:
            x2 = self.conv1_2(x[-1])
            x2 = self.bn2(x2)
            x3 = self.conv1_3(x[-1])
            x3 = self.bn3(x3)
            x1 = torch.cat([x1, x2, x3], dim=1)
        x1 = self.relu(x1)
        if self.up == True:
            x1 = self.up_layer(x1)
            x1 = self.bn_up(x1)
            x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn_up2(x1)
        x1 = self.relu(x1)

        x.append(x1)
        return x

class ExchangeLayer(nn.Module):
    def __init__(self, num_elements=4):
        super(ExchangeLayer, self).__init__()
        assert 2 <= num_elements <= 5, "num_elements must be between 2 and 5."

        self.num_elements = num_elements

        # 根据元素个数来初始化卷积层
        self.convs = nn.ModuleList()
        for i in range(num_elements):
            in_channels = sum([1, 64, 32, 16, 8][:num_elements])
            out_channels = [1, 64, 32, 16, 8][i]
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))

        # 初始化下采样卷积层
        self.downsampling_convs = nn.ModuleList([
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=4, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, stride=4, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, stride=8, padding=1),
        ])

    def forward(self, input_list):
        assert len(input_list) == self.num_elements, "Input list length must match num_elements."

        output_list = []

        for i in range(self.num_elements):
            concat_tensors = []

            # 处理所有元素
            for j in range(self.num_elements):
                if j < i:
                    if j in [0,1] and i in [0,1]:
                        concat_tensors.append(input_list[j])
                        continue
                    # 上采样
                    scale_factor_num = 2 ** (i - j)
                    if j== 0:
                        scale_factor_num = 2**(i-1-j)
                    upsampled_tensor = nn.functional.interpolate(input_list[j], scale_factor=scale_factor_num, mode='bilinear', align_corners=False)
                    concat_tensors.append(upsampled_tensor)
                elif j == i:
                    # 不做处理
                    concat_tensors.append(input_list[j])
                else:
                    # 下采样，除了(64,64,64)与(1,64,64)的情况
                    if not (i == 0 and j == 1):
                        if j==2 and i==0:
                            down_num = 0
                        elif j==2 and i==1:
                            down_num = 0
                        elif j==3 and i==0:
                            down_num = 3
                        elif j==3 and i==1:
                            down_num = 3
                        elif j==3 and i==2:
                            down_num = 1
                        elif j==4 and i==0:
                            down_num = 5
                        elif j==4 and i==1:
                            down_num = 5
                        elif j==4 and i==2:
                            down_num = 4
                        elif j==4 and i==3:
                            down_num = 2
                        downsampled_tensor = self.downsampling_convs[down_num](input_list[j])
                        concat_tensors.append(downsampled_tensor)
                    else:
                        concat_tensors.append(input_list[j])

            # 拼接所有的tensor并通过卷积生成新的元素
            concat_tensor = torch.cat(concat_tensors, dim=1)
            new_tensor = self.convs[i](concat_tensor)
            output_list.append(new_tensor)

        return output_list

class Convs(nn.Module):
    def __init__(self, num_elements=4):
        super(Convs, self).__init__()
        assert 2 <= num_elements <= 5, "num_elements must be between 2 and 5."

        self.num_elements = num_elements

        # 根据元素个数来初始化卷积层、批标准化层和激活函数
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relus = nn.ModuleList()
        for i in range(num_elements):
            in_channels = [1, 64, 32, 16, 8][i]
            out_channels = in_channels
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            self.bns.append(nn.BatchNorm2d(out_channels))
            self.relus.append(nn.ReLU(inplace=True))

    def forward(self, input_list):
        assert len(input_list) == self.num_elements, "Input list length must match num_elements."

        output_list = []

        # 对每个输入张量执行卷积、批标准化和ReLU激活操作
        for i in range(self.num_elements):
            x = input_list[i]
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = self.relus[i](x)
            output_list.append(x)

        return output_list



class Decoder(nn.Module):
    def __init__(self, out_channel=1,use_dilation=True,use_Exchange=True):
        super(Decoder, self).__init__()
        in_filters = [32, 64, 1]
        out_filters = [16, 32, 64]
        self.use_Exchange = use_Exchange

        self.linear = nn.Linear(128 * 32, 128 * 32)
        self.bn = nn.BatchNorm2d(1, track_running_stats=False)
        self.relu = nn.ReLU()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.ReLU())
        # self.exchange1 = ExchangeLayer(num_elements=2)
        self.up1 = UpLayer(in_size=64, out_size=32,use_dilation=use_dilation)

        self.conv2 = Convs(num_elements=3)
        # self.exchange2 = ExchangeLayer(num_elements=3)
        self.up2 = UpLayer(in_size=32, out_size=16,use_dilation=use_dilation)

        self.conv3 = Convs(num_elements=4)
        # self.exchange3 = ExchangeLayer(num_elements=4)
        self.up3 = UpLayer(in_size=16, out_size=8,use_dilation=use_dilation)

        self.conv4 = Convs(num_elements=5)
        # self.exchange4 = ExchangeLayer(num_elements=5)
        self.up4 = UpLayer(in_size=8, out_size=4,use_dilation=use_dilation)

        if use_Exchange:
            self.exchange1 = ExchangeLayer(num_elements=2)
            self.exchange2 = ExchangeLayer(num_elements=3)
            self.exchange3 = ExchangeLayer(num_elements=4)
            self.exchange4 = ExchangeLayer(num_elements=5)


        self.mask1 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(16),nn.ReLU())
        self.mask2 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=out_channel, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_channel),nn.ReLU())

    def forward(self,x):
        x = self.linear(x).reshape(-1,1,64,64)
        x = self.bn(x)
        x = self.relu(x)

        x1 = self.conv1(x)
        x = [x, x1]
        if self.use_Exchange:
            x = self.exchange1(x)
        x = self.up1(x)

        x = self.conv2(x)
        if self.use_Exchange:
            x = self.exchange2(x)
        x = self.up2(x)

        x = self.conv3(x)
        if self.use_Exchange:
            x = self.exchange3(x)
        x = self.up3(x)

        x = self.conv4(x)
        if self.use_Exchange:
            x = self.exchange4(x)

        x = self.mask1(x[3])
        x = self.mask2(x)
        return x








        return x
