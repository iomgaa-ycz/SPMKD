import torch
import torch.nn as nn
from models.MobileNet import MobileNetV3_Small


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = MobileNetV3_Small()
        self.coordinate = nn.MaxPool2d(kernel_size=4, return_indices=True)
        self.feature = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=4),
                                     nn.BatchNorm2d(16, track_running_stats=False), nn.ReLU())
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        keypoint = self.backbone(x)  # (k=128,64,64)

        value, index = self.coordinate(x)  # (4096,2)
        index_x = (((index / 256).int()) / 4).int()
        index_x = index_x / 64
        index_y = (((index % 256).int()) / 4).int()
        index_y = index_y / 64
        coordinate = torch.cat([value, index_x, index_y], dim=1)

        feature = self.feature(x)  # (f=16,64,64)
        keypoint = keypoint.permute(0, 2, 3, 1)  # (64,64,f=16)
        n, w, h, k = keypoint.size()

        keypoint = keypoint.reshape(n, w * h, k)
        keypoint = self.softmax(keypoint).reshape(n, w, h, k)
        return keypoint, coordinate, feature