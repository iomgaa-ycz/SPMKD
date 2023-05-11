import torch
import torch.nn as nn
from models.MobileNet import MobileNetV3_Small

class Fuser(nn.Module):
    def __init__(self):
        super(Fuser, self).__init__()
        self.backbone=MobileNetV3_Small()
        self.linear1=nn.Linear(128*3,128*16)

    def forward(self,keypoint,coordinate,feature):
        n,w,h,k=keypoint.size()

        feature = torch.bmm(feature.view(n, -1, w * h), keypoint.view(n, w * h, k)).permute(0,2,1)  # get feature (n,3,4096)*(n,4096,k)=(n,3,k)->(n,k,3)
        coordinate = torch.bmm(coordinate.view(n, -1, w * h), keypoint.view(n, w * h, k)).permute(0,2,1)  # get keypoint (n,3,4096)*(n,4096,k)=(n,3,k)->(n,k,3)

        coordinate_save = coordinate
        coordinate=self.linear1(coordinate.reshape(n,-1))

        feature=torch.cat([coordinate.reshape(n,1, -1),feature.reshape(n,1, -1)],dim=2)


        return feature,coordinate_save