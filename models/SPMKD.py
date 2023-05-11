from models.Encoder import Encoder
from models.Fuser import Fuser
from models.Decoder import Decoder
import torch
import torch.nn as nn
from model_keypoint.shift_keypoint import shift_keypoint
from model_keypoint.GCN import GCN
from model_keypoint.GAT import GAT
from model_keypoint.GraphSAGE import GraphSAGE
from models.densenet import DenseNet
from models.resnet import ResNet


class SPMKD(nn.Module):
    def __init__(self,config):
        super(SPMKD, self).__init__()
        self.config=config
        if self.config.train_encoder == True:
            self.encoder = Encoder()
            self.fuser = Fuser()
            if self.config.Mask == True:
                self.decoder = Decoder(out_channel=2)
            else:
                self.decoder = Decoder(out_channel=1,use_dilation=config.use_dilation,use_Exchange=config.use_Exchange)
        else:
            if self.config.use_Keypoint == True:
                if self.config.keypoint_type == "Ours":
                    self.num_features = 19
                    self.encoder = Encoder()
                    self.fuser = shift_keypoint()
                    self.one_step = 128
                else:
                    if self.config.keypoint_type == "HRpose":
                        from model_keypoint.HRpose import get_pose_net
                    elif self.config.keypoint_type == "ChainedPredictions":
                        from model_keypoint.ChainedPredictions import get_pose_net
                    elif self.config.keypoint_type == "PoseAttention":
                        from model_keypoint.PoseAttention import get_pose_net
                    elif self.config.keypoint_type == "PyraNet":
                        from model_keypoint.PyraNet import get_pose_net
                    elif self.config.keypoint_type == "RESpose":
                        from model_keypoint.RESpose import get_pose_net
                    elif self.config.keypoint_type == "StackedHourGlass":
                        from model_keypoint.StackedHourGlass import get_pose_net
                    self.encoder = get_pose_net(in_ch=1, out_ch=14)
                    if self.config.keypoint_type == "ChainedPredictions":
                        self.fuser = shift_keypoint(size=32)
                    else:
                        self.fuser = shift_keypoint()
                    self.one_step = 14
                if config.GNN_Network == "GCN":
                    self.decoder = GCN(num_features=self.num_features, num_classes=3, one_step=self.one_step)
                elif config.GNN_Network == "GAT":
                    self.decoder = GAT(num_features=self.num_features, num_classes=3, one_step=self.one_step)
                elif config.GNN_Network == "GraphSAGE":
                    self.decoder = GraphSAGE(num_features=self.num_features, num_classes=3, one_step=self.one_step)
            else:
                if self.config.use_Encoder == True:
                    self.encoder = Encoder()
                    self.fuser = Fuser()
                    self.linear = nn.Linear(128 * 32, 128 * 32)
                    self.bn = nn.BatchNorm2d(1, track_running_stats=False)
                    self.relu = nn.ReLU()
                    if self.config.detection_head =="ResNet":
                        self.decoder = ResNet(num_channels=2048)
                    elif self.config.detection_head =="DenseNet":
                        self.decoder = DenseNet()
                else:
                    if self.config.detection_head =="ResNet":
                        self.decoder = ResNet(num_channels=32768)
                    elif self.config.detection_head =="DenseNet":
                        self.decoder = DenseNet()
    def forward(self,x,config,epoch):
        if self.config.train_encoder == True:
            if epoch < self.config.unfreeze_epoch:
                with torch.no_grad():
                    keypoint, coordinate, feature = self.encoder(x)
                    feature, coordinate = self.fuser(keypoint, coordinate, feature)
            else:
                keypoint, coordinate, feature = self.encoder(x)
                feature,coordinate = self.fuser(keypoint, coordinate, feature)
            output = self.decoder(feature)
            return output,coordinate
        else:
            if self.config.use_Keypoint == True:
                keypoint, coordinate, feature = self.encoder(x)
                feature, edge_index = self.fuser((keypoint, coordinate, feature))
                output = self.decoder(feature, edge_index)
                return output
            else:
                if self.config.use_Encoder == True:
                    keypoint, coordinate, feature = self.encoder(x)
                    feature, coordinate = self.fuser(keypoint, coordinate, feature)
                    feature = self.linear(feature).reshape(-1,1,64,64)
                    feature = self.relu(feature)
                    output = self.decoder(feature)
                    return output
                else:
                    output = self.decoder(x)
                    return output

