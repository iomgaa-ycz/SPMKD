import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GlobalAttention
from torch_geometric.nn import BatchNorm
import torch.nn as nn

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes,one_step=14):
        super(GCN, self).__init__()
        self.one_step=one_step
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 64)
        self.conv4 = GCNConv(64, 32)

        self.bn1 = BatchNorm(128)
        self.bn2 = BatchNorm(256)
        self.bn3 = BatchNorm(64)
        self.bn4 = BatchNorm(32)


        self.fc = torch.nn.Linear(32, num_classes)
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU() , nn.Linear(16, 1), nn.BatchNorm1d(1), nn.ReLU()))
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        batch_size = x.shape[0]/self.one_step
        # batch is a tensor of shape [batch_size*14] containing the batch index of each node.
        batch = torch.cat([torch.full((self.one_step,), i, dtype=torch.long) for i in range(int(batch_size))])


        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = self.pool(x, batch.cuda())
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x