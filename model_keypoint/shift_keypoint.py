import torch
import torch.nn as nn
import numpy as np


class shift_keypoint(nn.Module):
    def __init__(self,size=64):
        super(shift_keypoint, self).__init__()
        self.max = nn.MaxPool2d(size,return_indices=True)


    def forward(self,x):
        # 如果x的类型为tuple
        if isinstance(x,tuple):
            keypoint, coordinate, feature = x
            n, w, h, k = keypoint.size()

            feature = torch.bmm(feature.view(n, -1, w * h), keypoint.view(n, w * h, k)).permute(0, 2,1)
            # (n,3,4096)*(n,4096,k)=(n,3,k)->(n,k,3)
            value = torch.bmm(coordinate.view(n, -1, w * h), keypoint.view(n, w * h, k)).permute(0, 2,1)
            # (n,3,4096)*(n,4096,k)=(n,3,k)->(n,k,3)

            # edge_weight=feature[:,:,0:2]
            # # Reshape A to be a 3D tensor with an extra singleton dimension
            # edge_weight = edge_weight.unsqueeze(2)
            #
            # # Calculate pairwise differences using broadcasting
            # edge_weight = edge_weight - edge_weight.transpose(1, 2)
            # edge_weight = edge_weight.reshape(n,-1, 2)
            # edge_weight = edge_weight[:, 0:int(k * k), :]
            #
            # edge_weight = edge_weight*edge_weight
            # edge_weight = edge_weight.sum(dim=2).reshape(n,-1,1)


            feature = torch.cat([value,feature],dim=2)

            n, c, w = feature.size()
            feature = feature.reshape(n * c, w)
            # edge_weight = edge_weight.reshape(int(n * c*c), -1)

            #coodinate的shape为[2,128*128]，coodinate[0,i]的值为i%64，coodinate[1,i]的值为i/64
            edge_index = np.zeros((2, 128*128))
            edge_index[0,:] = (np.arange(0,128*128)%128)
            edge_index[1,:] = (np.arange(0,128*128)/128)
            # turn the element of edge_index from float to int
            edge_index = edge_index.astype(np.int)

            index = np.zeros((2, n * 128*128))
            for i in range(n):
                index[:, i * 128*128:i * 128*128 + 128*128] = edge_index + i * 128
            edge_index= torch.from_numpy(index).cuda()
            return feature, edge_index.long()
        else:
            # 如果x的类型为list
            if isinstance(x,list):
                x = x[0]
            n,c,w,h=x.size()
            value, index = self.max(x)
            x = index % w
            y = index / w

            y = torch.round(y)

            # turn the shape of x and y from [batch_size,c,1,1] to [batch_size,c]
            x = x.view(x.shape[0], x.shape[1])
            y = y.view(y.shape[0], y.shape[1])
            value = value.view(value.shape[0], value.shape[1])

            # Normalize x and y
            x = x / w
            y = y / w

            # turn the shape of x and y from [batch_size,c] to [batch_size,1,c]
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            value = value.unsqueeze(1)

            feature = torch.cat([value, x, y], dim=1).permute(0, 2, 1)
            n, c, w = feature.size()
            feature = feature.reshape(n * c, w)

            coodiante = np.array([[12, 12, 8, 7, 12, 9, 10, 2, 1, 3, 4], [13, 8, 7, 6, 9, 10, 11, 1, 0, 4, 5]])
            # turn coodinate to tensor
            coodiante = torch.from_numpy(coodiante).long()

            # a [n,2,11] numpy array
            edge_index = np.zeros((2, n * 11))
            for i in range(n):
                edge_index[:, i * 11:i * 11 + 11] = coodiante + i * 14

            return feature, torch.from_numpy(edge_index).to(torch.long).cuda()







