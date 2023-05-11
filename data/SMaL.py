import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

def normalization(x):
    h, w = x.shape
    x = np.array(x, dtype=float).reshape(h, w)
    mu = np.average(x)
    sigma = np.std(x)
    x = (x - mu) / sigma
    Min = np.min(x)
    Max = np.max(x)
    x = (x - Min) / (Max - Min)

    return x



# load the PM.npy and label.npy
class SMaL(Dataset):
    def __init__(self, config, phase='train'):
        self.config = config
        if phase == 'train':
            self.PM = np.load(os.path.join(self.config.data_path, "PM_train.npy"))
            self.label = np.load(os.path.join(self.config.data_path, "label_train.npy"))
        else:
            self.PM = np.load(os.path.join(self.config.data_path, "PM_test.npy"))
            self.label = np.load(os.path.join(self.config.data_path, "label_test.npy"))

    def __getitem__(self, index):
        PM = self.PM[index]
        label = self.label[index].squeeze(-1)

        white = [0, 0, 0]
        PM = cv2.copyMakeBorder(PM, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=white)
        PM = normalization(PM)
        PM = torch.from_numpy(PM).float()
        # turn the size of PM to (1, 256, 256)
        PM = PM.unsqueeze(0)

        rst = {
            'pch': PM,
            'sleep_pose': label.astype(np.long),
        }
        return rst

    def __len__(self):
        return len(self.PM)