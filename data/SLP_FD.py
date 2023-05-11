from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import torch
import util.utils as ut


def normalization(x):
    h, w, c = x.shape
    x = np.array(x, dtype=float).reshape(c, h, w)
    mu = np.average(x)
    sigma = np.std(x)
    x = (x - mu) / sigma
    Min = np.min(x)
    Max = np.max(x)
    x = (x - Min) / (Max - Min)

    return x

def avg_threshold_pool(x,pool_size=(5,5),threshold=0.12,pad=2,stride=1):
    #如果区域内总和小于等于阀值，则输出为0
    # p_w,p_h=pool_size
    # Y=torch.zeros((X.shape[0],X.shape[1],X.shape[2]))
    # Z=torch.zeros((X.shape[0],X.shape[1]+padding*2,X.shape[2]+padding*2))
    # Z[:,2:X.shape[1]+2,2:X.shape[2]+2]=X
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         for k in range(X.shape[2]):
    #             Y[i,j,k]=Z[i,j:j+p_w,k:k+p_h].mean()
    #             if Y[i,j,k]<=threshold:
    #                 Y[i, j, k]=0
    # return Y
    c, h_in, w_in = x.shape
    k, j = pool_size
    x_pad = torch.zeros(c, h_in + 2 * pad, w_in + 2 * pad)  # 对输入进行补零操作
    if pad > 0:
        x_pad[ :, pad:-pad, pad:-pad] = x
    else:
        x_pad = x
    x_pad = x_pad.unfold(1, k, stride)
    x_pad = x_pad.unfold(2, j, stride)  # 按照滑动窗展开
    out = torch.einsum(  # 按照滑动窗相乘，
        'chwkj->chw',  # 并将所有输入通道卷积结果累加
        x_pad)
    out = out/(k*j)
    mask = torch.ge(out, threshold)
    out = out*mask
    return out



def generate_target(joints, joints_vis, sz_hm=[64, 64], sigma=3, gType='gaussian'):
    '''
	:param joints:  [num_joints, 3]
	:param joints_vis: n_jt vec     #  original n_jt x 3
	:param sigma: for gaussian gen, 3 sigma rule for effective area.  hrnet default 2.
	:return: target, target_weight(1: visible, 0: invisible),  n_jt x 1
	history: gen directly at the jt position, stride should be handled outside
	'''
    n_jt = len(joints)  #
    target_weight = np.ones((n_jt, 1), dtype=np.float32)
    # target_weight[:, 0] = joints_vis[:, 0]
    target_weight[:, 0] = joints_vis  # wt equals to vis

    assert gType == 'gaussian', \
        'Only support gaussian map now!'

    if gType == 'gaussian':
        target = np.zeros((n_jt,
                           sz_hm[1],
                           sz_hm[0]),
                          dtype=np.float32)

        tmp_size = sigma * 3

        for joint_id in range(n_jt):
            # feat_stride = self.image_size / sz_hm
            # mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            # mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            mu_x = int(joints[joint_id][0] + 0.5)  # in hm joints could be in middle,  0.5 to biased to the position.
            mu_y = int(joints[joint_id][1] + 0.5)

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= sz_hm[0] or ul[1] >= sz_hm[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], sz_hm[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], sz_hm[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], sz_hm[0])
            img_y = max(0, ul[1]), min(br[1], sz_hm[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    # print('min max', target.min(), target.max())
    # if self.use_different_joints_weight:
    # 	target_weight = np.multiply(target_weight, self.joints_weight)

    return target, target_weight


class SLP_FD(Dataset):
    # function dict for gettin function
    def __init__(self, ds, opts, phase='train', ):
        super(SLP_FD, self).__init__()
        self.phase = phase  # train data can be also used for test to generate result
        self.ds = ds  # all data in
        # dct_func_getData = {
        #     'jt_hm': self.jt_hm,
        #     'jt_hm_PrivateData': self.jt_hm_PrivateData
        # }
        # 1.Training the encoder and decoder networks on Private Data.
        # 2.Training the encoder and decoder networks on SLP.
        # 3. Training sleep pose classification on Private Data

        # prep = 'jt_hm'

        self.func_getData = self.jt_hm  # preparastioin

    def jt_hm(self, idx):
        '''
        joint heatmap format feeder.  get the img, hm(gaussian),  jts, l_std (head_size)
        :param index:
        :return:
        '''

        li_img = []
        out_shp = (64, 64, -1)
        out_shp = out_shp[:2]

        img, joints_ori, bb, id_frm = self.ds.get_array_joints(idx, mod="PM",
                                                               if_sq_bb=True)  # raw depth    # how transform
        joints_pch = joints_ori.copy()  # j
        joints_pch[:, 0] = joints_pch[:, 0] + 86
        joints_pch[:, 1] = joints_pch[:, 1] + 32  # j

        if id_frm < 15:
            sleep_pose = np.float64(0)
        elif id_frm < 30:
            sleep_pose = np.float64(1)
        else:
            sleep_pose = np.float64(2)
        img_height, img_width = img.shape[:2]  # first 2
        li_img.append(img)
        img_cb = np.concatenate(li_img, axis=-1)  # last dim, joint mods

        joints_vis = np.ones(14)  # n x 1
        for i in range(len(joints_pch)):  # only check 2d here
            # joints_ori [i, 2] = (joints_ori [i, 2] + 1.0) / 2.  # 0~1 normalize
            joints_vis[i] *= (
                    (joints_pch[i, 0] >= 0) & \
                    (joints_pch[i, 0] < 256) & \
                    (joints_pch[i, 1] >= 0) & \
                    (joints_pch[i, 1] < 256)
            )  # nice filtering  all in range visibile

        white = [0, 0, 0]
        img_patch = cv2.copyMakeBorder(img_cb, 32, 32, 86, 86, cv2.BORDER_CONSTANT, value=white)
        img_patch = img_patch[..., None]
        img_channels = 1  # add one channel

        pch_tch = normalization(img_patch)
        pch_tch = torch.from_numpy(pch_tch).float()
        stride = 256 / 64  # jt shrink
        joints_hm = joints_pch / stride
        n_jt=14
        # joints_vis = np.ones(n_jt)  # n x 1
        hms, jt_wt = generate_target(joints_hm, joints_vis, sz_hm=out_shp[::-1])  # n_gt x H X
        hms_tch = torch.from_numpy(hms)
        # hms_tch = torch.sum(hms_tch,dim=0).reshape(1,64,64)

        target = pch_tch

        #把一些床垫边缘的压力点去掉
        target = target.cpu().numpy()
        target[:,30:39,:]=0
        target[:, 220:226, :] = 0
        target[:, :, 82:92] = 0
        target[:, :, 163:171] = 0
        target = torch.from_numpy(target)


        # pool = torch.nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        # a = target.cpu().numpy()
        # target = pool(target)
        # a = target.cpu().numpy()
        input_mask = torch.ge(target, 0.0000001)
        # a = input_mask.cpu().numpy()
        target = torch.tensor(input_mask, dtype=int)

        target = avg_threshold_pool(target)
        input_mask = torch.ge(target, 0.0000001)
        target = torch.tensor(input_mask, dtype=int)
        
        pch_tch = pch_tch*target

        num_classes = 2
        seg_labels = np.eye(num_classes + 1)[target.cpu().numpy().reshape([-1])]
        seg_labels = seg_labels.reshape((256, 256, num_classes + 1))

        rst = {
            'pch': pch_tch,
            'hms': hms_tch,
            'sleep_pose': sleep_pose.astype(np.long),
            'target_mask': target,  # 蒙版图，也是要拟合的目标
            'seg_labels': seg_labels,
            'joints_vis': jt_wt,
        }
        return rst

    def __getitem__(self, index):
        # call the specific processing
        rst = self.func_getData(index)
        return rst

    def __len__(self):
        return self.ds.n_smpl

class SLP_FD_PrivateData(Dataset):
    # function dict for gettin function
    def __init__(self, ds, opts, phase='train', ):
        # super(SLP_FD_PrivateData, self).__init__()
        self.phase = phase  # train data can be also used for test to generate result
        self.ds = ds  # all data in
        # dct_func_getData = {
        #     'jt_hm': self.jt_hm,
        #     'jt_hm_PrivateData': self.jt_hm_PrivateData
        # }
        # 1.Training the encoder and decoder networks on Private Data.
        # 2.Training the encoder and decoder networks on SLP.
        # 3. Training sleep pose classification on Private Data

        prep = 'jt_hm_PrivateData'

        self.func_getData = self.jt_hm_PrivateData



    def jt_hm_PrivateData(self, idx):

        li_img = []
        img, PM_class = self.ds.get_array_joints(idx, self.phase)  # raw depth    # how transform
        PM_class = PM_class.split("_")[1]
        sleep_pose = np.float64(PM_class)
        img = img[..., None]
        li_img.append(img)
        img_cb = np.concatenate(li_img, axis=-1)  # last dim, joint mods

        white = [0, 0, 0]
        img_patch = cv2.copyMakeBorder(img_cb, 8, 8, 98, 98, cv2.BORDER_CONSTANT, value=white)
        img_patch = img_patch[..., None]

        pch_tch = torch.from_numpy(normalization(img_patch)).float()
        rst = {
            'pch': pch_tch,
            'sleep_pose': sleep_pose.astype(np.long),
        }

        return rst

    def __getitem__(self, index):
        # call the specific processing
        rst = self.func_getData(index)
        return rst

    def __len__(self):
        if self.phase == "train":
            return self.ds.train_len
        else:
            return self.ds.val_len


