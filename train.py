import os
import gc
import torch
import wandb
import collections
import numpy as np
import torch.nn as nn
from util.utils import init_weights
from models.SPMKD import SPMKD
from torch.nn import MSELoss,L1Loss,CrossEntropyLoss
from pytorch_msssim import SSIM
from loss import CE_Loss,f_score
from data.SLP_FD import SLP_FD
from data.SLP_RD import SLP_RD_SLP
from data.SMaL import SMaL
from torch.utils.data import DataLoader
import os.path as osp
from util.logger import Colorlogger
from thop import profile,clever_format
from one_epoch import one_epoch
from util.utils import print_options,make_folder



def train(config=None):


    config = wandb.config if config is None else config

    if config.use_Keypoint ==True:
        config.save_dir = osp.join(config.output_dir, '{}_{}_{}_{}'.format(config.keypoint_type,config.GNN_Network, config.lr,config.train_test_rate))
    elif config.use_Encoder == True:
        config.save_dir = osp.join(config.output_dir, '{}_Encoder_{}_{}'.format(config.detection_head, config.lr,config.train_test_rate))
    else:
        config.save_dir = osp.join(config.output_dir,
                                   '{}_{}_{}'.format(config.detection_head, config.lr, config.train_test_rate))
    log_dir = osp.join(config.save_dir, 'log')
    logger = Colorlogger(log_dir, '{}_logs.txt'.format(config.status))  # 生成一个logger，这是一个用于保存各种参数与信息的钩子
    config.model_dir = osp.join(config.save_dir, "models")  # 保存模型的路径
    print_options(config, if_sv=True,save_path='opts.txt')  # 把程序用到的所有保存在opt中的参数打印出来
    make_folder(config.model_dir)  # 创建文件夹


    model = SPMKD(config)  # 生成模型
    init_weights(model)

    criterion = {}
    if config.train_encoder == True and config.Mask == False:
        criterion["criterion_l1"] = L1Loss(reduction="mean").cuda()  # l1损失
        criterion["criterion_l2"] = MSELoss(reduction="mean").cuda()  # l2损失
        criterion["criterion_ssim"] = SSIM(data_range=1, size_average=True, channel=1)  # ssim损失

    elif config.train_encoder == True and config.Mask == True:
        criterion["criterion_mask"] = CE_Loss  # 蒙版分类损失
        criterion["f_score"] = f_score  # f_score损失
    else:
        criterion["criterion_sleep"] = CrossEntropyLoss()  # 睡姿分类的分类损失

    #data loader
    if config.data_type=="SLP":
        SLP_rd_test = SLP_RD_SLP(config, phase='test')  # all test data
        SLP_fd_test = SLP_FD(SLP_rd_test, config, phase='test')
        test_loader = DataLoader(dataset=SLP_fd_test, batch_size=config.batch_size,
                                shuffle=True, num_workers=1)
    elif config.data_type=="SMaL":
        SMaL_test = SMaL(config, phase='test')  # all test data
        test_loader = DataLoader(dataset=SMaL_test, batch_size=config.batch_size,
                                    shuffle=True, num_workers=1)

    checkpoint_file = config.pretrain

    if config.use_Keypoint == True and config.data_type=="SMaL":
        checkpoint_file = checkpoint_file.replace("PreTrain_Map", "{}_{}".format(config.keypoint_type, config.GNN_Network))
    # 如果use_Keypoint为True，那么将checkpoint_file中的PreTrain_Map替换为config.keypoint_type
    elif config.use_Keypoint == True and config.GNN_Network != "Ours":
        checkpoint_file = checkpoint_file.replace("PreTrain_Map", config.keypoint_type)

    if os.path.exists(checkpoint_file):  # from scratch
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        device = torch.device(config.device)
        model_dict = model.state_dict()
        checkpoint = torch.load(checkpoint_file, map_location=device)
        # checkpoint_epoch = checkpoint['epoch']
        checkpoint2 = collections.OrderedDict()
        for k, v in checkpoint.items():
            try:
                if k.startswith("module."):
                    name = k.replace("module.", "", 1)
                else:
                    name = k
                if np.shape(model_dict[name]) == np.shape(v):
                    checkpoint2[name] = v
            except:
                continue
        checkpoint = checkpoint2
        diff_keys = set(checkpoint.keys()) ^ set(model_dict.keys())

        # 输出差集
        print(diff_keys)
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict, strict=False)
        del checkpoint,checkpoint2,model_dict
        gc.collect()

        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, 0))

    model = model.cuda()
    input = torch.randn(2, 1, 256, 256).cuda()
    flops, params = profile(model, inputs=(input, config, 100))
    flops, params = clever_format([flops / 2, params / 2], "%.3f")
    logger.info("flops = {}, params = {}".format(
        flops, params))

    model = nn.DataParallel(model).cuda()


    n_iter = -1


    best_acc = 0
    for epoch in range(config.begin_epoch, config.end_epoch):
        rst_test=one_epoch(config=config, model=model, criterion=criterion, test_loader=test_loader, epoch=epoch, logger=logger, n_iter=n_iter)

