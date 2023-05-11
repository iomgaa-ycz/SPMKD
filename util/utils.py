import json
import argparse
import os
import torch.nn as nn
import numpy as np

import torch
import numpy as np
from PIL import Image
import os
import re
import sys
import numpy as np
from PIL import Image
import os.path as osp
import cv2
import matplotlib.pyplot as plt
from skimage import io, transform, img_as_ubyte
import math
import random
import time
from scipy.ndimage.filters import gaussian_filter
import torch.nn as nn

MAT_WIDTH = 0.762  # metres
MAT_HEIGHT = 1.854  # metres
MAT_HALF_WIDTH = MAT_WIDTH / 2
NUMOFTAXELS_X = 64  # 73 #taxels
NUMOFTAXELS_Y = 27  # 30
INTER_SENSOR_DISTANCE = 0.0286  # metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)

# txtfile = open("../FILEPATH.txt")
# FILEPATH = txtfile.read().replace("\n", "")
# txtfile.close()
FILEPATH = "../PresurePose"

try:
    import cPickle as pkl


    def load_pickle(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)
except:
    import pickle as pkl


    def load_pickle(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f, encoding='latin1')



#load the json file from ./config.json
def load_config(path='./config.json'):
    with open(path) as f:
        config = json.load(f)
    # turn dict to Namespace
    config = argparse.Namespace(**config)
    return config

#Print and save options
def print_options(opt, if_sv=False, save_path=""):
    """Print and save options

	It will print both current options and default values(if different).
	It will save options into a text file / [checkpoints_dir] / opt.txt
	"""
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        # 	comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    make_folder(opt.output_dir)  # ./output

    make_folder(opt.save_dir)

    # save to the disk
    if if_sv:
        file_name = os.path.join(opt.save_dir, save_path)  #'opts.txt'
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

#make folder
def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def init_weights(self, pretrained=''):
    # logger.info('=> init weights from normal distribution')
    for m in self.modules():
        if isinstance(m, (nn.Conv2d,nn.Conv1d,nn.Linear)):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.normal_(m.weight, std=0.001)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d,nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)

class TensorPrepLib():

    def load_files_to_database(self, database_file, creation_type, verbose=False, reduce_data=False, depth_in=False,
                               test=False):

        # load in the training or testing files.  This may take a while.
        dat = None
        for ss_idx in range(len(database_file[0])):

            some_subject = database_file[0][ss_idx]

            if creation_type in some_subject:
                print(creation_type, some_subject)
                dat_curr = load_pickle(some_subject)
                if depth_in == True:
                    dat_depth_curr = load_pickle(database_file[1][ss_idx])
                    print('     ', database_file[1][ss_idx])
                    for key in dat_depth_curr:
                        dat_curr[key] = dat_depth_curr[key]

                # print (some_subject, dat_curr['bed_angle_deg'][0])
                for key in dat_curr:
                    if np.array(dat_curr[key]).shape[0] != 0:
                        for inputgoalset in np.arange(len(dat_curr['images'])):

                            if math.isnan(np.sum(dat_curr['images'][inputgoalset])):
                                continue
                            else:

                                datcurr_to_append = dat_curr[key][inputgoalset]
                                try:
                                    if test == False:
                                        if reduce_data == True:
                                            if inputgoalset < len(dat_curr['images']) / 4:
                                                dat[key].append(datcurr_to_append)
                                        else:
                                            dat[key].append(datcurr_to_append)
                                    else:
                                        if len(dat_curr['images']) == 3000:
                                            if inputgoalset < len(dat_curr['images']) / 2:
                                                dat[key].append(datcurr_to_append)
                                        elif len(dat_curr['images']) == 1500:
                                            if inputgoalset < len(dat_curr['images']) / 3:
                                                dat[key].append(datcurr_to_append)
                                        else:
                                            dat[key].append(datcurr_to_append)

                                except:
                                    try:
                                        dat[key] = []
                                        dat[key].append(datcurr_to_append)
                                    except:
                                        dat = {}
                                        dat[key] = []
                                        dat[key].append(datcurr_to_append)
            else:
                pass

        if dat is not None and verbose == True:
            for key in dat:
                print('all data keys and shape', key, np.array(dat[key]).shape)
        return dat

    def prep_images(self, x, dat_f_real, dat_m_real, dat_f_synth, dat_m_synth, filter_sigma, start_map_idx):
        im_ct = 0
        for dat in [dat_f_real, dat_m_real]:
            if dat is not None:
                for entry in range(len(dat['images'])):
                    x[im_ct, start_map_idx, :, :] = gaussian_filter(
                        dat['images'][entry].reshape(64, 27).astype(np.float32),
                        sigma=0)  # the real ones are already normalized by mass
                    im_ct += 1

        if dat_f_synth is not None:
            for entry in range(len(dat_f_synth['images'])):
                pimg = gaussian_filter(dat_f_synth['images'][entry].reshape(64, 27).astype(np.float32),
                                       sigma=filter_sigma)
                pimg_mass = dat_f_synth['body_volume'][entry] * 62.5 / 0.06878933937454557
                x[im_ct, start_map_idx, :, :] = pimg * ((pimg_mass * 9.81) / (np.sum(pimg) * 0.0264 * 0.0286)) * (
                        1 / 133.322)  # normalize by body mass and convert to mmHg
                im_ct += 1
        if dat_m_synth is not None:
            for entry in range(len(dat_m_synth['images'])):
                pimg = gaussian_filter(dat_m_synth['images'][entry].reshape(64, 27).astype(np.float32),
                                       sigma=filter_sigma)
                pimg_mass = dat_m_synth['body_volume'][entry] * 78.4 / 0.0828308574658067
                x[im_ct, start_map_idx, :, :] = pimg * ((pimg_mass * 9.81) / (np.sum(pimg) * 0.0264 * 0.0286)) * (
                        1 / 133.322)  # normalize by body mass and convert to mmHg
                im_ct += 1
        return x

    def prep_reconstruction_gt(self, x, dat_f_real, dat_m_real, dat_f_synth, dat_m_synth, start_map_idx):
        im_ct = 0
        for dat in [dat_f_real, dat_m_real, dat_f_synth, dat_m_synth]:
            if dat is not None:
                for entry in range(len(dat['images'])):
                    # mesh_reconstruction_maps.append([dat['mesh_depth'][entry], dat['mesh_contact'][entry]*100, ])
                    x[im_ct, start_map_idx, :, :] = dat['mesh_depth'][entry].astype(np.float32)
                    x[im_ct, start_map_idx + 1, :, :] = dat['mesh_contact'][entry].astype(np.float32) * 100
                    im_ct += 1
        return x

    def prep_reconstruction_input_est(self, x, dat_f_real, dat_m_real, dat_f_synth, dat_m_synth, start_map_idx,
                                      cnn_type='resnet'):
        im_ct = 0
        for dat in [dat_f_real, dat_m_real, dat_f_synth, dat_m_synth]:
            if dat is not None:
                for entry in range(len(dat['images'])):
                    if cnn_type == 'resnet':
                        mdm_est_neg = np.copy(dat['mdm_est'][entry])
                        mdm_est_neg[mdm_est_neg > 0] = 0
                        mdm_est_neg *= -1
                        # reconstruction_input_est_list.append([mdm_est_neg, dat['cm_est'][entry]*100, ])
                        x[im_ct, start_map_idx, :, :] = mdm_est_neg.astype(np.float32)
                        x[im_ct, start_map_idx + 1, :, :] = dat['cm_est'][entry].astype(np.float32) * 100
                    elif cnn_type == 'resnetunet':
                        x[im_ct, start_map_idx, :, :] = dat['pimg_est'][entry].astype(np.float32)
                    im_ct += 1
        return x

    def prep_depth_input_images(self, x, dat_f_real, dat_m_real, dat_f_synth, dat_m_synth, start_map_idx,
                                depth_type='all_meshes', mix_bl_nobl=False, im_ct=0):
        ct = 0
        for dat in [dat_f_real, dat_m_real, dat_f_synth, dat_m_synth]:
            if dat is not None:
                for entry in range(len(dat['images'])):
                    ct += 1
                    if depth_type == 'all_meshes':
                        # print(ct, "adding blanket")
                        x[im_ct, start_map_idx + 0, :, :] = dat['overhead_depthcam'][entry][0:64, 0:27].astype(
                            np.float32)
                        x[im_ct, start_map_idx + 1, :, :] = dat['overhead_depthcam'][entry][0:64, 27:54].astype(
                            np.float32)
                        x[im_ct, start_map_idx + 2, :, :] = dat['overhead_depthcam'][entry][64:128, 0:27].astype(
                            np.float32)
                        x[im_ct, start_map_idx + 3, :, :] = dat['overhead_depthcam'][entry][64:128, 27:54].astype(
                            np.float32)
                        # print('appending blanketed')
                        if mix_bl_nobl == True and ct >= 2:  # with this we append 2/3 blanketed data and 1/3 without blanket
                            depth_type = 'no_blanket'
                            ct = 0
                    elif depth_type == 'no_blanket':
                        # print(ct, "no blanket")
                        x[im_ct, start_map_idx + 0, :, :] = dat['overhead_depthcam_noblanket'][entry][0:64,
                                                            0:27].astype(np.float32)
                        x[im_ct, start_map_idx + 1, :, :] = dat['overhead_depthcam_noblanket'][entry][0:64,
                                                            27:54].astype(np.float32)
                        x[im_ct, start_map_idx + 2, :, :] = dat['overhead_depthcam_noblanket'][entry][64:128,
                                                            0:27].astype(np.float32)
                        x[im_ct, start_map_idx + 3, :, :] = dat['overhead_depthcam_noblanket'][entry][64:128,
                                                            27:54].astype(np.float32)
                        # print('appending noblanket')
                        if mix_bl_nobl == True:
                            depth_type = 'all_meshes'
                            ct = 0
                    elif depth_type == 'human_only':
                        # x[im_ct, start_map_idx+0, :, :] = zoom(dat['overhead_depthcam_onlyhuman'][entry], 0.5).astype(np.float32)
                        x[im_ct, start_map_idx + 0, :, :] = dat['overhead_depthcam_onlyhuman'][entry][0:64,
                                                            0:27].astype(np.float32)
                        x[im_ct, start_map_idx + 1, :, :] = dat['overhead_depthcam_onlyhuman'][entry][0:64,
                                                            27:54].astype(np.float32)
                        x[im_ct, start_map_idx + 2, :, :] = dat['overhead_depthcam_onlyhuman'][entry][64:128,
                                                            0:27].astype(np.float32)
                        x[im_ct, start_map_idx + 3, :, :] = dat['overhead_depthcam_onlyhuman'][entry][64:128,
                                                            27:54].astype(np.float32)
                    elif depth_type == 'blanket_only':
                        x[im_ct, start_map_idx + 0, :, :] = dat['overhead_depthcam_onlyblanket'][entry][0:64,
                                                            0:27].astype(np.float32)
                        x[im_ct, start_map_idx + 1, :, :] = dat['overhead_depthcam_onlyblanket'][entry][0:64,
                                                            27:54].astype(np.float32)
                        x[im_ct, start_map_idx + 2, :, :] = dat['overhead_depthcam_onlyblanket'][entry][64:128,
                                                            0:27].astype(np.float32)
                        x[im_ct, start_map_idx + 3, :, :] = dat['overhead_depthcam_onlyblanket'][entry][64:128,
                                                            27:54].astype(np.float32)
                    elif depth_type == 'unet_est':
                        x[im_ct, start_map_idx + 0, :, :] = dat['dimg_est'][entry][0:64, 0:27].astype(np.float32)
                        x[im_ct, start_map_idx + 1, :, :] = dat['dimg_est'][entry][0:64, 27:54].astype(np.float32)
                        x[im_ct, start_map_idx + 2, :, :] = dat['dimg_est'][entry][64:128, 0:27].astype(np.float32)
                        x[im_ct, start_map_idx + 3, :, :] = dat['dimg_est'][entry][64:128, 27:54].astype(np.float32)

                    im_ct += 1

        # print ("depth array shape: ", np.shape(depth_images), np.max(depth_images))
        return x

    def append_trainxa_besides_pmat_edges(self, train_xa, CTRL_PNL, mesh_reconstruction_maps_input_est=None,
                                          mesh_reconstruction_maps=None, depth_images=None, depth_images_out_unet=None):
        train_xa[train_xa > 0] += 1.
        train_xa = train_xa.astype(np.float32)  # .astype(np.int16)

        print(np.shape(train_xa), 'shape before appending pmat contact input')
        if CTRL_PNL['recon_map_input_est'] == True:
            train_xa = np.concatenate((mesh_reconstruction_maps_input_est, train_xa), axis=1)

        print(np.shape(train_xa), 'shape before appending recon gt maps')
        if CTRL_PNL['recon_map_labels'] == True:
            mesh_reconstruction_maps = np.array(mesh_reconstruction_maps).astype(np.float32)  # GROUND TRUTH
            train_xa = np.concatenate((train_xa, mesh_reconstruction_maps), axis=1)

        print(np.shape(train_xa), 'shape before appending input depth images. we split a single into 4')
        if CTRL_PNL['depth_in'] == True:
            train_xa = train_xa.astype(np.float32)

            train_xa = np.concatenate((train_xa, depth_images[:, :, 0:64, 0:27]), axis=1)
            train_xa = np.concatenate((train_xa, depth_images[:, :, 0:64, 27:54]), axis=1)
            train_xa = np.concatenate((train_xa, depth_images[:, :, 64:128, 0:27]), axis=1)
            train_xa = np.concatenate((train_xa, depth_images[:, :, 64:128, 27:54]), axis=1)

        print(np.shape(train_xa),
              'shape before appending input depth output unet with no blanket. we split a single into 4')
        if CTRL_PNL['depth_out_unet'] == True:
            print(np.max(depth_images_out_unet), 'max depth ims')  # , np.std(depth_images_out_unet)

            train_xa = np.concatenate((train_xa, depth_images_out_unet[:, :, 0:64, 0:27]), axis=1)
            train_xa = np.concatenate((train_xa, depth_images_out_unet[:, :, 0:64, 27:54]), axis=1)
            train_xa = np.concatenate((train_xa, depth_images_out_unet[:, :, 64:128, 0:27]), axis=1)
            train_xa = np.concatenate((train_xa, depth_images_out_unet[:, :, 64:128, 27:54]), axis=1)

        print("TRAIN XA SHAPE", np.shape(train_xa))
        return train_xa

    def prep_labels(self, y_flat, dat, z_adj, gender, is_synth, loss_vector_type, initial_angle_est, cnn_type='resnet',
                    x_y_adjust_mm=[0, 0]):
        if gender == "f":
            g1 = 1
            g2 = 0
        elif gender == "m":
            g1 = 0
            g2 = 1
        if is_synth == True:
            s1 = 1
        else:
            s1 = 0
        z_adj_all = np.array(24 * [-x_y_adjust_mm[0], x_y_adjust_mm[1], z_adj * 1000])
        z_adj_one = np.array(1 * [-286. - x_y_adjust_mm[0], -286. + x_y_adjust_mm[1], z_adj * 1000])

        if dat is not None:
            for entry in range(len(dat['markers_xyz_m'])):
                gt_markers = np.array(dat['markers_xyz_m'][entry][0:72]).reshape(24, 3)
                gt_markers[:, 0:2] -= 0.286
                gt_markers = gt_markers.reshape(72)

                if gender == "f":
                    c = np.concatenate((gt_markers * 1000 + z_adj_all,
                                        dat['body_shape'][entry][0:10],
                                        dat['joint_angles'][entry][0:72],
                                        dat['root_xyz_shift'][entry][0:3] + z_adj_one / 1000.,
                                        [g1], [g2], [s1],
                                        [dat['body_volume'][entry] * 62.5 / 0.06878933937454557],
                                        [dat['body_height'][entry]],),
                                       axis=0)  # [x1], [x2], [x3]: female synth: 1, 0, 1.
                elif gender == "m":
                    c = np.concatenate((gt_markers * 1000 + z_adj_all,
                                        dat['body_shape'][entry][0:10],
                                        dat['joint_angles'][entry][0:72],
                                        dat['root_xyz_shift'][entry][0:3] + z_adj_one / 1000.,
                                        [g1], [g2], [s1],
                                        [dat['body_volume'][entry] * 78.4 / 0.0828308574658067],
                                        [dat['body_height'][entry]],),
                                       axis=0)  # [x1], [x2], [x3]: female synth: 1, 0, 1.

                if initial_angle_est == True:
                    c = np.concatenate((c,
                                        dat['betas_est'][entry][0:10],
                                        dat['angles_est'][entry][0:72],
                                        dat['root_xyz_est'][entry][0:3],
                                        dat['root_atan2_est'][entry][0:6]), axis=0)
                    if cnn_type == 'resnetunet':
                        c = np.concatenate((c, [0]), axis=0)
                    else:
                        c = np.concatenate((c, dat['bed_vertical_shift_est'][entry][0:1]), axis=0)

                y_flat.append(c)

        return y_flat

    def normalize_network_input(self, x, CTRL_PNL):

        for i in range(8):
            print(np.mean(x[:, i, :, :]), np.max(x[:, i, :, :]))

        if CTRL_PNL['recon_map_input_est'] == True:
            normalizing_std_constants = CTRL_PNL['norm_std_coeffs']

            if CTRL_PNL['cal_noise'] == True: normalizing_std_constants = normalizing_std_constants[
                                                                          1:]  # here we don't precompute the contact

            for i in range(x.shape[1]):
                x[:, i, :, :] *= normalizing_std_constants[i]

        else:
            normalizing_std_constants = []
            normalizing_std_constants.append(CTRL_PNL['norm_std_coeffs'][0])
            normalizing_std_constants.append(CTRL_PNL['norm_std_coeffs'][3])
            normalizing_std_constants.append(CTRL_PNL['norm_std_coeffs'][4])
            normalizing_std_constants.append(CTRL_PNL['norm_std_coeffs'][5])
            normalizing_std_constants.append(CTRL_PNL['norm_std_coeffs'][6])

            if CTRL_PNL['cal_noise'] == True: normalizing_std_constants = normalizing_std_constants[
                                                                          1:]  # here we don't precompute the contact

            for i in range(x.shape[1]):
                print("normalizing idx", i)
                x[:, i, :, :] *= normalizing_std_constants[i]

        for i in range(8):
            print(np.mean(x[:, i, :, :]), np.max(x[:, i, :, :]))
        return x

    def normalize_wt_ht(self, y, CTRL_PNL):
        # normalizing_std_constants = [1./30.216647403349857,
        #                             1./14.629298141231091]

        y = np.array(y)

        # y[:, 160] *= normalizing_std_constants[0]
        # y[:, 161] *= normalizing_std_constants[1]
        y[:, 160] *= CTRL_PNL['norm_std_coeffs'][7]
        y[:, 161] *= CTRL_PNL['norm_std_coeffs'][8]

        return y

class FileNameInputLib():
    def __init__(self, opt, FILEPATH, depth=False):
        # if opt.hd == False:
        #     self.data_fp_prefix = FILEPATH + "data_BP/"
        # else:
        #     self.data_fp_prefix = "/media/henry/git/multimodal_data_2/data_BR/"
        self.data_fp_prefix = FILEPATH

        self.data_fp_suffix = ''

def nameToIdx(name_tuple, joints_name):  # test, tp,
    '''
	from reference joints_name, change current name list into index form
	:param name_tuple:  The query tuple like tuple(int,) or tuple(tuple(int, ...), )
	:param joints_name:
	:return:
	'''
    jtNm = joints_name
    if type(name_tuple[0]) == tuple:
        # Transer name_tuple to idx
        return tuple(tuple([jtNm.index(tpl[0]), jtNm.index(tpl[1])]) for tpl in name_tuple)
    else:
        # direct transfer
        return tuple(jtNm.index(tpl) for tpl in name_tuple)

def get_bbox(joint_img, rt_margin=1.2, rt_xy=0):
    '''
	get the bounding box from joint gt min max, with a margin ratio.
	:param joint_img:
	:param rt_margin:
	:param rt_xy:   the ratio of x/y . 0 for original bb size. most times 1 for square input patch. Can be gotten from sz_pch[0]/sz_pch[1].
	:return:
	'''
    bb = np.zeros((4))
    xmin = np.min(joint_img[:, 0])
    ymin = np.min(joint_img[:, 1])
    xmax = np.max(joint_img[:, 0])
    ymax = np.max(joint_img[:, 1])
    width = xmax - xmin - 1
    height = ymax - ymin - 1
    if rt_xy:
        c_x = (xmin + xmax) / 2.
        c_y = (ymin + ymax) / 2.
        aspect_ratio = rt_xy
        w = width
        h = height
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        bb[2] = w * rt_margin
        bb[3] = h * rt_margin
        bb[0] = c_x - bb[2] / 2.
        bb[1] = c_y - bb[3] / 2.
    else:
        bb[0] = (xmin + xmax) / 2. - width / 2 * rt_margin
        bb[1] = (ymin + ymax) / 2. - height / 2 * rt_margin
        bb[2] = width * rt_margin
        bb[3] = height * rt_margin

    return bb


def get_max_preds(batch_heatmaps):
    '''
	get predictions from score maps
	heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
	:return preds [N x n_jt x 2]
	'''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)  # amax, array max

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)  # clean up if too low confidence (maxvals)

    preds *= pred_mask
    return preds, maxvals


def calc_dists(preds, target, normalize):
    # normalized distance on x, y seperately,  anyway
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1
	dist has already been normalized
	normalized is simply based on (h,w)/10 std
	'''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

class AverageMeter(object):
    """Computes and stores the average and current value, similar to timer"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def get_pretrain_path(config):
    if config.train_encoder == True:
        if config.Mask == True:
            return "./weights/SPMKD_Mask/checkpoint.pth"
        else:
            return "./weights/SPMKD_Map/checkpoint.pth"
    else:
        if config.use_Keypoint == True:
            if config.keypoint_type == "Ours":
                if config.GNN_Network == "GCN":
                    return "./weights/SPMKD_GCN_{}/checkpoint.pth".format(config.data_type)
                elif config.GNN_Network == "GAT":
                    return "./weights/SPMKD_GAT_{}/checkpoint.pth".format(config.data_type)
                elif config.GNN_Network == "GraphSAGE":
                    return "./weights/SPMKD_GraphSAGE_{}/checkpoint.pth".format(config.data_type)
            else:
                return "./weights/{}_{}_{}/checkpoint.pth".format(config.keypoint_type,config.GNN_Network,config.data_type)
        else:
            if config.use_Encoder == True:
                return "./weights/SPMKD_{}/checkpoint.pth".format(config.detection_head)
            else:
                return "./weights/{}/checkpoint.pth".format(config.detection_head)


def get_dataset_path(is_SMaL,config):
    if is_SMaL == False:
        config.data_path = "../SLP/danaLab/"
        config.data_type = "SLP"
    else:
        config.data_path = "../SMaL/"
        config.data_type = "SMaL"