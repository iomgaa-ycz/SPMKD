import util.utils as ut
import time
import torch
import numpy as np
import torch.nn.functional as F

def one_epoch(config=None,model=None,test_loader=None,criterion=None,epoch=None,logger=None,n_iter=-1):
    # #test
    if config.train_encoder== True:
        if config.Mask == True:
            batch_time = ut.AverageMeter()
            data_time = ut.AverageMeter()
            losses_mask = ut.AverageMeter()
            Score_numbers = ut.AverageMeter()

            best_Mask = None # 保存预测到最好的mask
            best_target_Mask = None # 保存最好的mask
            best_keypoint = None # 保存最好的关键点
            best_metric = 0 # 当前最好的指标
            Mask_all = None
            Target_Mask_all = None
            Keypoint_all = None

        elif config.Mask == False:
            batch_time = ut.AverageMeter()
            data_time = ut.AverageMeter()
            losses_l1 = ut.AverageMeter()
            losses_l2 = ut.AverageMeter()
            losses_ssim = ut.AverageMeter()

            best_metric = 0 # 当前最好的指标
            best_input = None # 保存最好的input
            best_rebuild = None # 保存最好的重建图
            best_keypoint = None # 保存最好的关键点

            Mask_all = None
            Target_Mask_all = None
            Keypoint_all = None
    else:
        batch_time = ut.AverageMeter()
        data_time = ut.AverageMeter()
        losses_sleep = ut.AverageMeter()
        Acces = ut.AverageMeter()

    end = time.time()
    data_time.update(time.time() - end)

    with torch.no_grad():
        for i, inp_dct in enumerate(test_loader):
            # get items
            if i >= n_iter and n_iter > 0:  # break if iter is set and i is greater than that
                break

            if config.train_encoder == True:
                if config.Mask == True:
                    input = inp_dct['pch'].cuda()  # 压力图
                    target_mask = inp_dct['target_mask']  # 蒙版图
                    seg_labels = inp_dct['seg_labels']

                    data_time.update(time.time() - end)

                    output_mask, coordinate = model(input, config, epoch)

                    input = input.cuda()
                    target_mask = target_mask.cuda(non_blocking=True)
                    seg_labels = seg_labels.cuda(non_blocking=True)

                    loss_mask = criterion["criterion_mask"](output_mask, target_mask)
                    score_class = criterion["f_score"](output_mask, seg_labels)

                    losses_mask.update(loss_mask.item(), input.size(0))
                    Score_numbers.update(score_class, input.size(0))

                    the_metric = score_class
                    if the_metric > best_metric:
                        best_metric = the_metric

                        Mask_all = output_mask.cpu().detach().numpy()
                        Target_Mask_all = target_mask.cpu().detach().numpy()
                        Keypoint_all = coordinate.cpu().detach().numpy()

                        best_Mask = output_mask[0, :, :, :]
                        best_Mask = torch.ge(best_Mask[1, :, :], best_Mask[0, :, :]).int()
                        best_target_mask = target_mask[0, :, :, :]
                        target_mask_max = torch.max(best_target_mask)
                        best_Mask = (best_Mask / target_mask_max).cpu().detach().numpy()
                        best_target_mask = (best_target_mask / target_mask_max).cpu().detach().numpy()
                        best_keypoint = coordinate[0, :, :].cpu().detach().numpy()

                        # keypoint is a numpy, which size is (64,64)
                        keypoint = np.zeros((64, 64))
                        for i in range(64):
                            keypoint[int(best_keypoint[i, 1]*64), int(best_keypoint[i, 2]*64)] = 1
                        best_keypoint = keypoint

                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % config.print_freq == 0:
                        msg = 'Epoch: [{0}][{1}/{2}]\t' \
                              'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                              'Speed {speed:.1f} samples/s\t' \
                              'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                              'Loss_mask {loss_mask.val:.5f} ({loss_mask.avg:.5f})\t' \
                              'Acc {acc.val:.5f} ({acc.avg:.5f})\t' \
                            .format(
                            epoch, i, len(test_loader)-1, batch_time=batch_time,
                            speed=input.size(0) / batch_time.val,
                            data_time=data_time, loss_mask=losses_mask,
                            acc=Score_numbers)
                        logger.info(msg)
                else:
                    input = inp_dct['pch'].cuda()  # 压力图

                    data_time.update(time.time() - end)

                    output_map,coordinate = model(input, config, epoch)

                    input = input.to(config.device)

                    loss_l1 = criterion["criterion_l1"](output_map, input)
                    loss_l2 = criterion["criterion_l2"](output_map, input)
                    loss_ssim = 1 - criterion["criterion_ssim"](output_map, input)

                    losses_l1.update(loss_l1.item(), input.size(0))
                    losses_l2.update(loss_l2.item(), input.size(0))
                    losses_ssim.update(loss_ssim.item(), input.size(0))

                    the_metric = 1 / (loss_l1.item() * 10 + loss_l2.item() * 100 + loss_ssim.item())
                    if the_metric > best_metric:
                        best_metric = the_metric

                        Mask_all = output_map.cpu().detach().numpy()
                        Target_Mask_all = input.cpu().detach().numpy()
                        Keypoint_all = coordinate.cpu().detach().numpy()

                        best_input = input[0, :, :, :].cpu().detach().numpy()
                        best_rebuild = output_map[0, :, :, :].cpu().detach().numpy()
                        best_keypoint = coordinate[0, :, :].cpu().detach().numpy()  # (n,k,2)

                        # keypoint is a numpy, which size is (64,64)
                        keypoint = np.zeros((64, 64))
                        for i in range(64):
                            keypoint[int(best_keypoint[i, 1] * 64), int(best_keypoint[i, 2] * 64)] = 1
                        best_keypoint = keypoint

                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % config.print_freq == 0:
                        msg = 'Epoch: [{0}][{1}/{2}]\t' \
                              'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                              'Speed {speed:.1f} samples/s\t' \
                              'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                              'Loss_l1 {loss_l1.val:.5f} ({loss_l1.avg:.5f})\t' \
                              'Loss_l2 {loss_l2.val:.5f} ({loss_l2.avg:.5f})\t' \
                              'Loss_ssim {loss_ssim.val:.5f} ({loss_ssim.avg:.5f})\t' \
                            .format(
                            epoch, i, len(test_loader)-1, batch_time=batch_time,
                            speed=input.size(0) / batch_time.val,
                            data_time=data_time, loss_l1=losses_l1, loss_l2=losses_l2, loss_ssim=losses_ssim,)
                        logger.info(msg)

            else:
                input = inp_dct['pch'].cuda()  # 压力图
                sleep_target = inp_dct['sleep_pose']  # 睡姿分类

                data_time.update(time.time() - end)

                output_sleep = model(input, config, epoch)

                input = input.to(config.device)
                sleep_target = sleep_target.to(config.device)

                loss_sleep = criterion["criterion_sleep"](output_sleep, sleep_target)
                acc_sleep = (output_sleep.argmax(dim=1) == sleep_target).sum().cpu().item() / input.size(0)

                losses_sleep.update(loss_sleep.item(), input.size(0))
                Acces.update(acc_sleep, input.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.print_freq == 0:
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Loss_sleep {loss_sleep.val:.5f} ({loss_sleep.avg:.5f})\t' \
                          'Acc {acc.val:.5f} ({acc.avg:.5f})\t' \
                        .format(
                        epoch, i, len(test_loader)-1, batch_time=batch_time,
                        speed=input.size(0) / batch_time.val,
                        data_time=data_time, loss_sleep=losses_sleep, acc=Acces, )
                    logger.info(msg)


    if config.train_encoder == True:
        if config.Mask == True:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss_mask {loss_mask.val:.5f} ({loss_mask.avg:.5f})\t' \
                  'Acc {acc.val:.5f} ({acc.avg:.5f})\t' \
                .format(
                epoch, i, len(test_loader)-1, batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss_mask=losses_mask,
                acc=Score_numbers)
            logger.info(msg)
            return {"loss_mask": losses_mask.avg, "acc": Score_numbers.avg, "keypoint": best_keypoint,
                    "target_mask": best_target_mask, "mask": best_Mask,"mask_all": Mask_all,
                    "target_mask_all": Target_Mask_all, "keypoint_all": Keypoint_all}
        else:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss_l1 {loss_l1.val:.5f} ({loss_l1.avg:.5f})\t' \
                  'Loss_l2 {loss_l2.val:.5f} ({loss_l2.avg:.5f})\t' \
                  'Loss_ssim {loss_ssim.val:.5f} ({loss_ssim.avg:.5f})\t' \
                .format(
                epoch, i, len(test_loader)-1, batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss_l1=losses_l1, loss_l2=losses_l2, loss_ssim=losses_ssim, )
            logger.info(msg)
            return {
                "loss_l1": losses_l1.avg, "loss_l2": losses_l2.avg, "loss_ssim": losses_ssim.avg,
                "input": best_input, "map": best_rebuild, "keypoint": best_keypoint,
                "mask_all": Mask_all, "target_mask_all": Target_Mask_all, "keypoint_all": Keypoint_all}
    else:
        msg = 'Epoch: [{0}][{1}/{2}]\t' \
              'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
              'Speed {speed:.1f} samples/s\t' \
              'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
              'Loss_sleep {loss_sleep.val:.5f} ({loss_sleep.avg:.5f})\t' \
              'Acc {acc.val:.5f} ({acc.avg:.5f})\t' \
            .format(
            epoch, i, len(test_loader)-1, batch_time=batch_time,
            speed=input.size(0) / batch_time.val,
            data_time=data_time, loss_sleep=losses_sleep, acc=Acces, )
        logger.info(msg)
        return {"loss_sleep": losses_sleep.avg, "acc": Acces.avg}


