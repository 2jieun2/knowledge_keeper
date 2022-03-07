import numpy as np
# import cv2
# from torch.autograd import Variable
# import math
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from skimage.metrics import structural_similarity as cal_ssim
import torch


def cal_psnr_ssim_list(real_y_list, pred_y_list, return_total=False):
    total_psnr = np.zeros(len(real_y_list))
    total_ssim = np.zeros(len(real_y_list))

    for idx in range(len(real_y_list)):
        # total_psnr[idx] = cal_psnr(real_y_list[idx], pred_y_list[idx])
        total_psnr[idx] = cal_psnr(real_y_list[idx], pred_y_list[idx], data_range=1)
        # total_ssim[idx] = cal_ssim(real_y_list[idx], pred_y_list[idx])
        total_ssim[idx] = cal_ssim(real_y_list[idx], pred_y_list[idx], data_range=1)

    if return_total:
        return total_psnr.mean(), total_ssim.mean(), total_psnr, total_ssim
    else:
        return total_psnr.mean(), total_ssim.mean()


def cal_dice_score(real, pred, smooth=1e-7):  # soft dice loss
    batch_size = real.shape[0]  # real shape: b, c, h, w, d
    num_labels = real.shape[1]

    if isinstance(real, torch.Tensor):
        real = real.view(batch_size, num_labels, -1)
        pred = pred.view(batch_size, num_labels, -1)
        intersection = torch.sum(pred * real, dim=(0, 2))
        cardinality = torch.sum(pred + real, dim=(0, 2))
    elif isinstance(real, np.ndarray):
        real = real.reshape(batch_size, num_labels, -1)
        pred = pred.reshape(batch_size, num_labels, -1)
        intersection = np.sum(pred * real, axis=(0, 2))
        cardinality = np.sum(pred + real, axis=(0, 2))
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)

    return dice_score


def cal_dice_score_list(real_list, pred_list, return_total=False):  # soft dice loss
    score_bg = np.zeros(len(real_list))
    score_CSF = np.zeros(len(real_list))
    score_GM = np.zeros(len(real_list))
    score_WM = np.zeros(len(real_list))

    for idx in range(len(real_list)):
        real = real_list[idx]
        pred = pred_list[idx]
        if len(real) == 4:
            real = np.expand_dims(real, axis=0)  # (C, H, W, D) -> (1, C, H, W, D)
        if len(pred) == 4:
            pred = np.expand_dims(pred, axis=0)  # (C, H, W, D) -> (1, C, H, W, D)
        bg, CSF, GM, WM = cal_dice_score(real, pred)
        score_bg[idx] = bg
        score_CSF[idx] = CSF
        score_GM[idx] = GM
        score_WM[idx] = WM
    scores = [score_bg, score_CSF, score_GM, score_WM]
    scores_mean = [score.mean() for score in scores]

    # return scores_mean
    if return_total:
        return scores_mean, scores
    else:
        return scores_mean
