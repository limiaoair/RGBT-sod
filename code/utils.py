import os
import torch
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from Code.utils4.my2_data import get_loader, test_dataset
from Code.utils4.my2_lib import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
import argparse
import cv2
from Code.lib4.model4 import VMUNet


def Edge_pre(mask, a, b):
    # 假设seg_map是分割映射（numpy数组
    # 转换为灰度图像（如果已经是二值图像，则可能不需要这一步）  
    # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.ndim == 3 else mask  
    
    # 使用Canny边缘检测  
    edges = cv2.Canny(mask, a, b)
    return edges


# import torch
# import torch.nn.functional as F
# from utils.AF.Fsmish import smish as Fsmish

# def bdcn_loss2(inputs, targets, l_weight=1.1):
#     # bdcn loss modified in DexiNed

#     targets = targets.long()
#     mask = targets.float()
#     num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
#     num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

#     mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
#     mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
#     inputs= torch.sigmoid(inputs)
#     cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
#     cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum
#     return l_weight*cost

# # ------------ cats losses ----------
# def bdrloss(predicti~on, label, radius,device='cpu'):
#     '''
#     The boundary tracing loss that handles the confusing pixels.
#     '''

#     filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
#     filt.requires_grad = False
#     filt = filt.to(device)

#     bdr_pred = prediction * label
#     pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)

#     texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
#     mask = (texture_mask != 0).float()
#     mask[label == 1] = 0
#     pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

#     softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
#     cost = -label * torch.log(softmax_map)
#     cost[label == 0] = 0

#     return torch.sum(cost.float().mean((1, 2, 3)))

# def textureloss(prediction, label, mask_radius, device='cpu'):
#     '''
#     The texture suppression loss that smooths the texture regions.
#     '''
#     filt1 = torch.ones(1, 1, 3, 3)
#     filt1.requires_grad = False
#     filt1 = filt1.to(device)
#     filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
#     filt2.requires_grad = False
#     filt2 = filt2.to(device)

#     pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
#     label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

#     mask = 1 - torch.gt(label_sums, 0).float()

#     loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
#     loss[mask == 0] = 0

#     return torch.sum(loss.float().mean((1, 2, 3)))

# def cats_loss(prediction, label, l_weight=[0.,0.], device='cpu'):
#     # tracingLoss

#     tex_factor,bdr_factor = l_weight
#     balanced_w = 1.1
#     label = label.float()
#     prediction = prediction.float()
#     with torch.no_grad():
#         mask = label.clone()

#         num_positive = torch.sum((mask == 1).float()).float()
#         num_negative = torch.sum((mask == 0).float()).float()
#         beta = num_negative / (num_positive + num_negative)
#         mask[mask == 1] = beta
#         mask[mask == 0] = balanced_w * (1 - beta)
#         mask[mask == 2] = 0

#     prediction = torch.sigmoid(prediction)

#     cost = torch.nn.functional.binary_cross_entropy(
#         prediction.float(), label.float(), weight=mask, reduction='none')
#     cost = torch.sum(cost.float().mean((1, 2, 3)))  # by me
#     label_w = (label != 0).float()
#     textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4, device=device)
#     bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4, device=device)

#     return cost + bdr_factor * bdrcost + tex_factor * textcost


# def Infrared_edge_loss(pre, mask):


#     return loss1


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def confident_loss(pred, gt, beta=2):
    y = torch.sigmoid(pred)
    weight = beta * y * (1 - y)
    weight = weight.detach()
    loss = (F.binary_cross_entropy(pred, gt, reduction='none') * weight).mean()
    loss2 = 0.5 * beta * (y * (1 - y)).mean()
    return loss + loss2

def MY_loss(pre1, pre2, pre3, mask):
    pool  = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    # pre2  = pool(pre)
    # pre4  = pool(pre2)
    gt2   = pool(mask)
    gt4   = pool(gt2)
    out1  = structure_loss(pre1, mask)
    out2  = structure_loss(pre2, gt2)
    out4  = structure_loss(pre3, gt4)
    out   = out1*0.7 + out2*0.15 + out4*0.15
    # out   = out1*0.7 + out2*0.15 + out4*0.15
    return out


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        self.avg = self.sum / self.count
 
