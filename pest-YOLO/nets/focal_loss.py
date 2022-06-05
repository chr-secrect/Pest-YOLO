#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 20:22:07 2021

@author: chr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from nets.yolo_training import BCELoss

def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def BCELoss(pred,target):
    
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    
    return output

class FocalLoss(nn.Module):
    def __init__(self, gamma,alpha,pred,target,reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = BCELoss
        

    def forward(self,pred,target):
        
        loss = BCELoss(pred,target)
        loss *= self.__alpha * torch.pow(
            torch.abs(target - torch.sigmoid(pred)), self.__gamma
        )

        return loss



# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()
# class FocalLoss(nn.Module):
#     '''
#     This loss is proposed in this paper: https://arxiv.org/abs/1909.11932
#     It does not work in my projects, hope it will work well in your projects.
#     Hope you can correct me if there are any mistakes in the implementation.
#     '''

#     def __init__(self, ignore_lb=255, eps=1e-5, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.ignore_lb = ignore_lb
#         self.eps = eps
#         self.reduction = reduction
#         self.mse = nn.MSELoss(reduction='none')

#     def forward(self, logits, label):
#         ignore = label.data.cuda() == self.ignore_lb
#         n_valid = (ignore == 0).sum()
#         label = label.clone()
#         label[ignore] = 0
#         lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1).detach()

#         pred = torch.softmax(logits, dim=1)
#         loss = -torch.log(self.eps + 1. - self.mse(pred, lb_one_hot)).sum(dim=1)
#         loss[ignore] = 0
#         if self.reduction == 'mean':
#             loss = loss.sum() / n_valid
#         elif self.reduction == 'sum':
#             loss = loss.sum()
#         elif self.reduction == 'none':
#             loss = loss
#         return loss