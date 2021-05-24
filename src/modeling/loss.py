# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:59:17 2020

@author: Lim
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def _neg_loss(pred, gt, alpha=2, beta=4, weights=None):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    part1 = torch.pow(pos_inds + (2 * neg_inds - 1) * pred, alpha)
    part2 = torch.pow(neg_inds + (2 * pos_inds - 1) * gt, beta)
    part3 = torch.log(neg_inds + (2 * pos_inds - 1) * pred)
    loss_tensor = -1.0 * part1 * part2 * part3
    if weights is not None:
        for i, w in enumerate(weights):
            loss_tensor[:, i] *= w
    loss = torch.sum(loss_tensor)

    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        num_pos = pred.gt(0.3).sum()
        num_pos = num_pos if num_pos > 0 else 1
        
    loss /= num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
      super(FocalLoss, self).__init__()
      self.neg_loss = _neg_loss

    def forward(self, pred_tensor, target_tensor, weights=None):
      return self.neg_loss(pred_tensor, target_tensor, weights=None)


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, pred, mask, ind, target):
    pred = _transpose_and_gather_feat(pred, ind)  
    mask = mask.unsqueeze(2).expand_as(pred).float() 
    loss = F.smooth_l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4) # 每个目标的平均损失
    return loss

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y
def _relu(x):
    y = torch.clamp(x.relu_(), min = 0., max=179.99)
    return y

class CtdetLoss(torch.nn.Module):
    # loss_weight={'hm_weight':1,'wh_weight':0.1,'reg_weight':0.1}
    def __init__(self, loss_weight):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = RegL1Loss()
        self.loss_weight = loss_weight
        
    def forward(self, pred_tensor, target_tensor): 
        hm_weight = self.loss_weight['heatmap_weight']
        wh_weight = self.loss_weight['wh_weight']
        reg_weight = self.loss_weight['offset_weight']
        classes_weight = self.loss_weight['classes_weight']
        assert len(classes_weight) == pred_tensor['heatmap'].size(1)
        #ang_weight = self.loss_weight['ang_weight']

        # hm_loss, wh_loss, off_loss, ang_loss = 0, 0, 0, 0
        device = pred_tensor['heatmap'].device
        hm_loss, wh_loss, off_loss = torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0., device=device)
        pred_tensor['heatmap'] = _sigmoid(pred_tensor['heatmap'])
        hm_loss += self.crit(pred_tensor['heatmap'], target_tensor['heatmap'], weights=classes_weight)
        '''
        if ang_weight > 0:
            pred_tensor['ang'] = _relu(pred_tensor['ang'])
            ang_loss += self.crit_wh(pred_tensor['ang'], target_tensor['reg_mask'],target_tensor['ind'], target_tensor['ang'])  
        '''
        if wh_weight > 0:
            wh_loss += self.crit_wh(pred_tensor['wh'], target_tensor['offset_mask'],target_tensor['indexes'], target_tensor['wh']) 
        if reg_weight > 0:
            off_loss += self.crit_reg(pred_tensor['offset'], target_tensor['offset_mask'],target_tensor['indexes'], target_tensor['offset']) 
        return hm_loss, wh_loss, off_loss, hm_weight * hm_loss + wh_weight * wh_loss + reg_weight * off_loss
