# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
# from torch.autograd import Variable
from data import wheat as cfg
from ..box_utils import match, log_sum_exp
import numpy as np


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=1):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        if self.use_gpu == 1:
              # handbook
#             loc_t = loc_t.cuda()
#             conf_t = conf_t.cuda()
              device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif self.use_gpu == 2:
            try:
                device = xm.xla_device()
            except:
                device = 'cpu'
        else:
            device = 'cpu'
            
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        
        loc_t = loc_t.to(device)
        conf_t = conf_t.to(device, dtype=torch.long)
        
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            
            defaults = defaults.to(device)
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
            
              # handbook
             
        # wrap targets
        # handbook
#         loc_t = Variable(loc_t, requires_grad=False)
#         conf_t = Variable(conf_t, requires_grad=False)
        # handbook

        # if self.use_gpu == 2:
#             loc_data = loc_data.data.to(device)
#             conf_data = conf_data.data.to(device)
            
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        ##user_warning
#         loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # https://github.com/amdegroot/ssd.pytorch/issues/161
        conf_tt = conf_t.view(-1,1)
        conf_tt_index = (conf_tt != 0).nonzero()
        conf_tt[conf_tt_index] = 1
#         loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_tt)

        # Hard Negative Mining
        # handbook
        #loss_c[pos] = 0  # filter out pos boxes for now
        #loss_c = loss_c.view(num, -1)
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        
        # if self.use_gpu == 2:
#             pos_idx = pos_idx.to(device)
#             neg_idx = neg_idx.to(device)
         
        if self.use_gpu == 2:   
            both_idx = (pos_idx.float() + neg_idx.float()).gt(0)
            conf_p =  (both_idx.float() * conf_data)
            conf_p = conf_p.view(-1, self.num_classes)
        else:
            conf_p = conf_p[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
            
        if self.use_gpu == 2:
            both = (pos.float() + neg.float()).gt(0)
            targets_weighted = (both.float() * conf_t)
        else:
            targets_weighted = conf_t[(pos+neg).gt(0)]
        ##user_warning
        # loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False, reduction='sum')
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

         # handbook
#         N = num_pos.data.sum()
        N = num_pos.data.sum().double()
        loss_l = loss_l.double()
        loss_c = loss_c.double()
         # handbook
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c

    # def forward(self, predictions, targets):     
#     # predictions (tuple) : A tuple (loc, conf, prior boxes) from SSD net
#         # loc   : [batch_size, num_priors, 4]
# #         conf  : [batch_size, num_priors, num_classes]
# #         prior : [num_priors, 4]
#     # targets : Ground truth boxes and labels for a batch,
#         # [num_objs, 5],
# #         [x1,y1,x2,y2,class] format
# 
#         loc_data, conf_data, priors = predictions
#         num = loc_data.size(0)
#         priors = priors[:loc_data.size(1), :]
#         num_priors = (priors.size(0))
#         num_classes = self.num_classes
# 
#         # match priors (default boxes) and ground truth boxes
#         loc_t = torch.Tensor(num, num_priors, 4)
#         conf_t = torch.LongTensor(num, num_priors)
#         for idx in range(num):
#             truths = targets[idx][:, :-1].data
#             labels = targets[idx][:, -1].data
#             defaults = priors.data
#             match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
#             
#         #handbook    
# #         loc_t = loc_t.cuda()
# #         conf_t = conf_t.cuda()
#         if self.use_gpu:
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#             
#         loc_t = loc_t.to(device)
#         conf_t = conf_t.to(device)
#         #handbook
#     
#         # [batch, num_priors]
#         pos = conf_t > 0
#         num_pos = pos.sum(dim=1, keepdim=True)
# 
#         # Localization Loss (Smooth L1)
#         # Shape: [batch,num_priors,4]
#         pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
#         loc_p = loc_data[pos_idx].view(-1, 4)
#         loc_t = loc_t[pos_idx].view(-1, 4)
#         loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
# 
#         # Compute max conf across batch for hard negative mining
#         batch_conf = conf_data.view(-1, self.num_classes)
#         loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
# 
#         # Hard Negative Mining
#         # [batch, num_priors]
#         loss_c = loss_c.view(num, -1)
#         nonzero_pos = pos.nonzero()
# 
#         loss_c[pos] = 0  # filter out pos boxes for now
# 
#         _, loss_idx = loss_c.sort(1, descending=True)
#         _, idx_rank = loss_idx.sort(1)
#         num_pos = pos.long().sum(1, keepdim=True)
#         num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
#         neg = idx_rank < num_neg.expand_as(idx_rank)
#     
#         # Confidence Loss Including Positive and Negative Examples
#         pos_idx = pos.unsqueeze(2).expand_as(conf_data)
#         neg_idx = neg.unsqueeze(2).expand_as(conf_data)
# 
#         # .gt(input) computes input > other element-wise.
#         conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
#         targets_weighted = conf_t[(pos+neg).gt(0)]
# 
#         loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
# 
#         # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#         N = num_pos.data.sum().double()
#         loss_l = loss_l.double()
#         loss_c = loss_c.double()
# 
#         loss_l /= N
#         loss_c /= N
#         return loss_l, loss_c
