from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.7, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels, mask=None):
        features = features.to(self.device)
        labels = labels.to(self.device)
        #print("labels:",labels)
        #print("labels shape:", labels.shape)
        batch_size = features.shape[0]
        features = torch.where(torch.isnan(features), torch.tensor(0.0).to(self.device), features) 
        features = features.view(batch_size, -1)        
        # p:标准化的范数
        features = F.normalize(features , p=2, dim=1)
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # For code vulnerability detection, assume labels are 0 for normal, 1 for vulnerable
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            #将原始的labels张量转化为一个列向量
            labels = labels.contiguous().view(-1, 1)
            #print("labels:", labels)
            #print("labels shape:", labels.shape)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
            #print("labels.T:", labels.T)
            #print("mask:", mask)
        else:
            mask = mask.float().to(self.device)


        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        #print("anchor_dot_contrast:",anchor_dot_contrast)
        
        postives_mask = mask.to(self.device)
        negatives_mask = 1. - postives_mask
        #print("negatives_mask:",negatives_mask)
            
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        #print("logits_max:", logits_max)
        logits = anchor_dot_contrast - logits_max.detach()
        #print("logits:", logits)
        
        #去掉对角线的距离，做标准化
        exp_logits = torch.exp(logits)#距离的指数
        #print("exp_logits:", exp_logits)
        #除自己之外，正样本的个数
        num_positives_per_now = torch.sum(postives_mask, axis=1)
        #print("num_positives_per_now:", num_positives_per_now)
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdim=True)
        + torch.sum(exp_logits * postives_mask, axis=1, keepdim=True)
        #print("denominator:", denominator)
        #print("log(denominator)", torch.log(denominator))
        
        log_probs = logits - torch.log(denominator)
        #print("log_probs:",log_probs)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("log_probs has nan!")
        
        log_probs = torch.sum(log_probs*postives_mask, axis=1)[num_positives_per_now > 0] / num_positives_per_now[num_positives_per_now > 0]

        # Adjusted loss computation
        loss = - log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
