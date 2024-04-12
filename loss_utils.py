import torch.nn as nn
import torch
import torchsort
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pt, targets):
        epsilon = 1e-6
        loss = - self.alpha * (1 - pt) ** self.gamma * targets * torch.log(pt+epsilon) -\
              (1 - self.alpha) * pt ** self.gamma * (1 - targets) * torch.log(1 - pt+epsilon)
        loss = torch.mean(loss, dim=-1)
        return loss

class AUC_loss(nn.Module):
  def __init__(self, num_thresholds, device, tau=1e-3, reg=1e-3):
    super(AUC_loss, self).__init__()
    self.thresholds = torch.linspace(0, 1, steps=num_thresholds+2)[1:-1].to(device)
    self.tau = tau
    self.reg = reg
    self.focal_month = FocalLoss(alpha=0.85, gamma=2)
    self.focal_day = FocalLoss(alpha=0.99, gamma=2)
    self.device = device

  def sigmoid(self, x):
    exponent = -x / self.tau
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y
  
  def forward(self, pred, label, phi, phi_next, label_day, pred_day, q0, val=False):
    # ap
    T_pred = pred[..., None] - self.thresholds.reshape(1, 1, -1) # [B, N, t]
    T_pred = self.sigmoid(T_pred)
    T_pred_T = torch.sum(T_pred * label[..., None], dim=1) # [B, t]
    sorted_idx = torch.argsort(T_pred_T, dim=-1)
    sorted_T_pred_T = torch.gather(T_pred_T, dim=-1, index=sorted_idx) # [B, t]
    sorted_T_pred_all = torch.gather(torch.sum(T_pred, dim=1), dim=-1, index=sorted_idx) # [B, t]
    precision = sorted_T_pred_T / sorted_T_pred_all # [B, t]
    recall = sorted_T_pred_T / torch.sum(label[..., None], dim=1) # [B, t]
    pr_auc = torch.trapz(precision, recall).mean()
    loss_ap = (1 - pr_auc) * 0.2

    # cycle
    loss_cyc = torch.abs(phi - phi_next).mean() * 100
    
    # focal
    focal_label = self.focal_month(pred, label).mean() * 100
    focal_diffuse = self.focal_day(pred_day.flatten(-2, -1), label_day).mean() * 100
    loss_focal = focal_label + focal_diffuse

    # entropy
    loss_rg = torch.abs(q0).mean() / 2

    return loss_ap, loss_cyc, loss_focal, loss_rg
