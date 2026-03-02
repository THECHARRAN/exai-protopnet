# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        logp = torch.log(pt + 1e-12)
        loss = -((1 - pt) ** self.gamma) * logp
        if self.alpha is not None:
            a = self.alpha[targets].to(inputs.device)
            loss = loss * a
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device, lambda_c=0.003):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        B = features.size(0)
        features = features.view(B, -1)
        centers_batch = self.centers[labels]
        loss = ((features - centers_batch) ** 2).sum() / 2.0 / B
        return self.lambda_c * loss

def consistency_loss(logits1, logits2):
    p1 = F.log_softmax(logits1, dim=1)
    p2 = F.softmax(logits2, dim=1)
    return F.kl_div(p1, p2, reduction='batchmean')