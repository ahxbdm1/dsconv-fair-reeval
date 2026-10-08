"""Soft clDice loss (Shit et al., CVPR 2021) for the optional 'topology loss' variant. Inputs are probabilities (B,1,H,W)."""
import torch
import torch.nn.functional as F

def _soft_erode(x):
    return torch.min(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))

def _soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)

def _soft_open(x):
    return _soft_dilate(_soft_erode(x))

def soft_skel(x, iters=10):
    x1 = _soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(iters):
        x = _soft_erode(x)
        x1 = _soft_open(x)
        delta = F.relu(x - x1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

class SoftClDiceLoss(torch.nn.Module):
    def __init__(self, iters=10, smooth=1.0):
        super().__init__(); self.iters = iters; self.smooth = smooth
    def forward(self, y_true, y_pred):
        sp, st = soft_skel(y_pred, self.iters), soft_skel(y_true, self.iters)
        tprec = ((sp * y_true).sum() + self.smooth) / (sp.sum() + self.smooth)
        tsens = ((st * y_pred).sum() + self.smooth) / (st.sum() + self.smooth)
        return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)
