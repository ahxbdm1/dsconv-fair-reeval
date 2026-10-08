"""
Evaluation metrics for thin/tubular structure segmentation (binary masks, 2D numpy arrays).
  dice, cldice (Shit et al., CVPR 2021, skeleton-based), betti0_error, betti1_error, n_components_error.
Betti numbers in 2D: b0 = #connected components of foreground (8-connectivity),
                     b1 = #holes = #background components (4-connectivity) not touching the border.
Report both whole-image values and patch-averaged values (patch=64, stride=64), since papers differ;
state in every table which one is used.
"""
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label


def dice(pred, gt, eps=1e-8):
    pred = pred.astype(bool); gt = gt.astype(bool)
    return 2.0 * (pred & gt).sum() / (pred.sum() + gt.sum() + eps)


def cldice(pred, gt, eps=1e-8):
    pred = pred.astype(bool); gt = gt.astype(bool)
    sp, sg = skeletonize(pred), skeletonize(gt)
    tprec = (sp & gt).sum() / (sp.sum() + eps)   # topology precision
    tsens = (sg & pred).sum() / (sg.sum() + eps)  # topology sensitivity
    return 2 * tprec * tsens / (tprec + tsens + eps)


def betti(mask):
    mask = mask.astype(bool)
    b0 = label(mask, connectivity=2).max()
    bg = label(~mask, connectivity=1)
    border = set(np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]])))
    b1 = len([k for k in range(1, bg.max() + 1) if k not in border])
    return b0, b1


def betti_errors(pred, gt):
    p0, p1 = betti(pred); g0, g1 = betti(gt)
    return abs(p0 - g0), abs(p1 - g1)


def patch_betti_errors(pred, gt, patch=64, stride=64):
    e0, e1 = [], []
    H, W = gt.shape
    for i in range(0, H - patch + 1, stride):
        for j in range(0, W - patch + 1, stride):
            a, b = betti_errors(pred[i:i + patch, j:j + patch], gt[i:i + patch, j:j + patch])
            e0.append(a); e1.append(b)
    return float(np.mean(e0)), float(np.mean(e1))


def evaluate(pred, gt, fov=None):
    """pred, gt: binary 2D arrays. fov: optional field-of-view mask (DRIVE provides one); pixels outside FOV are
    set to 0 in both before computing metrics. Returns a dict."""
    pred = pred.astype(bool); gt = gt.astype(bool)
    if fov is not None:
        fov = fov.astype(bool); pred = pred & fov; gt = gt & fov
    e0, e1 = betti_errors(pred, gt)
    pe0, pe1 = patch_betti_errors(pred, gt)
    return {'dice': float(dice(pred, gt)), 'cldice': float(cldice(pred, gt)),
            'betti0_err': float(e0), 'betti1_err': float(e1),
            'patch_betti0_err': pe0, 'patch_betti1_err': pe1}
