"""python tests/test_dsconv_fix.py <path/to/DSCNet_2D_opensource/Code/DRIVE/DSCNet>
Checks: (1) original DSConv is batch-dependent for samples >= 1; (2) fixed version is batch-invariant;
(3) with batch size 1 the fixed version is bit-identical to the original (so DSCNet's paper setting is unchanged)."""
import sys, os, importlib, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, sys.argv[1])
import S3_DSConv as orig
from dsconv_fix import patch_dsc
torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
res = {}
for morph in (0, 1):
    m = orig.DSConv(8, 16, 9, 1.0, morph, True, dev).to(dev).eval()
    x = torch.randn(3, 8, 40, 36, device=dev)
    orig_fn = orig.DSC._bilinear_interpolate_3D
    with torch.no_grad():
        yb = m(x); ys = torch.cat([m(x[i:i + 1]) for i in range(3)])
        y1_orig = m(x[1:2])
        res[f'orig_morph{morph}_batch_dep_maxdiff'] = (yb - ys).abs().max().item()
        patch_dsc(orig)
        ybf = m(x); ysf = torch.cat([m(x[i:i + 1]) for i in range(3)])
        res[f'fixed_morph{morph}_batch_dep_maxdiff'] = (ybf - ysf).abs().max().item()
        res[f'fixed_vs_orig_b1_maxdiff_morph{morph}'] = (m(x[1:2]) - y1_orig).abs().max().item()
    orig.DSC._bilinear_interpolate_3D = orig_fn
print(res)
assert res['orig_morph0_batch_dep_maxdiff'] > 1e-2, 'expected the original bug to be visible'
assert res['fixed_morph0_batch_dep_maxdiff'] < 1e-4 and res['fixed_morph1_batch_dep_maxdiff'] < 1e-4
assert res['fixed_vs_orig_b1_maxdiff_morph0'] == 0 and res['fixed_vs_orig_b1_maxdiff_morph1'] == 0
print('dsconv fix tests passed')
