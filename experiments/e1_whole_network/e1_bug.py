# Copyright (c) 2025 Yaolei Qi
# Copyright (c) 2026 Heng Zhang
# SPDX-License-Identifier: MIT
# Adapted from DSCNet; see THIRD_PARTY_NOTICES.md.
"""E1: characterise the cross-sample indexing bug in the original DSConv (S3_DSConv.py, DSCNet commit 0e35b2b).
Runs on CPU. Usage: python e1_bug.py <DSCNet_2D_opensource dir> <best.pt of a batch-1 trained DSCNet> <out dir>
Parts
  A  synthetic layer-level relative error vs batch size / sample index (orig, fixed, pro)
  B  real-feature layer-level error inside a trained DSCNet (inputs captured with the batch-invariant fixed version)
  C  training-step gradient corruption at batch 8 (orig vs fixed)
  D  whole-network inference on the 20 DRIVE test images: upstream sliding window, ROI patches run one-by-one
     (upstream behaviour) vs packed in batches of 8
  E  qualitative: centre-tap deformed feature of sample 1 under orig vs fixed (input of the first DSConv = image)
"""
import sys, os, json, time, copy, io, contextlib
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(2)
ROOT = Path(sys.argv[1]); CK = Path(sys.argv[2]); OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'Code/DRIVE/DSCNet'), str(ROOT / 'Code/DRIVE'), str(HERE.parents[1] / 'modules')]
import SimpleITK as sitk
import S3_DSConv as orig_mod
from S3_DSCNet import DSCNet
from S3_DSConv_pro import DSConv_pro
from S3_Train_Process import predict
from dsconv_fix import patch_dsc
from topo_metrics import evaluate
_np_reshape = np.reshape
def _reshape_compat(a, *args, newshape=None, **kw):   # upstream predict() uses np.reshape(newshape=...), removed in NumPy 2.x
    return _np_reshape(a, newshape, **kw) if newshape is not None else _np_reshape(a, *args, **kw)
np.reshape = _reshape_compat

CPU = torch.device('cpu')
ORIG_FN = orig_mod.DSC._bilinear_interpolate_3D
patch_dsc(orig_mod); FIX_FN = orig_mod.DSC._bilinear_interpolate_3D
orig_mod.DSC._bilinear_interpolate_3D = ORIG_FN

class impl:
    """context manager selecting 'orig' or 'fix' bilinear interpolation"""
    def __init__(self, which): self.which = which
    def __enter__(self): orig_mod.DSC._bilinear_interpolate_3D = FIX_FN if self.which == 'fix' else ORIG_FN
    def __exit__(self, *a): orig_mod.DSC._bilinear_interpolate_3D = ORIG_FN

def rel_err(yb, ys):
    d = (yb - ys).flatten(1).norm(dim=1); n = ys.flatten(1).norm(dim=1).clamp(min=1e-12)
    return (d / n).tolist()

def save(name, obj): (OUT / name).write_text(json.dumps(obj, indent=1))

BS = [1, 2, 4, 8, 16]
res = {}
# ---------------------------------------------------------------- A synthetic
def part_a():
    rows = []
    for morph in (0, 1):
        for C in (16, 64):
            for S in (28, 56, 112):
                torch.manual_seed(0)
                layers = {'orig': orig_mod.DSConv(C, C, 9, 1.0, morph, True, CPU).eval(),
                          'pro': DSConv_pro(C, C, 9, 1.0, morph, True, CPU).eval()}
                x = torch.randn(max(BS), C, S, S)
                for name in ('orig', 'fix', 'pro'):
                    m = layers['pro' if name == 'pro' else 'orig']
                    with torch.no_grad(), impl('fix' if name == 'fix' else 'orig'):
                        ys = torch.cat([m(x[i:i + 1]) for i in range(max(BS))])
                        for B in BS:
                            e = rel_err(m(x[:B]), ys[:B])
                            rows.append(dict(impl=name, morph=morph, C=C, S=S, B=B, per_sample=e))
                print('A', morph, C, S, flush=True)
    save('A_synthetic.json', rows); return rows

# ---------------------------------------------------------------- data helpers
DATA = ROOT / 'Data/DRIVE'
def files(folder): return sorted((DATA / folder).glob('*.nii.gz'), key=lambda p: int(p.name.split('.')[0]))
def meanstd(paths):
    arrays = [sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32) for p in paths]
    cnt = sum(a.size for a in arrays); mean = 0.0; var = 0.0
    for a in arrays: mean += np.sum(a)
    mean = mean / cnt
    for a in arrays: var += np.sum(np.square(a - mean))
    return float(mean), float(np.sqrt(var / cnt))
TE = files('Image_Te'); TE_LAB = files('Label_Te')
MEAN, STD = meanstd(TE)

def load_model():
    m = DSCNet(1, 1, 9, 1.0, True, CPU, 16, 1)
    sd = torch.load(CK, map_location='cpu', weights_only=False)
    m.load_state_dict(sd['model'] if 'model' in sd else sd); return m.eval()

def rois(n=16, seed=0, size=224):
    rng = np.random.RandomState(seed); out = []
    for k in range(n):
        p = TE[k % len(TE)]
        im = (sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32) - MEAN) / STD
        y = rng.randint(0, im.shape[0] - size + 1); x = rng.randint(0, im.shape[1] - size + 1)
        out.append(im[y:y + size, x:x + size])
    return torch.from_numpy(np.stack(out)[:, None])

# ---------------------------------------------------------------- B real features
def dsconv_layers(model): return [(n, m) for n, m in model.named_modules() if isinstance(m, orig_mod.DSConv)]

def part_b(model):
    x = rois(16, size=128)
    caps = {}
    hooks = [m.register_forward_hook(lambda mod, inp, out, n=n: caps.setdefault(n, []).append(inp[0].detach()))
             for n, m in dsconv_layers(model)]
    with torch.no_grad(), impl('fix'):
        for i in range(len(x)): model(x[i:i + 1])   # fixed version is batch-invariant -> exact per-sample inputs
    for h in hooks: h.remove()
    caps = {k: torch.cat(v) for k, v in caps.items()}
    rows = []
    for n, m in dsconv_layers(model):
        f = caps[n]
        with torch.no_grad():
            with impl('fix'): ys = torch.cat([m(f[i:i + 1]) for i in range(16)])
            for name in ('orig', 'fix'):
                with impl(name):
                    for B in BS:
                        rows.append(dict(layer=n, impl=name, B=B, roi=128, C=f.shape[1], S=f.shape[-1], per_sample=rel_err(m(f[:B]), ys[:B])))
        print('B', n, tuple(f.shape), flush=True)
    save('B_real_layers.json', rows); return rows

# ---------------------------------------------------------------- C gradient corruption
def part_c(model):
    m = copy.deepcopy(model).train()
    x = rois(8, seed=1, size=96)
    lab = (x > x.mean()).float()  # dummy target; only gradient discrepancy is reported
    grads = {}
    for name in ('fix', 'orig'):
        torch.manual_seed(0); m.zero_grad()
        with impl(name):
            out = m(x); loss = torch.nn.functional.binary_cross_entropy(out.clamp(1e-6, 1 - 1e-6), lab); loss.backward()
        grads[name] = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]).clone()
        grads[name + '_out'] = out.detach()
    g0, g1 = grads['fix'], grads['orig']
    r = dict(batch=8, roi=96, grad_rel_err=float((g1 - g0).norm() / g0.norm()),
             grad_cosine=float(torch.nn.functional.cosine_similarity(g0, g1, dim=0)),
             train_mode_output_rel_err_per_sample=rel_err(grads['orig_out'], grads['fix_out']),
             note='train mode (BatchNorm in offset branch uses batch statistics in both implementations); dummy target')
    save('C_gradient.json', r); print('C', r['grad_rel_err'], r['grad_cosine'], flush=True); return r

# ---------------------------------------------------------------- D whole network inference
class Recorder(torch.nn.Module):
    def __init__(self): super().__init__(); self.inputs = []
    def forward(self, x): self.inputs.append(x.clone()); return torch.zeros_like(x)
class Replayer(torch.nn.Module):
    def __init__(self, outs): super().__init__(); self.outs = list(outs)
    def forward(self, x): return self.outs.pop(0)

def run_predict(fake, path, outdir, args):
    one = outdir / 'one.txt'; one.write_text(str(path) + '\n')
    with contextlib.redirect_stdout(io.StringIO()): predict(fake, str(one), str(outdir) + '/', args)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(outdir / path.name)))

def batched_via_pairs(model, patches, bsz, single):
    """Exact, memory-light evaluation of the ORIGINAL DSConv at batch size bsz.
    In the original code, sample j>0 of a batch reads values only from sample 0's feature map (at its own coordinates);
    sample 0 is computed correctly; GroupNorm is per-sample and BatchNorm is in eval mode. Hence the batch output of
    sample j equals the output of the 2-sample batch [first sample of its chunk, sample j]. Verified in verify_pairs()."""
    outs = []
    for c in range(0, len(patches), bsz):
        chunk = patches[c:c + bsz]; outs.append(single[c:c + 1])
        for j in range(1, len(chunk)):
            outs.append(model(torch.stack([chunk[0], chunk[j]]))[1:2])
    return torch.cat(outs)

def verify_pairs(model, bsz=8, size=96):
    x = rois(bsz, seed=5, size=size)
    with torch.no_grad(), impl('orig'):
        full = model(x); single = torch.cat([model(x[i:i + 1]) for i in range(bsz)])
        pairs = batched_via_pairs(model, x, bsz, single)
    r = dict(roi=size, bsz=bsz, max_abs_diff_full_vs_pairs=float((full - pairs).abs().max()),
             max_abs_diff_full_vs_single=float((full - single).abs().max()))
    save('D_pairs_equivalence.json', r); print('verify', r, flush=True); return r

def part_d(model, bsz=8):
    verify_pairs(model, bsz)
    from types import SimpleNamespace
    tmp = OUT / 'tmp_pred'; tmp.mkdir(exist_ok=True)
    np.save(tmp / 'te.npy', [MEAN, STD])
    args = SimpleNamespace(root_dir=str(tmp) + '/', Te_Meanstd_name='te.npy', ROI_shape=224, n_classes=1)
    rows = []
    start = int(os.environ.get('E1_START', '0'))
    for p, lp in list(zip(TE, TE_LAB))[start:]:
        t0 = time.time()
        rec = Recorder(); run_predict(rec, p, tmp, args); patches = torch.cat(rec.inputs)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(lp))); gt = gt / gt.max() > 0.5
        outs = {}
        with torch.no_grad():
            with impl('orig'):
                outs['orig_b1'] = torch.cat([model(patches[i:i + 1]) for i in range(len(patches))])
                outs['orig_b8'] = batched_via_pairs(model, patches, bsz, outs['orig_b1'])
        probs = {k: run_predict(Replayer(list(v.split(1))), p, tmp, args) for k, v in outs.items()}
        row = dict(image=p.name, n_patches=len(patches))
        for k, pr in probs.items():
            row[k] = evaluate(pr > 0.5, gt)
        b1 = probs['orig_b1'] > 0.5
        for k in ('orig_b8',):
            row[k]['pixel_flip_frac'] = float(((probs[k] > 0.5) != b1).mean())
            row[k]['max_abs_prob_diff'] = float(np.nanmax(np.abs(probs[k] - probs['orig_b1'])))
        rows.append(row)
        print('D', p.name, len(patches), {k: round(row[k]['dice'], 4) for k in probs}, f'{time.time() - t0:.0f}s', flush=True)
        save('D_whole_network.json', rows)
    return rows

# ---------------------------------------------------------------- E qualitative
def part_e(model):
    x = rois(2, seed=3)
    layer = dict(dsconv_layers(model))['conv00'] if 'conv00' in dict(dsconv_layers(model)) else dsconv_layers(model)[0][1]
    feats = {}
    for name in ('fix', 'orig'):
        with torch.no_grad(), impl(name):
            off = torch.tanh(layer.bn(layer.offset_conv(x)))
            dsc = orig_mod.DSC(x.shape, layer.kernel_size, layer.extend_scope, layer.morph, CPU)
            feats[name] = dsc.deform_conv(x, off, True).numpy()
    K = layer.kernel_size; m = layer.morph
    def tap(f, k):  # morph 0: rows interleaved (W*K, H); morph 1: cols interleaved
        return f[:, 0, k::K, :] if m == 0 else f[:, 0, :, k::K]
    np.savez_compressed(OUT / 'E_qualitative.npz', image=x[:, 0].numpy(),
                        fix_center=tap(feats['fix'], K // 2), orig_center=tap(feats['orig'], K // 2),
                        fix_tap0=tap(feats['fix'], 0), orig_tap0=tap(feats['orig'], 0), morph=m, K=K)
    print('E saved, morph', m, flush=True)

if __name__ == '__main__':
    parts = sys.argv[4] if len(sys.argv) > 4 else 'ABCDE'
    model = load_model()
    print('DSConv layers:', [n for n, _ in dsconv_layers(model)], flush=True)
    if 'E' in parts: part_e(model)
    if 'C' in parts: part_c(model)
    if 'B' in parts: part_b(model)
    if 'A' in parts: part_a()
    if 'D' in parts: part_d(model)
    print('done')
