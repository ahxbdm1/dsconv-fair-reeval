"""CPU-only mechanistic characterization; no network or training entry points."""
import argparse,csv,hashlib,importlib.util,json,pathlib,sys,time
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'modules'))
from dsconv_fix import patch_dsc
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=pathlib.Path,default=ROOT/'DSCNet/DSCNet_2D_opensource/Code/DRIVE/DSCNet/S3_DSConv.py')
parser.add_argument('--output-dir',type=pathlib.Path,default=ROOT/'runs/e1_synthetic')
args=parser.parse_args();SOURCE=args.source
spec=importlib.util.spec_from_file_location('e1_original_dsconv',SOURCE)
orig=importlib.util.module_from_spec(spec);spec.loader.exec_module(orig)
ORIGINAL=orig.DSC._bilinear_interpolate_3D
OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
PLAN={'owner':'Codex','device':'cpu','torch':torch.__version__,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'upstream_commit':'0e35b2b6ea748a5e8ce7557ea3e580b9951471d6','batches':[1,2,4,8,16],'seeds':list(range(20)),'shape_per_sample':[8,40,36],'out_channels':16,'kernel_size':9,'morphs':[0,1],'model_mode':'eval; random weights, frozen BatchNorm statistics','input':'independent standard normal synthetic feature maps','relative_error':'L2(original_batch - singleton_reference) / L2(singleton_reference), per sample, then batch mean','controls':['fixed same weights','identical inputs in every batch slot','batch permutation','fixed-coordinate gather-path gradient'],'scientific_scope':'operator mechanism characterization, not dataset prevalence or trained-model performance'}
plan_file=OUT/'PLAN_CODEX.json'
if plan_file.exists():assert json.loads(plan_file.read_text())==PLAN
else:plan_file.write_text(json.dumps(PLAN,indent=2))

def use_fix(on):
    orig.DSC._bilinear_interpolate_3D=ORIGINAL
    if on:patch_dsc(orig)

def rel(a,b):
    return (a-b).flatten(1).norm(dim=1)/b.flatten(1).norm(dim=1).clamp_min(1e-12)

rows=[];controls=[]
for morph in PLAN['morphs']:
    for seed in PLAN['seeds']:
        torch.manual_seed(seed)
        m=orig.DSConv(8,16,9,1.0,morph,True,'cpu').eval()
        all_x=torch.randn(16,8,40,36)
        for B in PLAN['batches']:
            x=all_x[:B]
            with torch.no_grad():
                use_fix(False)
                baseline=torch.cat([m(x[i:i+1]) for i in range(B)])
                y=m(x)
                use_fix(True)
                fixed=m(x)
            if B==1:assert torch.equal(y,fixed)
            assert torch.allclose(fixed,baseline,atol=2e-5,rtol=2e-5),(morph,seed,B)
            error=rel(y,baseline);fixed_error=rel(fixed,baseline)
            for j in range(B):
                rows.append({'morph':morph,'seed':seed,'batch':B,'sample_index':j,'original_relative_l2':error[j].item(),'fixed_relative_l2':fixed_error[j].item(),'original_max_abs':(y[j]-baseline[j]).abs().max().item(),'fixed_max_abs':(fixed[j]-baseline[j]).abs().max().item()})
        x=all_x[:4]
        permutation=torch.tensor([2,0,3,1])
        with torch.no_grad():
            use_fix(False);a=m(x);p=m(x[permutation]);dup=m(x[0:1].repeat(4,1,1,1));dup_ref=m(x[0:1]).repeat(4,1,1,1)
            orig_perm=(p-a[permutation]).abs().max().item()
            use_fix(True);af=m(x);pf=m(x[permutation])
            fix_perm=(pf-af[permutation]).abs().max().item()
        assert torch.allclose(pf,af[permutation],atol=2e-5,rtol=2e-5)
        assert torch.allclose(dup,dup_ref,atol=2e-5,rtol=2e-5)
        controls.append({'morph':morph,'seed':seed,'original_permutation_max_abs':orig_perm,'fixed_permutation_max_abs':fix_perm,'identical_inputs_max_abs':(dup-dup_ref).abs().max().item()})
    print(json.dumps({'morph_complete':morph,'rows':len(rows)}),flush=True)

# Hold coordinates fixed to isolate the feature-gather gradient; a complete
# learned-offset module additionally has a gradient through its offset path.
gradient_rows=[];examples={}
for morph in (0,1):
    torch.manual_seed(123)
    x=torch.randn(4,1,12,12)
    offset=torch.zeros(4,18,12,12)
    dsc=orig.DSC(x.shape,9,1.0,morph,'cpu')
    ycoord,xcoord=dsc._coordinate_map_3D(offset,False)
    for fixed in (False,True):
        use_fix(fixed);xx=x.clone().requires_grad_(True)
        yy=dsc._bilinear_interpolate_3D(xx,ycoord,xcoord)
        yy[1].square().mean().backward()
        g=xx.grad.flatten(1).norm(dim=1)
        expected=1 if fixed else 0
        assert g[expected]>0 and torch.count_nonzero(g)==1
        for donor,norm in enumerate(g.tolist()):
            gradient_rows.append({'morph':morph,'implementation':'fixed' if fixed else 'original','loss_sample':1,'input_sample':donor,'gradient_l2':norm})
        if not fixed:
            assert all(torch.equal(yy[0],yy[j]) for j in range(1,4))
    # Distinct constant sample features expose the source index directly.
    values=torch.arange(1,5,dtype=torch.float32).reshape(4,1,1,1).expand(4,1,12,12).clone()
    for fixed in (False,True):
        use_fix(fixed)
        z=dsc._bilinear_interpolate_3D(values,ycoord,xcoord)
        # Interior point avoids original boundary-weight behavior.
        centers=z[:,:,3:6,3:6].mean((1,2,3)).tolist()
        examples[f'morph{morph}_'+('fixed' if fixed else 'original')]=centers
use_fix(False)

def csv_write(name,rs):
    p=OUT/name
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
csv_write('per_sample.csv',rows);csv_write('controls.csv',controls);csv_write('gather_gradient.csv',gradient_rows)
summary=[]
for morph in (0,1):
    for B in PLAN['batches']:
        original=[];fixed=[];nonfirst=[]
        for seed in PLAN['seeds']:
            rs=[r for r in rows if r['morph']==morph and r['batch']==B and r['seed']==seed]
            original.append(np.mean([r['original_relative_l2'] for r in rs]))
            fixed.append(np.mean([r['fixed_relative_l2'] for r in rs]))
            if B>1:nonfirst.append(np.mean([r['original_relative_l2'] for r in rs if r['sample_index']>0]))
        summary.append({'morph':morph,'batch':B,'n_seeds':20,'original_batch_mean_relative_l2':float(np.mean(original)),'original_between_seed_sd':float(np.std(original,ddof=1)),'original_nonfirst_relative_l2':float(np.mean(nonfirst)) if nonfirst else 0.,'fixed_batch_mean_relative_l2':float(np.mean(fixed)),'fixed_between_seed_sd':float(np.std(fixed,ddof=1))})
csv_write('batch_summary.csv',summary)
source_lines=SOURCE.read_text(encoding='utf-8').splitlines()
upstream_lines={'batch_base':[i for i,line in enumerate(source_lines,1) if 'base_y0 = base +' in line or 'base_y1 = base +' in line], 'cancellation':[i for i,line in enumerate(source_lines,1) if ' = base_y' in line and ' - base +' in line]}
assert len(upstream_lines['cancellation'])==4
result={'complete':True,'device':'cpu','records':len(rows),'plan':PLAN,'batch_summary':summary,'gradient_rows':gradient_rows,'constant_feature_example':examples,'max_fixed_batch_dependence':max(r['fixed_max_abs'] for r in rows),'max_original_permutation_error':max(r['original_permutation_max_abs'] for r in controls),'max_fixed_permutation_error':max(r['fixed_permutation_max_abs'] for r in controls),'max_identical_input_error':max(r['identical_inputs_max_abs'] for r in controls),'batch1_bitwise_identity':True,'upstream_lines':upstream_lines,'completed_epoch':time.time()}
(OUT/'RESULTS.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ('complete','device','records','max_fixed_batch_dependence','batch1_bitwise_identity')}),flush=True)
