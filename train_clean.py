# Copyright (c) 2025 Yaolei Qi
# Copyright (c) 2026 Heng Zhang
# SPDX-License-Identifier: MIT
# Adapted from DSCNet; see THIRD_PARTY_NOTICES.md.
"""DRIVE experiments. Upstream files are imported unmodified; all outputs are separate."""
import argparse, contextlib, copy, hashlib, io, json, os, random, shutil, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
UP = Path(os.environ.get('DSCNET_DIR',ROOT/'DSCNet'))/'DSCNet_2D_opensource/Code/DRIVE/DSCNet'
DATA = Path(os.environ.get('DSCNET_DIR',ROOT/'DSCNet'))/'DSCNet_2D_opensource/Data/DRIVE'
sys.path[:0] = [str(UP), str(ROOT/'modules')]
import SimpleITK as sitk
from S3_DSCNet import DSCNet
from S3_Dataloader import Dataloader
from S3_Loss import cross_loss
from S3_Train_Process import predict, Dice
from S1_Pre_Getmeanstd import Getmeanstd
from dscnet_plus import DSCNetPlus
from topo_metrics import evaluate
from cldice_loss import SoftClDiceLoss
from monai.utils import set_determinism
from monai.transforms import Randomizable

def atomic(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)); tmp.replace(path)

def save_pt(path,obj):
    tmp=Path(str(path)+'.tmp');torch.save(obj,tmp);tmp.replace(path)

def files(folder):
    return sorted((DATA/folder).glob('*.nii.gz'),key=lambda p:int(p.name.split('.')[0]))

def txt(path, values):
    path.write_text('\n'.join(map(str,values))+'\n');return str(path)

def prepare(dataset='drive'):
    if dataset=='chase':
        from prepare_chase import prepare_split
        return prepare_split()
    dest=ROOT/'runs/data';dest.mkdir(parents=True,exist_ok=True)
    alltr=files('Image_Tr');te=files('Image_Te')
    assert len(alltr)==20 and len(te)==20
    val=[p for p in alltr if int(p.name.split('.')[0]) in {21,26,31,36}]
    if len(val)!=4:val=[alltr[i] for i in (0,5,10,15)]
    tr=[p for p in alltr if p not in val]
    split={'train':[p.name for p in tr],'val':[p.name for p in val],'test':[p.name for p in te]}
    if (dest/'split.json').exists():assert json.loads((dest/'split.json').read_text())==split
    atomic(dest/'split.json',split)
    # Exactly upstream dataset-wise, label-free intensity normalization.
    for name,im in [('original_train',alltr),('clean_train',tr),('val',val),('test',te)]:
        txt(dest/(name+'.txt'),im)
        lab='Label_Te' if name=='test' else 'Label_Tr'
        txt(dest/(name+'_labels.txt'),[DATA/lab/p.name for p in im])
        if not (dest/(name+'.npy')).exists():
            # Preserve upstream float32 per-image summation and image ordering.
            arrays=[sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32) for p in im]
            count=sum(x.size for x in arrays);mean=0.0;var=0.0
            for x in arrays:mean+=np.sum(x)
            mean=mean/count
            for x in arrays:var+=np.sum(np.square(x-mean))
            np.save(dest/(name+'.npy'),[mean,np.sqrt(var/count)])
    return dest

def model_for(a):
    if a.variant=='unet':
        from unet_same_skeleton import UNetSameSkeleton
        return UNetSameSkeleton().cuda()
    cls=DSCNet
    if a.pro:
        from S3_DSCNet_pro import DSCNet as cls
    base=cls(1,1,9,1.0,True,torch.device('cuda'),16,1)
    return (base if a.kind=='none' else DSCNetPlus(base,tuple(a.insert.split(',')),a.kind)).cuda()

def measure(model):
    from fvcore.nn import FlopCountAnalysis
    # fvcore tracing turns shape scalars into CPU tensors in upstream DSConv's
    # clamp; trace a CPU copy to avoid device mismatch without editing upstream.
    traced=copy.deepcopy(model).cpu().eval()
    for module in traced.modules():
        if hasattr(module,'device'):module.device=torch.device('cpu')
    # Official Mamba kernels cannot execute on CPU. Profile its backbone by
    # fvcore and explicitly report plugin operations as uncounted.
    mamba_omitted=[]
    if hasattr(traced,'kind') and traced.kind=='mamba':
        mamba_omitted=list(traced.extra)
        for k in mamba_omitted:traced.extra[k]=torch.nn.Identity()
    with torch.no_grad(),contextlib.redirect_stderr(io.StringIO()):
        counter=FlopCountAnalysis(traced,torch.zeros(1,1,224,224))
        counter.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        flops=counter.total();unsupported=dict(counter.unsupported_ops())
    return {'parameters':sum(p.numel() for p in model.parameters()),'flops_224_fvcore':flops,
            'flops_unsupported_ops':unsupported,'mamba_blocks_not_counted':mamba_omitted,'flops_note':'fvcore CPU tracing counts one multiply-add as one FLOP; unsupported operations and Mamba plugins excluded, so this is a partial count, not full scan complexity.'}

def infer(model, data, subset, out, resume=False):
    out.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(root_dir=str(data)+'/',Te_Meanstd_name=subset+'.npy',ROI_shape=224,n_classes=1)
    paths=(data/(subset+'.txt')).read_text().splitlines()
    # Upstream inference has singular center weights. Retain it exactly: original
    # threshold >.5 maps its NaNs to background. Never apply a hidden kernel fix.
    with torch.no_grad(),contextlib.redirect_stdout(io.StringIO()):
        for source in paths:
            dest=out/Path(source).name
            if resume and dest.exists():
                sitk.ReadImage(str(dest));continue
            one=out/'one_image.txt';one.write_text(source+'\n')
            predict(model,str(one),str(out)+'/',args)
        scores=Dice(str(data/(subset+'_labels.txt')),str(out)+'/')
    return float(np.mean(scores))

def metrics(data,subset,out):
    rows=[]
    for lab in (data/(subset+'_labels.txt')).read_text().splitlines():
        prob=sitk.GetArrayFromImage(sitk.ReadImage(str(out/Path(lab).name)))
        gt=sitk.GetArrayFromImage(sitk.ReadImage(lab));gt=gt/gt.max()
        row={'image':Path(lab).name,**evaluate(prob>0.5,gt>0.5),
             'upstream_inference_nonfinite_pixels':int((~np.isfinite(prob)).sum())}
        rows.append(row)
    keys=('dice','cldice','betti0_err','betti1_err','patch_betti0_err','patch_betti1_err')
    return {'per_image':rows,'mean':{k:float(np.mean([r[k] for r in rows])) for k in keys}}

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    Randomizable.R=np.random.RandomState(seed)
    # Preserve FP32 and disable TF32. GPU reductions can still be nondeterministic.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True);p.add_argument('--model',choices=['dscnet','plus'],default='dscnet')
    p.add_argument('--kind',choices=['none','trop','sumprod','conv','mamba'],default='none')
    p.add_argument('--insert',default='b3');p.add_argument('--loss',choices=['ce','ce+cldice'],default='ce')
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=400);p.add_argument('--protocol',choices=['original','clean'],default='clean')
    p.add_argument('--evaluate-only',action='store_true');p.add_argument('--defer-test',action='store_true')
    p.add_argument('--dsconv',choices=['orig','fix','shuffle'],default='orig')
    p.add_argument('--study_role',choices=['pre_registered','posthoc_lr','mechanism_posthoc'],default='pre_registered')
    p.add_argument('--bs',type=int,default=1);p.add_argument('--roi_per_image',type=int,default=1)
    p.add_argument('--diag_test',action='store_true')
    p.add_argument('--variant',choices=['dscnet','unet'],default='dscnet')
    p.add_argument('--dataset',choices=['drive','chase'],default='drive')
    p.add_argument('--trace_equivalence',action='store_true')
    p.add_argument('--pro',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);seed_all(a.seed)
    out=ROOT/'runs'/a.run;out.mkdir(parents=True,exist_ok=True);data=prepare(a.dataset)
    config={k:v for k,v in vars(a).items() if k not in ('evaluate_only','defer_test','study_role')}
    if a.study_role!='pre_registered':config['study_role']=a.study_role
    config.update(batch=a.bs,roi=224,precision='FP32',optimizer='AdamW',betas=[.9,.95],weight_decay=.01,
                  scheduler='constant',shared_gpu=True,checkpoint_selection='test_dice' if a.protocol=='original' else 'validation_dice',
                  threshold=.5,cldice_weight=.5,cldice_iters=10)
    if a.protocol=='clean':config['implementation']='upstream DSCNet + validated static-coordinate/source-image cache; inference batch1'
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    atomic(out/'config.json',config)
    if a.dsconv in ('fix','shuffle'):
        import S3_DSConv
        from dsconv_fix import patch_dsc
        patch_dsc(S3_DSConv)
    model=model_for(a)
    if not (out/'complexity.json').exists():atomic(out/'complexity.json',measure(model))
    if a.protocol=='clean':
        from performance_cache import enable
        enable()
    if a.dsconv=='shuffle':
        from dsconv_shuffle import patch_shuffle
        patch_shuffle(S3_DSConv)
    # Reset data RNG after model construction/profiling, preventing variant-specific
    # parameter initialization from shifting augmentation/shuffle random streams.
    seed_all(a.seed)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.9,.95))
    ce=cross_loss();cl=SoftClDiceLoss(iters=10)
    subset='original_train' if a.protocol=='original' else 'clean_train'
    args=SimpleNamespace(root_dir=str(data)+'/',Tr_Meanstd_name=subset+'.npy',ROI_shape=224,
                         Image_Tr_txt=str(data/(subset+'.txt')),Label_Tr_txt=str(data/(subset+'_labels.txt')))
    dataset=Dataloader(args)
    if a.roi_per_image!=1:
        class RepeatedROI(torch.utils.data.Dataset):
            def __len__(self):return len(dataset)*a.roi_per_image
            def __getitem__(self,index):return dataset[index//a.roi_per_image]
        dataset_repeated=RepeatedROI()
    else:dataset_repeated=dataset
    loader=torch.utils.data.DataLoader(dataset_repeated,batch_size=a.bs,shuffle=True,num_workers=0)
    assert len(dataset_repeated)%a.bs==0
    start=0;best=-1;best_epoch=0;best_val=None
    if (out/'last.pt').exists() and not a.evaluate_only:
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        start=ck['epoch'];best=ck['best'];best_epoch=ck['best_epoch'];best_val=ck['best_val']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
        Randomizable.R.set_state(ck['monai_rng'])
        # Reconcile logs to the durable checkpoint. A kill after a diagnostic
        # append but before last.pt must not duplicate the repeated epoch.
        for filename in ('epochs.jsonl','diag_test.jsonl','forward_trace.jsonl'):
            log=out/filename
            if log.exists():
                retained=[json.loads(line) for line in log.read_text().splitlines() if line.strip() and json.loads(line)['epoch']<=start]
                if filename=='epochs.jsonl' and ck.get('last_row') and not any(row['epoch']==start for row in retained):retained.append(ck['last_row'])
                log.write_text(''.join(json.dumps(row)+'\n' for row in retained))
    if not a.evaluate_only:
        for epoch in range(start,a.epochs):
            t=time.monotonic();model.train();losses=[]
            for image,label in loader:
                image=image.cuda();label=label.cuda();opt.zero_grad();model.zero_grad()
                pred=model(image)
                if a.trace_equivalence:
                    with (out/'forward_trace.jsonl').open('a') as f:
                        f.write(json.dumps({'epoch':epoch+1,'sha256':hashlib.sha256(pred.detach().cpu().numpy().tobytes()).hexdigest()})+'\n')
                if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite model output')
                loss=ce(label,pred)+(0.5*cl(label,pred) if a.loss=='ce+cldice' else 0)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradient')
                opt.step();losses.append(loss.item())
            torch.cuda.synchronize();training_seconds=time.monotonic()-t
            verify=epoch>=200 and (a.protocol=='original' or (epoch-200)%5==0)
            score=None
            if verify:
                target='test' if a.protocol=='original' else 'val'
                score=infer(model,data,target,out/'current_predictions')
                if score>best:
                    best=score;best_epoch=epoch+1
                    save_pt(out/'best.pt',{'model':model.state_dict(),'epoch':best_epoch,'score':best})
                    if a.protocol=='original':
                        (out/'best_predictions').mkdir(exist_ok=True)
                        for f in (out/'current_predictions').glob('*.nii.gz'):shutil.copy2(f,out/'best_predictions'/f.name)
                    else:best_val=metrics(data,'val',out/'current_predictions')
                    atomic(out/'selection.json',{'best_dice':best,'best_epoch':best_epoch,'validation_metrics':best_val})
            if verify and a.diag_test:
                diagnostic_dice=infer(model,data,'test',out/'diag_predictions')
                with (out/'diag_test.jsonl').open('a') as f:
                    f.write(json.dumps({'epoch':epoch+1,'test_dice':diagnostic_dice,'validation_dice':score})+'\n')
            ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'best':best,'best_epoch':best_epoch,
                'best_val':best_val,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'monai_rng':Randomizable.R.get_state()}
            row={'epoch':epoch+1,'loss':float(np.mean(losses)),'selection_dice':score,'best_dice':best,
                 'train_seconds':training_seconds,'epoch_seconds':time.monotonic()-t,
                 'peak_cuda_mb':torch.cuda.max_memory_allocated()/2**20,'time':time.time()}
            ck['last_row']=row
            save_pt(out/'last.pt',ck)
            with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            atomic(out/'progress.json',row);print(json.dumps(row),flush=True)
        atomic(out/'training_done.json',{'best_dice':best,'best_epoch':best_epoch,'validation_metrics':best_val})
    if a.defer_test and a.protocol=='clean':return
    if (out/'metrics.json').exists() and (out/'last_metrics.json').exists() and (a.bs!=8 or (out/'metrics_batch8.json').exists()):return
    final=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    if not (out/'last_metrics.json').exists():
        model.load_state_dict(final['model']);model.eval()
        infer(model,data,'test',out/'last_predictions',resume=True)
        last_result=metrics(data,'test',out/'last_predictions');last_result['epoch']=final['epoch']
        atomic(out/'last_metrics.json',last_result)
    selected=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model']);model.eval()
    if a.protocol=='original':predpath=out/'best_predictions'
    else:
        predpath=out/'test_predictions'
        atomic(out/'test_checkpoint.json',{'epoch':selected['epoch'],'sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()})
        infer(model,data,'test',predpath,resume=True)
    result=metrics(data,'test',predpath)
    with contextlib.redirect_stdout(io.StringIO()):orig=Dice(str(data/'test_labels.txt'),str(predpath)+'/')
    result.update(original_dice=float(np.mean(orig)),selected_epoch=selected['epoch'],protocol=a.protocol,
                  scope='whole image, no FOV; patch64 stride64, remainder strips excluded',test_evaluated_once=False,
                  primary_result_role='validation-selected checkpoint' if a.protocol=='clean' else 'test-selected checkpoint',
                  diagnostic_test_logging=bool(a.diag_test),additional_final_checkpoint_test=True)
    if a.study_role!='pre_registered':result['study_role']=a.study_role
    atomic(out/'metrics.json',result)
    if a.bs==8:
        from batch_inference import infer_batch8
        infer_batch8(model,data,'test',out/'test_predictions_batch8')
        packed=metrics(data,'test',out/'test_predictions_batch8');packed['selected_epoch']=selected['epoch']
        atomic(out/'metrics_batch8.json',packed)

if __name__=='__main__':main()
