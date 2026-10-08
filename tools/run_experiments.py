"""Serial reproduction commands with validation-only LR selection and stable run names."""
import argparse,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--phase',choices=['tuning','e2','main','posthoc','all'],required=True)
parser.add_argument('--dry-run',action='store_true')
parser.add_argument('--use-published-lr',action='store_true',help='Use reported choices instead of newly measured validation choices')
parser.add_argument('--include-chase-sumprod',action='store_true',help='Unreported optional five-seed extension; keep it separate from primary CHASE results')
args=parser.parse_args()
lr=json.loads((ROOT/'configs/lr_selected.json').read_text())
runroot=ROOT/'runs'
def run(dataset='drive',variant='dscnet',kind='none',dsconv='orig',batch=1,rate=.0002,seed=0,role='pre_registered'):
    name=f'{dataset}_{variant}_{kind}_{dsconv}_b{batch}_lr{rate:g}_s{seed}'
    command=[sys.executable,str(ROOT/'train_clean.py'),'--run',name,'--dataset',dataset,'--variant',variant,'--kind',kind,'--model','plus' if kind!='none' else 'dscnet','--insert','b2,b3,b4','--lr',str(rate),'--seed',str(seed),'--epochs','400','--bs',str(batch),'--roi_per_image',str(batch),'--dsconv',dsconv,'--study_role',role,'--diag_test']
    print(' '.join(command),flush=True)
    if not args.dry_run:subprocess.run(command,cwd=ROOT,check=True)
def select(dataset,variant,dsconv,batch,rates,key):
    if args.use_published_lr:return lr[key]
    candidates=[]
    for rate in rates:
        path=runroot/f'{dataset}_{variant}_none_{dsconv}_b{batch}_lr{rate:g}_s0'/'selection.json'
        if not path.exists():
            if args.dry_run:return lr[key]
            raise RuntimeError(f'Run tuning first: missing {path.relative_to(ROOT)}')
        candidates.append((json.loads(path.read_text())['best_dice'],rate))
    return max(candidates,key=lambda p:p[0])[1]
phases=['tuning','e2','main','posthoc'] if args.phase=='all' else [args.phase]
for phase in phases:
    if phase=='tuning':
        for dataset,variant in [('drive','unet'),('chase','unet'),('chase','dscnet')]:
            for rate in (.00005,.0001,.0002):run(dataset,variant,rate=rate)
        for rate in (.0002,.0005,.001):run(dsconv='fix',batch=8,rate=rate)
        for dataset in ('drive','chase'):
            for seed in range(5):run(dataset,rate=.0001,seed=seed)
    if phase in ('e2','main'):
        lr['drive_unet']=select('drive','unet','orig',1,(.00005,.0001,.0002),'drive_unet')
        lr['chase_unet']=select('chase','unet','orig',1,(.00005,.0001,.0002),'chase_unet')
        lr['chase_dscnet']=select('chase','dscnet','orig',1,(.00005,.0001,.0002),'chase_dscnet')
        lr['E2']=select('drive','dscnet','fix',8,(.0002,.0005,.001),'E2')
        if not args.dry_run:
            runroot.mkdir(exist_ok=True);(runroot/'lr_selected.json').write_text(json.dumps(lr,indent=2))
    if phase=='e2':
        for dsconv in ('orig','fix'):
            for seed in range(5):run(dsconv=dsconv,batch=8,rate=lr['E2'],seed=seed)
    if phase=='main':
        for dataset in ('drive','chase'):
            for variant,kind in [('unet','none'),('dscnet','none'),('dscnet','conv'),('dscnet','sumprod'),('dscnet','trop'),('dscnet','mamba')]:
                if dataset=='chase' and kind=='sumprod' and not args.include_chase_sumprod:continue
                rate=lr[dataset+'_unet'] if variant=='unet' else (.0002 if dataset=='drive' else lr['chase_dscnet'])
                for seed in range(5):run(dataset,variant,kind,rate=rate,seed=seed)
    if phase=='posthoc':
        plan=json.loads((ROOT/'configs/POSTHOC_EXPERIMENT_PLAN.json').read_text())
        for c in plan['configurations']:run(c['dataset'],c['variant'],c['kind'],c['dsconv'],c['bs'],c['lr'],c['seed'],c['study_role'])
        if not args.dry_run:
            runroot.mkdir(exist_ok=True);(runroot/'POSTHOC_EXPERIMENT_PLAN.json').write_text(json.dumps(plan,indent=2))
