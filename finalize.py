"""Generate all owned E5/E7 assets from finished runs; never fabricate results."""
import json,shutil,subprocess,hashlib,sys,os
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import SimpleITK as sitk
from analysis_stats import paired,holm,power
ROOT=Path(__file__).resolve().parent;R=ROOT/'runs';A=ROOT/'reproduced';T=A/'tables';F=A/'figures'
from ops.audit_completed import audit
audit(ROOT)
T.mkdir(parents=True,exist_ok=True);F.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(20261006)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':9,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#333333']
def read(p):return json.loads(Path(p).read_text())
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def dump(p,x):Path(p).write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False))
def table(name,rows):
    df=pd.DataFrame(rows);df.to_csv(T/(name+'.csv'),index=False)
    (T/(name+'.md')).write_text(df.to_markdown(index=False)+'\n')
    return df
def figure(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),dpi=400,bbox_inches='tight');plt.close(fig)
lr=read(R/'lr_selected.json') if (R/'lr_selected.json').exists() else read(ROOT/'configs/lr_selected.json')
def path(dataset='drive',variant='dscnet',kind='none',learning_rate=2e-4,seed=0,bs=1,dsconv='orig'):
    return R/f'{dataset}_{variant}_{kind}_{dsconv}_b{bs}_lr{learning_rate:g}_s{seed}'
groups={}
drive={}
for label,kind in [('dscnet','none'),('+conv','conv'),('+sumprod','sumprod'),('+trop','trop'),('+mamba','mamba')]:
    pos='b3' if kind=='none' else 'b2-b3-b4'
    drive[label]=[(R/'reused'/f'clean_{kind}_{pos}_lr0.0002_s{s}_ce') if (R/'reused'/f'clean_{kind}_{pos}_lr0.0002_s{s}_ce'/'metrics.json').exists() else path(kind=kind,seed=s) for s in range(5)]
drive['unet']=[path(variant='unet',learning_rate=lr['drive_unet'],seed=s) for s in range(5)]
groups['drive']=drive
chase={}
for label,variant,kind in [('unet','unet','none'),('dscnet','dscnet','none'),('+conv','dscnet','conv'),('+trop','dscnet','trop'),('+mamba','dscnet','mamba')]:
    chase[label]=[path(dataset='chase',variant=variant,kind=kind,learning_rate=lr['chase_unet' if variant=='unet' else 'chase_dscnet'],seed=s) for s in range(5)]
groups['chase']=chase
def results(paths,file='metrics.json'):return [read(p/file) for p in paths]
def matrix(paths,metric,file='metrics.json'):
    records=results(paths,file);ids=sorted(x['image'] for x in records[0]['per_image'])
    assert all(sorted(x['image'] for x in record['per_image'])==ids for record in records)
    return np.array([[next(row[metric] for row in record['per_image'] if row['image']==i) for i in ids] for record in records])
metrics=('dice','cldice','betti0_err','betti1_err','patch_betti0_err','patch_betti1_err')
numbers={'protocol_version':'CODEX_STAGE4 v2 2026-10-06','groups':{},'reuse_provenance':'runs/reuse_provenance.json','statistics':{},'inflation':{},'sources':{},'learning_rates':{'values':lr,'source':'runs/lr_selected.json'},'units':'Dice/clDice stored as proportions in JSON; paper effect fields explicitly pp; beta metrics absolute error.'}
for dataset,g in groups.items():
    rows=[];numbers['groups'][dataset]={}
    perseed=[]
    for label,paths in g.items():
        rs=results(paths);complexity=read(paths[0]/'complexity.json');values=np.array([[x['mean'][m] for m in metrics] for x in rs])
        row={'dataset':dataset,'model':label,'seeds':len(paths),'parameters':complexity['parameters'],'partial_fvcore_flops':complexity['flops_224_fvcore'],'FOV':'whole image, no FOV','flops_note':complexity['flops_note']}
        summary={}
        for j,m in enumerate(metrics):
            mult=100 if m in ('dice','cldice') else 1
            row[m+'_percent' if mult==100 else m]=f'{values[:,j].mean()*mult:.3f} ± {values[:,j].std(ddof=1)*mult:.3f}'
            summary[m]={'mean':float(values[:,j].mean()),'sample_sd':float(values[:,j].std(ddof=1))}
        rows.append(row)
        numbers['groups'][dataset][label]={'summary':summary,'source_files':[str(p.relative_to(ROOT)/'metrics.json') for p in paths],'parameters':complexity['parameters'],'partial_flops':complexity['flops_224_fvcore'],'complexity_source':str(paths[0].relative_to(ROOT)/'complexity.json')}
        for i,record in enumerate(rs):perseed.append({'dataset':dataset,'model':label,'seed':i,'selected_epoch':record['selected_epoch'],**record['mean']})
    table('main_'+dataset,rows);table('per_seed_'+dataset,perseed)
    contrasts=[(label,'dscnet') for label in g if label not in ('dscnet','unet')]+[('dscnet','unet')]
    statrows=[]
    for label,reference in contrasts:
        for metric in ('dice','cldice','betti0_err'):
            statrows.append({'dataset':dataset,'model':label,'reference':reference,'metric':metric,**paired(matrix(g[label],metric),matrix(g[reference],metric),rng)})
    for row in statrows:
        row['delta_unit']='proportion' if row['metric'] in ('dice','cldice') else 'absolute error'
        if row['metric'] in ('dice','cldice'):
            for key in ('delta','ci_low','ci_high'):row[key+'_pp']=100*row[key]
    # Holm family: all planned model/metric contrasts within each dataset.
    holm(statrows);table('stats_'+dataset,statrows);numbers['statistics'][dataset]=statrows
    raw=[]
    for label,paths in g.items():
        for s,p in enumerate(paths):
            raw += [{'model':label,'seed':s,**row} for row in read(p/'metrics.json')['per_image']]
    table('per_image_'+dataset,raw)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained')
    for j,metric in enumerate(('dice','cldice')):
        for index,(label,paths) in enumerate(g.items()):
            values=matrix(paths,metric).mean(1)*100;offset=np.linspace(-.12,.12,len(values))
            axes[j].scatter(index+offset,values,c=colors[index%len(colors)],s=16,zorder=3)
            axes[j].errorbar(index,values.mean(),yerr=values.std(ddof=1),c='black',fmt='_',capsize=3,lw=.8)
        axes[j].set_xticks(range(len(g)),g.keys(),rotation=25,ha='right');axes[j].set_ylabel(metric+' (%)');axes[j].set_title(dataset.upper()+'; points=seeds, bars=sample SD')
    figure(fig,'fig3_seeds_'+dataset)
# Tuning scans and defaults, including reused DRIVE tuning evidence.
tuning=[]
for p in list(R.glob('drive_*'))+list(R.glob('chase_*')):
    if not (p/'config.json').exists() or not (p/'training_done.json').exists():continue
    c=read(p/'config.json');sel=read(p/'training_done.json');tuning.append({'study_role':c.get('study_role','pre_registered'),'source_run':str(p.relative_to(ROOT)),'dataset':c['dataset'],'variant':c['variant'],'kind':c['kind'],'dsconv':c['dsconv'],'batch':c['bs'],'lr':c['lr'],'seed':c['seed'],'validation_dice':sel['best_dice'],'selected_epoch':sel['best_epoch']})
for p in (R/'reused').glob('*'):
    c=read(p/'config.json');sel=read(p/'training_done.json');tuning.append({'source_run':str(p.relative_to(ROOT)),'dataset':'drive','variant':'dscnet','kind':c['kind'],'dsconv':'orig','batch':1,'lr':c['lr'],'seed':c['seed'],'validation_dice':sel['best_dice'],'selected_epoch':sel['best_epoch']})
table('tuning',tuning)
# E2 all seed0 scans plus selected five-seed groups, both inference batches.
e2={fix:[path(bs=8,dsconv=fix,learning_rate=lr['E2'],seed=s) for s in range(5)] for fix in ('orig','fix')}
rows=[];estat=[]
for fix,paths in e2.items():
    for file,infer_batch in [('metrics.json',1),('metrics_batch8.json',8)]:
        values=np.array([[v['mean'][m] for m in metrics] for v in results(paths,file)])
        row={'implementation':fix,'training_batch':8,'inference_batch':infer_batch,'lr':lr['E2'],'seeds':5}
        for j,m in enumerate(metrics):
            mult=100 if m in ('dice','cldice') else 1;row[m+'_percent' if mult==100 else m]=f'{values[:,j].mean()*mult:.3f} ± {values[:,j].std(ddof=1)*mult:.3f}'
        rows.append(row)
for file,infer_batch in [('metrics.json',1),('metrics_batch8.json',8)]:
    for metric in ('dice','cldice','betti0_err','betti1_err'):
        estat.append({'inference_batch':infer_batch,'metric':metric,**paired(matrix(e2['fix'],metric,file),matrix(e2['orig'],metric,file),rng)})
refvalues=np.array([[v['mean'][m] for m in metrics] for v in results(groups['drive']['dscnet'])])
refrow={'implementation':'orig (stage2 reference)','training_batch':1,'inference_batch':1,'lr':2e-4,'seeds':5}
for j,m in enumerate(metrics):
    mult=100 if m in ('dice','cldice') else 1;refrow[m+'_percent' if mult==100 else m]=f'{refvalues[:,j].mean()*mult:.3f} ± {refvalues[:,j].std(ddof=1)*mult:.3f}'
rows.append(refrow)
table('E2_bug_impact',rows);holm(estat);table('stats_E2',estat);numbers['E2']={'lr':lr['E2'],'groups':{k:[str(p.relative_to(ROOT)) for p in v] for k,v in e2.items()},'statistics':estat,'batch1_reference':numbers['groups']['drive']['dscnet']}
# E3 diagnostic checkpoint inflation, best seed and fixed-vs-tuned learning rate.
inflation=[];posthoc_inflation=[]
for p in R.glob('*'):
    log=p/'diag_test.jsonl'
    if not log.exists() or not (p/'metrics.json').exists():continue
    records=[json.loads(line) for line in log.read_text().splitlines()];c=read(p/'config.json');selected=read(p/'metrics.json')
    assert len(records)==40 and [x['epoch'] for x in records]==list(range(201,397,5))
    # Use identical upstream float32 mean Dice for the difference.
    winner=max(records,key=lambda x:x['test_dice'])
    diagnostic_selected=next(row for row in records if row['epoch']==selected['selected_epoch'])
    assert abs(diagnostic_selected['test_dice']-selected['original_dice'])<1e-5,(p,diagnostic_selected,selected['original_dice'])
    destination=inflation if c.get('study_role','pre_registered')=='pre_registered' else posthoc_inflation
    destination.append({'study_role':c.get('study_role','pre_registered'),'run':str(p.relative_to(ROOT)),'dataset':c['dataset'],'variant':c['variant'],'kind':c['kind'],'dsconv':c['dsconv'],'batch':c['bs'],'lr':c['lr'],'seed':c['seed'],'optimistic_test_epoch':winner['epoch'],'validation_selected_epoch':selected['selected_epoch'],'inflation_pp':100*(winner['test_dice']-selected['original_dice'])})
if posthoc_inflation:table('E3_posthoc_checkpoint_inflation',posthoc_inflation)
idf=table('E3_checkpoint_inflation_per_run',inflation)
summary=[]
for keys,frame in idf.groupby(['dataset','variant','kind','dsconv','batch','lr'],sort=True):
    row=dict(zip(['dataset','variant','kind','dsconv','batch','lr'],keys));row.update(mean_pp=frame.inflation_pp.mean(),sample_sd_pp=frame.inflation_pp.std(ddof=1),max_pp=frame.inflation_pp.max(),runs=len(frame));summary.append(row)
table('E3_checkpoint_inflation',summary)
seedbias=[]
for dataset,g in groups.items():
    for label,paths in g.items():
        values=matrix(paths,'dice').mean(1)*100;seedbias.append({'dataset':dataset,'model':label,'best_seed_minus_mean_pp':float(values.max()-values.mean()),'sample_sd_pp':float(values.std(ddof=1)),'seeds':len(paths)})
table('E3_best_seed_inflation',seedbias)
default_paths={'drive':[R/'reused'/f'clean_none_b3_lr0.0001_s{s}_ce' if (R/'reused'/f'clean_none_b3_lr0.0001_s{s}_ce'/'metrics.json').exists() else path(learning_rate=1e-4,seed=s) for s in range(5)],'chase':[path(dataset='chase',learning_rate=1e-4,seed=s) for s in range(5)]}
lrbias=[]
for dataset in groups:
    delta=matrix(groups[dataset]['dscnet'],'dice').mean(1)-matrix(default_paths[dataset],'dice').mean(1)
    lrbias.append({'dataset':dataset,'default_lr':1e-4,'tuned_lr':2e-4 if dataset=='drive' else lr['chase_dscnet'],'tuned_minus_default_mean_pp':float(delta.mean()*100),'sample_sd_pp':float(delta.std(ddof=1)*100),'per_seed_difference_pp':json.dumps((delta*100).tolist())})
table('E3_lr_effect',lrbias);numbers['inflation']={'checkpoint':summary,'seed':seedbias,'learning_rate':lrbias,'source':'tables/E3_*'}
fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
for ax,(dataset,g) in zip(axes,groups.items()):
    vals=[];labels=[]
    for label,paths in g.items():
        if label in ('unet','dscnet'):continue
        vals.append(100*(matrix(paths,'dice').mean()-matrix(g['dscnet'],'dice').mean()));labels.append(label+' − DSC')
    subset=idf[idf.dataset==dataset];vals += [subset.inflation_pp.mean(),max(v['best_seed_minus_mean_pp'] for v in seedbias if v['dataset']==dataset),next(v['tuned_minus_default_mean_pp'] for v in lrbias if v['dataset']==dataset)]
    labels += ['Test checkpoint choice','Best seed (maximum)','Learning-rate tuning']
    ax.barh(range(len(vals)),vals,color=[colors[i%len(colors)] for i in range(len(vals))]);ax.axvline(0,color='black',lw=.6);ax.set_yticks(range(len(vals)),labels);ax.set_xlabel('Dice difference (percentage points)');ax.set_title(dataset.upper())
figure(fig,'fig2_inflation_vs_modules')
# Fixed examples: seed0 DSC defaults, selected without viewing test rankings.
for dataset in groups:
    p=default_paths[dataset][2] if dataset=='drive' else groups[dataset]['dscnet'][0]
    train=[json.loads(line) for line in (p/'epochs.jsonl').read_text().splitlines()];diag=[json.loads(line) for line in (p/'diag_test.jsonl').read_text().splitlines()]
    fig,ax=plt.subplots(figsize=(3.5,2.5),layout='constrained')
    val=[v for v in train if v['selection_dice'] is not None];ax.plot([v['epoch'] for v in val],[v['selection_dice']*100 for v in val],label='Validation',color=colors[0]);ax.plot([v['epoch'] for v in diag],[v['test_dice']*100 for v in diag],label='Diagnostic test',color=colors[1],linestyle='--')
    ax.axvline(read(p/'metrics.json')['selected_epoch'],color='black',ls=':',lw=.8,label='Validation choice');ax.set_xlabel('Epoch');ax.set_ylabel('Dice (%)');ax.legend(frameon=False,fontsize=7);figure(fig,'fig4_curves_'+dataset)
# E2 validation curves and both inference-batch results.
fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained')
for i,(fix,paths) in enumerate(e2.items()):
    records=[json.loads(line) for line in (paths[0]/'epochs.jsonl').read_text().splitlines()];records=[v for v in records if v['selection_dice'] is not None]
    axes[0].plot([v['epoch'] for v in records],[v['selection_dice']*100 for v in records],label=fix,color=colors[i],ls='-' if i==0 else '--')
    for j,file in enumerate(('metrics.json','metrics_batch8.json')):
        vals=matrix(paths,'dice',file).mean(1)*100;axes[1].scatter(np.full(5,i*3+j)+np.linspace(-.12,.12,5),vals,c=colors[i],s=16)
axes[0].set_xlabel('Epoch');axes[0].set_ylabel('Validation Dice (%)');axes[0].legend(frameon=False)
axes[1].set_xticks([0,1,3,4],['Original\nB1 infer','Original\nB8 infer','Fixed\nB1 infer','Fixed\nB8 infer']);axes[1].set_ylabel('Test Dice (%); five seeds');figure(fig,'fig7_E2_bug_effect')
# E5 conditional bootstrap sensitivity, reporting null calibration too.
power_records={}
for dataset,g in groups.items():
    rows=power(matrix(g['dscnet'],'dice'),rng);table('power_'+dataset,rows);power_records[dataset]=rows
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained')
    for ax,axis in zip(axes,('seeds','images')):
        subset=[row for row in rows if row['axis']==axis];xx=[row[axis] for row in subset]
        ax.plot(xx,[row['power'] for row in subset],marker='o',ms=3,label='+0.5 pp Dice',color=colors[0]);ax.plot(xx,[row['null_rejection_rate'] for row in subset],ls='--',label='Null rejection',color=colors[1]);ax.axhline(.8,color='black',ls=':',lw=.8);ax.set_xlabel('Seeds per model' if axis=='seeds' else 'Resampled test images');ax.set_ylabel('Rejection probability');ax.set_ylim(0,1.02);ax.legend(frameon=False,fontsize=7)
    figure(fig,'fig6_power_'+dataset)
numbers['power']={'records':power_records,'source':'tables/power_*.csv','interpretation':'Conditional sensitivity simulation from only five DSC seeds, independently resampled model groups +0.005 shift, whole seed rows then paired image bootstrap. Test-image extrapolation assumes exchangeability; null rejection calibration reported. Not post-hoc achieved power or guarantee.'}
# Typical qualitative example: median DSC seed0 test Dice image, declared rule.
qual=[]
for dataset,g in groups.items():
    records=read(g['dscnet'][0]/'metrics.json')['per_image'];ordered=sorted(records,key=lambda v:(v['dice'],v['image']));image=ordered[len(ordered)//2]['image']
    image_root=Path(os.environ.get('DSCNET_DIR',ROOT/'DSCNet'))/'DSCNet_2D_opensource/Data/DRIVE' if dataset=='drive' else R/'chase_data'
    raw=sitk.GetArrayFromImage(sitk.ReadImage(str(image_root/'Image_Te'/image)));gt=sitk.GetArrayFromImage(sitk.ReadImage(str(image_root/'Label_Te'/image)))>0
    base=sitk.GetArrayFromImage(sitk.ReadImage(str(g['dscnet'][0]/'test_predictions'/image)))>.5
    from skimage.morphology import skeletonize
    missing=skeletonize(gt)&~base;ys,xs=np.nonzero(missing)
    # Locate densest 64x64 missed-skeleton region with deterministic tie rule.
    centers=[(y,x) for y in range(32,gt.shape[0]-32,32) for x in range(32,gt.shape[1]-32,32)]
    cy,cx=max(centers,key=lambda p:int(missing[p[0]-32:p[0]+32,p[1]-32:p[1]+32].sum()))
    names=['Input','Ground truth']+list(g);maps=[raw,gt]+[sitk.GetArrayFromImage(sitk.ReadImage(str(v[0]/'test_predictions'/image)))>.5 for v in g.values()]
    fig,axes=plt.subplots(2,len(maps),figsize=(7.1,3.1),layout='constrained')
    for j,(name,img) in enumerate(zip(names,maps)):
        axes[0,j].imshow(img,cmap='gray');axes[0,j].set_title(name,fontsize=7);axes[0,j].add_patch(plt.Rectangle((cx-32,cy-32),64,64,fill=False,edgecolor='#E69F00',lw=.7))
        axes[1,j].imshow(img[cy-32:cy+32,cx-32:cx+32],cmap='gray');axes[0,j].axis('off');axes[1,j].axis('off')
    figure(fig,'fig5_predictions_'+dataset);qual.append({'dataset':dataset,'image':image,'seed':0,'center_y':cy,'center_x':cx,'rule':'median DSC seed0 test Dice image; ROI maximizes missed GT skeleton count on32px grid; illustration, not unbiased performance evidence'})
table('qualitative_example_selection',qual)
from posthoc_report import report as posthoc_report
primary_evidence,posthoc_evidence,e2_findings,posthoc_details=posthoc_report(ROOT,e2,table)
numbers['E2'].update(primary_evidence);numbers['E2_posthoc']=posthoc_evidence
dump(A/'numbers.json',numbers)
