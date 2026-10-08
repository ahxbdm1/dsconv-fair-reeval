import argparse,csv,hashlib,json,pathlib,shutil,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=pathlib.Path,default=ROOT/'results/e1_synthetic');parser.add_argument('--output-dir',type=pathlib.Path,default=ROOT/'reproduced/e1');args=parser.parse_args();OUT=args.input_dir;ASSETS=args.output_dir
d=json.loads((OUT/'RESULTS.json').read_text());s=d['batch_summary']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(7.1,5.7),layout='constrained')
a=axes[0,0];a.set_axis_off();a.set_title('(a) Flattened feature-source index',loc='left',fontweight='bold')
a.text(.02,.83,'Sample j occupies flat positions\njHW through (j + 1)HW − 1.',transform=a.transAxes,va='top')
a.text(.02,.55,'Original: (jHW + yW) − jHW + x\n              = yW + x  →  sample 0',transform=a.transAxes,va='top',color='#B84C3A',linespacing=1.6)
a.text(.02,.21,'Fixed: jHW + yW + x  →  sample j',transform=a.transAxes,va='top',color='#00798C')
b=axes[0,1];example=d['constant_feature_example']
arr=np.array([example['morph0_original'],example['morph0_fixed']])
im=b.imshow(arr,vmin=1,vmax=4,cmap='viridis',aspect='auto')
b.set_xticks(range(4),['0','1','2','3']);b.set_yticks([0,1],['Original','Fixed']);b.set_xlabel('Output sample index')
b.set_title('(b) Distinct constant input features',loc='left',fontweight='bold')
for row in range(2):
    for col in range(4):b.text(col,row,f'{arr[row,col]:.0f}',ha='center',va='center',color='white' if arr[row,col]<3 else 'black',fontweight='bold')
fig.colorbar(im,ax=b,label='Gathered feature value',ticks=[1,2,3,4],fraction=.06)
c=axes[1,0]
for morph,color,marker in ((0,'#B84C3A','o'),(1,'#375DA8','s')):
    rs=[r for r in s if r['morph']==morph]
    c.errorbar([r['batch'] for r in rs],[r['original_batch_mean_relative_l2'] for r in rs],yerr=[r['original_between_seed_sd'] for r in rs],color=color,marker=marker,markersize=3,capsize=2,label=f'Original, morph {morph}')
c.plot([1,2,4,8,16],[0]*5,'--',color='#00798C',label='Fixed (≤ 6 × 10⁻⁷ mean error)')
c.set_xscale('log',base=2);c.set_xticks([1,2,4,8,16],['1','2','4','8','16'])
c.set_ylim(-.045,1.22);c.set_xlabel('Batch size');c.set_ylabel('Mean per-sample relative L2 error')
c.set_title('(c) Full DSConv: batch vs singleton',loc='left',fontweight='bold');c.legend(loc='lower right',frameon=False);c.grid(axis='y',alpha=.15)
e=axes[1,1];g=np.zeros((2,4))
for r in d['gradient_rows']:
    if r['morph']==0:g[0 if r['implementation']=='original' else 1,r['input_sample']]=float(r['gradient_l2']>0)
e.imshow(g,vmin=0,vmax=1,cmap='Blues',aspect='auto')
e.set_xticks(range(4),['0','1','2','3']);e.set_yticks([0,1],['Original','Fixed']);e.set_xlabel('Input sample receiving gather gradient')
e.set_title('(d) Loss on sample 1; coordinates fixed',loc='left',fontweight='bold')
for row in range(2):
    for col in range(4):e.text(col,row,'nonzero' if g[row,col] else 'zero',ha='center',va='center',fontsize=7,color='white' if g[row,col] else 'black')
fig.suptitle('Cross-sample feature gathering in the original 2D DSConv',fontsize=10,fontweight='bold')
figures=ASSETS/'figures';figures.mkdir(parents=True,exist_ok=True)
for ext in ('png','pdf'):
    path=figures/f'fig1_bug_characterization_codex.{ext}'
    assert not path.exists(),'Do not overwrite an existing figure; remove own failed rendering explicitly first.'
    fig.savefig(path,dpi=400,facecolor='white')
plt.close(fig)
tables=ASSETS/'tables';tables.mkdir(parents=True,exist_ok=True)
shutil.copy2(OUT/'batch_summary.csv',tables/'E1_batch_error_codex.csv')
shutil.copy2(OUT/'gather_gradient.csv',tables/'E1_gather_gradient_codex.csv')
header='|morph|batch|Mean relative L2 (original)|Seed SD|Mean relative L2 (fixed)|\n|---|---|---|---|---|\n'
table=header+'\n'.join(f"|{r['morph']}|{r['batch']}|{r['original_batch_mean_relative_l2']:.9f}|{r['original_between_seed_sd']:.9f}|{r['fixed_batch_mean_relative_l2']:.9g}|" for r in s)+'\n'
(tables/'E1_batch_error_codex.md').write_text(table,encoding='utf-8')
