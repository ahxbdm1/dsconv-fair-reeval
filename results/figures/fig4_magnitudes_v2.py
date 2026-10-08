import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
EVAL, ARCH, MOD = '#eb6834', '#1baf7a', '#2a78d6'
data = {
 'DRIVE': [('Learning-rate tuning (selected − default)',0.557,EVAL),('Best seed − mean (largest model)',0.309,EVAL),
           ('Test-chosen − validation-chosen checkpoint',0.071,EVAL),('DSCNet − matched U-Net',0.358,ARCH),
           ('+mamba − DSCNet',-0.0152,MOD),('+trop − DSCNet',0.0149,MOD),('+sumprod − DSCNet',-0.147,MOD),('+conv − DSCNet',-0.078,MOD)],
 'CHASE_DB1': [('Learning-rate tuning (selected − default)',0.414,EVAL),('Best seed − mean (largest model)',0.422,EVAL),
           ('Test-chosen − validation-chosen checkpoint',0.339,EVAL),('DSCNet − matched U-Net',0.713,ARCH),
           ('+mamba − DSCNet',-0.064,MOD),('+trop − DSCNet',-0.124,MOD),('+sumprod − DSCNet',None,MOD),('+conv − DSCNet',-0.107,MOD)]}
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans'})
fig, axes = plt.subplots(1,2,figsize=(7.2,3.2),sharey=True,sharex=True)
for ax,(name,rows) in zip(axes,data.items()):
    y = list(range(len(rows)))[::-1]
    for yi,(lab,v,c) in zip(y,rows):
        if v is None:
            ax.text(0.01,yi,'not run',va='center',ha='left',color='#52514e',fontsize=8); continue
        ax.barh(yi,v,color=c,height=0.62)
        ax.text(v+(0.015 if v>=0 else -0.015),yi,f'{v:+.2f}'.replace('-','−'),va='center',ha='left' if v>=0 else 'right',fontsize=8,color='#0b0b0b')
    ax.axvline(0,color='#0b0b0b',lw=0.8)
    ax.set_title(name,fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-0.45,0.95)
    ax.grid(axis='x',color='#e6e5e0',lw=0.6); ax.set_axisbelow(True)
    for s in ('top','right'): ax.spines[s].set_visible(False)
fig.legend(handles=[Patch(color=EVAL,label='Evaluation choice'),Patch(color=ARCH,label='DSConv vs. matched U-Net'),Patch(color=MOD,label='Plug-in module')],
           loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(0.6,-0.02))
fig.supxlabel('Dice difference (percentage points)',fontsize=9,x=0.62,y=0.08)
fig.tight_layout(rect=(0,0.1,1,1))
fig.savefig('fig4_magnitudes_v2.png',dpi=300,bbox_inches='tight')
fig.savefig('fig4_magnitudes_v2.pdf',bbox_inches='tight')
print('ok')
