"""Recompute primary image-paired comparisons using the released numeric tables."""
import argparse,csv,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis_stats import paired,holm
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path,default=ROOT/'reproduced/statistics');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(20261006)
for dataset in ('drive','chase'):
    rows=list(csv.DictReader((ROOT/f'results/tables/per_image_{dataset}.csv').open(encoding='utf-8-sig')))
    labels=list(dict.fromkeys(r['model'] for r in rows));contrasts=[(x,'dscnet') for x in labels if x not in ('dscnet','unet')]+[('dscnet','unet')]
    def matrix(label,metric):
        subset=[r for r in rows if r['model']==label];seeds=sorted({int(r['seed']) for r in subset});images=sorted({r['image'] for r in subset})
        assert len(subset)==len(seeds)*len(images)
        return np.array([[float(next(r[metric] for r in subset if int(r['seed'])==s and r['image']==i)) for i in images] for s in seeds])
    result=[]
    for label,reference in contrasts:
        for metric in ('dice','cldice','betti0_err'):
            result.append({'dataset':dataset,'model':label,'reference':reference,'metric':metric,**paired(matrix(label,metric),matrix(reference,metric),rng)})
    holm(result)
    (a.output/f'stats_{dataset}.json').write_text(json.dumps(result,indent=2))
    print(dataset,len(result),'contrasts written')
