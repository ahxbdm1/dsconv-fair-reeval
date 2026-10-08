"""Data-free fixture: exploratory controls do not change primary source digests."""
import hashlib,json,pathlib,sys,tempfile
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from posthoc_report import report
with tempfile.TemporaryDirectory() as tmp:
    root=pathlib.Path(tmp);runs=root/'runs';runs.mkdir();e2={k:[] for k in ('orig','fix')}
    before={}
    for kind in e2:
        for seed in range(5):
            p=runs/f'primary_{kind}_{seed}';p.mkdir();e2[kind].append(p)
            (p/'metrics.json').write_text(json.dumps({'selected_epoch':201,'mean':{'dice':.8}}))
            (p/'epochs.jsonl').write_text(''.join(json.dumps({'epoch':e,'loss':1/e})+'\n' for e in range(1,401)))
            before[str(p.relative_to(root)/'metrics.json')]=hashlib.sha256((p/'metrics.json').read_bytes()).hexdigest()
    tables={};primary,extra,_,_=report(root,e2,lambda k,v:tables.update({k:v}))
    assert primary['primary_metric_sha256']==before
    assert len(tables['E2_training_loss_per_seed'])==10 and len(tables['E2_training_loss_curves'])==4000
    assert extra['results']==[] and extra['excluded_from_primary_E2_and_E3']
print('PASS: data-free loss summaries and separation of exploratory evidence')
