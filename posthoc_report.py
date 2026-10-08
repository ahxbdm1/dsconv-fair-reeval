"""Report exploratory controls separately; new reproduction outputs get new digests."""
import hashlib,json
import numpy as np
def report(root,e2,table):
    r=root/'runs';read=lambda p:json.loads(p.read_text())
    hashes={str(p.relative_to(root)/'metrics.json'):hashlib.sha256((p/'metrics.json').read_bytes()).hexdigest() for paths in e2.values() for p in paths}
    rows=[];curves=[]
    for implementation,paths in e2.items():
        for seed,p in enumerate(paths):
            log=[json.loads(x) for x in (p/'epochs.jsonl').read_text().splitlines() if x.strip()]
            assert [v['epoch'] for v in log]==list(range(1,401))
            epoch=read(p/'metrics.json')['selected_epoch']
            rows.append({'implementation':implementation,'seed':seed,'source':str(p.relative_to(root)/'epochs.jsonl'),'final_epoch_loss':log[-1]['loss'],'last50_mean_loss':float(np.mean([v['loss'] for v in log[-50:]])),'last200_mean_loss':float(np.mean([v['loss'] for v in log[-200:]])),'validation_selected_epoch':epoch,'loss_at_selected_epoch':log[epoch-1]['loss']})
            curves.extend({'implementation':implementation,'seed':seed,'epoch':v['epoch'],'loss':v['loss']} for v in log)
    table('E2_training_loss_per_seed',rows);table('E2_training_loss_curves',curves)
    summary=[]
    for implementation,paths in e2.items():
        row={'implementation':implementation,'seeds':len(paths)}
        for k in ('final_epoch_loss','last50_mean_loss','last200_mean_loss','loss_at_selected_epoch'):
            values=[v[k] for v in rows if v['implementation']==implementation]
            row[k+'_mean']=float(np.mean(values));row[k+'_sample_sd']=float(np.std(values,ddof=1))
        summary.append(row)
    table('E2_training_loss_comparison',summary)
    planpath=r/'POSTHOC_EXPERIMENT_PLAN.json';plan=read(planpath) if planpath.exists() else {'configurations':[]}
    extra=[];sources={}
    for c in plan['configurations']:
        p=r/f"{c['dataset']}_{c['variant']}_{c['kind']}_{c['dsconv']}_b{c['bs']}_lr{c['lr']:g}_s{c['seed']}"
        if not (p/'metrics.json').exists():continue
        actual=read(p/'config.json');assert all(actual[k]==v for k,v in c.items())
        for filename,batch,checkpoint in [('metrics.json',1,'validation_selected'),('metrics_batch8.json',8,'validation_selected'),('last_metrics.json',1,'epoch400')]:
            result=read(p/filename);extra.append({'label':'post-hoc supplement','study_role':c['study_role'],'implementation':c['dsconv'],'lr':c['lr'],'seed':c['seed'],'training_batch':8,'roi_per_image':8,'inference_batch':batch,'checkpoint':checkpoint,'epoch':result.get('selected_epoch',result.get('epoch')),'source':str(p.relative_to(root)/filename),**result['mean']})
        sources[str(p.relative_to(root)/'metrics.json')]=hashlib.sha256((p/'metrics.json').read_bytes()).hexdigest()
    if extra:table('E2_posthoc',extra)
    text='The primary five-seed E2 comparison uses the shared validation-selected LR. The fixed LR scan winner is at its upper boundary; the original was not independently tuned. Training losses are measured under different cross-sample feature contexts and do not establish a generalization mechanism.'
    details='Exploratory LR controls have one seed each; shuffle has two seeds. Keep these outside the original E2/E3 inferential family. Shuffle draws other-sample donors with replacement, retains recipient coordinates, and is identity at batch1.'
    return {'primary_metric_sha256':hashes,'training_loss':summary},{'label':'post-hoc supplement','plan':plan,'results':extra,'source_sha256':sources,'excluded_from_primary_E2_and_E3':True},text,details
