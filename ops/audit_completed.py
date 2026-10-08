"""Audit completed new runs; correct inherited descriptive metadata only."""
import json,math,hashlib
from pathlib import Path
def audit(root):
    root=Path(root);runs=root/'runs';rows=[];corrections={}
    log=runs/'METADATA_CORRECTIONS.json'
    if log.exists():corrections=json.loads(log.read_text())
    for p in sorted(runs.iterdir()):
        if not p.name.startswith(('drive_','chase_')) or not (p/'metrics.json').exists() or not (p/'last_metrics.json').exists():continue
        cfg=json.loads((p/'config.json').read_text());result=json.loads((p/'metrics.json').read_text())
        assert cfg['protocol']=='clean' and cfg['epochs']==400 and cfg['diag_test'],p
        epochs=[json.loads(v) for v in (p/'epochs.jsonl').read_text().splitlines() if v.strip()]
        diag=[json.loads(v) for v in (p/'diag_test.jsonl').read_text().splitlines() if v.strip()]
        assert [v['epoch'] for v in epochs]==list(range(1,401)),p
        assert [v['epoch'] for v in diag]==list(range(201,397,5)),p
        assert all(math.isfinite(v['loss']) for v in epochs),p
        val=[v for v in epochs if v['selection_dice'] is not None]
        assert [v['epoch'] for v in val]==list(range(201,397,5)),p
        winner=max(val,key=lambda v:v['selection_dice'])
        assert result['selected_epoch']==winner['epoch'],p
        selected=next(v for v in diag if v['epoch']==result['selected_epoch'])
        assert abs(selected['test_dice']-result['original_dice'])<1e-5,p
        last=json.loads((p/'last_metrics.json').read_text());assert last['epoch']==400,p
        expected=20 if cfg['dataset']=='drive' else 8
        assert len(result['per_image'])==expected and len({v['image'] for v in result['per_image']})==expected,p
        assert all(math.isfinite(v) for item in result['per_image'] for k,v in item.items() if k!='image'),p
        metadata=dict(test_evaluated_once=False,primary_result_role='validation-selected checkpoint',diagnostic_test_logging=True,additional_final_checkpoint_test=True)
        if any(result.get(k)!=v for k,v in metadata.items()):
            original=(p/'metrics.json').read_bytes()
            corrections.setdefault(p.name,{'original_sha256':hashlib.sha256(original).hexdigest(),'old_metadata':{k:result.get(k) for k in metadata},'reason':'Stage2 inherited test-once flag does not describe Stage4 diagnostic plus epoch400 evaluation; numerical results unchanged.'})
            result.update(metadata);tmp=p/'metrics.audit.tmp';tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False));tmp.replace(p/'metrics.json')
        rows.append({'run':p.name,'epochs':400,'diagnostic_test_evaluations':40,'validation_selected_epoch':result['selected_epoch'],'per_image_count':expected,'validation_selection_verified':True})
    for name,data in [('COMPLETED_RUN_AUDIT.json',rows),('METADATA_CORRECTIONS.json',corrections)]:
        target=runs/name;tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False));tmp.replace(target)
    return rows
if __name__=='__main__':print(json.dumps(audit(Path(__file__).resolve().parents[1]),indent=2))
