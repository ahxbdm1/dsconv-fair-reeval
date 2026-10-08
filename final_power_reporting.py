"""Report the observed power-grid crossings without hiding failed null calibration."""
import json
from pathlib import Path

def annotate(assets):
    assets=Path(assets);p=assets/'numbers.json';numbers=json.loads(p.read_text())
    rows=[]
    for dataset,records in numbers['power']['records'].items():
        for axis in ('seeds','images'):
            subset=sorted((x for x in records if x['axis']==axis),key=lambda x:x[axis])
            crossed=next((x for x in subset if x['power']>=.8),None)
            rows.append({'dataset':dataset,'axis':axis,'minimum_tested_grid_value_crossing_80_percent':crossed[axis] if crossed else None,'fixed_other_axis':{'images' if axis=='seeds' else 'seeds':subset[0]['images' if axis=='seeds' else 'seeds']},'raw_rejection_probability_at_crossing':crossed['power'] if crossed else None,'null_rejection_probability_at_crossing':crossed['null_rejection_rate'] if crossed else None,'source':f'tables/power_{dataset}.csv','reliable_prospective_sample_size':None,'interpretation':'Exploratory conditional grid crossing only. Empirical null rejection exceeds nominal5%; no calibrated80%-power sample-size recommendation is supported.'})
    numbers['power']['grid_crossings']=rows
    numbers['power']['calibration_conclusion']='The reported conditional simulations fail nominal5% null calibration. Increasing resampled images repeats a finite empirical test set and can amplify pseudo-replication. Raw80% crossings are reported for transparency, but reliable required seeds/images are not established; no new calibration method or scientific settings were introduced.'
    p.write_text(json.dumps(numbers,ensure_ascii=False,indent=2,allow_nan=False))
    note='\n\n## Power calibration and sample-size limitation\n\n'
    note+='The raw conditional sensitivity grid first crosses80% with1 seed per model on the observed20 DRIVE or8 CHASE images (rejection96.20% and81.84%, respectively); the images grid first crosses at its smallest tested count6, holding5 seeds per model fixed (93.93% and98.94%). These are minima of the tested grid, not validated sample-size requirements. At the corresponding seed-grid crossings, null rejection is13.40% (DRIVE) and13.71% (CHASE), exceeding nominal5%; even the6-image crossings have null rejection5.79% and10.49%. Increasing resampled images can repeat the same small empirical test set and inflate null rejection. Therefore this analysis does not establish how many independent seeds/images reliably achieve80% power at controlled5% type-I error. All10000-replicate results and null curves are retained as conditional sensitivity diagnostics. Sources: tables/power_drive.csv, tables/power_chase.csv, numbers.json/power/grid_crossings.\n'
    for name in ('FINDINGS.md','EXPERIMENT_DETAILS.md'):
        file=assets/name;text=file.read_text();marker='\n\n## Power calibration and sample-size limitation\n\n'
        if marker in text:text=text.split(marker)[0]
        file.write_text(text.rstrip()+note)
    captions=assets/'figures/CAPTIONS.md'
    text=captions.read_text() if captions.exists() else ''
    warning='Fig6 limitation: empirical null rejection exceeds nominal5%; raw80% crossings do not give calibrated prospective sample-size recommendations.\n'
    if warning not in text:captions.write_text(text+warning)
    return rows

if __name__=='__main__':
    print(json.dumps(annotate(Path(__file__).resolve().parent/'PAPER_ASSETS'),ensure_ascii=True))
