"""Verify accelerated small-sample p-values against the original SciPy method."""
import sys,time,json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis_stats import power_pvalues,power

def reference(d):
    nz=np.any(d!=0,axis=1);p=np.ones(len(d))
    p[nz]=wilcoxon(d[nz],axis=1,alternative='two-sided',zero_method='wilcox',method='auto').pvalue
    return p

rng=np.random.default_rng(717)
cases=[]
for n in (6,8,10,13,15,20):
    # Half-rank ties, zeros, both signs, all-zero rows and untied rows.
    cases.append(rng.integers(-3,4,size=(12,n)).astype(float))
    cases.append(rng.normal(size=(12,n)))
    cases.append(np.zeros((2,n)))
    cases.append(np.tile(np.arange(1,n+1),(2,1))*np.array([1,-1])[:,None])
root=Path(__file__).resolve().parents[1]
import csv
rows=list(csv.DictReader((root/'results/tables/per_image_chase.csv').open(encoding='utf-8-sig')))
matrix=[]
for seed in range(5):
    selected=[r for r in rows if r['model']=='dscnet' and int(r['seed'])==seed]
    matrix.append([float(r['dice']) for r in sorted(selected,key=lambda r:r['image'])])
matrix=np.asarray(matrix)
for n in (6,8,10,13):
    a=matrix[rng.integers(5,size=(12,5))].mean(1)
    b=matrix[rng.integers(5,size=(12,5))].mean(1)
    delta=np.take_along_axis(b-a,rng.integers(8,size=(12,n)),axis=1)
    cases.extend([np.round(delta,12),np.round(delta+.005,12)])
start=time.perf_counter()
for i,d in enumerate(cases):
    expected=reference(d);actual=power_pvalues(d)
    np.testing.assert_array_equal(actual,expected,err_msg=f'case {i}, shape {d.shape}')
    np.testing.assert_array_equal(actual<.05,expected<.05)
before=np.random.default_rng(717).bit_generator.state
test_rng=np.random.default_rng(717);power_pvalues(cases[0])
assert test_rng.bit_generator.state==before
# Identical bootstrap draws and output records for the complete configuration
# grid with a small replication count; the production count remains 10000.
import analysis_stats
fast_rng=np.random.default_rng(717);slow_rng=np.random.default_rng(717)
fast=power(matrix,fast_rng,repetitions=12)
analysis_stats.power_pvalues=reference
slow=power(matrix,slow_rng,repetitions=12)
assert fast==slow
assert fast_rng.bit_generator.state==slow_rng.bit_generator.state
print(json.dumps({'passed':True,'cases':len(cases),'complete_power_grid_identical':True,'rng_state_identical':True,'seconds':time.perf_counter()-start}))
