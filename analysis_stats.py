"""Pre-registered image-paired inference and transparent bootstrap sensitivity."""
import numpy as np
from scipy.stats import wilcoxon,skew,rankdata
def power_pvalues(differences):
    """Same two-sided Wilcoxon auto p-values, with exact small-n enumeration.

    For at most 13 columns, SciPy auto uses the exact distribution or exhausts
    every sign permutation when ties/zeros occur. Average ranks are half-integers,
    so a subset-sum count on doubled ranks gives the identical distribution while
    avoiding reranking all 2**n sign assignments. No random draws are consumed.
    Larger samples still call the original SciPy auto implementation.
    """
    differences=np.asarray(differences)
    nonzero=np.any(differences!=0,axis=1);p=np.ones(len(differences))
    if not np.any(nonzero):return p
    d=differences[nonzero]
    if d.shape[1]>13:
        p[nonzero]=wilcoxon(d,axis=1,alternative='two-sided',zero_method='wilcox',method='auto').pvalue
        return p
    absolute=np.where(d!=0,np.abs(d),np.nan)
    ranks=np.nan_to_num(rankdata(absolute,axis=1,method='average',nan_policy='omit'))
    weights=np.rint(2*ranks).astype(np.int64)
    observed=(weights*(d>0)).sum(1)
    width=int(weights.sum(1).max())+1
    counts=np.zeros((len(d),width),dtype=np.int64);counts[:,0]=1
    columns=np.arange(width)[None,:];row=np.arange(len(d))[:,None]
    for weight in weights.T:
        source=columns-weight[:,None]
        counts=counts+np.where(source>=0,counts[row,np.maximum(source,0)],0)
    denominator=2**d.shape[1]
    lower=np.where(columns<=observed[:,None],counts,0).sum(1)
    upper=np.where(columns>=observed[:,None],counts,0).sum(1)
    p[nonzero]=np.minimum(1.,2*np.minimum(lower,upper)/denominator)
    return p
def paired(matrix_a,matrix_b,rng,iterations=10000):
    assert matrix_a.shape[1]==matrix_b.shape[1]
    differences=np.round(matrix_a.mean(0)-matrix_b.mean(0),12)
    test=wilcoxon(differences,alternative='two-sided',zero_method='wilcox',method='auto') if np.any(differences) else None
    boot=differences[rng.integers(len(differences),size=(iterations,len(differences)))].mean(1)
    return {'delta':float(differences.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),
        'p':float(test.pvalue) if test else 1.,'wilcoxon_statistic':float(test.statistic) if test else 0.,
        'n_images':len(differences),'n_seeds_a':len(matrix_a),'n_seeds_b':len(matrix_b),
        'paired_difference_skewness':float(skew(differences)) if np.ptp(differences)>0 else 0.,
        'assumption_note':'Matched images asserted; seed mean per image; Wilcoxon symmetric-difference location interpretation may be limited for skewed differences. Bootstrap CI resamples images, conditional on observed seeds.'}
def holm(rows):
    order=np.argsort([r['p'] for r in rows]);previous=0.
    for rank,index in enumerate(order):
        value=min(1.,max(previous,rows[index]['p']*(len(rows)-rank)));rows[index]['p_holm']=value;previous=value
    return rows
def power(matrix,rng,repetitions=10000):
    # Two hypothetical independently trained DSCNet groups. Resample whole seed
    # rows to preserve cross-image seed correlations, then paired image columns.
    # Shift second group by +0.005 Dice. This is simulated sensitivity, not
    # evidence that a particular module has that effect or a new data guarantee.
    configs=[('seeds',s,matrix.shape[1]) for s in (1,2,3,4,5,6,8,10,15,20,30)]
    configs += [('images',5,n) for n in (6,8,10,15,20,30,40,60,80,100,150,200)]
    rows=[]
    for axis,seeds,images in configs:
        print(f'Power: axis={axis}, seeds={seeds}, images={images}, repetitions={repetitions}',flush=True)
        hits=0;null_hits=0;done=0
        while done<repetitions:
            size=min(500,repetitions-done)
            a=matrix[rng.integers(matrix.shape[0],size=(size,seeds))].mean(1)
            b=matrix[rng.integers(matrix.shape[0],size=(size,seeds))].mean(1)
            columns=rng.integers(matrix.shape[1],size=(size,images))
            delta=np.take_along_axis(b-a,columns,axis=1)
            shifted=np.round(delta+.005,12);null=np.round(delta,12)
            # Vectorized test uses the same pre-registered auto/zero rules.
            pn=power_pvalues(shifted)
            hits+=int(np.sum(pn<.05))
            p0=power_pvalues(null)
            null_hits+=int(np.sum(p0<.05));done+=size
        rows.append({'axis':axis,'seeds':seeds,'images':images,'power':hits/repetitions,'null_rejection_rate':null_hits/repetitions,
          'monte_carlo_se':float(np.sqrt(hits/repetitions*(1-hits/repetitions)/repetitions)),'repetitions':repetitions,'dice_shift':.005})
    return rows
