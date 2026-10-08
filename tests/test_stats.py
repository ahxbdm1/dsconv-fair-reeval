import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis_stats import paired,holm
rng=np.random.default_rng(0)
same=np.tile(np.arange(8)/100,(5,1));zero=paired(same,same,rng)
assert zero['p']==1 and zero['delta']==0 and zero['ci_low']==0 and zero['ci_high']==0
offset=paired(same+.005,same,rng)
assert abs(offset['delta']-.005)<1e-12 and abs(offset['ci_low']-.005)<1e-12
rows=holm([{'p':.01},{'p':.04},{'p':.03}])
assert np.allclose([r['p_holm'] for r in rows],[.03,.06,.06])
print('Statistical edge cases and Holm verified')
