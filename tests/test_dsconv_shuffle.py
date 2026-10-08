"""Small GPU checks for the exact donor rule, gradients and RNG restoration."""
import sys,os,torch,json
sys.path[:0]=[sys.argv[1],os.path.join(os.path.dirname(__file__),'..','modules')]
from dsconv_shuffle import donor_indices,patch_shuffle
from dsconv_fix import patch_dsc
import S3_DSConv as upstream
device='cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(12)
for batch in (1,2,8):
    for _ in range(50):
        donors=donor_indices(batch,device)
        assert donors[0].item()==0
        assert ((donors>=0)&(donors<batch)).all()
        assert (donors[1:]!=torch.arange(batch,device=device)[1:]).all()
        if batch==2:assert donors.tolist()==[0,0]
# Recipient-specific coordinates; source gradients must accumulate into the
# declared donors. The reference directly gathers each donor's scalar field.
class Sampling:pass
sampler=Sampling();sampler.num_batch=8;sampler.num_channels=2
sampler.width=4;sampler.height=5;sampler.num_points=1;sampler.device=device;sampler.morph=0
class Module:pass
Module.DSC=Sampling
from dsconv_fix import _fixed_bilinear_interpolate_2D as fixed
Sampling._bilinear_interpolate_3D=fixed
patch_shuffle(Module)
features=torch.arange(8*2*4*5,dtype=torch.float32,device=device).reshape(8,2,4,5).requires_grad_()
yy=torch.arange(4,device=device).reshape(1,4,1).expand(8,4,5).float()+.25
xx=torch.arange(5,device=device).reshape(1,1,5).expand(8,4,5).float()+.25
torch.manual_seed(123);donors=donor_indices(8,device)
torch.manual_seed(123);actual=sampler._bilinear_interpolate_3D(features,yy,xx)
expected=fixed(sampler,features.index_select(0,donors),yy,xx)
assert torch.equal(actual,expected)
grad_actual=torch.autograd.grad(actual.sum(),features,retain_graph=True)[0]
grad_expected=torch.autograd.grad(expected.sum(),features)[0]
assert torch.equal(grad_actual,grad_expected)
# CUDA/CPU RNG checkpoint restoration repeats the next donor mapping exactly.
torch.manual_seed(42);rng=torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state()
first=donor_indices(8,device)
if device=='cuda':torch.cuda.set_rng_state(rng)
else:torch.set_rng_state(rng)
assert torch.equal(first,donor_indices(8,device))
# At batch1 the entire DSConv remains bitwise identical to the fixed version
# and no stochastic donor draw consumes RNG. Both spatial morphs are checked.
orig=upstream.DSC._bilinear_interpolate_3D
patch_dsc(upstream)
models=[upstream.DSConv(8,16,9,1.,m,True,device).to(device).eval() for m in (0,1)]
x=torch.randn(1,8,40,36,device=device)
with torch.no_grad():reference=[m(x) for m in models]
patch_shuffle(upstream)
with torch.no_grad():assert all(torch.equal(m(x),y) for m,y in zip(models,reference))
upstream.DSC._bilinear_interpolate_3D=orig
# The actual training path patches the fix, enables the cache, then wraps it.
# Batch2 donors are exactly [0,0], so this mechanism must reproduce the
# original cross-sample gather, including each recipient's own offsets.
x2=torch.randn(2,8,40,36,device=device)
with torch.no_grad():original_b2=[m(x2) for m in models]
patch_dsc(upstream)
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
from performance_cache import enable
enable(source_image_cache=False);patch_shuffle(upstream)
with torch.no_grad():assert all(torch.equal(m(x2),y) for m,y in zip(models,original_b2))
print(json.dumps({'passed':True,'device':device,'donors':donors.tolist(),'sampling_forward_and_gradients_bitwise_match_reference':True,'batch1_matches_fixed':True,'rng_resume_repeats_donors':True}))
