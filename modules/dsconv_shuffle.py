"""Post-hoc stochastic cross-sample feature mixing; offsets stay sample-specific."""
import torch

def donor_indices(batch,device):
    donors=torch.arange(batch,device=device)
    if batch>1:
        # Uniform independent draws with replacement from all samples except
        # the recipient. Sample0 retains its own features, as requested.
        recipients=donors[1:]
        draws=torch.randint(batch-1,(batch-1,),device=device)
        donors[1:]=draws+(draws>=recipients).long()
    return donors

def patch_shuffle(module):
    """Install AFTER patch_dsc and performance_cache.enable, once per process."""
    base=module.DSC._bilinear_interpolate_3D
    def shuffled(self,input_feature,y,x):
        donors=donor_indices(self.num_batch,input_feature.device)
        if self.num_batch==1:return base(self,input_feature,y,x)
        mixed=input_feature.index_select(0,donors)
        return base(self,mixed,y,x)
    module.DSC._bilinear_interpolate_3D=shuffled
    return module
