# Copyright (c) 2025 Yaolei Qi
# Copyright (c) 2026 Heng Zhang
# SPDX-License-Identifier: MIT
# Adapted from DSCNet; see THIRD_PARTY_NOTICES.md.
"""Cache only input-independent tensors in the pinned upstream DSC implementation.

No upstream file is edited. AST extraction retains the exact original expressions
and operation order. Offset-dependent expressions and gradients are untouched.
"""
import ast,copy,functools,inspect,textwrap,os
from pathlib import Path
import torch

_GRIDS={};_BASES={};_ENABLED=False

def _target(node,name):
    return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)

def _compile(tree,name,scope):
    code=compile(ast.fix_missing_locations(ast.Module(body=[tree],type_ignores=[])),__file__,'exec')
    exec(code,scope);return scope[name]

def enable(source_image_cache=True):
    global _ENABLED
    if _ENABLED:return
    import S3_DSConv as upstream
    cls=upstream.DSC
    coord=ast.parse(textwrap.dedent(inspect.getsource(cls._coordinate_map_3D))).body[0]
    branch=next(n for n in coord.body if isinstance(n,ast.If))
    split=coord.body[0]
    centers=coord.body[1:coord.body.index(branch)]
    iy=next(i for i,n in enumerate(branch.body) if _target(n,'y_offset_new'))
    ix=next(i for i,n in enumerate(branch.orelse) if _target(n,'x_offset_new'))
    grid_branch=copy.deepcopy(branch);grid_branch.body=copy.deepcopy(branch.body[:iy]);grid_branch.orelse=copy.deepcopy(branch.orelse[:ix])
    static=ast.parse('def _original_grid(self):\n    pass').body[0]
    static.body=copy.deepcopy(centers)+[grid_branch,ast.parse('return y_new, x_new').body[0]]
    scope={'torch':torch};grid_builder=_compile(static,'_original_grid',scope)
    def key(s):return (int(s.num_batch),int(s.width),int(s.height),int(s.num_points),int(s.morph),str(s.device))
    def grid(s):
        k=key(s)
        if k not in _GRIDS:_GRIDS[k]=grid_builder(s)
        return _GRIDS[k]
    replacement=ast.parse('y_new, x_new = _cached_grid(self)').body[0]
    branch.body=[copy.deepcopy(replacement)]+branch.body[iy:]
    branch.orelse=[copy.deepcopy(replacement)]+branch.orelse[ix:]
    coord.body=[split,branch];scope['_cached_grid']=grid
    cls._coordinate_map_3D=_compile(coord,coord.name,scope)

    bilinear=ast.parse(textwrap.dedent(inspect.getsource(cls._bilinear_interpolate_3D))).body[0]
    begin=next(i for i,n in enumerate(bilinear.body) if _target(n,'dimension'))
    end=next(i for i,n in enumerate(bilinear.body) if _target(n,'base_y0'))
    static_base=ast.parse('def _original_base(self):\n    pass').body[0]
    static_base.body=copy.deepcopy(bilinear.body[begin:end])+[ast.parse('return base').body[0]]
    base_builder=_compile(static_base,'_original_base',scope)
    def base(s):
        k=key(s)
        if k not in _BASES:_BASES[k]=base_builder(s)
        return _BASES[k]
    bilinear.body[begin:end]=[ast.parse('base = _cached_base(self)').body[0]]
    scope['_cached_base']=base
    cls._bilinear_interpolate_3D=_compile(bilinear,bilinear.name,scope)

    if not source_image_cache:
        _ENABLED=True
        return
    import SimpleITK as sitk
    original_read=sitk.ReadImage
    @functools.lru_cache(maxsize=96)
    def source_image(filename):return original_read(filename)
    data_roots=[str(p.resolve())+os.sep for p in (Path(os.environ.get('DSCNET_DIR',Path(__file__).resolve().parent/'DSCNet'))/'DSCNet_2D_opensource/Data/DRIVE',Path(__file__).resolve().parent/'runs/chase_data')]
    def read_image(filename,*args,**kwargs):
        if not args and not kwargs and any(str(Path(filename).resolve()).startswith(prefix) for prefix in data_roots):
            return sitk.Image(source_image(str(filename)))
        return original_read(filename,*args,**kwargs)
    sitk.ReadImage=read_image
    _ENABLED=True
