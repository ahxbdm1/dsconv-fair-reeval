# Copyright (c) 2025 Yaolei Qi
# Copyright (c) 2026 Heng Zhang
# SPDX-License-Identifier: MIT
# Adapted from DSCNet; see THIRD_PARTY_NOTICES.md.
"""
DSCNetPlus: wraps the ORIGINAL DSCNet (ICCV 2023, github.com/YaoleiQi/DSCNet, file S3_DSCNet.py) and inserts a
plug-in block after chosen stages, WITHOUT modifying the original file. The original forward is re-implemented
line-by-line below with hooks; `insert=None` must reproduce the original output exactly (checked in tests).

Insertion points (224x224 training patches, n_basic_layer=16):
  'b2'  after block2 output  x_2_1  (64 ch, 56x56)
  'b3'  after block3 output  x_3_1  (128 ch, 28x28)   <- bottleneck, default
  'b4'  after block4 output  x_2_3  (64 ch, 56x56)
Block types: 'none' | 'trop' (TropScan2D) | 'sumprod' (SumProd2D ablation) | 'conv' (param-matched control)
             | 'mamba' (bidirectional Mamba2 from mamba_ssm on raster sequences; GPU only, independently reported parameters)
"""
import torch
from torch import nn, cat
from tropscan2d import TropScan2D, SumProd2D, ParamMatchedConv, n_params, matched_hidden


class BiMamba2Block(nn.Module):
    """Stitched global-context block from mamba_ssm (official). Residual, zero-init output projection."""
    def __init__(self, channels, d_state=32, headdim=16, expand=2):
        super().__init__()
        from mamba_ssm import Mamba2  # requires GPU build of mamba_ssm
        self.norm = nn.LayerNorm(channels)
        self.f = Mamba2(d_model=channels, d_state=d_state, headdim=headdim, expand=expand)
        self.b = Mamba2(d_model=channels, d_state=d_state, headdim=headdim, expand=expand)
        self.out = nn.Linear(channels, channels)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        t = self.norm(x.flatten(2).transpose(1, 2))
        y = self.f(t) + self.b(t.flip(1)).flip(1)
        return x + self.out(y).transpose(1, 2).reshape(B, C, H, W)


def make_block(kind, channels, match_to=None):
    if kind == 'trop':
        return TropScan2D(channels)
    if kind == 'sumprod':
        return SumProd2D(channels)
    if kind == 'conv':
        tgt = n_params(TropScan2D(channels)) if match_to is None else match_to
        return ParamMatchedConv(channels, matched_hidden(channels, tgt))
    if kind == 'mamba':
        return BiMamba2Block(channels)
    raise ValueError(kind)


class DSCNetPlus(nn.Module):
    CH = {'b2': 4, 'b3': 8, 'b4': 4}   # multiples of n_basic_layer

    def __init__(self, base, insert=('b3',), kind='none'):
        super().__init__()
        self.base = base
        self.kind = kind
        self.extra = nn.ModuleDict()
        if kind != 'none':
            for pos in insert:
                self.extra[pos] = make_block(kind, self.CH[pos] * base.number)

    def _hook(self, name, x):
        return self.extra[name](x) if name in self.extra else x

    def forward(self, x):
        n = self.base
        x_00_0 = n.conv00(x); x_0x_0 = n.conv0x(x); x_0y_0 = n.conv0y(x)
        x_0_1 = n.conv1(cat([x_00_0, x_0x_0, x_0y_0], dim=1))
        x = n.maxpooling(x_0_1)
        x_1_1 = n.conv3(cat([n.conv20(x), n.conv2x(x), n.conv2y(x)], dim=1))
        x = n.maxpooling(x_1_1)
        x_2_1 = n.conv5(cat([n.conv40(x), n.conv4x(x), n.conv4y(x)], dim=1))
        x_2_1 = self._hook('b2', x_2_1)
        x = n.maxpooling(x_2_1)
        x_3_1 = n.conv7(cat([n.conv60(x), n.conv6x(x), n.conv6y(x)], dim=1))
        x_3_1 = self._hook('b3', x_3_1)
        x = n.up(x_3_1)
        xx = cat([x, x_2_1], dim=1)
        x_2_3 = n.conv13(cat([n.conv120(xx), n.conv12x(xx), n.conv12y(xx)], dim=1))
        x_2_3 = self._hook('b4', x_2_3)
        x = n.up(x_2_3)
        xx = cat([x, x_1_1], dim=1)
        x_1_3 = n.conv15(cat([n.conv140(xx), n.conv14x(xx), n.conv14y(xx)], dim=1))
        x = n.up(x_1_3)
        xx = cat([x, x_0_1], dim=1)
        x_0_3 = n.conv17(cat([n.conv160(xx), n.conv16x(xx), n.conv16y(xx)], dim=1))
        return n.sigmoid(n.out_conv(x_0_3))
