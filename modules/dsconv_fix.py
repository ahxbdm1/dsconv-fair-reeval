# Copyright (c) 2025 Yaolei Qi
# Copyright (c) 2026 Heng Zhang
# SPDX-License-Identifier: MIT
# Adapted from DSCNet; see THIRD_PARTY_NOTICES.md.
"""Minimal fix for the cross-sample indexing bug in the ORIGINAL DSConv (DSCNet, ICCV 2023, file S3_DSConv.py).

Bug: in DSC._bilinear_interpolate_3D (sic: the 2D file uses this name) the flat gather indices are computed as
        index_a0 = base_y0 - base + x0   (and the same for index_c0, index_a1, index_c1)
     where base = batch_index * H * W.  Subtracting `base` removes the per-sample offset, so for batch size B > 1
     every sample gathers its deformed features from SAMPLE 0's feature map. With B = 1 (DSCNet's own setting) the
     result is correct, which is why the paper's numbers are unaffected.
Fix: keep the per-sample offset:  index_a0 = base_y0 + x0  (etc.).
`patch_dsc(module)` monkey-patches the DSC class of an imported S3_DSConv module, leaving the file untouched.
"""
import types, torch


def _fixed_bilinear_interpolate_2D(self, input_feature, y, x):
    y = y.reshape([-1]).float()
    x = x.reshape([-1]).float()
    zero = torch.zeros([]).int()
    max_y = self.width - 1
    max_x = self.height - 1
    y0 = torch.floor(y).int(); y1 = y0 + 1
    x0 = torch.floor(x).int(); x1 = x0 + 1
    y0 = torch.clamp(y0, zero, max_y); y1 = torch.clamp(y1, zero, max_y)
    x0 = torch.clamp(x0, zero, max_x); x1 = torch.clamp(x1, zero, max_x)
    input_feature_flat = input_feature.flatten()
    input_feature_flat = input_feature_flat.reshape(self.num_batch, self.num_channels, self.width, self.height)
    input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
    input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
    dimension = self.height * self.width
    base = torch.arange(self.num_batch) * dimension
    base = base.reshape([-1, 1]).float()
    repeat = torch.ones([self.num_points * self.width * self.height]).unsqueeze(0).float()
    base = torch.matmul(base, repeat).reshape([-1]).to(self.device)
    base_y0 = base + y0 * self.height
    base_y1 = base + y1 * self.height
    index_a0 = base_y0 + x0          # FIX: was  base_y0 - base + x0
    index_c0 = base_y0 + x1          # FIX
    index_a1 = base_y1 + x0          # FIX
    index_c1 = base_y1 + x1          # FIX
    value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
    value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(self.device)
    value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(self.device)
    value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(self.device)
    y0 = torch.floor(y).int(); y1 = y0 + 1
    x0 = torch.floor(x).int(); x1 = x0 + 1
    y0 = torch.clamp(y0, zero, max_y + 1); y1 = torch.clamp(y1, zero, max_y + 1)
    x0 = torch.clamp(x0, zero, max_x + 1); x1 = torch.clamp(x1, zero, max_x + 1)
    x0_float, x1_float, y0_float, y1_float = x0.float(), x1.float(), y0.float(), y1.float()
    vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
    vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
    vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
    vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)
    outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 + value_c1 * vol_c1)
    if self.morph == 0:
        outputs = outputs.reshape([self.num_batch, self.num_points * self.width, 1 * self.height, self.num_channels])
        outputs = outputs.permute(0, 3, 1, 2)
    else:
        outputs = outputs.reshape([self.num_batch, 1 * self.width, self.num_points * self.height, self.num_channels])
        outputs = outputs.permute(0, 3, 1, 2)
    return outputs


def patch_dsc(dsconv_module):
    """dsconv_module: the imported S3_DSConv module object. Replaces DSC._bilinear_interpolate_3D in place."""
    dsconv_module.DSC._bilinear_interpolate_3D = _fixed_bilinear_interpolate_2D
    return dsconv_module
