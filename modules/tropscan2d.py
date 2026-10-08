"""
TropScan2D: plug-in max-plus (tropical) selective 2D scan block for (B, C, H, W) feature maps.

Design (what is and is not claimed):
  * y_i = max_j [ v_j - cost(j -> i) ], where cost is the sum of NON-NEGATIVE, INPUT-DEPENDENT,
    ANISOTROPIC per-pixel step costs (c_x for horizontal moves, c_y for vertical moves) along
    two families of L-shaped paths (horizontal-then-vertical, vertical-then-horizontal).
  * This is computed EXACTLY for these two L-path families in O(C*H*W) via cumsum + cummax.
    It is NOT an exact solver for arbitrary 2D shortest paths (paths with >1 turn can be missed
    within one layer; stacking layers enlarges the path family).
  * `SumProd2D` below uses the SAME costs and the SAME L-path families but sum-product aggregation
    (exp(-cost) weighted sum). It is the key ablation: the difference between the two is only the
    aggregation semiring, not the path structure.
Block layout (residual, out_proj zero-initialised => identity at init):
  x -> GroupNorm -> 1x1 conv (v, z, costs) -> 3x3 depthwise conv on v -> SiLU
    -> [max-plus | sum-product] L-path scan (+ D * v skip) -> GroupNorm -> * SiLU(z) -> 1x1 conv -> + x
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _scan_axis_maxplus(v, cost, dim):
    """out_i = max_j (v_j - |S_i - S_j|) along `dim`, S = cumsum(cost). Exact, O(L)."""
    S = torch.cumsum(cost, dim)
    f = torch.cummax(v + S, dim)[0] - S                                  # j <= i
    q = torch.flip(torch.cummax(torch.flip(v - S, [dim]), dim)[0], [dim])  # max_{j>=i}(v_j - S_j)
    L = v.shape[dim]
    pad = torch.full_like(q.narrow(dim, 0, 1), -1e4)
    b = torch.cat([q.narrow(dim, 1, L - 1), pad], dim) + S               # j > i
    return torch.maximum(f, b)


def _scan_axis_sumprod(v, cost, dim):
    """out_i = sum_j exp(-|S_i - S_j|) v_j along `dim` (row-normalised). O(L^2) per line; fine for
    the low-resolution stages where the block is inserted (<= 64 per axis)."""
    S = torch.cumsum(cost, dim)
    v_ = v.movedim(dim, -1)
    S_ = S.movedim(dim, -1)
    W = torch.exp(-(S_.unsqueeze(-1) - S_.unsqueeze(-2)).abs())          # ..., i, j
    W = W / W.sum(-1, keepdim=True)
    out = torch.einsum('...ij,...j->...i', W, v_)
    return out.movedim(-1, dim)


class _LPathScan2D(nn.Module):
    def __init__(self, channels, groups=4, expand=1, cost_init=(1e-3, 1e-1), mode='maxplus'):
        super().__init__()
        assert mode in ('maxplus', 'sumprod')
        self.mode = mode
        self.h = groups
        self.di = channels * expand
        assert self.di % groups == 0
        self.norm_in = nn.GroupNorm(1, channels)
        self.in_proj = nn.Conv2d(channels, 2 * self.di + 2 * groups, 1)
        self.dw = nn.Conv2d(self.di, self.di, 3, padding=1, groups=self.di)
        lo, hi = cost_init
        dt = torch.exp(torch.rand(2 * groups) * (math.log(hi) - math.log(lo)) + math.log(lo))
        self.cost_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # softplus^-1(dt)
        self.A_log = nn.Parameter(torch.log(torch.empty(groups).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(groups))
        self.norm_out = nn.GroupNorm(groups, self.di)
        self.out_proj = nn.Conv2d(self.di, channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def costs(self, x):
        """Return non-negative anisotropic costs (B, groups, 1, H, W) for x- and y-moves (for visualisation)."""
        B, _, H, W = x.shape
        _, _, c = torch.split(self.in_proj(self.norm_in(x)), [self.di, self.di, 2 * self.h], 1)
        c = F.softplus(c + self.cost_bias.view(1, -1, 1, 1)) * torch.exp(self.A_log).repeat(2).view(1, -1, 1, 1)
        return c[:, :self.h].unsqueeze(2), c[:, self.h:].unsqueeze(2)

    def forward(self, x):
        B, C, H, W = x.shape
        h, p = self.h, self.di // self.h
        v, z, c = torch.split(self.in_proj(self.norm_in(x)), [self.di, self.di, 2 * h], 1)
        v = F.silu(self.dw(v)).view(B, h, p, H, W)
        c = F.softplus(c + self.cost_bias.view(1, -1, 1, 1)) * torch.exp(self.A_log).repeat(2).view(1, -1, 1, 1)
        cx = c[:, :h].unsqueeze(2).expand(B, h, p, H, W)
        cy = c[:, h:].unsqueeze(2).expand(B, h, p, H, W)
        scan = _scan_axis_maxplus if self.mode == 'maxplus' else _scan_axis_sumprod
        y1 = scan(scan(v, cx, 4), cy, 3)   # horizontal then vertical
        y2 = scan(scan(v, cy, 3), cx, 4)   # vertical then horizontal
        y = torch.maximum(y1, y2) if self.mode == 'maxplus' else 0.5 * (y1 + y2)
        y = y + v * self.D.view(1, h, 1, 1, 1)
        y = self.norm_out(y.reshape(B, self.di, H, W)) * F.silu(z)
        return x + self.out_proj(y)


class TropScan2D(_LPathScan2D):
    def __init__(self, channels, groups=4, expand=1, cost_init=(1e-3, 1e-1)):
        super().__init__(channels, groups, expand, cost_init, mode='maxplus')


class SumProd2D(_LPathScan2D):
    """Ablation: identical block, identical costs and L-paths, sum-product aggregation."""
    def __init__(self, channels, groups=4, expand=1, cost_init=(1e-3, 1e-1)):
        super().__init__(channels, groups, expand, cost_init, mode='sumprod')


class ParamMatchedConv(nn.Module):
    """Parameter-matched control: residual block of GN + 1x1 + dw3x3 + 1x1 (zero-init out), no global scan.
    Use `hidden` to match the parameter count of TropScan2D at the same insertion point."""
    def __init__(self, channels, hidden):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.a = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.b = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.b.weight); nn.init.zeros_(self.b.bias)

    def forward(self, x):
        return x + self.b(F.silu(self.dw(self.a(self.norm(x)))))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def matched_hidden(channels, target_params):
    """Smallest `hidden` for ParamMatchedConv whose params >= target_params."""
    for hdn in range(1, 16 * channels):
        if n_params(ParamMatchedConv(channels, hdn)) >= target_params:
            return hdn
    return 16 * channels
