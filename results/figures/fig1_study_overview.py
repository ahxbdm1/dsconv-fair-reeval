"""Figure 1 (study overview), v2. Double-column (183 mm), Python/matplotlib.
Four evidence cards (a-d) with mini charts drawn from the paper's own numbers, plus a recommendation strip.
usage: python results/figures/fig1_study_overview.py <per_seed_drive.csv> <per_seed_chase.csv> <out stem>"""
import sys, csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.spines.top": False, "axes.spines.right": False,
})
INK, MUTED, FAINT = "#1b1f24", "#5b6270", "#9aa3ae"
CARD_FILL, CARD_EDGE = "#f6f7f9", "#d7dbe1"
NAVY, VERM, TEAL, GREY = "#1f4e79", "#c8414b", "#2a8c82", "#a7afba"

W, H = 183.0, 104.0
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
base = fig.add_axes([0, 0, 1, 1]); base.set_xlim(0, W); base.set_ylim(0, H); base.axis("off")


def ax_mm(x, y, w, h):
    a = fig.add_axes([x / W, y / H, w / W, h / H]); a.set_facecolor("none"); return a


def card(x, y, w, h, letter, title, subtitle):
    base.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                  fc=CARD_FILL, ec=CARD_EDGE, lw=0.6))
    base.text(x + 2.4, y + h - 3.2, letter, fontsize=9, weight="bold", color=INK, va="center")
    base.text(x + 6.6, y + h - 3.2, title, fontsize=7.2, weight="bold", color=INK, va="center")
    base.text(x + 6.6, y + h - 6.6, subtitle, fontsize=6.2, color=MUTED, va="center")


# ---------------------------------------------------------------- header
base.text(2.0, H - 3.0, "Case study", fontsize=7.2, weight="bold", color=INK, va="center")
base.text(17.5, H - 3.0,
          "DSCNet with Dynamic Snake Convolution (ICCV 2023)   ·   DRIVE and CHASE_DB1   ·   "
          "86 training runs   ·   5 seeds   ·   validation-only checkpoint selection",
          fontsize=6.5, color=MUTED, va="center")
base.plot([2.0, W - 2.0], [H - 6.0, H - 6.0], color=CARD_EDGE, lw=0.6)

GX, GW, GAP = 2.0, 88.5, 2.0          # card grid
TOP_Y, CARD_H = 56.0, 41.0
BOT_Y = 13.0

# ---------------------------------------------------------------- (a) defect
card(GX, TOP_Y, GW, CARD_H, "a", "A silent, context-dependent defect (Q1–Q2)",
     "Gather index drops the batch offset: every sample reads sample 0")
sch = ax_mm(GX + 3.0, TOP_Y + 3.0, 36.0, 27.0); sch.set_xlim(0, 36); sch.set_ylim(0, 27); sch.axis("off")
SHADES = ["#d9e3ef", "#b7cbe2", "#8eaed3", "#5f88bb"]
X0, STEP, SZ = 14.0, 5.6, 4.6
sch.text(X0 + (3 * STEP + SZ) / 2, 22.6, "output for sample", fontsize=5.8, color=MUTED, ha="center", va="center")
for i in range(4):
    sch.text(X0 + i * STEP + SZ / 2, 19.6, str(i), fontsize=5.8, color=MUTED, ha="center", va="center")
for lab, y0, colour, reads in [("Original", 12.0, VERM, [0, 0, 0, 0]), ("Corrected", 5.0, TEAL, [0, 1, 2, 3])]:
    sch.text(0.0, y0 + SZ / 2, lab, fontsize=6.2, weight="bold", color=colour, va="center")
    for i, src in enumerate(reads):
        x0 = X0 + i * STEP
        wrong = (lab == "Original" and i > 0)
        sch.add_patch(Rectangle((x0, y0), SZ, SZ, fc=SHADES[src], ec=VERM if wrong else NAVY, lw=0.9 if wrong else 0.5))
        sch.text(x0 + SZ / 2, y0 + SZ / 2, str(src), fontsize=5.8, color="#ffffff" if src == 3 else INK,
                 ha="center", va="center", weight="bold" if wrong else "normal")
sch.text(0.0, 1.0, "cell = sample whose features are read", fontsize=5.6, color=MUTED, va="center")

bar = ax_mm(GX + 66.0, TOP_Y + 8.5, 19.0, 19.5)
hb = [("Batch-1 model \u00b7 single tiles", 81.73, NAVY), ("Batch-1 model \u00b7 batched tiles", 18.32, VERM),
      ("Batch-8 training \u00b7 original", 81.65, VERM), ("Batch-8 training \u00b7 corrected", 80.80, NAVY)]
ypos = [3.3, 2.3, 1.0, 0.0]
for (lab, val, colour), yy in zip(hb, ypos):
    bar.barh(yy, val, 0.72, color=colour, lw=0)
    bar.text(val + 2.0, yy, f"{val:.1f}", va="center", fontsize=5.8, color=VERM if val < 50 else INK,
             weight="bold" if val < 50 else "normal")
bar.set_yticks(ypos); bar.set_yticklabels([h[0] for h in hb], fontsize=5.8)
bar.tick_params(axis="y", length=0, pad=1.5)
bar.set_xlim(0, 100); bar.set_xticks([0, 50, 100]); bar.set_xlabel("Test Dice (%)", fontsize=6, labelpad=1)
bar.set_ylim(-0.6, 3.9); bar.spines["left"].set_visible(False)
_kx, _ky = GX + 47.0, TOP_Y + 31.0
for _c, _t, _dx in [(VERM, "defect active", 0.0), (NAVY, "defect inactive", 18.0)]:
    base.add_patch(Rectangle((_kx + _dx, _ky - 0.9), 1.8, 1.8, fc=_c, lw=0))
    base.text(_kx + _dx + 2.6, _ky, _t, fontsize=5.6, color=MUTED, va="center")

# ---------------------------------------------------------------- (b) fair comparison
bx = GX + GW + GAP
card(bx, TOP_Y, GW, CARD_H, "b", "Fair comparison of plug-in modules (Q3)",
     "Dots: seeds; line: mean; band: DSCNet mean \u00b1 SD. No plug-in beats DSCNet")


def read(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        out.setdefault(r["model"], []).append(100 * float(r["dice"]))
    return out


drive, chase = read(sys.argv[1]), read(sys.argv[2])
order = ["unet", "dscnet", "+conv", "+sumprod", "+trop", "+mamba"]
names = {"unet": "U-Net", "dscnet": "DSCNet", "+conv": "+conv", "+sumprod": "+sumprod", "+trop": "+trop", "+mamba": "+mamba"}
b_axes = []
for j, (ds, data) in enumerate([("DRIVE", drive), ("CHASE_DB1", chase)]):
    ax = ax_mm(bx + 12.0 + j * 41.0, TOP_Y + 8.0, 33.0, 21.5); b_axes.append(ax)
    models = [m for m in order if m in data]
    ref = np.mean(data["dscnet"]); sd = np.std(data["dscnet"], ddof=1)
    ax.axhspan(ref - sd, ref + sd, color=NAVY, alpha=0.10, lw=0)
    ax.axhline(ref, color=NAVY, lw=0.6, ls=(0, (2, 1.5)))
    for i, m in enumerate(models):
        v = np.array(data[m]); c = NAVY if m == "dscnet" else (FAINT if m == "unet" else GREY)
        ax.scatter(i + np.linspace(-0.12, 0.12, len(v)), v, s=5, color=c, lw=0, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=INK if m != "unet" else MUTED, lw=0.9, zorder=4)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([("\n" if k % 2 else "") + names[m] for k, m in enumerate(models)], ha="center", fontsize=5.6, linespacing=1.15)
    ax.tick_params(axis="x", length=0, pad=1)
    lo = min(min(v) for v in data.values()); hi = max(max(v) for v in data.values())
    ax.set_ylim(lo - 0.15, hi + 0.15)
    ax.set_title(ds, fontsize=6.2, color=MUTED, pad=2)
    if j == 0:
        ax.set_ylabel("Test Dice (%)", fontsize=6)

# ---------------------------------------------------------------- (c) evaluation choices
card(GX, BOT_Y, GW, CARD_H, "c", "Evaluation choices outweigh module effects (Q4)",
     "Measured on the same runs; Dice points relative to the fair estimate")
ax = ax_mm(GX + 36.0, BOT_Y + 7.0, 47.0, 24.0)
rows = ["Learning-rate tuning", "Best of 5 seeds", "Test-set checkpoint choice", "Largest plug-in gain"]
d_vals = [0.557, 0.309, 0.071, 0.0149]
c_vals = [0.414, 0.422, 0.339, -0.064]
y = np.arange(len(rows))[::-1].astype(float)
ax.barh(y + 0.18, d_vals, 0.34, color=[VERM, VERM, VERM, GREY], lw=0)
ax.barh(y - 0.18, c_vals, 0.34, color=[VERM, VERM, VERM, GREY], alpha=0.55, lw=0)
for yy, a_, b_ in zip(y, d_vals, c_vals):
    ax.text(max(a_, 0) + 0.015, yy + 0.18, f"{a_:+.2f}".replace("-", "−"), va="center", fontsize=5.6, color=INK)
    ax.text(max(b_, 0) + 0.015, yy - 0.18, f"{b_:+.2f}".replace("-", "−"), va="center", fontsize=5.6, color=INK)
ax.axvline(0, color=INK, lw=0.6)
ax.set_yticks(y); ax.set_yticklabels([])
for yy, r in zip(y, rows):
    base.text(GX + 3.0, BOT_Y + 7.0 + 24.0 * (yy + 0.5) / len(rows), r, fontsize=6.2, color=INK, va="center")
ax.set_xlim(-0.1, 0.72); ax.set_xticks([0, 0.25, 0.5]); ax.set_xlabel("Dice difference (points)", fontsize=6, labelpad=1)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.add_patch(Rectangle((0.43, 0.30), 0.04, 0.22, color=VERM, transform=ax.transData, clip_on=False))
ax.text(0.485, 0.41, "DRIVE", fontsize=5.6, va="center", color=MUTED)
ax.add_patch(Rectangle((0.43, -0.12), 0.04, 0.22, color=VERM, alpha=0.55, transform=ax.transData, clip_on=False))
ax.text(0.485, -0.01, "CHASE_DB1", fontsize=5.6, va="center", color=MUTED)

# ---------------------------------------------------------------- (d) public code
dx = GX + GW + GAP
card(dx, BOT_Y, GW, CARD_H, "d", "Practices in public code (Q5)",
     "Commit-pinned audit with file- and line-level evidence")
wa = ax_mm(dx + 3.0, BOT_Y + 3.0, 82.0, 28.0); wa.set_xlim(0, 82); wa.set_ylim(0, 28); wa.axis("off")


def waffle(y0, n, k, colour, title, note):
    wa.text(0.0, y0 + 9.2, title, fontsize=6.2, weight="bold", color=INK, va="center")
    for i in range(n):
        filled = i < k
        wa.add_patch(FancyBboxPatch((i * 5.2, y0), 4.2, 4.2, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=colour if filled else "#ffffff", ec=colour if filled else FAINT, lw=0.6))
    wa.text(n * 5.2 + 1.5, y0 + 2.1, note, fontsize=6.2, color=INK, va="center")


waffle(15.0, 11, 4, VERM, "DSConv ports (11 repositories)", "4 retain the defect")
waffle(1.0, 12, 2, VERM, "Vessel training pipelines (12 repositories)", "2 select on test data")

# ---------------------------------------------------------------- recommendations
base.text(2.0, 7.2, "Recommendations", fontsize=7.2, weight="bold", color=INK, va="center")
pills = ["Validation-only selection", "Equal tuning budget", "≥3–5 seeds, paired tests",
         "Capacity-matched control", "Batch-invariance unit test"]
widths = [3.0 + 1.0 * len(p) for p in pills]
gap = (W - 2.0 - 27.0 - sum(widths)) / (len(pills) - 1)
x = 27.0
for p, w in zip(pills, widths):
    base.add_patch(FancyBboxPatch((x, 4.4), w, 5.6, boxstyle="round,pad=0,rounding_size=2.8",
                                  fc="#ffffff", ec=TEAL, lw=0.7))
    base.text(x + w / 2, 7.2, p, fontsize=6.2, color=INK, ha="center", va="center")
    x += w + gap

fig.canvas.draw()
out = sys.argv[3]
# Only the two comparable data panels in card b form an alignment group; the other axes are
# free-positioned schematic/inset elements inside fixed-size cards drawn on one canvas.
fig.savefig(f"{out}.svg"); fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png", dpi=600)
fig.savefig(f"{out}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("saved", out, "pills end at", round(x - gap, 1), "mm")
