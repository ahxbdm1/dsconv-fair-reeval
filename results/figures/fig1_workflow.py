"""Figure 1 (study workflow), v3. Single-panel flowchart, 183 mm wide (double column), Python/matplotlib.
Input -> four analyses -> one key finding each -> recommendations. 1 data unit = 1 mm.
usage: python fig_flow_v3.py <nature-figure scripts dir> <out stem>"""
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
})
INK, MUTED, LINE = "#1b1f24", "#5b6270", "#8d95a1"
EDGE, FIND_FILL = "#c9ced6", "#edf2f8"
NAVY, NAVY_L, VERM, TEAL, GREY = "#1f4e79", "#9fb8d3", "#c8414b", "#2a8c82", "#b3bac4"

W, H = 183.0, 76.0
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

# ------------------------------------------------------------------ grid
COL = {"in": (2.0, 28.0), "an": (34.0, 80.0), "fd": (86.0, 138.0), "out": (144.0, 181.0)}
ROW_H, ROW_GAP, TOP = 13.0, 4.0, 67.5
rows_y = [TOP - i * (ROW_H + ROW_GAP) for i in range(4)]          # top edge of each row
rows_c = [y - ROW_H / 2 for y in rows_y]                           # row centres
BOT = rows_y[-1] - ROW_H
MID = (TOP + BOT) / 2


def node(x0, x1, y0, y1, fill="#ffffff", edge=EDGE, lw=0.6):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fill, ec=edge, lw=lw, zorder=2))


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5.5, lw=0.7,
                                 color=LINE, shrinkA=0, shrinkB=0, zorder=1))


def line(xs, ys):
    ax.plot(xs, ys, color=LINE, lw=0.7, solid_capstyle="round", zorder=1)


# column headings
for key, label in [("in", "Input"), ("an", "Analysis"), ("fd", "Key finding"), ("out", "Output")]:
    x0, x1 = COL[key]
    ax.text((x0 + x1) / 2, TOP + 4.6, label, ha="center", va="center", fontsize=7, weight="bold", color=MUTED)

# ------------------------------------------------------------------ input node (retina icon + case study)
x0, x1 = COL["in"]; cx = (x0 + x1) / 2
node(x0, x1, BOT, TOP)
icx, icy, r = cx, MID + 12.0, 8.0
ax.add_patch(Circle((icx, icy), r, fc="#f4e9e4", ec="#d9bfb4", lw=0.6, zorder=3))
ax.add_patch(Circle((icx + 3.0, icy + 0.6), 1.5, fc="#fbf5f2", ec="#d9bfb4", lw=0.5, zorder=4))
od = np.array([icx + 3.0, icy + 0.6])
ax.add_patch(Circle((icx - 2.6, icy - 0.2), 1.1, fc="#e8d3ca", lw=0, zorder=3.5))            # macula


def vessel(pts, lw):
    pts = [np.array(p, float) for p in pts]
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(pts) - 1)
    ax.add_patch(PathPatch(Path(pts, codes), fc="none", ec=VERM, lw=lw, alpha=0.85, zorder=4, capstyle="round"))


def bez(P, t):
    P = [np.array(q, float) for q in P]
    return ((1 - t) ** 3) * P[0] + 3 * ((1 - t) ** 2) * t * P[1] + 3 * (1 - t) * t * t * P[2] + (t ** 3) * P[3]


for sgn in (1, -1):
    arc = [od, od + [-0.6, 4.4 * sgn], od + [-6.4, 5.8 * sgn], od + [-10.4, 2.0 * sgn]]   # temporal arcade
    vessel(arc, 0.9)
    for t, d1, d2, lw in [(0.38, [-0.4, 1.6], [-2.4, 2.6], 0.5), (0.62, [-0.6, -1.5], [-1.4, -2.9], 0.45),
                          (0.80, [-0.3, 0.8], [-1.0, 1.1], 0.45)]:
        p0 = bez(arc, t); d1 = np.array(d1) * [1, sgn]; d2 = np.array(d2) * [1, sgn]
        vessel([p0, p0 + d1 * 0.5, p0 + (d1 + d2) * 0.5, p0 + d2], lw)
    vessel([od, od + [1.0, 1.6 * sgn], od + [2.2, 3.0 * sgn], od + [3.4, 4.2 * sgn]], 0.55)  # nasal branch
    vessel([od, od + [1.4, 0.4 * sgn], od + [2.8, 0.9 * sgn], od + [4.0, 1.8 * sgn]], 0.4)
ty = MID - 3.0
ax.text(cx, ty, "DSCNet + DSConv", ha="center", va="center", fontsize=7, weight="bold", color=INK)
ax.text(cx, ty - 5.0, "DRIVE", ha="center", va="center", fontsize=6.5, color=INK)
ax.text(cx, ty - 9.0, "CHASE_DB1", ha="center", va="center", fontsize=6.5, color=INK)
ax.text(cx, ty - 15.0, "86 training runs", ha="center", va="center", fontsize=6.5, color=MUTED)

# ------------------------------------------------------------------ icons for the four analyses (centred at x, y)
def icon_defect(x, y):
    s, g = 1.7, 0.45
    for row, dy in [(0, 1.3), (1, -1.3)]:
        for i in range(4):
            xx = x - 2 * s - 1.5 * g + i * (s + g)
            bad = row == 1 and i > 0
            ax.add_patch(Rectangle((xx, y + dy - s / 2), s, s, fc="#ffffff" if bad else NAVY_L if i else NAVY,
                                   ec=VERM if bad else NAVY, lw=0.6, zorder=3))


def icon_seeds(x, y):
    for i, (c, m) in enumerate([(GREY, -0.6), (NAVY, 0.9), (GREY, 0.1)]):
        xx = x - 2.6 + i * 2.6
        for dy in (-1.4, 0.0, 1.3):
            ax.add_patch(Circle((xx, y + m + dy * 0.9), 0.42, fc=c, lw=0, zorder=3))
        ax.plot([xx - 0.9, xx + 0.9], [y + m] * 2, color=INK, lw=0.7, zorder=4)
    ax.plot([x - 3.8, x + 3.8], [y + 0.9] * 2, color=NAVY, lw=0.5, ls=(0, (2, 1.2)), zorder=2)


def icon_bars(x, y):
    base = y - 2.6
    for i, h in enumerate([1.6, 3.2, 4.8]):
        ax.add_patch(Rectangle((x - 3.3 + i * 2.4, base), 1.7, h, fc=VERM, alpha=0.35 + 0.25 * i, lw=0, zorder=3))
    ax.plot([x - 3.8, x + 3.8], [base, base], color=INK, lw=0.6, zorder=4)


def icon_code(x, y):
    w, h, f = 5.4, 6.6, 1.5
    x0, y0 = x - w / 2, y - h / 2
    ax.add_patch(Polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h - f), (x0 + w - f, y0 + h), (x0, y0 + h)],
                         closed=True, fc="#ffffff", ec=NAVY, lw=0.6, zorder=3))
    for k, (ln, c) in enumerate([(3.2, NAVY_L), (2.2, NAVY_L), (3.4, VERM), (2.6, NAVY_L)]):
        ax.plot([x0 + 0.9, x0 + 0.9 + ln], [y0 + h - 1.9 - k * 1.25] * 2, color=c, lw=0.8 if c == VERM else 0.6,
                solid_capstyle="round", zorder=4)


analyses = [(icon_defect, "Implementation check", "Batch-invariance test"),
            (icon_seeds, "Fair comparison", "5 seeds, validation-only"),
            (icon_bars, "Evaluation choices", "Re-scored on the same runs"),
            (icon_code, "Public-code audit", "11 ports, 12 pipelines")]
findings = [("81.7 → 18.3", "Dice (%) when tiles are batched"),
            ("−0.15 to +0.01", "Plug-in gain over DSCNet"),
            ("up to +0.56", "Dice gained from evaluation choices"),
            ("4/11 · 2/12", "Defective ports · test-set selection")]

xa0, xa1 = COL["an"]; xf0, xf1 = COL["fd"]
for (icon, title, sub), (big, small), yt, yc in zip(analyses, findings, rows_y, rows_c):
    node(xa0, xa1, yt - ROW_H, yt)
    icon(xa0 + 6.0, yc)
    ax.text(xa0 + 12.0, yc + 2.1, title, ha="left", va="center", fontsize=7, weight="bold", color=INK)
    ax.text(xa0 + 12.0, yc - 2.5, sub, ha="left", va="center", fontsize=6.5, color=MUTED)
    node(xf0, xf1, yt - ROW_H, yt, fill=FIND_FILL, edge=NAVY_L)
    ax.text((xf0 + xf1) / 2, yc + 2.1, big, ha="center", va="center", fontsize=8.5, weight="bold", color=NAVY)
    ax.text((xf0 + xf1) / 2, yc - 2.7, small, ha="center", va="center", fontsize=6.5, color=MUTED)
    arrow(xa1, yc, xf0, yc)

# input -> analyses (bus)
xb = (COL["in"][1] + xa0) / 2
line([COL["in"][1], xb], [MID, MID]); line([xb, xb], [rows_c[-1], rows_c[0]])
for yc in rows_c:
    arrow(xb, yc, xa0, yc)

# findings -> output (bus)
xo0, xo1 = COL["out"]
xm = (xf1 + xo0) / 2
for yc in rows_c:
    line([xf1, xm], [yc, yc])
line([xm, xm], [rows_c[-1], rows_c[0]]); arrow(xm, MID, xo0, MID)

# ------------------------------------------------------------------ output node: recommendations
node(xo0, xo1, BOT, TOP, edge=TEAL, lw=0.8)
ax.text((xo0 + xo1) / 2, TOP - 5.0, "Recommendations", ha="center", va="center", fontsize=7, weight="bold", color=INK)
recs = ["Validation-only selection", "Equal tuning budget", "≥3–5 seeds, paired tests",
        "Capacity-matched control", "Batch-invariance unit test"]
y_first, y_last = TOP - 14.0, BOT + 6.0
for k, rtxt in enumerate(recs):
    yy = y_first - k * (y_first - y_last) / (len(recs) - 1)
    ax.plot([xo0 + 2.6, xo0 + 3.5, xo0 + 5.0], [yy, yy - 0.9, yy + 0.9], color=TEAL, lw=0.9,
            solid_capstyle="round", solid_joinstyle="round")
    ax.text(xo0 + 6.6, yy, rtxt, ha="left", va="center", fontsize=6.5, color=INK)

fig.canvas.draw()
out = sys.argv[1]
fig.savefig(f"{out}.svg"); fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png", dpi=600)
fig.savefig(f"{out}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("saved", out)
