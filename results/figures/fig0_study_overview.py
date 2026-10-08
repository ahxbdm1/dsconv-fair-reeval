"""Figure 1 (study overview) for the manuscript. Single-panel schematic, 183 mm wide (double column).
Backend: Python/matplotlib. 1 data unit = 1 mm."""
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
})
W, H = 183.0, 92.0
INK, MUTED = "#1f1f1f", "#55534f"
NEUTRAL_FILL, NEUTRAL_EDGE = "#f3f3f1", "#9a9893"
ACCENT_FILL, ACCENT_EDGE = "#e3ecf7", "#0F4D92"      # result boxes (signal family)

fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x0, y0, x1, y1, text, fill=NEUTRAL_FILL, edge=NEUTRAL_EDGE, weight="normal", ha="left", lw=0.6):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fill, ec=edge, lw=lw))
    tx = x0 + 2.2 if ha == "left" else (x0 + x1) / 2
    ax.text(tx, (y0 + y1) / 2, text, ha=ha, va="center", fontsize=7, color=INK, weight=weight, linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.7,
                                 color=MUTED, shrinkA=0, shrinkB=0))


C0 = (2.0, 29.0)      # starting point
C1 = (34.0, 58.0)     # question
C2 = (62.0, 123.0)    # what we did
C3 = (127.0, 181.0)   # what we found
TOP = 90.0
ax.text((C1[0] + C1[1]) / 2, TOP - 1.5, "Question", ha="center", va="center", fontsize=7, color=MUTED, weight="bold")
ax.text(C2[0] + 2.2, TOP - 1.5, "What we did", ha="left", va="center", fontsize=7, color=MUTED, weight="bold")
ax.text(C3[0] + 2.2, TOP - 1.5, "What we found", ha="left", va="center", fontsize=7, color=MUTED, weight="bold")

lanes = [
    ("Q1–Q2\nImplementation",
     "Inspect the DSConv gather index;\nbatch the tiles of a batch-1 model (B = 8);\ntrain at B = 8: original, corrected, shuffle",
     "Batched inference: Dice 81.7% → 18.3%\nCorrected − original: −0.85 pts\nSections 3, 5.1, 5.2"),
    ("Q3\nFair comparison",
     "Validation-only checkpoint selection,\nper-dataset lr, 5 seeds, image-paired tests;\nU-Net, DSCNet and four plug-in variants",
     "No plug-in improves DSCNet's Dice\n(−0.15 to +0.02 pts)\nSection 5.3"),
    ("Q4\nEvaluation\nchoices",
     "Same runs plus diagnostic test logs\n(logged only, never used for selection):\nlr tuning, best seed, test-chosen checkpoint",
     "Each choice > largest plug-in gain;\ntogether: plug-ins look ≈0.8 pts better\nSection 5.4"),
    ("Q5\nPublic code",
     "Commit-pinned audit, file/line evidence:\n12 vessel-segmentation training pipelines\nand 11 DSConv ports",
     "2 of 12 select checkpoints on test data\n4 of 11 ports retain the defect\nSection 5.5"),
]
lane_h, gap = 14.0, 3.0
y_top = TOP - 5.0
centers = []
for i, (q, did, found) in enumerate(lanes):
    y1 = y_top - i * (lane_h + gap); y0 = y1 - lane_h; yc = (y0 + y1) / 2; centers.append(yc)
    box(C1[0], y0, C1[1], y1, q, ha="center", weight="bold")
    box(C2[0], y0, C2[1], y1, did)
    box(C3[0], y0, C3[1], y1, found, fill=ACCENT_FILL, edge=ACCENT_EDGE)
    arrow(C1[1], yc, C2[0], yc)
    arrow(C2[1], yc, C3[0], yc)

y_lo = centers[-1] - lane_h / 2
box(C0[0], y_lo, C0[1], y_top,
    "Case study\n\nDSCNet with\nDSConv\n(ICCV 2023)\n\nDRIVE\n16 / 4 / 20\n\nCHASE_DB1\n16 / 4 / 8\n\n(train / val / test)",
    ha="center")
for yc in centers:
    arrow(C0[1], yc, C1[0], yc)

rb1 = y_lo - 4.0; rb0 = 2.5
box(C1[0], rb0, C3[1], rb1,
    "Recommendations (Section 6): select checkpoints and hyper-parameters on validation data only;\n"
    "tune baselines with the same budget; report ≥3–5 seeds with image-paired tests;\n"
    "include a capacity-matched control; unit-test every module for batch invariance",
    fill="#ffffff", edge=MUTED)
for x in ((C1[0] + C1[1]) / 2, (C2[0] + C2[1]) / 2, (C3[0] + C3[1]) / 2):
    arrow(x, y_lo, x, rb1)

fig.canvas.draw()
out = sys.argv[1] if len(sys.argv) > 1 else 'fig0_study_overview'
fig.savefig(f"{out}.svg"); fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png", dpi=600)
print("saved", out)
