"""Fig. 2 (v3). Reads tables/paper_numbers.json, paper_numbers_v2.json and audit_v3.json.
(a) bit-identical share by episode length: same-period reruns vs June run paired with reruns
(b) SR difference from the rerun mean per configuration, route-clustered paired 95% CI (no rerun band:
    rerun agreement understates the variability of comparisons between configurations)
(c) Cohen's kappa with reruns on all episodes (filled) and on episodes where the reruns diverged (open)"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ["/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf"]:
    try: font_manager.fontManager.addfont(f)
    except Exception: pass
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Liberation Serif"],
                     "font.size": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8d8c88", "#0b0b0b", "#52514e", "#e4e3df"
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("FIG_DIR", HERE)
N = json.load(open(os.path.join(HERE, "tables", "paper_numbers.json")))
V = json.load(open(os.path.join(HERE, "tables", "paper_numbers_v2.json")))
A3 = {c["config"]: c for c in json.load(open(os.path.join(HERE, "tables", "audit_v3.json")))["configs"]}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 1.72),
                                    gridspec_kw=dict(width_ratios=[1.15, 1, 1], wspace=0.42))
# (a)
labels = ["≤1k", "1–2k", "2–3k", "3–4k", "4–5k", ">5k"]
ys, ns = N["ident_by_length"]["sweep"]["pct"], N["ident_by_length"]["sweep"]["n"]
yj, nj = N["ident_by_length"]["june"]["pct"], N["ident_by_length"]["june"]["n"]
x = np.arange(6); w = 0.36
ax1.bar(x - w / 2 - 0.03, ys, w, color=BLUE, label="same-period reruns", zorder=2)
ax1.bar(x + w / 2 + 0.03, yj, w, color=ORANGE, label="June run vs. reruns", zorder=2)
for xi, (a, n) in enumerate(zip(ys, ns)):
    ax1.text(xi - w / 2 - 0.03, a + 1.5, f"{n}", ha="center", va="bottom", fontsize=5.6, color=MUTED, rotation=90)
for xi, (a, n) in enumerate(zip(yj, nj)):
    ax1.text(xi + w / 2 + 0.03, a + 1.5, f"{n}", ha="center", va="bottom", fontsize=5.6, color=MUTED, rotation=90)
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_xlabel("Episode length (control steps)")
ax1.set_ylabel("Pairs with identical records (%)")
ax1.set_ylim(0, 100); ax1.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
ax1.legend(frameon=False, loc="upper right", handlelength=1.0)
ax1.set_title("(a) Exact reproduction", fontsize=8, loc="left")

# (b)
C = V["configs"]
# camera height above the floor in the one measured episode (dataset entry 0, robot standing
# after the reset warm-up: trunk 0.497 m above the floor plus the camera offset above the trunk);
# † = runs with the gray robot-camera region at episode start (Section V-B)
names = {"crop": "Crop", "pad": "Pad", "h060": "0.57 m", "h095": "0.92 m†", "h110": "1.07 m†",
         "h130": "1.27 m", "h150": "1.47 m"}
order = ["crop", "pad", "h060", "h095", "h110", "h130", "h150"]
cd = {c["config"]: c for c in C}
yy = np.arange(len(order))[::-1]
ax2.axvline(0, color="#c9c8c3", linewidth=0.6, zorder=1)
for yi, k in zip(yy, order):
    c = A3[k]["SR"]
    ax2.plot(c["ci"], [yi, yi], color="#9dc0ea", linewidth=1.2, zorder=2)
    ax2.plot([c["delta"]], [yi], "o", ms=3.8, color=BLUE, zorder=3)
ax2.set_yticks(yy); ax2.set_yticklabels([names[k] for k in order])
ax2.set_xlabel("SR minus rerun mean (pp)")
ax2.set_xlim(-9, 14)
ax2.set_ylim(-0.7, len(order) - 0.3)
ax2.set_title("(b) Aggregate shift", fontsize=8, loc="left")

# (c)
rows = [("Reruns (300)", V["configs"][0]["kappa_R"], V["configs"][0]["kappa_R_div"], GRAY),
        ("Reruns (200)", V["configs"][2]["kappa_R"], V["configs"][2]["kappa_R_div"], GRAY),
        ("June run", V["J"]["kappa_all"], V["J"]["kappa_div"], ORANGE)] + \
       [(names[k], cd[k]["kappa_X"], cd[k]["kappa_X_div"], BLUE) for k in order]
yy3 = np.arange(len(rows))[::-1]
for yi, (lab, ka, kd, col) in zip(yy3, rows):
    ax3.plot([kd, ka], [yi, yi], color="#d6d5d0", linewidth=0.8, zorder=1)
    ax3.plot([ka], [yi], "o", ms=3.8, color=col, zorder=3)
    ax3.plot([kd], [yi], "o", ms=3.8, markerfacecolor="white", markeredgecolor=col, markeredgewidth=0.9, zorder=3)
ax3.set_yticks(yy3); ax3.set_yticklabels([r[0] for r in rows])
ax3.set_xlabel("Cohen's κ with reruns")
ax3.set_xlim(0.1, 0.7); ax3.set_ylim(-0.7, len(rows) - 0.3)
ax3.axhline(len(order) - 0.5, color=GRID, linewidth=0.6)
from matplotlib.lines import Line2D
ax3.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.8, color=MUTED, label="all episodes"),
                    Line2D([], [], marker="o", ls="", ms=3.8, markerfacecolor="white", markeredgecolor=MUTED,
                           label="reruns diverged")], frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.36), handletextpad=0.2,
           borderaxespad=0.0)
ax3.set_title("(c) Episode-level agreement", fontsize=8, loc="left")
fig.savefig(os.path.join(OUTDIR, "fig_results.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(OUTDIR, "fig_results.png"), dpi=250, bbox_inches="tight", pad_inches=0.02)
print("ok")
