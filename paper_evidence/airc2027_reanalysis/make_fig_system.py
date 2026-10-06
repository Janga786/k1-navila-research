"""Fig. 1: (a) the evaluated closed loop; (b) the simulated K1 as the benchmark rendered it at the
start of episode 0 (dataset entry 0, "Exit the bedroom and turn left. ..."); (c) the robot camera at
the same start pose for each of the six camera heights.

All images are re-renders made on the sweep workstation on 2026-10-01 with the unmodified evaluation
code, robot model, scene and render settings (private NaVILA-Complete-Archive,
14_airc2027_renders/images/as_evaluated/; provenance in fig1_assets/SOURCES.json):
- (b) third_person/behind_left.png: path traced (256 samples per pixel, OptiX denoiser) from a camera
  behind the robot after the evaluator's 1-s reset warm-up; cropped to the central 1080x1080 region
  around the robot and resized to 612x612. The head, arms and chest logo are drawn at the trunk's spawn
  pose, above the settled torso, exactly as the evaluator rendered them.
- (c) robot_camera/idx0_rec0_camz<offset>_initial.png: the evaluator's robot-camera frame after the
  reset warm-up (history frame 0, which enters every model query), RTX real-time, 1280x720, lossless;
  resized to 450x253. Each matches frame 0 of its sweep run's episode-0 video to within H.264 noise
  (mean absolute difference 1.0-1.8 of 255).
Heights are the camera's height above the floor in this episode with the robot standing after the
reset warm-up (trunk 0.497 m above the floor plus the camera offset 0.07-0.97 m above the trunk)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
from PIL import Image

for f in ["/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf",
          "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf"]:
    try: font_manager.fontManager.addfont(f)
    except Exception: pass
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Liberation Serif"],
                     "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "fig1_assets")
OUTDIR = os.environ.get("FIG_DIR", HERE)
INK, MUTED, BOX, EDGE, ACC = "#0b0b0b", "#52514e", "#f3f2ee", "#8d8c88", "#2a78d6"

FW, FH = 7.16, 1.60                          # figure size (in): IEEE two-column width
fig = plt.figure(figsize=(FW, FH))


def axes_in(x, y, w, h):                     # position in inches -> figure fraction
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])


# ---------------- (a) loop diagram ----------------
ax = axes_in(0.0, 0.02, 2.45, 1.38); ax.set_xlim(-1, 101); ax.set_ylim(2, 56); ax.axis("off")


def box(x, y, w, h, title, sub):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                facecolor=BOX, edgecolor=EDGE, linewidth=0.6))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=6.7, color=INK, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=5.8, color=MUTED)


def arrow(p, q, label=None, lx=0, ly=0, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, linewidth=0.7, color=INK,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, ha=ha, va="center", fontsize=5.8, color=MUTED)


W_ = 29
box(0.3, 36, W_, 17, "Isaac Sim", "K1 physics, 200 Hz")
box(35.8, 36, W_, 17, "Preprocess", "stretch, crop or pad")
box(71.0, 36, W_, 17, "NaVILA 8B", "8 frames, instruction")
box(71.0, 6, W_, 17, "Parser, executor", "closed loop on pose")
box(29.5, 6, W_, 17, "Walking policy", "12 joints, 50 Hz")
arrow((29.4, 44.5), (35.7, 44.5))
arrow((64.9, 44.5), (70.9, 44.5))
arrow((85.5, 35.5), (85.5, 23.5), "text action", 1.3, 0, ha="left")
arrow((70.9, 14.5), (58.6, 14.5), "velocity", 0, 4.2)
arrow((29.4, 14.5), (14.5, 14.5), None)
arrow((14.5, 14.5), (14.5, 35.5), "joint\ntargets", -1.5, 0, ha="right")
fig.text(0.02 / FW, 1.51 / FH, "(a) Closed loop", ha="left", va="center", fontsize=7.2, color=INK)

# ---------------- (b) the K1 as rendered ----------------
CH = 1.36
axb = axes_in(2.58, 0.04, CH, CH)
axb.imshow(np.asarray(Image.open(os.path.join(ASSETS, "k1_as_rendered_ep0_behind.png"))), interpolation="none")
axb.set_xticks([]); axb.set_yticks([])
for sp in axb.spines.values(): sp.set_linewidth(0.4); sp.set_color(EDGE)
axb.annotate("head and arms\nat spawn pose", xy=(352, 96), xytext=(404, 40), textcoords="data",
             fontsize=5.6, color=INK, ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
             arrowprops=dict(arrowstyle="-|>", mutation_scale=5, linewidth=0.6, color=INK, shrinkA=1, shrinkB=1))
axb.annotate("torso", xy=(282, 300), xytext=(30, 300), textcoords="data",
             fontsize=5.6, color=INK, ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
             arrowprops=dict(arrowstyle="-|>", mutation_scale=5, linewidth=0.6, color=INK, shrinkA=1, shrinkB=1))
fig.text(2.58 / FW, 1.51 / FH, "(b) K1 as rendered", ha="left", va="center", fontsize=7.2, color=INK)

# ---------------- (c) robot camera at six heights ----------------
runs = [("h060_camz0.07", "0.57 m"), ("h078_camz0.25", "0.75 m"), ("h095_camz0.42", "0.92 m"),
        ("h110_camz0.57", "1.07 m"), ("h130_camz0.77", "1.27 m"), ("h150_camz0.97", "1.47 m")]
tw = 1.00; th = tw * 9 / 16; gx = 0.045; gy = 0.15
x0 = 4.06; ytop = 1.40 - th
for i, (r, lab) in enumerate(runs):
    col, row = i % 3, i // 3
    x = x0 + col * (tw + gx); y = ytop - row * (th + gy)
    a = axes_in(x, y, tw, th)
    im = np.asarray(Image.open(os.path.join(ASSETS, f"robotcam_ep0_{r}_initial.png")))
    a.imshow(im, interpolation="none"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_linewidth(0.4); sp.set_color(EDGE)
    if r.startswith("h078"):                       # crop keeps the central 720 of 1280 columns
        for cx in (280 / 1280 * im.shape[1], 1000 / 1280 * im.shape[1]):
            a.axvline(cx - 0.5, color=ACC, linewidth=0.8, linestyle="--")
    fig.text((x + tw / 2) / FW, (y - 0.06) / FH, lab, ha="center", va="center", fontsize=6.6, color=INK)
fig.text(x0 / FW, 1.51 / FH, "(c) Robot camera at each height, same start", ha="left", va="center", fontsize=7.2, color=INK)

fig.savefig(os.path.join(OUTDIR, "fig_system.pdf"), pad_inches=0.0)
fig.savefig(os.path.join(OUTDIR, "fig_system.png"), dpi=300, pad_inches=0.0)
print("ok")
