"""Step 3 figure (CPU): forward-speed and yaw-rate traces, robot E (as evaluated) against robot T (training robot model).

Panels: forward trial (forward speed), turn-left and turn-right trials (yaw rate), and the 30-s executor-like sequence
(forward speed, yaw rate). Base-frame quantities as in metrics.py (forward speed of the Trunk origin; yaw rate =
root_ang_vel_b z), repeat 1. 12-s panels: thin translucent lines = every 0.02-s control step (unsmoothed); bold lines
= centred moving average over 33 steps (0.66 s, about one 1.5-Hz gait cycle, which removes the 3-Hz stepping
oscillation; drawn only where the full window exists). Sequence panels: unsmoothed per-step traces (a 0.66-s average
would blur the 0.16-s zero segments). Gray step line = command; shaded band = steady-state window (last 8 s).
Colours: slots 1-2 of the dataviz skill's reference categorical palette (blue #2a78d6 = E, orange #eb6834 = T),
light surface #fcfcfb, hairline grid #e1e0d9, text #0b0b0b / #52514e / #898781.
Usage: python make_figure.py [--states_dir DIR] [--out_dir DIR] [--rep 1]
"""
import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import load, trial_series  # noqa: E402

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL = {"E": "#2a78d6", "T": "#eb6834"}
LABEL = {"E": "E — as evaluated (K1_locomotion, head/arms merged)", "T": "T — training robot model (K1_22dof)"}
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
CMD_COL = "#898781"
BAND = "#f0efec"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8, width=0.6)


SMOOTH_N = 33  # 0.66 s = about one gait cycle (1.5-Hz gait clock) at 0.02-s steps


def moving_average(t, y, n=SMOOTH_N):
    """Centred n-sample mean; only where the full window exists."""
    if len(y) < n:
        return t[:0], y[:0]
    m = np.convolve(y, np.ones(n) / n, mode="valid")
    h = n // 2
    return t[h:h + len(m)], m


def panel(ax, series, comp, title, ylabel, steady=False, fell=None, smooth=False):
    style(ax)
    for robot in ("E", "T"):
        s = series.get(robot)
        if s is None:
            continue
        y = s["v"][:, 0] if comp == "vx" else s["w"][:, 2]
        if smooth:
            ax.plot(s["t"][1:], y[1:], color=COL[robot], linewidth=0.6, alpha=0.35, zorder=2)
            ts, ys = moving_average(s["t"][1:], y[1:])
            ax.plot(ts, ys, color=COL[robot], linewidth=1.8, solid_joinstyle="round", solid_capstyle="round",
                    label=LABEL[robot], zorder=3)
        else:
            ax.plot(s["t"][1:], y[1:], color=COL[robot], linewidth=0.9, alpha=0.9, solid_joinstyle="round",
                    solid_capstyle="round", label=LABEL[robot], zorder=3 if robot == "T" else 2)
        if fell and fell.get(robot):
            tf = fell[robot]
            ax.axvline(tf, color=COL[robot], linewidth=0.8, linestyle="-", alpha=0.6)
            ax.annotate(f"{robot} fell, t = {tf:.2f} s", xy=(tf, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -10), textcoords="offset points", fontsize=7, color=INK2)
    ref = next((series[r] for r in ("E", "T") if series.get(r) is not None), None)
    if ref is not None:
        c = ref["cmd"][:, 0] if comp == "vx" else ref["cmd"][:, 2]
        ax.step(ref["t"][1:], c[1:], where="pre", color=CMD_COL, linewidth=1.0, label="command", zorder=4)
        if steady:
            ax.axvspan(4.0, ref["t"][-1] if len(ref["t"]) else 12.0, color=BAND, zorder=0, linewidth=0)
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=4)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    ax.set_xlabel("time from command start (s)", fontsize=8, color=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", default=os.path.join(W, "logs", "states"))
    ap.add_argument("--out_dir", default=os.path.join(W, "figures"))
    ap.add_argument("--rep", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    S, F = {}, {}
    for trial in ("forward", "turn_left", "turn_right", "sequence"):
        S[trial], F[trial] = {}, {}
        for robot in ("E", "T"):
            try:
                d, meta = load(a.states_dir, robot, trial, a.rep)
            except FileNotFoundError:
                S[trial][robot] = None
                continue
            s = trial_series(d, meta)
            S[trial][robot] = s
            if meta.get("fall") and meta["fall"]["phase"] == "command" and s is not None:
                F[trial][robot] = (s["n_cmd"] + 1) * s["dt"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    fig = plt.figure(figsize=(11.0, 8.4), dpi=150)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.62, wspace=0.28,
                          left=0.065, right=0.985, top=0.865, bottom=0.07)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    ax5 = fig.add_subplot(gs[2, :], sharex=ax4)
    panel(ax1, S["forward"], "vx", "Forward (0.5, 0, 0): forward speed", "m/s", steady=True, fell=F["forward"],
          smooth=True)
    panel(ax2, S["turn_left"], "wz", "Turn left (0, 0, +π/6): yaw rate", "rad/s", steady=True, fell=F["turn_left"],
          smooth=True)
    panel(ax3, S["turn_right"], "wz", "Turn right (0, 0, −π/6): yaw rate", "rad/s", steady=True,
          fell=F["turn_right"], smooth=True)
    panel(ax4, S["sequence"], "vx", "Executor-like sequence (forward 1.5 s, zero 0.16 s, turn left 1.5 s, zero, "
          "forward, zero, turn right, zero, …): forward speed", "m/s", fell=F["sequence"])
    panel(ax5, S["sequence"], "wz", "Executor-like sequence: yaw rate", "rad/s", fell=F["sequence"])
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, 12)
    ax4.set_xlim(0, 30)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.065, 0.975), ncol=3, frameon=False, fontsize=8.5,
               labelcolor=INK, handlelength=2.2, columnspacing=1.6)
    fig.text(0.065, 0.915, "Base-frame forward speed of the Trunk origin and yaw rate; Isaac Sim 4.1, flat plane, "
             "model_14498, repeat 1 (repeat 2: see repeatability table). Gray = command.\n"
             "Top row: thin = every 0.02-s control step, bold = 0.66-s (one gait cycle) centred moving average, "
             "shaded = steady-state window (last 8 s). Sequence rows: every 0.02-s step, unsmoothed.",
             fontsize=7.2, color=INK2, linespacing=1.5)
    out = os.path.join(a.out_dir, "speed_yawrate_E_vs_T")
    fig.savefig(out + ".png", facecolor=SURFACE)
    fig.savefig(out + ".pdf", facecolor=SURFACE)
    print("saved", out + ".png/.pdf")


if __name__ == "__main__":
    main()
