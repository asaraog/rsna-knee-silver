import json, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
W = os.path.dirname(os.path.abspath(__file__)) + "/"
os.makedirs(W + "fig", exist_ok=True)
SURF, T1, T2, GRID, MUTE = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e0", "#b9b8b1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 9.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": T2, "xtick.color": T2, "ytick.color": T2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": T1, "xtick.major.size": 0, "ytick.major.size": 0})
def title(fig, t, s):
    fig.text(0.02, 0.965, t, fontsize=13, fontweight="bold", color=T1, va="top")
    fig.text(0.02, 0.905, s, fontsize=9.5, color=T2, va="top")
D = json.load(open(W + "data/label_auc.json"))
LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

# ---------- Fig 1: pipeline ----------
fig = plt.figure(figsize=(9, 4.8), dpi=200); ax = fig.add_axes([0, 0, 1, 0.86]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
title(fig, "Solution overview", "Two halves, rank-blended per finding. Specialist members each overwrite a share of their own findings.")
def box(x, y, w, h, txt, col, alpha=0.13, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=col, alpha=alpha, ec="none"))
    lines = txt.split("\n"); n = len(lines); step = 3.9
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h / 2 + (n - 1) * step / 2 - i * step, l, ha="center", va="center", fontsize=fs + (0.4 if i == 0 else 0),
                fontweight="bold" if i == 0 else "normal", color=T1 if i == 0 else T2)
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.1, color="#8a8983", shrinkA=0, shrinkB=0))
box(0.5, 22, 12, 16, "Test study\nsagittal, coronal\nand axial MRI", "#8a8983", 0.12)
ax.text(16, 56.5, "HYBRID HALF", fontsize=7.6, color=BLUE, fontweight="bold"); ax.text(16, 25.2, "PUBLIC HALF", fontsize=7.6, color=ORANGE, fontweight="bold")
box(16, 44, 22, 11, "Window-MIL stack\n8 arms, CoAtNet-2 + ConvNeXt-V2", BLUE)
box(16, 31.5, 22, 11, "Sagittal CNN pair\nConvNeXt-Tiny, 16 slices", BLUE)
box(16, 12.5, 22, 11, "R7 (heliosli)\ncompact ConvNeXt bundle", ORANGE)
box(16, 0.5, 22, 11, "S6 (nartaa)\nwindow-MIL ConvNeXt-small", ORANGE)
for y in (49.5, 37, 18, 6): arrow(12.5, 30, 16, y)
box(41.5, 37, 12, 13, "Rank fuse\n0.7 / 0.3", BLUE)
arrow(38, 49.5, 41.5, 46); arrow(38, 37, 41.5, 41)
box(57, 31, 19, 25, "Referee + specialists\nmedial/lateral referee\n12 specialist members\ncrops, multi-view, 3D", BLUE, fs=7.9)
arrow(53.5, 43.5, 57, 43.5)
box(41.5, 5.5, 12, 13, "Rank mean\n0.5 / 0.5", ORANGE)
arrow(38, 18, 41.5, 14); arrow(38, 6, 41.5, 10)
box(80, 14, 19.5, 25, "Per-finding blend\n0.5 hybrid, 0.5 public\nSynovitis 0.2 public\nPF OA 0.4 public\nprivate 0.947", "#8a8983", 0.12, fs=7.9)
arrow(76, 37, 80, 31); arrow(53.5, 12, 80, 21)
fig.savefig(W + "fig/fig1_overview.png"); plt.close(fig)

# ---------- Fig 2: label sets on gold-58 ----------
L = D["labels"]; keys = list(L.keys()); cols = [BLUE, ORANGE, AQUA]
order = sorted(LAB, key=lambda c: L[keys[2]][c] - L[keys[0]][c])
rows = order + ["Macro (12)"]
fig = plt.figure(figsize=(9, 5.6), dpi=200); ax = fig.add_axes([0.2, 0.08, 0.76, 0.72])
title(fig, "Image-read labels beat our report-only labels most on Synovitis, Effusion and Contusion",
      "AUC of each training-label set against the 58 expert-labeled studies. Rows sorted by the gap to the 1st-place set.")
for i, r in enumerate(rows):
    c = "macro" if r.startswith("Macro") else r
    v = [L[k][c] for k in keys]
    ax.plot([min(v), max(v)], [i, i], color=GRID, lw=2, zorder=1, solid_capstyle="round")
    for j, k in enumerate(keys): ax.scatter(v[j], i + (j - 1) * 0.2, s=44, color=cols[j], edgecolor=SURF, linewidth=1.3, zorder=3)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows); ax.get_yticklabels()[-1].set_fontweight("bold")
ax.axhline(len(rows) - 1.5, color=GRID, lw=0.8); ax.set_xlim(0.78, 1.006); ax.grid(axis="y", visible=False); ax.set_xlabel("AUC on gold-58")
for j, k in enumerate(keys): ax.text(L[k]["macro"], len(rows) - 0.42 + (j - 1) * 0.2, f"{L[k]['macro']:.3f}", ha="center", fontsize=8, color=T2)
hs = [plt.Line2D([], [], marker="o", ls="", color=cols[j], markeredgecolor=SURF, markersize=7, label=k + f"  ({L[k]['macro']:.3f})") for j, k in enumerate(keys)]
fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.2, 0.86), ncol=3, frameon=False, fontsize=8.4, handletextpad=0.2, columnspacing=1.2)
fig.savefig(W + "fig/fig2_labels.png"); plt.close(fig)

# ---------- Fig 3: how the solution developed ----------
import matplotlib.dates as mdates
R = pd.read_csv(W + "data/submissions.csv")
R = R[R.late == 0].copy()
R["t"] = pd.to_datetime(R.date); R = R.sort_values("t"); R["pub"] = R.public.astype(float); R["priv"] = R.private.astype(float)
R = R[R.pub >= 0.90]
best, rows = -1, []
for _, r in R.iterrows():
    if r.pub > best + 1e-9: best = r.pub; rows.append(r)
Bst = pd.DataFrame(rows)
fin = R[R.final_pick == 1].iloc[0]
Bst = pd.concat([Bst, fin.to_frame().T]).reset_index(drop=True); Bst["pub"] = Bst.pub.astype(float); Bst["priv"] = Bst.priv.astype(float)
end = pd.Timestamp("2026-10-22 23:59")
fig = plt.figure(figsize=(9, 5.2), dpi=200); ax = fig.add_axes([0.08, 0.1, 0.9, 0.66])
title(fig, "How the solution developed", "Each step is a new best public score, with that submission's private score beneath it.")
xs = list(Bst.t) + [end]
ax.step(xs, list(Bst.pub) + [Bst.pub.iloc[-1]], where="post", color=BLUE, lw=2, zorder=3, label="Best public so far")
ax.step(xs, list(Bst.priv) + [Bst.priv.iloc[-1]], where="post", color=ORANGE, lw=2, zorder=3, label="Private of that submission")
def note(day, txt, dy, dx=0):
    t = pd.Timestamp(day); r = Bst[Bst.t <= t + pd.Timedelta(hours=23)].iloc[-1]
    ax.annotate(txt, (r.t, r.pub), xytext=(dx, dy), textcoords="offset points", fontsize=7.8, color=T1, ha="center",
                arrowprops=dict(arrowstyle="-", color=T2, lw=0.7))
note("2026-08-20", "Solo ConvNeXt\n0.931", 30, 4)
note("2026-08-22", "30-model CNN\nensemble, 0.935", 22, 62)
note("2026-09-14", "Window-MIL\nstack, 0.949", 22, -14)
note("2026-09-24", "Stack + CNN pair\nfused, 0.953", 30, -6)
note("2026-10-07", "SKM-TEA crop\nreaders, 0.954-0.958", 30, -22)
note("2026-10-14", "+ public notebooks\n0.960", 26, 10)
ax.scatter([fin.t], [fin.priv], s=46, color=ORANGE, edgecolor=SURF, linewidth=1.5, zorder=4)
ax.annotate("final pick\nprivate 0.947", (fin.t, fin.priv), xytext=(0, -34), textcoords="offset points", fontsize=7.8, color=T1, ha="center",
            arrowprops=dict(arrowstyle="-", color=T2, lw=0.7))
ax.set_ylim(0.90, 0.972); ax.set_xlim(pd.Timestamp("2026-08-18"), pd.Timestamp("2026-10-26"))
ax.set_xticks([pd.Timestamp(d) for d in ("2026-08-24", "2026-09-07", "2026-09-21", "2026-10-05", "2026-10-19")]); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
ax.set_ylabel("AUC"); ax.legend(loc="lower right", frameon=False, fontsize=8.2)
fig.savefig(W + "fig/fig3_history.png"); plt.close(fig)
print("history n", len(R), "best steps", len(Bst))
print("done")
