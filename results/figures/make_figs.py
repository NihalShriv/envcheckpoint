"""Paper figures. Data in data/ (copied from the raw outputs in ../raw)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.4, "lines.markersize": 4,
                     "savefig.bbox": "tight", "font.family": "serif"})
C = ["#2a6fdb", "#e07b39", "#3a9e6e"]  # fixed order
W1 = 3.4

# Fig 1: environment memory vs k, square YC6, D=1024, with model env0*(1/k + k/N)
d = np.loadtxt("data/nscan.dat")
fig, ax = plt.subplots(figsize=(W1, 2.3))
for c, L in zip(C, (16, 32, 64)):
    s = d[d[:, 0] == L]; N = 6 * L; e0 = s[s[:, 1] == 0, 3][0]
    m = s[s[:, 1] > 0]
    kk = np.geomspace(3, 100, 200)
    ax.plot(kk, e0 * (1 / kk + kk / N), color=c, lw=0.8, ls="--")
    ax.plot(m[:, 1], m[:, 3], "o", color=c, label=f"$N={N}$")
    ax.axvline(np.sqrt(N), color=c, lw=0.5, alpha=0.5)
ax.set_xscale("log"); ax.set_yscale("log")
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator
ax.yaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 40, 60])); ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel("checkpoint spacing $k$"); ax.set_ylabel("environment memory (GB)")
ax.legend(frameon=False); ax.set_title("dashed: Eq. (2); vertical: $k=\\sqrt{N}$", fontsize=7)
fig.savefig("fig1_k.pdf")

# Fig 2: peak GPU memory breakdown, triangular XC6 (job 418486)
rows = [("1024\nstore", 2.10, 39.85, 9.89), ("1024\nckpt", 2.10, 6.99, 10.01),
        ("1536\nstore", 4.68, 88.80, 18.40), ("1536\nckpt", 4.68, 15.72, 18.60),
        ("2048\nckpt", 8.28, 27.95, 30.37)]
fig, ax = plt.subplots(figsize=(W1, 2.3))
x = np.arange(len(rows)); bot = np.zeros(len(rows))
for i, (lab, col) in enumerate(zip(("MPS", "environments", "working"), (C[2], C[0], "#9a9a9a"))):
    v = np.array([r[i + 1] for r in rows])
    ax.bar(x, v, 0.6, bottom=bot, color=col, label=lab, edgecolor="white", linewidth=0.8)
    bot += v
for xi, b in zip(x, bot):
    ax.text(xi, b + 2, f"{b:.0f}", ha="center", fontsize=7)
ax.set_xticks(x, [r[0] for r in rows]); ax.set_ylabel("peak GPU memory (GB)")
ax.legend(frameon=False, fontsize=7)
fig.savefig("fig2_breakdown.pdf")

# Fig 3: peak saving vs MPO width, D=1024
w = [("YC10", 1.5), ("YC8", 1.8), ("XC6", 2.7), ("XC8", 3.4)]
fig, ax = plt.subplots(figsize=(W1, 1.9))
ax.bar(range(4), [s for _, s in w], 0.6, color=C[0])
for i, (_, s) in enumerate(w):
    ax.text(i, s + 0.05, f"{s}$\\times$", ha="center", fontsize=7)
ax.set_xticks(range(4), ["square\nYC10", "square\nYC8", "triangular\nXC6", "triangular\nXC8"])
ax.set_ylabel("peak saving"); ax.set_ylim(0, 3.9)
fig.savefig("fig3_width.pdf")

# Fig 6: energy vs summed truncation error, last sweep at each D >= 1024; fit through the four largest D
fig, axs = plt.subplots(2, 1, figsize=(W1, 3.6))
for ax, f, name in zip(axs, ("sq8_conv.csv", "tri_conv.csv"),
                       ("square YC8, $J_2/J_1=0.5$", "triangular XC6, $J_2/J_1=0.12$")):
    t = pd.read_csv("data/" + f).groupby("D_cap").tail(1)
    t = t[t.D_cap >= 1024]
    e, eps = t.energy.values, t.sum_disc.values * 1e3
    p = np.polyfit(eps[-4:], e[-4:], 1)
    xx = np.linspace(0, eps.max() * 1.08, 50)
    ax.plot(xx, np.polyval(p, xx), color="#9a9a9a", lw=0.8, ls="--")
    ax.plot(eps, e, "o", color=C[0])
    ax.plot([0], [p[1]], "D", color=C[1], ms=4)
    for D, a, b in zip(t.D_cap, eps, e):
        ax.annotate(str(D), (a, b), textcoords="offset points", xytext=(6, -2), fontsize=6.5, va="top")
    ax.set_xlim(left=-0.05 * eps.max()); ax.set_ylabel("energy")
    ax.text(0.98, 0.06, name, transform=ax.transAxes, ha="right", fontsize=7.5)
    ax.set_xlim(right=eps.max() * 1.25)
    print(f, "extrapolated", p[1])
axs[1].set_xlabel("summed truncation error ($10^{-3}$)")
fig.tight_layout()
fig.savefig("fig6_extrap.pdf")

# Fig 7: ITensorMPS.jl real-time TDVP, time and GC time per step, stored vs checkpointed, two runs each (job 422828,
# whole EPYC 7542 node). Stored run 2 was stopped by its 110 min limit inside step 12 (lower bound marked).
fig, ax = plt.subplots(figsize=(W1, 2.4))
for s, col, lab in (("proj", C[0], "stored"), ("ckpt", C[1], "checkpointed")):
    for rep, mk in ((1, "o"), (2, "s")):
        t = pd.read_csv(f"data/speed_rome/tdvp_{s}_{rep}.csv")
        ax.plot(t.maxlinkdim, t.step_s, mk + "-", color=col, ms=3, lw=1, label=f"{lab}, run {rep}")
        ax.plot(t.maxlinkdim, t.gc_s, mk + ":", color=col, ms=3, lw=0.8, mfc="white")
t = pd.read_csv("data/speed_rome/tdvp_proj_2.csv")
lb = 6600 - t.step_s.sum() - 60  # time left in the 110 min limit, minus start-up
ax.annotate("", (758, lb * 1.9), (758, lb), arrowprops=dict(arrowstyle="->", color=C[0], lw=0.8))
ax.plot([758], [lb], "s", color=C[0], ms=3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(top=2e4)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel("bond dimension $D$ at that step"); ax.set_ylabel("time per step (s)")
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.text(0.98, 0.04, "solid: step time; dotted, open: GC time", transform=ax.transAxes, ha="right", fontsize=6.5)
fig.savefig("fig7_tdvp.pdf")
print("tdvp stored run 2 step 12 lower bound", lb)
