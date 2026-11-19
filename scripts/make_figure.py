"""
make_figure.py
==============
Fig. 1 of the letter, drawn directly from the raw measurement files.
  (a) option-scoring energy per question relative to Q4_K_M at 2 threads (run 1),
      every format x thread setting, paired ratios with 95% bootstrap intervals over questions;
  (b) per-prompt ratio of generation energy per token, Q8_0 / Q4_K_M, against prompt length,
      27 prompts at 1, 2 and 4 threads (verified loads).
Usage: python make_figure.py results/raw/sweep_items.csv results/raw/decode_v2.csv figures
"""
import csv
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

items_csv, decode_csv, out = sys.argv[1], sys.argv[2], sys.argv[3]
Q4, Q8 = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
TH = {"1": ("o", "#7a7a7a"), "2": ("s", INK), "4": ("^", "#b0b0b0")}

per = defaultdict(dict)
for r in csv.DictReader(open(items_csv)):
    if r["mode"] == "loglik" and r["rep"] == "1":
        per[r["model"]][(r["task"], r["doc_id"])] = float(r["joules"])
keys = sorted(per["medilite_Q4_K_M_t2"])
ref = np.array([per["medilite_Q4_K_M_t2"][k] for k in keys])
rng = np.random.default_rng(0)
idx = rng.integers(0, len(keys), (5000, len(keys)))

dec = defaultdict(dict)
for r in csv.DictReader(open(decode_csv)):
    dec[r["config"]][(r["task"], r["doc_id"])] = r

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "pdf.fonttype": 42})
fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.8), gridspec_kw={"width_ratios": [1, 1.1]})
for ax in (a, b):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

# (a)
a.axhline(1.0, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
for i, (q, col, mk) in enumerate([("Q4_K_M", Q4, "o"), ("Q8_0", Q8, "s")]):
    for j, t in enumerate(("1", "2", "4")):
        v = np.array([per[f"medilite_{q}_t{t}"][k] for k in keys])
        m = v.mean() / ref.mean()
        bs = v[idx].mean(1) / ref[idx].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        x = j + (i - 0.5) * 0.32
        a.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt=mk, ms=3.3, color=col, mfc=col if i == 0 else "white",
                   mec=col, mew=0.9, elinewidth=0.8, capsize=1.5, label=q if j == 0 else None)
        a.annotate(f"{m:.2f}", xy=(x, m), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=5.3, color=INK)
a.set_xticks([0, 1, 2], ["1", "2", "4"]); a.set_xlim(-0.6, 2.6); a.set_ylim(0.9, 1.45)
a.set_xlabel("Threads"); a.set_ylabel("Energy per question\nrel. to Q4_K_M, 2 threads")
a.set_title("(a) Option scoring", fontsize=7.5, loc="left", color=INK)
a.legend(frameon=False, fontsize=6, loc="center", handlelength=1.0)

# (b)
for t, (mk, col) in TH.items():
    q4, q8 = dec[f"medilite_Q4_K_M_t{t}"], dec[f"medilite_Q8_0_t{t}"]
    ks = sorted(q4)
    x = np.array([int(q4[k]["prefill_tokens"]) for k in ks])
    y = np.array([float(q8[k]["decode_j"]) / float(q4[k]["decode_j"]) for k in ks])
    b.scatter(x, y, s=7, marker=mk, facecolors=col if t != "4" else "white", edgecolors=col if t != "4" else MUTED,
              linewidths=0.5, label=f"{t} thr.", zorder=3)
b.set_xscale("log"); b.set_xlim(30, 1600); b.set_ylim(1.2, 1.5)
b.set_xlabel("Prompt tokens"); b.set_ylabel("Q8_0 / Q4_K_M energy\nper generated token")
b.set_title("(b) Generation", fontsize=7.5, loc="left", color=INK)
b.legend(frameon=False, fontsize=5.8, loc="lower left", handlelength=0.8, ncol=3, columnspacing=0.6, handletextpad=0.2)
fig.tight_layout(pad=0.3, w_pad=0.9)
fig.savefig(f"{out}/fig_energy.pdf")
fig.savefig(f"{out}/fig_energy.png", dpi=300)
print("written", out)
