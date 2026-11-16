"""Figure 1 of the letter: (a) energy per question (option scoring) by thread count and
quantization; (b) energy per token for prompt processing vs generation at 2 threads, runs 1 and 2."""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

full, rep, out = sys.argv[1], sys.argv[2], sys.argv[3]
Q4, Q8 = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated); Q8 also hatched for grayscale
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

E = {r["config"]: r for r in csv.DictReader(open(f"{full}/energy_by_config.csv"))}
def per_tok(path, cfg_suffix=""):
    d = {}
    for r in csv.DictReader(open(path)):
        c = r["config"].replace("_r2", "")
        a = d.setdefault(c, [0, 0, 0, 0])
        a[0] += float(r["prefill_j"]); a[1] += float(r["prefill_tokens"])
        a[2] += float(r["decode_j"]);  a[3] += float(r["decode_tokens"])
    return {c: (a[0] / a[1], a[2] / a[3]) for c, a in d.items()}
r1 = per_tok(f"{rep}/sweep_items_decode.csv")
r2 = per_tok(f"{rep}/sweep_items_decode_r2.csv")

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "hatch.linewidth": 0.6,
                     "pdf.fonttype": 42})
fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.9), gridspec_kw={"width_ratios": [1.15, 1]})
w = 0.36
for ax in (a, b):
    ax.grid(axis="y", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

# (a) option-scoring energy relative to Q4_K_M at 2 threads (paired ratio, 95% bootstrap CI over questions)
a.axhline(1.0, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
x = np.arange(3)
for i, (q, col, mk, lab) in enumerate([("Q4_K_M", Q4, "o", "Q4_K_M"), ("Q8_0", Q8, "s", "Q8_0")]):
    for j, t in enumerate((1, 2, 4)):
        r = E.get(f"medilite_{q}_t{t}")
        xi = x[j] + (i - 0.5) * 0.3
        if r is None:
            a.text(xi, 1.02, "n/m", ha="center", va="bottom", fontsize=5.5, color=MUTED)
            continue
        m, lo, hi = float(r["ratio_vs_ref"]), float(r["ratio_lo"]), float(r["ratio_hi"])
        a.errorbar(xi, m, yerr=[[m - lo], [hi - m]], fmt=mk, ms=3.5, color=col, mfc=col if i == 0 else "white",
                   mec=col, mew=0.9, elinewidth=0.8, capsize=1.5, label=lab if j == 1 else None)
        a.annotate(f"{m:.2f}", xy=(xi, m), xytext=(0, 4), textcoords="offset points", ha="center",
                   fontsize=5.5, color=INK)
a.set_xticks(x, ["1", "2", "4"]); a.set_xlabel("Threads"); a.set_xlim(-0.6, 2.6)
a.set_ylabel("Energy relative to\nQ4_K_M at 2 threads")
a.set_title("(a) Option scoring", fontsize=7.5, loc="left", color=INK)
a.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.0)

# (b) energy per token, 2 threads
x = np.arange(2)
for i, (q, col, hatch) in enumerate([("Q4_K_M", Q4, None), ("Q8_0", Q8, "////")]):
    c = f"medilite_{q}_t2"
    v1, v2 = np.array(r1[c]), np.array(r2[c])
    m = (v1 + v2) / 2
    xi = x + (i - 0.5) * w
    b.bar(xi, m, w * 0.92, color=col, hatch=hatch, edgecolor="white", linewidth=0.4)
    for v in (v1, v2):
        b.plot(xi, v, ls="none", marker="o", ms=2.2, mfc="white", mec=INK, mew=0.5)
b.set_xticks(x, ["Prompt", "Generation"]); b.set_ylabel("Energy per token (J)")
b.set_ylim(0, 4.5); b.set_title("(b) 2 threads", fontsize=7.5, loc="left", color=INK)
q4g, q8g = (r1["medilite_Q4_K_M_t2"][1] + r2["medilite_Q4_K_M_t2"][1]) / 2, (r1["medilite_Q8_0_t2"][1] + r2["medilite_Q8_0_t2"][1]) / 2
b.annotate(f"+{100 * (q8g / q4g - 1):.0f}%", xy=(1 + w / 2, max(r1["medilite_Q8_0_t2"][1], r2["medilite_Q8_0_t2"][1])), xytext=(0, 5), textcoords="offset points",
           ha="center", fontsize=6.5, color=INK)
fig.tight_layout(pad=0.3, w_pad=1.0)
fig.savefig(f"{out}/fig_energy.pdf"); fig.savefig(f"{out}/fig_energy.png", dpi=300)
print({k: [round(x, 3) for x in v] for k, v in r1.items()}, {k: [round(x, 3) for x in v] for k, v in r2.items()})
