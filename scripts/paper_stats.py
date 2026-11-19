"""
paper_stats.py
==============
Recompute every number reported in the letter from the archived raw data, and
check the validity of each model load.

Usage (from the repository root):
    python scripts/paper_stats.py results/raw > results/analysis/paper_stats.txt

Inputs in results/raw/:
    sweep_items.csv     option scoring, one row per question, setting and repetition
    decode_v2.csv       prompt processing + 128-token generation, 27 prompts, verified loads
    superseded/decode_9prompts_session1.csv   first-session generation (for run-to-run)
    logs/vmstat_night*.txt, sweep_items_runs.jsonl   swap activity per setting

All intervals: 95% percentile bootstrap (10,000 resamples, seed 0) over questions or
prompts, paired between settings. They describe question-set variability only.
"""
import csv
import datetime as dt
import json
import os
import sys
from collections import Counter, defaultdict
from math import comb

import numpy as np

RAW = sys.argv[1] if len(sys.argv) > 1 else "results/raw"
rng = np.random.default_rng(0)
B = 10000
SIZE_GB = {"Q4_K_M": 2.393231616, "Q8_0": 4.061221632}
PEAK_GBS = 25.6


def boot_ratio(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, len(a), (B, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return a.mean() / b.mean(), *np.percentile(r, [2.5, 97.5])


def boot_mean(x):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (B, len(x)))
    return tuple(float(v) for v in (x.mean(), *np.percentile(x[idx].mean(1), [2.5, 97.5])))


def fmt(m, scale=100):
    return f"{scale * m[0]:.1f} [{scale * m[1]:.1f}, {scale * m[2]:.1f}]"


def pct(r):
    return f"{100 * (r[0] - 1):+.1f}% [{100 * (r[1] - 1):+.1f}, {100 * (r[2] - 1):+.1f}]"


# ------------------------------------------------------------------ option scoring
per = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(RAW, "sweep_items.csv"))):
    if r["mode"] == "loglik":
        per[(r["model"], r["rep"])][(r["task"], r["doc_id"])] = r
keys = sorted(per[("medilite_Q4_K_M_t2", "1")])
col = lambda m, rep, f: np.array([float(per[(f"medilite_{m}", rep)][k][f]) for k in keys])
print(f"questions: {len(keys)}; settings x repetitions: {sorted(per)}")

print("\n== Load validity: every accepted load of a format gives identical scores")
for q in ("Q4_K_M", "Q8_0"):
    loads = sorted(k for k in per if k[0].startswith(f"medilite_{q}_t"))
    base = loads[0]
    for L in loads[1:]:
        same = sum(per[L][k]["conf"] == per[base][k]["conf"] and per[L][k]["pred"] == per[base][k]["pred"] for k in keys)
        print(f"  {L} vs {base}: {same}/{len(keys)} identical")
inv = os.path.join(RAW, "invalid_Q8_0_t1_night1.csv")
if os.path.exists(inv):
    bad = {(r["task"], r["doc_id"]): r for r in csv.DictReader(open(inv))}
    ref = per[("medilite_Q8_0_t2", "1")]
    print(f"  rejected load (Q8_0, 1 thread, session 3): identical scores {sum(bad[k]['conf'] == ref[k]['conf'] for k in keys)}/198,"
          f" same answer {sum(bad[k]['pred'] == ref[k]['pred'] for k in keys)}/198 -> excluded")

print("\n== Table III, option scoring (first measurement)")
for q in ("Q4_K_M", "Q8_0"):
    for t in ("1", "2", "4"):
        j, s, n = col(f"{q}_t{t}", "1", "joules"), col(f"{q}_t{t}", "1", "seconds"), col(f"{q}_t{t}", "1", "prompt_tokens")
        print(f"  {q:7} {t} thr: {j.mean():6.1f} J/q  {s.mean():5.1f} s/q  {j.sum() / s.sum():4.2f} W  {n.sum() / s.sum():4.2f} prompt tok/s")

print("\n== Run-to-run, option scoring (second / first measurement)")
for q, t in (("Q4_K_M", "1"), ("Q4_K_M", "2"), ("Q8_0", "2")):
    print(f"  {q} {t} thr: {pct(boot_ratio(col(f'{q}_t{t}', '2', 'joules'), col(f'{q}_t{t}', '1', 'joules')))}")

print("\n== Q8_0 vs Q4_K_M, option scoring energy per question")
for t in ("1", "2", "4"):
    print(f"  {t} thr: {pct(boot_ratio(col(f'Q8_0_t{t}', '1', 'joules'), col(f'Q4_K_M_t{t}', '1', 'joules')))}")
print(f"  2 thr, second measurement: {pct(boot_ratio(col('Q8_0_t2', '2', 'joules'), col('Q4_K_M_t2', '2', 'joules')))}")

print("\n== Thread count, option scoring (relative to 2 threads, same format)")
for q in ("Q4_K_M", "Q8_0"):
    for t in ("1", "4"):
        e = boot_ratio(col(f"{q}_t{t}", "1", "joules"), col(f"{q}_t2", "1", "joules"))
        tm = col(f"{q}_t{t}", "1", "seconds").mean() / col(f"{q}_t2", "1", "seconds").mean()
        print(f"  {q} {t} thr: energy {pct(e)}, time {100 * (tm - 1):+.1f}%")
e = boot_ratio(col("Q4_K_M_t1", "2", "joules"), col("Q4_K_M_t2", "2", "joules"))
tm = col("Q4_K_M_t1", "2", "seconds").mean() / col("Q4_K_M_t2", "2", "seconds").mean()
print(f"  Q4_K_M 1 thr, second measurement: energy {pct(e)}, time {100 * (tm - 1):+.1f}%")

print("\n== Energy model: E = a + b * prompt tokens (OLS), first measurement")
for q in ("Q4_K_M", "Q8_0"):
    for t in ("1", "2", "4"):
        x, y = col(f"{q}_t{t}", "1", "prompt_tokens"), col(f"{q}_t{t}", "1", "joules")
        b, a = np.polyfit(x, y, 1)
        r2 = 1 - ((y - a - b * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(x), len(x))
            bs.append(np.polyfit(x[i], y[i], 1)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {q} {t} thr: b = {b:.3f} J/token [{lo:.3f}, {hi:.3f}], a = {a:.1f} J, R2 = {r2:.4f}")
print(f"  mean prompt tokens {col('Q4_K_M_t2', '1', 'prompt_tokens').mean():.0f}, "
      f"mean energy at Q4_K_M 2 thr = {col('Q4_K_M_t2', '1', 'joules').mean() / 3600:.3f} Wh")

print("\n== Answer changes, Q4_K_M vs Q8_0 (2 threads, first measurement)")
g = col("Q4_K_M_t2", "1", "gold").astype(int)
p4, p8 = col("Q4_K_M_t2", "1", "pred").astype(int), col("Q8_0_t2", "1", "pred").astype(int)
c4, c8 = p4 == g, p8 == g
b_, c_ = int((c4 & ~c8).sum()), int((~c4 & c8).sum())
n = b_ + c_
p = min(1.0, 2 * sum(comb(n, i) for i in range(min(b_, c_) + 1)) / 2 ** n)
d = c4.astype(float) - c8.astype(float)
print(f"  changed answers {int((p4 != p8).sum())}/198: only Q4 correct {b_}, only Q8 correct {c_}, both wrong {int(((p4 != p8) & ~c4 & ~c8).sum())}")
print(f"  agreement % {fmt(boot_mean(p4 == p8))}")
print(f"  accuracy % Q4 {fmt(boot_mean(c4))}, Q8 {fmt(boot_mean(c8))}")
print(f"  paired difference Q4 - Q8, percentage points {fmt(boot_mean(d))}; exact McNemar p = {p:.3f}")
print(f"  changed answers by benchmark: {dict(Counter(k[0] for k, x in zip(keys, p4 != p8) if x))}")

# ------------------------------------------------------------------ generation
dec = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(RAW, "decode_v2.csv"))):
    dec[r["config"]][(r["task"], r["doc_id"])] = r
gk = sorted(dec["medilite_Q4_K_M_t2"])
gc = lambda c, f: np.array([float(dec[c][k][f]) for k in gk])
pt = gc("medilite_Q4_K_M_t2", "prefill_tokens")
print(f"\n== Generation benchmark: {len(gk)} prompts, {pt.min():.0f}-{pt.max():.0f} prompt tokens, 128 generated tokens each")
print(f"  all loads verified: {all(r['load_verified'] == 'True' for c in dec.values() for r in c.values())}")
for q in ("Q4_K_M", "Q8_0"):
    same = sum(len({dec[f'medilite_{q}_t{t}'][k]['gen_hash'] for t in '124'}) == 1 for k in gk)
    print(f"  {q}: generated text identical at 1, 2 and 4 threads for {same}/{len(gk)} prompts")
print("\n== Table III, generation")
for q in ("Q4_K_M", "Q8_0"):
    for t in ("1", "2", "4"):
        c = f"medilite_{q}_t{t}"
        tok = 128 * len(gk)
        print(f"  {q:7} {t} thr: {tok / gc(c, 'decode_s').sum():4.2f} tok/s  {gc(c, 'decode_j').sum() / tok:4.2f} J/tok  "
              f"{gc(c, 'decode_j').sum() / gc(c, 'decode_s').sum():4.2f} W   prompt phase {gc(c, 'prefill_j').sum() / pt.sum():.3f} J/token")
print("\n== Q8_0 / Q4_K_M per prompt (mean of per-prompt ratios, bootstrap over prompts)")
allr = []
for t in ("1", "2", "4"):
    rg = gc(f"medilite_Q8_0_t{t}", "decode_j") / gc(f"medilite_Q4_K_M_t{t}", "decode_j")
    rp = gc(f"medilite_Q8_0_t{t}", "prefill_j") / gc(f"medilite_Q4_K_M_t{t}", "prefill_j")
    tr = gc(f"medilite_Q8_0_t{t}", "decode_s").sum() / gc(f"medilite_Q4_K_M_t{t}", "decode_s").sum()
    allr += list(rg)
    m = boot_mean(rg)
    print(f"  {t} thr: generation energy/token {pct(m)}; per-prompt range {rg.min():.3f}-{rg.max():.3f}; "
          f"r(ratio, prompt length) = {np.corrcoef(rg, pt)[0, 1]:.2f}; saving of 4-bit {100 * (1 - 1 / m[0]):.1f}%; "
          f"time/token ratio {tr:.3f}; prompt-phase energy/token {pct(boot_mean(rp))}")
print(f"  all prompts and thread counts: ratio {min(allr):.3f}-{max(allr):.3f}")
print("\n== Thread count, generation energy per token (relative to 2 threads)")
for q in ("Q4_K_M", "Q8_0"):
    for t in ("1", "4"):
        r = gc(f"medilite_{q}_t{t}", "decode_j").sum() / gc(f"medilite_{q}_t2", "decode_j").sum()
        print(f"  {q} {t} thr: {100 * (r - 1):+.1f}%")
print("\n== Effective memory bandwidth at 2 threads (file size x generated tokens/s)")
for q in ("Q4_K_M", "Q8_0"):
    tps = 128 * len(gk) / gc(f"medilite_{q}_t2", "decode_s").sum()
    print(f"  {q}: {tps * SIZE_GB[q]:.2f} GB/s = {100 * tps * SIZE_GB[q] / PEAK_GBS:.0f}% of {PEAK_GBS} GB/s")
print(f"  file size ratio {SIZE_GB['Q8_0'] / SIZE_GB['Q4_K_M']:.2f}")

old = os.path.join(RAW, "superseded", "decode_9prompts_session1.csv")
if os.path.exists(old):
    o = {(r["config"], r["task"]): r for r in csv.DictReader(open(old))}
    first = {}
    for k in gk:
        first.setdefault(k[0], k)
    print("\n== Generation run-to-run: same nine prompts, session 4 / session 1, 2 threads")
    for c in ("medilite_Q4_K_M_t2", "medilite_Q8_0_t2"):
        a = sum(float(dec[c][first[t]]["decode_j"]) for t in first)
        b = sum(float(o[(c, t)]["decode_j"]) for t in first)
        print(f"  {c}: {100 * (a / b - 1):+.1f}%")

# ------------------------------------------------------------------ swap
print("\n== Swap-in during measured settings (vmstat, sessions 3-4)")
runs = [json.loads(l) for l in open(os.path.join(RAW, "sweep_items_runs.jsonl"))]
for name in ("vmstat_night.txt", "vmstat_night2.txt"):
    path = os.path.join(RAW, "logs", name)
    if not os.path.exists(path):
        continue
    vm = []
    for line in open(path):
        p = line.split()
        if len(p) > 18 and p[0].isdigit():
            vm.append((dt.datetime.fromisoformat(p[-2] + "T" + p[-1]), int(p[6])))
    lo, hi = vm[0][0], vm[-1][0]
    for r in runs:
        s, e = dt.datetime.fromisoformat(r["started"][:19]), dt.datetime.fromisoformat(r["finished"][:19])
        w = [v for v in vm if s <= v[0] <= e]
        if len(w) > 10:
            print(f"  {name}: {r['label']} ({r['started'][:16]}): swap-in in {sum(v[1] > 0 for v in w)}/{len(w)} minutes")
    last = max(dt.datetime.fromisoformat(r["finished"][:19]) for r in runs)
    w = [v for v in vm if v[0] > last]
    if w:
        print(f"  {name}: generation period: swap-in in {sum(v[1] > 0 for v in w)}/{len(w)} minutes")
