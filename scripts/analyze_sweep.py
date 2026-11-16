"""
analyze_sweep.py
================
Tables for the quantization x thread-count sweep.

Energy and time (laptop, the 198-question set):
  per configuration: mean gross and net joules per question, seconds per question,
  prompt tokens per second, with 95% bootstrap intervals (questions resampled);
  paired ratio of joules per question against a reference configuration.
  Repetitions of the same question are averaged first.

Energy vs prompt length: per-configuration linear fit, joules = a + b * prompt tokens.

Answers on the laptop (no extra runs): answer fidelity of each quant against the
--fidelity-ref quant (share of questions with the same answer), accuracy on the
laptop set with its CI, and whether answers change with the thread count.

Optional accuracy (full benchmarks, from GGUF scoring logs, e.g. Kaggle):
  --acc QUANT=log_folder for each quantization level. Micro-averaged accuracy over
  all questions, per-benchmark accuracy, and the paired difference against the
  reference quantization (bootstrap over questions). Laptop predictions on the
  198 questions are also compared with the logged predictions of the same file
  (platform agreement).

Pareto set: configurations not beaten on both full-benchmark accuracy and energy.

Usage
-----
    python analyze_sweep.py --csv sweep_items.csv --decode sweep_items_decode.csv \
        --acc Q4_K_M=logs/medilite_q4 Q5_K_M=logs/medilite_q5 Q8_0=logs/medilite_q8 \
        --ref-config medilite_Q4_K_M_t4 --out sweep_out
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def task_of(path):
    name = os.path.basename(path)[len("samples_"):-len(".jsonl")]
    return re.sub(r"_\d{4}-\d{2}-\d{2}T.*$", "", name)


def latest_files(d):
    best = {}
    for f in glob.glob(os.path.join(d, "**", "samples_*.jsonl"), recursive=True):
        t = task_of(f)
        if t not in best or os.path.basename(f) > os.path.basename(best[t]):
            best[t] = f
    return best


def first_number(x):
    while isinstance(x, (list, tuple)):
        x = x[0]
    return float(x)


def load_logs(d):
    """(task, doc_id) -> (correct, pred)"""
    out = {}
    for task, f in latest_files(d).items():
        for line in open(f):
            r = json.loads(line)
            ll = [first_number(x) for x in r["resps"]]
            out[(task, r["doc_id"])] = (float(r["acc"]), int(np.argmax(ll)))
    if not out:
        raise SystemExit(f"no samples_*.jsonl under {d}")
    return out


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ratio(a, b, n=5000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return float(a.mean() / b.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def parse_label(label):
    m = re.match(r"medilite_(.+)_t(\d+)$", label)
    return (m.group(1), int(m.group(2))) if m else (label, None)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--decode", default=None)
    ap.add_argument("--acc", nargs="*", default=[], help="QUANT=folder of GGUF scoring logs")
    ap.add_argument("--ref-config", default="medilite_Q4_K_M_t4")
    ap.add_argument("--ref-quant", default="Q4_K_M", help="reference quant for the optional --acc logs")
    ap.add_argument("--fidelity-ref", default="Q8_0", help="quant whose laptop answers are the fidelity reference")
    ap.add_argument("--rep", default=None, help="use only this repetition (e.g. 1); default: average all")
    ap.add_argument("--out", default="sweep_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = []
    say = lambda s="": (print(s), report.append(s))

    # ---------------------------------------------------------- energy per question
    per = defaultdict(lambda: defaultdict(list))  # label -> (task, doc) -> rows
    for r in csv.DictReader(open(args.csv)):
        if r["mode"] == "loglik" and (args.rep is None or r["rep"] == args.rep):
            per[r["model"]][(r["task"], int(r["doc_id"]))].append(r)
    if args.ref_config not in per:
        raise SystemExit(f"reference {args.ref_config} not in {sorted(per)}")
    common = set.intersection(*[set(v) for v in per.values()])
    keys = sorted(common)
    say(f"questions measured in every configuration: {len(keys)}")

    def avg(label, field):
        return np.array([np.mean([float(x[field]) for x in per[label][k]]) for k in keys])

    ref_j = avg(args.ref_config, "joules")
    energy_rows = []
    say("\nEnergy and time per question (laptop; mean, 95% bootstrap CI over questions)")
    say(f"{'config':<24}{'J gross':>9}{'CI':>17}{'J net':>8}{'s':>8}{'tok/s':>8}{'reps':>5}"
        f"{'ratio vs ref':>14}{'CI':>16}")
    for label in sorted(per, key=lambda l: (parse_label(l)[0], parse_label(l)[1] or 0)):
        j, jn, s = avg(label, "joules"), avg(label, "joules_net"), avg(label, "seconds")
        toks = avg(label, "prompt_tokens")
        lo, hi = boot_ci(j)
        ratio, rlo, rhi = boot_ratio(j, ref_j)
        reps = int(np.mean([len(per[label][k]) for k in keys]))
        q, t = parse_label(label)
        energy_rows.append(dict(config=label, quant=q, threads=t, n=len(keys), joules=j.mean(), joules_lo=lo,
                                joules_hi=hi, joules_net=jn.mean(), seconds=s.mean(),
                                prompt_tok_per_s=toks.sum() / s.sum(), mean_watts=j.sum() / s.sum(),
                                ratio_vs_ref=ratio, ratio_lo=rlo, ratio_hi=rhi, reps=reps))
        say(f"{label:<24}{j.mean():9.1f}  [{lo:6.1f}, {hi:6.1f}]{jn.mean():8.1f}{s.mean():8.2f}"
            f"{toks.sum() / s.sum():8.1f}{reps:5d}{ratio:14.3f}  [{rlo:5.3f}, {rhi:5.3f}]")
    write_csv(os.path.join(args.out, "energy_by_config.csv"), energy_rows)

    # ---------------------------------------------------------- energy vs prompt length
    say("\nEnergy vs prompt length, per configuration: J = a + b * prompt_tokens (OLS; 95% bootstrap CI for b)")
    say(f"{'config':<24}{'a (J)':>8}{'b (J/token)':>13}{'CI':>18}{'R2':>7}")
    fit_rows = []
    rng = np.random.default_rng(0)
    for label in sorted(per, key=lambda l: (parse_label(l)[0], parse_label(l)[1] or 0)):
        x, y = avg(label, "prompt_tokens"), avg(label, "joules")
        b, a = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - (a + b * x)) ** 2) / np.sum((y - y.mean()) ** 2)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(x), len(x))
            bs.append(np.polyfit(x[i], y[i], 1)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        fit_rows.append(dict(config=label, intercept_j=a, j_per_token=b, b_lo=lo, b_hi=hi, r2=r2))
        say(f"{label:<24}{a:8.2f}{b:13.4f}  [{lo:7.4f}, {hi:7.4f}]{r2:7.3f}")
    write_csv(os.path.join(args.out, "energy_vs_length.csv"), fit_rows)

    # ---------------------------------------------------------- answers on the laptop
    pred = {l: np.array([int(per[l][k][0]["pred"]) for k in keys]) for l in per}
    gold = np.array([int(per[args.ref_config][k][0]["gold"]) for k in keys])
    by_q = defaultdict(list)
    for l in per:
        by_q[parse_label(l)[0]].append(l)
    say(f"\nSame-question answers across thread counts (determinism check)")
    for q, ls in sorted(by_q.items()):
        ls = sorted(ls, key=lambda l: parse_label(l)[1] or 0)
        same = np.mean([len({int(pred[l][i]) for l in ls}) == 1 for i in range(len(keys))])
        say(f"  {q:<8} identical answer in all of {len(ls)} thread settings on {100 * same:.1f}% of questions")
    fid_ref = args.fidelity_ref
    rows = []
    say(f"\nAnswer fidelity vs {fid_ref} and accuracy on the {len(keys)} laptop questions "
        f"(first thread setting of each quant; 95% bootstrap CI)")
    ref_labels = sorted(by_q.get(fid_ref, []), key=lambda l: parse_label(l)[1] or 0)
    for q, ls in sorted(by_q.items()):
        l = sorted(ls, key=lambda l: parse_label(l)[1] or 0)[0]
        c = (pred[l] == gold).astype(float)
        lo, hi = boot_ci(c)
        row = dict(quant=q, config=l, n=len(keys), acc=100 * c.mean(), acc_lo=100 * lo, acc_hi=100 * hi)
        line = f"  {q:<8} acc {100 * c.mean():5.1f} [{100 * lo:5.1f}, {100 * hi:5.1f}]"
        if ref_labels and q != fid_ref:
            ag = (pred[l] == pred[ref_labels[0]]).astype(float)
            alo, ahi = boot_ci(ag)
            row.update(agree_vs_ref=100 * ag.mean(), agree_lo=100 * alo, agree_hi=100 * ahi,
                       flips=int(len(ag) - ag.sum()))
            line += f"   same answer as {fid_ref}: {100 * ag.mean():5.1f}% [{100 * alo:5.1f}, {100 * ahi:5.1f}]"
        rows.append(row)
        say(line)
    say("  Accuracy differences between quants on this set are not resolved (CI width ~13 points);\n"
        "  report fidelity (agreement with the reference) as the quality measure.")
    write_csv(os.path.join(args.out, "fidelity_by_quant.csv"), rows)

    # ---------------------------------------------------------- prefill / decode
    if args.decode and os.path.exists(args.decode):
        d = defaultdict(list)
        for r in csv.DictReader(open(args.decode)):
            d[r["config"]].append(r)
        rows = []
        say("\nPrefill and decode (one prompt per benchmark)")
        say(f"{'config':<24}{'prefill tok/s':>14}{'J/prompt tok':>14}{'decode tok/s':>14}{'J/gen tok':>11}"
            f"{'decode W':>10}")
        for label in sorted(d, key=lambda l: (parse_label(l)[0], parse_label(l)[1] or 0)):
            rs = d[label]
            f = lambda k: sum(float(x[k]) for x in rs)
            row = dict(config=label, prompts=len(rs),
                       prefill_tok_s=f("prefill_tokens") / f("prefill_s"),
                       prefill_j_per_tok=f("prefill_j") / f("prefill_tokens"),
                       decode_tok_s=f("decode_tokens") / f("decode_s"),
                       decode_j_per_tok=f("decode_j") / f("decode_tokens"),
                       decode_watts=f("decode_j") / f("decode_s"))
            rows.append(row)
            say(f"{label:<24}{row['prefill_tok_s']:14.1f}{row['prefill_j_per_tok']:14.3f}"
                f"{row['decode_tok_s']:14.2f}{row['decode_j_per_tok']:11.3f}{row['decode_watts']:10.2f}")
        write_csv(os.path.join(args.out, "prefill_decode.csv"), rows)

    # ---------------------------------------------------------- accuracy
    acc = {}
    if args.acc:
        logs = {q: load_logs(p) for q, p in (a.split("=", 1) for a in args.acc)}
        ref = logs.get(args.ref_quant)
        say("\nFull-benchmark accuracy (GGUF scoring logs)")
        acc_rows = []
        for q, lg in logs.items():
            ks = sorted(lg)
            a = np.array([lg[k][0] for k in ks])
            lo, hi = boot_ci(a)
            row = dict(quant=q, n=len(ks), acc=100 * a.mean(), acc_lo=100 * lo, acc_hi=100 * hi)
            line = f"  {q:<8} n={len(ks):5d}  acc {100 * a.mean():6.2f}  [{100 * lo:6.2f}, {100 * hi:6.2f}]"
            if ref is not None and q != args.ref_quant:
                shared = sorted(set(lg) & set(ref))
                diff = np.array([lg[k][0] - ref[k][0] for k in shared])
                dlo, dhi = boot_ci(diff)
                same = np.mean([lg[k][1] == ref[k][1] for k in shared])
                row.update(diff_vs_ref=100 * diff.mean(), diff_lo=100 * dlo, diff_hi=100 * dhi,
                           same_pred_vs_ref=100 * same)
                line += (f"   vs {args.ref_quant}: {100 * diff.mean():+5.2f} [{100 * dlo:+5.2f}, {100 * dhi:+5.2f}]"
                         f", same answer on {100 * same:.1f}%")
            by_task = defaultdict(list)
            for k in ks:
                by_task[k[0]].append(lg[k][0])
            row.update({f"acc_{t}": 100 * np.mean(v) for t, v in sorted(by_task.items())})
            acc_rows.append(row)
            acc[q] = 100 * a.mean()
            say(line)
            # laptop vs logged predictions, same file, same 198 questions
            for label in per:
                lq, lt = parse_label(label)
                if lq != q:
                    continue
                agree = [int(per[label][k][0]["pred"]) == lg[k][1] for k in keys if k in lg]
                if agree:
                    say(f"      laptop {label}: same answer as the log on {100 * np.mean(agree):.1f}% "
                        f"of {len(agree)} questions")
        write_csv(os.path.join(args.out, "accuracy_by_quant.csv"), acc_rows)

        # ------------------------------------------------------ Pareto set
        pts = [(r["config"], acc[r["quant"]], r["joules"]) for r in energy_rows if r["quant"] in acc]
        pareto = [p for p in pts if not any((o[1] >= p[1] and o[2] < p[2]) or (o[1] > p[1] and o[2] <= p[2])
                                             for o in pts)]
        say("\nPareto set (higher full-benchmark accuracy, lower joules per question):")
        for c, a, j in sorted(pareto, key=lambda p: p[2]):
            say(f"  {c:<24} acc {a:6.2f}   {j:7.1f} J")
        say("  Note: accuracy differences inside the bootstrap intervals above are not resolved;\n"
            "  do not rank configurations on them.")

    open(os.path.join(args.out, "REPORT.txt"), "w").write("\n".join(report) + "\n")
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
