"""
decode_bench.py
===============
Split one inference into its two phases and measure time and CPU-package energy
for each:

  prefill  the prompt is processed in one batch (compute-bound)
  decode   a fixed number of tokens is generated one at a time, greedy, with the
           end-of-sequence token suppressed so every prompt yields exactly
           --gen-tokens tokens (memory-bandwidth-bound)

One prompt per benchmark is used (the first item of each task in --items), so the
prompt set is the same for every configuration. The text produced is not scored;
this script measures throughput and energy only.

Usage
-----
    python decode_bench.py --model MediLITE-QA.Q4_K_M.gguf --label medilite_Q4_K_M_t4 \
        --threads 4 --items items.jsonl --gen-tokens 128 --out decode.csv
"""

import argparse
import hashlib
import csv
import json
import os
import time

import numpy as np

from energy_per_item import Rapl, idle_power

FIELDS = ["config", "threads", "task", "doc_id", "prefill_tokens", "prefill_s", "prefill_j",
          "decode_tokens", "decode_s", "decode_j", "idle_w", "gen_hash", "load_verified"]


def last_logits(llm):
    ptr = llm._ctx.get_logits_ith(-1)
    return np.ctypeslib.as_array(ptr, shape=(llm.n_vocab(),)).copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--per-task", type=int, default=1,
                    help="prompts per benchmark, taken in file order (default 1: the first of each task)")
    ap.add_argument("--no-mmap", action="store_true", help="read the whole model into RAM at load time")
    ap.add_argument("--fingerprint", default=None, help="fingerprints.json: verify the load (see load_check.py)")
    ap.add_argument("--fingerprint-key", default=None, help="key in fingerprints.json, e.g. Q8_0")
    ap.add_argument("--idle-secs", type=float, default=30.0)
    ap.add_argument("--rapl", default="/sys/class/powercap/intel-rapl:0")
    ap.add_argument("--out", default="decode.csv")
    args = ap.parse_args()

    from llama_cpp import Llama

    items, count = [], {}
    for line in open(args.items):
        it = json.loads(line)
        if count.get(it["task"], 0) < args.per_task:
            count[it["task"]] = count.get(it["task"], 0) + 1
            items.append(it)

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done = {(r["config"], r["task"], str(r["doc_id"])) for r in csv.DictReader(fh)}
    new_file = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    rapl = Rapl(args.rapl)
    make = lambda: Llama(model_path=args.model, n_ctx=args.n_ctx, n_threads=args.threads, n_threads_batch=args.threads,
                         n_gpu_layers=0, logits_all=False, use_mmap=not args.no_mmap, verbose=False)
    if args.fingerprint:
        from load_check import load_verified
        all_items = [json.loads(line) for line in open(args.items)]
        llm, check = load_verified(make, args.model, args.fingerprint, args.fingerprint_key, all_items)
    else:
        llm, check = make(), {"verified": None}
    eos = llm.token_eos()

    def run(it):
        toks = llm.tokenize(it["context"].encode("utf-8"), add_bos=False, special=False)
        llm.reset()
        e0, t0 = rapl.read(), time.perf_counter()
        llm.eval(toks)
        t1, e1 = time.perf_counter(), rapl.read()
        gen = []
        for _ in range(args.gen_tokens):
            logits = last_logits(llm)
            logits[eos] = -np.inf
            gen.append(int(np.argmax(logits)))
            llm.eval([gen[-1]])
        t2, e2 = time.perf_counter(), rapl.read()
        digest = hashlib.sha256(",".join(map(str, gen)).encode()).hexdigest()[:16]
        return len(toks), t1 - t0, rapl.joules(e0, e1), t2 - t1, rapl.joules(e1, e2), digest

    for it in items[: args.warmup]:
        run(it)
    idle_w = idle_power(rapl, args.idle_secs)
    print(f"[{args.label} / decode] idle package power {idle_w:.2f} W")
    for it in items:
        if (args.label, it["task"], str(it["doc_id"])) in done:
            continue
        n, ps, pj, ds, dj, gh = run(it)
        w.writerow(dict(config=args.label, threads=args.threads, task=it["task"], doc_id=it["doc_id"],
                        prefill_tokens=n, prefill_s=f"{ps:.4f}", prefill_j=f"{pj:.4f}",
                        decode_tokens=args.gen_tokens, decode_s=f"{ds:.4f}", decode_j=f"{dj:.4f}",
                        idle_w=f"{idle_w:.3f}", gen_hash=gh, load_verified=check.get("verified")))
        fh.flush()
        print(f"  {it['task']:<28} prefill {n / ps:6.1f} tok/s  decode {args.gen_tokens / ds:5.2f} tok/s  "
              f"{dj / args.gen_tokens:.2f} J/token")
    fh.close()


if __name__ == "__main__":
    main()
