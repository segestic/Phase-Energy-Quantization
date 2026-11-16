"""
sweep_energy.py
===============
Measure option-scoring energy and time per question for every combination of
quantization level and CPU thread count, on one laptop (CPU only, Intel RAPL).

For each configuration it runs energy_per_item.py (mode loglik, one repetition)
and, with --decode, decode_bench.py (prefill/decode split). Configurations run in
a seeded random order, a new order for each pass, so slow drifts (temperature,
background activity) do not line up with one quantization level or thread count.
Before each configuration the script waits until the CPU package has cooled to
--max-start-temp and records the system state (governor, frequency, temperature,
mains power, free memory) to <out>_system.jsonl.

Labels written to the CSV: medilite_<QUANT>_t<THREADS>, e.g. medilite_Q4_K_M_t2.
Interrupted runs resume: finished (config, pass, question) rows are skipped.

Usage
-----
    # pilot: 5 questions per configuration, short idle and cooldown (prints a time projection)
    python sweep_energy.py --models Q4_K_M=MediLITE-QA.Q4_K_M.gguf Q5_K_M=MediLITE-QA.Q5_K_M.gguf \
        Q8_0=MediLITE-QA.Q8_0.gguf --threads 1 2 4 --pilot 5 --out pilot.csv

    # full run
    python sweep_energy.py --models Q4_K_M=... Q5_K_M=... Q8_0=... --threads 1 2 4 \
        --passes 1 --decode --out sweep_items.csv

RAPL must be readable:  sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj
"""

import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def read(path, default=None):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return default


def pkg_temp_c():
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        if read(os.path.join(z, "type")) == "x86_pkg_temp":
            return int(read(os.path.join(z, "temp"), "0")) / 1000
    temps = [int(read(f, "0")) / 1000 for f in glob.glob("/sys/class/hwmon/hwmon*/temp1_input")]
    return max(temps) if temps else None


def meminfo(key):
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 2**20
    return None


def mem_available_gib():
    return meminfo("MemAvailable")


def storage_of(path):
    """Device, filesystem type and whether the disk is rotational, for the methods section."""
    try:
        out = subprocess.run(["df", "-T", path], capture_output=True, text=True).stdout.splitlines()[-1].split()
        dev, fstype = out[0], out[1]
        base = re.sub(r"p?\d+$", "", os.path.basename(dev))
        rot = read(f"/sys/block/{base}/queue/rotational")
        removable = read(f"/sys/block/{base}/removable")
        return {"device": dev, "fstype": fstype, "rotational": rot, "removable": removable}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def system_state():
    cpus = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq"))
    mains = [read(os.path.join(p, "online")) for p in glob.glob("/sys/class/power_supply/*")
             if read(os.path.join(p, "type")) == "Mains"]
    return {
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "epp": read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
        "no_turbo": read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "cur_mhz": [int(read(os.path.join(c, "scaling_cur_freq"), "0")) // 1000 for c in cpus],
        "pkg_temp_c": pkg_temp_c(),
        "mains_online": mains,
        "mem_available_gib": round(mem_available_gib() or 0, 2),
        "swap_used_gib": round((meminfo("SwapTotal") or 0) - (meminfo("SwapFree") or 0), 2),
        "loadavg": read("/proc/loadavg"),
    }


def wait_cool(limit_c, max_wait):
    t0 = time.time()
    while True:
        t = pkg_temp_c()
        if t is None or t <= limit_c or time.time() - t0 > max_wait:
            return t
        time.sleep(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="QUANT=path.gguf, e.g. Q4_K_M=MediLITE-QA.Q4_K_M.gguf")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--items", default=os.path.join(HERE, "items.jsonl"))
    ap.add_argument("--skip", nargs="*", default=[], help="configurations to leave out, e.g. Q8_0_t1")
    ap.add_argument("--passes", type=int, default=1, help="repetitions, each a full pass in a new random order")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pilot", type=int, default=None, help="only the first N questions, short idle/cooldown")
    ap.add_argument("--decode", action="store_true", help="also run decode_bench.py for every configuration")
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--idle-secs", type=float, default=30.0)
    ap.add_argument("--cooldown", type=float, default=60.0)
    ap.add_argument("--max-start-temp", type=float, default=50.0, help="degrees C; wait until cooler (max 10 min)")
    ap.add_argument("--mmap", action="store_true",
                    help="memory-map the model (llama.cpp default). Off by default here: the model is read "
                         "fully into RAM at load time, so timed inference never reads from the drive")
    ap.add_argument("--out", default="sweep_items.csv")
    args = ap.parse_args()

    models = dict(m.split("=", 1) for m in args.models)
    for q, p in models.items():
        if not os.path.exists(p):
            raise SystemExit(f"missing model file for {q}: {p}")
    configs = [(q, t) for q in models for t in args.threads if f"{q}_t{t}" not in args.skip]
    idle = 5.0 if args.pilot else args.idle_secs
    cool = 10.0 if args.pilot else args.cooldown
    sys_log = os.path.splitext(args.out)[0] + "_system.jsonl"
    decode_out = os.path.splitext(args.out)[0] + "_decode.csv"

    print("system:", json.dumps(system_state()))
    with open(sys_log, "a") as fh:
        fh.write(json.dumps({"phase": "setup", "use_mmap": args.mmap,
                             "storage": {q: storage_of(p) for q, p in models.items()},
                             "mem_total_gib": round(meminfo("MemTotal") or 0, 2)}) + "\n")
    timings = []
    for p in range(1, args.passes + 1):
        order = configs[:]
        random.Random(args.seed + p).shuffle(order)
        print(f"pass {p} order: {[f'{q}/t{t}' for q, t in order]}")
        for q, t in order:
            label = f"medilite_{q}_t{t}"
            size_gib = os.path.getsize(models[q]) / 2**30
            free = mem_available_gib()
            if free is not None and free < size_gib + 0.8:
                print(f"[skip] {label}: model {size_gib:.1f} GiB, only {free:.1f} GiB available")
                continue
            temp = wait_cool(args.max_start_temp, 600)
            state = dict(system_state(), label=label, quant=q, threads=t, pass_=p, phase="start",
                         start_temp_c=temp, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            with open(sys_log, "a") as fh:
                fh.write(json.dumps(state) + "\n")
            cmd = [sys.executable, os.path.join(HERE, "energy_per_item.py"), "--model", models[q],
                   "--label", label, "--items", args.items, "--modes", "loglik", "--reps", "1",
                   "--rep-start", str(p), "--threads", str(t), "--n-ctx", str(args.n_ctx),
                   "--idle-secs", str(idle), "--cooldown", "0", "--out", args.out]
            if not args.mmap:
                cmd += ["--no-mmap"]
            if q == "Q4_K_M":
                cmd += ["--ref", "medilite_q4"]
            if args.pilot:
                cmd += ["--limit", str(args.pilot)]
            t0 = time.time()
            subprocess.run(cmd, check=True)
            secs = time.time() - t0
            timings.append((label, secs))
            if args.decode and p == 1:
                time.sleep(cool)
                subprocess.run([sys.executable, os.path.join(HERE, "decode_bench.py"), "--model", models[q],
                                "--label", label, "--threads", str(t), "--items", args.items,
                                "--gen-tokens", str(args.gen_tokens if not args.pilot else 16),
                                "--n-ctx", str(args.n_ctx), "--idle-secs", str(idle), "--out", decode_out]
                               + ([] if args.mmap else ["--no-mmap"]),
                               check=True)
            with open(sys_log, "a") as fh:
                fh.write(json.dumps(dict(system_state(), label=label, pass_=p, phase="end",
                                         seconds=round(time.time() - t0, 1),
                                         time=time.strftime("%Y-%m-%dT%H:%M:%S"))) + "\n")
            time.sleep(cool)

    if args.pilot:
        n_items = sum(1 for _ in open(args.items))
        print(f"\nprojection for the full set ({n_items} questions, per pass; model load and idle included):")
        total = 0.0
        for label, secs in timings:
            est = secs / args.pilot * n_items
            total += est
            print(f"  {label:<26}{est / 3600:6.2f} h")
        print(f"  total per pass (loglik only, excluding decode and cooldowns) ~ {total / 3600:.1f} h")


if __name__ == "__main__":
    main()
