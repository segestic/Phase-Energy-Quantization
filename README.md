# Phase-Energy-Quantization

Prompt-processing vs. generation energy of a 4-bit and 8-bit medical language model on a two-core laptop CPU.

Code and data for the letter

> O. O. Odewole and S. Başaran, "When Does 4-Bit Quantization Save Energy? Prompt Processing Versus Generation of a Medical Language Model on a Two-Core Laptop CPU," submitted to *IEEE Embedded Systems Letters*.

The study measures CPU-package energy (Intel RAPL) of MediLITE-QA, a fine-tuned Phi-3.5-mini-instruct model (3.8B parameters), in its published 4-bit (Q4_K_M) and 8-bit (Q8_0) GGUF files at 1, 2 and 4 threads on an Intel Core i3-4005U laptop. Energy is measured separately for option scoring (multiple-choice answering, dominated by prompt processing) and for generation. The model is described in O. O. Odewole and S. Başaran, *Sustainability* 18(21), 11245, 2026, https://doi.org/10.3390/su182111245.

## Model files

| File | SHA-256 | Bytes |
|---|---|---|
| MediLITE-QA.Q4_K_M.gguf | `df4a535a65c1822c132dafd6f6a5b2afbbded09020bbcd70dc8919d41ae2ac25` | 2,393,231,616 |
| MediLITE-QA.Q8_0.gguf | `b25d1840ee7681b7c732f209e2af51b217009c01596ed54cb9823b13b68d78ad` | 4,061,221,632 |

Both are the files published at https://huggingface.co/segestic/MediLITE-QA-GGUF (uploaded 2026-08-29, unchanged since). `scripts/verify_gguf_sha.py` compares local copies with the published hashes.

## Measurement sessions and load verification

| Session | Date (2026) | Measured |
|---|---|---|
| 1 | 15–16 Nov | option scoring: Q4_K_M at 1, 2, 4 threads; Q8_0 at 2, 4 threads (superseded generation runs on 9 prompts) |
| 2 | 16 Nov | repeat: option scoring, both formats at 2 threads |
| 3 | 18 Nov (night) | repeat: option scoring, Q4_K_M at 1 thread; Q8_0 at 1 thread **rejected** (see below) |
| 4 | 18–19 Nov (night) | option scoring, Q8_0 at 1 thread; generation on 27 prompts, both formats at 1, 2, 4 threads |

The external USB disk holding the model files developed read errors during the study and sometimes returned corrupted data without reporting an error. Every model load was therefore checked:

* All accepted loads of a format give **identical** scores (predicted option and confidence) on all 198 questions, across thread counts and sessions (`scripts/paper_stats.py`, section "Load validity").
* One load (Q8_0, 1 thread, session 3) did not (0/198 identical scores); its rows are kept for the record in `results/raw/invalid_Q8_0_t1_night1.csv` and the setting was remeasured in session 4.
* In session 4, `scripts/load_check.py` scored five reference questions (`scripts/fingerprints.json`) right after every load and repeated any load that did not reproduce them; two loads were rejected and repeated (`results/raw/logs/night2_console.txt`).
* Generation produced identical text (hash of the 128 token ids, column `gen_hash`) at 1, 2 and 4 threads for all 27 prompts of each format.

## Repository structure

```
.
├── README.md, LICENSE, requirements.txt
├── data/items_les.jsonl               # 198 questions (22 per benchmark, seed 2027), lm-eval prompts and options
├── scripts/
│   ├── export_items.py                # draws the question set from lm-evaluation-harness per-item logs
│   ├── verify_gguf_sha.py             # local GGUF files vs. published SHA-256
│   ├── check_gguf.py                  # quick sanity check of a GGUF file
│   ├── load_check.py, fingerprints.json   # verified model loading (session 4)
│   ├── energy_per_item.py             # option scoring: answer, time and RAPL energy per question
│   ├── decode_bench.py                # prompt processing and 128-token generation, timed and metered separately
│   ├── sweep_energy.py                # runs settings, waits for cool-down, logs system state
│   ├── analyze_sweep.py               # per-setting tables
│   ├── paper_stats.py                 # every number in the letter, plus load and swap checks
│   └── make_figure.py                 # Fig. 1
├── results/
│   ├── raw/
│   │   ├── sweep_items.csv            # option scoring: one row per question, setting and repetition (rep 1, rep 2)
│   │   ├── decode_v2.csv              # generation benchmark, 27 prompts x 6 settings, verified loads
│   │   ├── invalid_Q8_0_t1_night1.csv # rejected load (excluded)
│   │   ├── sweep_items_runs.jsonl     # software versions, load checks, idle power, start/end time per setting
│   │   ├── sweep_items_system.jsonl   # governor, frequency, temperature, mains, memory, swap, storage
│   │   ├── gguf_sha256.txt
│   │   ├── logs/                      # console and vmstat logs of all sessions
│   │   └── superseded/                # earlier generation runs (9 prompts; unverified loads), not used in the letter
│   ├── pilot/                         # five-question pilot (not used in the letter)
│   └── analysis/
│       ├── paper_stats.txt            # output of paper_stats.py
│       └── run1/                      # output of analyze_sweep.py (first measurement of each setting)
└── figures/fig_energy.pdf / .png      # Fig. 1
```

## Hardware and conditions

| Item | Setting |
|---|---|
| Laptop | HP 250 G4 |
| CPU | Intel Core i3-4005U (Haswell), 2 cores / 4 threads, 1.6 GHz, no turbo |
| Memory | 6 GB DDR3-1600 (4 GB + 2 GB) |
| OS | Ubuntu 22.04.5, kernel 6.8, `schedutil` governor |
| Runtime | llama-cpp-python 0.3.35, CPU only, context 2048, weights loaded fully into RAM (`--no-mmap`) |
| Energy | RAPL package domain (`/sys/class/powercap/intel-rapl:0`) |
| Model storage | external USB hard disk (read errors, see above); OS on internal SATA SSD |

Measurements were taken on mains power with Wi-Fi off and the laptop not in use (sessions 3–4: desktop stopped). Before each setting the script waits for the package to cool to 50 °C, runs three warm-up questions and measures idle power for 30 s.

## Reproducing the numbers and figure from the archived data

```bash
pip install numpy matplotlib
python scripts/paper_stats.py results/raw > results/analysis/paper_stats.txt
python scripts/analyze_sweep.py --csv results/raw/sweep_items.csv --decode results/raw/decode_v2.csv \
  --rep 1 --ref-config medilite_Q4_K_M_t2 --fidelity-ref Q8_0 --out results/analysis/run1
python scripts/make_figure.py results/raw/sweep_items.csv results/raw/decode_v2.csv figures
```

| Letter item | Source |
|---|---|
| Table III, option scoring | `paper_stats.txt` "Table III, option scoring" |
| Table III, generation | `paper_stats.txt` "Table III, generation" |
| Run-to-run variation | `paper_stats.txt` "Run-to-run" sections |
| Quantization comparison, Fig. 1 | `paper_stats.txt` "Q8_0 vs Q4_K_M" and "per prompt"; `make_figure.py` |
| Answer changes, McNemar test | `paper_stats.txt` "Answer changes" |
| Thread count | `paper_stats.txt` "Thread count" sections |
| Energy model | `paper_stats.txt` "Energy model" |
| Load verification, swap | `paper_stats.txt` "Load validity", "Swap-in" |

Intervals are 95% percentile bootstrap intervals (10,000 resamples) over questions or prompts, paired between settings. They describe question-set variability only; run-to-run variation is reported separately from the repeated settings.

## Reproducing the measurements

```bash
pip install -r requirements.txt
sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj      # once per boot
python scripts/verify_gguf_sha.py MediLITE-QA.Q4_K_M.gguf MediLITE-QA.Q8_0.gguf

# option scoring, all settings (verified loads use scripts/fingerprints.json automatically)
python scripts/sweep_energy.py --items data/items_les.jsonl \
  --models Q4_K_M=MediLITE-QA.Q4_K_M.gguf Q8_0=MediLITE-QA.Q8_0.gguf --threads 1 2 4 --out sweep_items.csv

# generation, 27 prompts, one setting at a time
python scripts/decode_bench.py --model MediLITE-QA.Q4_K_M.gguf --label medilite_Q4_K_M_t2 --threads 2 \
  --per-task 3 --items data/items_les.jsonl --no-mmap \
  --fingerprint scripts/fingerprints.json --fingerprint-key Q4_K_M --out decode_v2.csv
```

Interrupted runs resume from the last finished question. The Q8_0 model needs about 4.6 GiB of free memory.

## Data licences

`data/items_les.jsonl` contains questions from MedMCQA, MedQA, PubMedQA and MMLU in the lm-evaluation-harness prompt format; they remain under the licences of those datasets. Code is released under the MIT License.
