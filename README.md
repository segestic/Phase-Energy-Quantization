# Prompt-processing vs. generation energy of a 4-bit and 8-bit medical language model on a two-core laptop CPU

Code and data for the letter

> O. O. Odewole and S. Başaran, "When Does 4-Bit Quantization Save Energy? Prompt Processing Versus Generation of a Medical Language Model on a Two-Core Laptop CPU," submitted to *IEEE Embedded Systems Letters*.

The study measures CPU-package energy (Intel RAPL) of MediLITE-QA, a fine-tuned Phi-3.5-mini-instruct model (3.8B parameters), in 4-bit (Q4_K_M) and 8-bit (Q8_0) GGUF formats at 1, 2 and 4 threads on an Intel Core i3-4005U laptop. Energy is measured separately for option scoring (multiple-choice answering, dominated by prompt processing) and for generation. The model itself is described in O. O. Odewole and S. Başaran, *Sustainability* 18(21), 11245, 2026, https://doi.org/10.3390/su182111245.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   └── items_les.jsonl            # 198 questions (22 per benchmark, seed 2027) with lm-eval prompts and options
├── scripts/
│   ├── export_items.py            # draws the question set from lm-evaluation-harness per-item logs
│   ├── check_gguf.py              # sanity check of a GGUF file before measuring
│   ├── energy_per_item.py         # option scoring: answer, time and RAPL energy per question
│   ├── decode_bench.py            # prompt processing and 128-token generation, timed and metered separately
│   ├── sweep_energy.py            # runs every format x thread setting in a seeded random order, logs system state
│   ├── analyze_sweep.py           # tables: energy, ratios, energy-vs-length fits, fidelity, thread checks
│   └── make_figure.py             # Fig. 1
├── results/
│   ├── raw/                       # measurements behind the letter
│   │   ├── sweep_items.csv        # one row per question, setting and run (rep 1 = main run, rep 2 = repeat)
│   │   ├── sweep_items_decode.csv     # prompt processing / generation benchmark, run 1
│   │   ├── sweep_items_decode_r2.csv  # same, repeat run (2 threads)
│   │   ├── sweep_items_runs.jsonl     # software versions, idle power, start/end times per setting
│   │   ├── sweep_items_system.jsonl   # governor, frequency, temperature, mains, memory, swap, storage
│   │   ├── sweep_console.txt          # console log, main run
│   │   └── repeat_console.txt         # console log, repeat run
│   ├── pilot/                     # five-question pilot and smoke test (not used in the letter's results)
│   └── analysis/
│       ├── run1/                  # analysis of the main run (Table III, Fig. 1a)
│       └── both_runs/             # 2-thread settings averaged over both runs, with run-to-run summary
└── figures/
    └── fig_energy.pdf / .png      # Fig. 1
```

Model files are not included. The Q4_K_M and Q8_0 GGUF files are on Hugging Face (`segestic/MediLITE-QA-GGUF`).

## Hardware and conditions

| Item | Setting |
|---|---|
| Laptop | HP 250 G4 |
| CPU | Intel Core i3-4005U (Haswell), 2 cores / 4 threads, 1.6 GHz, no turbo |
| Memory | 6 GB DDR3-1600 (4 GB + 2 GB) |
| OS | Ubuntu 22.04.5, kernel 6.8, `schedutil` governor |
| Runtime | llama-cpp-python 0.3.35, CPU only, context 2048, weights loaded fully into RAM (`--no-mmap`) |
| Energy | RAPL package domain (`/sys/class/powercap/intel-rapl:0`) |

Measurements were taken on mains power with Wi-Fi off, other applications closed and the laptop not in use. Before each setting the script waits for the package to cool to 50 °C, runs three warm-up questions and measures idle power for 30 s.

## Reproducing the measurements

```bash
pip install -r requirements.txt
sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj      # once per boot

python scripts/check_gguf.py MediLITE-QA.Q4_K_M.gguf MediLITE-QA.Q8_0.gguf

# main run: five settings, option scoring and generation (about 12 h)
python scripts/sweep_energy.py --items data/items_les.jsonl \
  --models Q4_K_M=MediLITE-QA.Q4_K_M.gguf Q8_0=MediLITE-QA.Q8_0.gguf \
  --threads 1 2 4 --skip Q8_0_t1 --decode --out sweep_items.csv

# repeat run: both formats at 2 threads (about 5 h)
python scripts/sweep_energy.py --items data/items_les.jsonl \
  --models Q4_K_M=MediLITE-QA.Q4_K_M.gguf Q8_0=MediLITE-QA.Q8_0.gguf \
  --threads 2 --passes 2 --out sweep_items.csv
python scripts/decode_bench.py --model MediLITE-QA.Q8_0.gguf --label medilite_Q8_0_t2_r2 \
  --threads 2 --items data/items_les.jsonl --no-mmap --out sweep_items_decode_r2.csv
python scripts/decode_bench.py --model MediLITE-QA.Q4_K_M.gguf --label medilite_Q4_K_M_t2_r2 \
  --threads 2 --items data/items_les.jsonl --no-mmap --out sweep_items_decode_r2.csv
```

Interrupted runs resume from the last finished question.

## Reproducing the tables and figure from the archived data

```bash
python scripts/analyze_sweep.py --csv results/raw/sweep_items.csv --decode results/raw/sweep_items_decode.csv \
  --rep 1 --ref-config medilite_Q4_K_M_t2 --fidelity-ref Q8_0 --out results/analysis/run1
python scripts/analyze_sweep.py --csv results/raw/sweep_items.csv --decode results/raw/sweep_items_decode.csv \
  --ref-config medilite_Q4_K_M_t2 --fidelity-ref Q8_0 --out results/analysis/both_runs
python scripts/make_figure.py results/analysis/run1 results/raw figures
```

| Letter item | Source |
|---|---|
| Table III, Fig. 1a | `results/analysis/run1/energy_by_config.csv`, `prefill_decode.csv` |
| Fig. 1b, run-to-run variation | `results/raw/sweep_items_decode*.csv`, `results/analysis/both_runs/REPORT.txt` |
| Answer fidelity, accuracy | `results/analysis/run1/fidelity_by_quant.csv` |
| Energy model | `results/analysis/run1/energy_vs_length.csv` |

`export_items.py` builds the question set from lm-evaluation-harness 0.4.11 per-item logs:
`python scripts/export_items.py --logs lm_eval_logs --source medilite --per-task 22 --seed 2027 --out items_les.jsonl`.

## Data licences

`data/items_les.jsonl` contains questions from MedMCQA, MedQA, PubMedQA and MMLU in the lm-evaluation-harness prompt format; they remain under the licences of those datasets. Code is released under the MIT License.
