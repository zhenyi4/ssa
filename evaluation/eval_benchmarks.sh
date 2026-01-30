#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="zen-E/SSA-1B"
OUTDIR="outputs/benchmarks_multiseed_results_ssa1b"
INFERENCE_MODE="full"

mkdir -p "$OUTDIR"

SEEDS=(1 12 123 1234 12345)

echo "=== Running ARC (arc_easy, arc_challenge) with multiple seeds ==="
for SEED in "${SEEDS[@]}"; do
    OUTFILE="${OUTDIR}/arc_seed${SEED}.json"
    echo ">>> [ARC] Running seed ${SEED}, saving to ${OUTFILE}"

    python -m lm_eval \
        --model_args pretrained="${MODEL_PATH}",max_length=8192,dtype=auto,trust_remote_code=true,inference_mode="${INFERENCE_MODE}" \
        --tasks arc_easy,arc_challenge \
        --batch_size 8 \
        --output_path "${OUTFILE}" \
        --log_samples \
        --num_fewshot 25 \
        --device cuda \
        --seed "${SEED}"
done

echo ""
echo "=== Running PIQA with multiple seeds (num_fewshot=3) ==="
for SEED in "${SEEDS[@]}"; do
    OUTFILE="${OUTDIR}/piqa_seed${SEED}.json"
    echo ">>> [PIQA] Running seed ${SEED}, saving to ${OUTFILE}"

    python -m lm_eval \
        --model_args pretrained="${MODEL_PATH}",max_length=8192,dtype=auto,trust_remote_code=true,inference_mode="${INFERENCE_MODE}" \
        --tasks piqa \
        --batch_size 8 \
        --output_path "${OUTFILE}" \
        --log_samples \
        --num_fewshot 3 \
        --device cuda \
        --seed "${SEED}"
done

echo ""
echo "=== Running HellaSwag with multiple seeds (num_fewshot=10) ==="
for SEED in "${SEEDS[@]}"; do
    OUTFILE="${OUTDIR}/hellaswag_seed${SEED}.json"
    echo ">>> [HellaSwag] Running seed ${SEED}, saving to ${OUTFILE}"

    python -m lm_eval \
        --model_args pretrained="${MODEL_PATH}",max_length=8192,dtype=auto,trust_remote_code=true,inference_mode="${INFERENCE_MODE}" \
        --tasks hellaswag \
        --batch_size 8 \
        --output_path "${OUTFILE}" \
        --log_samples \
        --num_fewshot 10 \
        --device cuda \
        --seed "${SEED}"
done

echo ""
echo "=== Running Wikitext once (no randomness) ==="
WIKI_OUTFILE="${OUTDIR}/wikitext.json"
python -m lm_eval \
    --model_args pretrained="${MODEL_PATH}",max_length=8192,dtype=auto,trust_remote_code=true,inference_mode="${INFERENCE_MODE}" \
    --tasks wikitext \
    --batch_size 8 \
    --output_path "${WIKI_OUTFILE}" \
    --log_samples \
    --device cuda

echo ""

echo "=== Computing mean / std of acc_norm per task (in %) and wikitext word_perplexity ==="

python << 'EOF'
import json, glob, statistics as stats
from pathlib import Path

OUTDIR = "outputs/benchmarks_multiseed_results_ssa1b"
PATTERN = str(Path(OUTDIR) / "*.json")

files = sorted(glob.glob(PATTERN))
if not files:
    raise SystemExit(f"No results found: {PATTERN}")

print("Found files:")
for f in files:
    print("  ", f)

# --- First pass: collect all metric keys per task across all files ---
metric_keys_by_task = {}

for f in files:
    with open(f, "r") as fh:
        data = json.load(fh)
    results = data.get("results", {})
    for task, metrics in results.items():
        keys = metric_keys_by_task.setdefault(task, set())
        for k in metrics.keys():
            keys.add(k)

print("\n=== Metric keys per task (for debugging) ===")
for task, keys in metric_keys_by_task.items():
    print(f"  Task: {task}, metric keys: {sorted(keys)}")

# --- Determine acc_norm key per task (for classification benchmarks) ---
acc_norm_key_for_task = {}
for task, keys in metric_keys_by_task.items():
    candidates = [k for k in keys if "acc_norm" in k]
    if not candidates:
        continue
    # if multiple acc_norm-like keys, choose the first one deterministically
    chosen = sorted(candidates)[0]
    acc_norm_key_for_task[task] = chosen

if not acc_norm_key_for_task:
    print("\nNo acc_norm-like keys found for any task.")
else:
    print("\nUsing acc_norm keys per task:")
    for task, key in acc_norm_key_for_task.items():
        print(f"  {task}: {key}")

# --- Determine word_perplexity key(s) for wikitext-like tasks ---
wikitext_key_for_task = {}
for task, keys in metric_keys_by_task.items():
    if "wikitext" in task.lower():
        candidates = [k for k in keys if "word_perplexity" in k]
        if not candidates:
            continue
        chosen = sorted(candidates)[0]
        wikitext_key_for_task[task] = chosen

# --- Second pass: aggregate acc_norm (scaled to %) and wikitext word_perplexity ---
acc_norm_per_task = {task: [] for task in acc_norm_key_for_task.keys()}
wikitext_ppl_per_task = {task: [] for task in wikitext_key_for_task.keys()}

for f in files:
    with open(f, "r") as fh:
        data = json.load(fh)
    results = data.get("results", {})

    # acc_norm aggregation
    for task, key in acc_norm_key_for_task.items():
        metrics = results.get(task)
        if not isinstance(metrics, dict):
            continue
        if key not in metrics:
            continue
        v = metrics[key]
        try:
            v_num = float(v)
        except (TypeError, ValueError):
            continue
        acc_norm_per_task[task].append(v_num * 100.0)  # convert to %

    # wikitext word_perplexity aggregation
    for task, key in wikitext_key_for_task.items():
        metrics = results.get(task)
        if not isinstance(metrics, dict):
            continue
        if key not in metrics:
            continue
        v = metrics[key]
        try:
            v_num = float(v)
        except (TypeError, ValueError):
            continue
        wikitext_ppl_per_task[task].append(v_num)

# --- Print acc_norm stats ---
print("\n=== Mean ± Std of acc_norm per task (in percentage points) ===\n")
if not acc_norm_per_task:
    print("No acc_norm values found.")
else:
    for task, vals in sorted(acc_norm_per_task.items()):
        if not vals:
            print(f"{task:15s} acc_norm = N/A (no values collected)")
            continue
        n = len(vals)
        if n == 1:
            mean = vals[0]
            std = 0.0
        else:
            mean = stats.mean(vals)
            std = stats.stdev(vals)  # sample std, n-1
        print(f"{task:15s} acc_norm = {mean:.2f} ± {std:.2f}  (n={n})")

# --- Print wikitext word_perplexity ---
print("\n=== Wikitext word_perplexity ===\n")
if not wikitext_ppl_per_task:
    print("No wikitext word_perplexity found.")
else:
    for task, vals in sorted(wikitext_ppl_per_task.items()):
        if not vals:
            print(f"{task:15s} word_perplexity = N/A")
            continue
        n = len(vals)
        if n == 1:
            mean = vals[0]
            std = 0.0
        else:
            mean = stats.mean(vals)
            std = stats.stdev(vals)
        print(f"{task:15s} word_perplexity = {mean:.3f}  (n={n})")
EOF

echo ""
echo "=== Done! ==="
