#!/usr/bin/env bash
set -euo pipefail

export HF_ALLOW_CODE_EVAL=1

OUTDIR="outputs/ruler_niah_results"
mkdir -p "$OUTDIR"

MODELS=("zen-E/FullAttn-1B" "zen-E/SSA-1B" "zen-E/MoBA-1B")
MODEL_NAMES=("FullAttn-1B" "SSA-1B" "MoBA-1B")
TASKS=("niah_multikey_3" "ruler_vt" "ruler_fwe")
METADATA='{"max_seq_lengths":[4096,8192,16384,32768]}'

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"

    for TASK in "${TASKS[@]}"; do
        RUN_OUTDIR="${OUTDIR}/${MODEL_NAME}/${TASK}"
        mkdir -p "${RUN_OUTDIR}"

        echo ">>> Running ${TASK} on ${MODEL_NAME}"

        accelerate launch --num_processes 8 -m lm_eval \
            --model_args "{\"pretrained\":\"${MODEL}\",\"dtype\":\"auto\",\"trust_remote_code\":true,\"inference_mode\":\"full\"}" \
            --tasks "${TASK}" \
            --metadata="${METADATA}" \
            --gen_kwargs use_cache=True \
            --batch_size 1 \
            --output_path "${RUN_OUTDIR}" \
            --log_samples \
            --device cuda

        echo ">>> Finished ${TASK} on ${MODEL_NAME}"
        echo ""
    done
done

echo "=== All evaluations complete. Aggregating results... ==="

python << 'EOF'
import json, os, glob

OUTDIR = "outputs/ruler_niah_results"
MODELS = ["FullAttn-1B", "SSA-1B", "MoBA-1B"]
TASKS = ["niah_multikey_3", "ruler_vt", "ruler_fwe"]
LENGTHS = [4096, 8192, 16384, 32768]

all_results = {}  # (model, task, length) -> score

for model in MODELS:
    for task in TASKS:
        search_path = os.path.join(OUTDIR, model, task)
        json_files = glob.glob(os.path.join(search_path, "**", "results_*.json"), recursive=True)
        if not json_files:
            print(f"WARNING: No results.json found for {model}/{task}")
            continue

        results_file = sorted(json_files)[-1]
        with open(results_file) as f:
            data = json.load(f)

        results = data.get("results", {})
        metrics = results.get(task, {})

        for length in LENGTHS:
            score = metrics.get(f"{length},none")
            all_results[(model, task, length)] = score

# Print formatted table
header = f"{'Model':<15} {'Task':<20} " + " ".join(f"{l:>8}" for l in LENGTHS)
separator = "-" * len(header)

lines = []
lines.append(separator)
lines.append(header)
lines.append(separator)

for model in MODELS:
    for task in TASKS:
        scores = []
        for length in LENGTHS:
            val = all_results.get((model, task, length))
            if val is not None:
                scores.append(f"{val*100:>7.1f}%")
            else:
                scores.append(f"{'N/A':>8}")
        lines.append(f"{model:<15} {task:<20} " + " ".join(scores))
    lines.append("")

lines.append(separator)

output = "\n".join(lines)
print(output)

summary_path = os.path.join(OUTDIR, "results_summary.txt")
with open(summary_path, "w") as f:
    f.write(output + "\n")
print(f"\nResults written to {summary_path}")
EOF

echo "=== Done! ==="
