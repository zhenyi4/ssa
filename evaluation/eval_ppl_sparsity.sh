export HF_ALLOW_CODE_EVAL=1

# Ensure the adasplash package is importable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}/adasplash:${PYTHONPATH}"

# Usage: bash evaluation/eval_ppl_sparsity.sh [model_path] [max_length] [batch_size]
# Defaults: model_path=<project_root>/adasplash-1b, max_length=8192, batch_size=8

python evaluation/eval_ppl_with_sparsity.py "$@"
