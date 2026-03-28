export HF_ALLOW_CODE_EVAL=1

# Usage: bash evaluation/eval_ppl_sparsity.sh [model_path] [max_length] [batch_size]
# Defaults: model_path=/Users/zhenyishen/Downloads/adasplash_dir/adasplash-1b-init, max_length=8192, batch_size=8

python evaluation/eval_ppl_with_sparsity.py "$@"
