# since niah does not support dict type of model_args, we have to add the rope scaling config manually.
export HF_ALLOW_CODE_EVAL=1

AFILE=lm-evaluation-harness-outputs/

python -m lm_eval \
    --model_args '{"pretrained":"zen-E/SSA-1B","dtype":"auto","trust_remote_code":true, "inference_mode":"sparse"}' \
    --tasks niah_single_2 \
    --metadata='{"max_seq_lengths":[4096,8192,16384,32768]}' \
    --gen_kwargs use_cache=False \
    --batch_size 1 \
    --output_path ${AFILE} \
    --log_samples \
    --device cuda
