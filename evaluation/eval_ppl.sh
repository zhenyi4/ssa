export HF_ALLOW_CODE_EVAL=1

AFILE=lm-evaluation-harness-outputs/

python -m lm_eval \
            --model_args pretrained="zen-E/SSA-1B",max_length=8192,dtype=auto,trust_remote_code=true,inference_mode="sparse" \
            --tasks wikitext \
            --batch_size 8 \
            --output_path ${AFILE} \
            --log_samples \
            --device cuda
