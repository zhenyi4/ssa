# 其它参数
export HF_ALLOW_CODE_EVAL=1

AFILE=lm-evaluation-harness-outputs/

python -m lm_eval \
            --model_args '{"pretrained":"zen-E/SSA-1B","dtype":"auto","trust_remote_code":true,"rope_scaling":{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,"original_max_position_embeddings":8192,"rope_type":"llama3"}, "inference_mode":"sparse"}' \
            --tasks longbench_narrativeqa,longbench_qasper,longbench_multifieldqa_en,longbench_hotpotqa,longbench_2wikimqa,longbench_musique,longbench_gov_report,longbench_qmsum,longbench_multi_news,longbench_trec,longbench_triviaqa,longbench_samsum,longbench_passage_count,longbench_passage_retrieval_en,longbench_lcc,longbench_repobench-p \
            --batch_size 1 \
            --gen_kwargs use_cache=False \
            --output_path ${AFILE} \
            --log_samples \
            --device cuda \
