# ==============================Env Variables=================================
export NCCL_DEBUG=WARN
export CUDA_HOME=$CONDA_PREFIX

# Single-node training defaults
export HOST_NUM=1
export INDEX=0
export HOST_GPU_NUM=2  # Change to your number of GPUs
export CHIEF_IP=localhost

# Others
export WANDB_MODE=disabled

# ==============================Training Parameters=================================
MAXLEN=8192  # max len of data
postfix=ssa-1b-smollm100b  # name of output dir
PER_GPU_BATCH=24  # Micro batch size
GRA_ACC=8  # Accumulation steps
model=configs/ssa-1b-init  # name of start configs
root=/root/ssa  # root dir
data_cache=/workspace/smollm_cache  # cache dir of data
ds_config=ds_config_zero2.json  # deepspeed config
resume_checkpoint=None  # resuming dir, None refers to new start
skipnum=-1  # num of skipping instances from the beginning, -1 refers to no skip
maxnum=0  # max num of training instances, 0 refers to run with all

raw_model_path=${root}/${model}/
train_data_path=${data_cache}
deepspeed_config_path=${root}/${ds_config}
model_output_path=${root}/${postfix}
cache_path=${root}/${model}-cache  # dummy parameter
resume_path=${model_output_path}/${resume_checkpoint}

LR=1e-3
EPOCH=1
SEED=42
MIN_LR=0
SAVESTEP=0.2
optim="adamw_torch"
warmup_ratio=0.01
warmup_steps=0
lr_scheduler="cosine_with_min_lr"

# ==============================Training=================================
torchrun --nnodes=$HOST_NUM \
    --node_rank=$INDEX \
    --nproc_per_node $HOST_GPU_NUM \
    --master_addr $CHIEF_IP \
    --master_port 19198 \
    ${root}/train_pt.py \
    --model_name_or_path ${raw_model_path} \
    --bf16 True \
    --optim ${optim} \
    --output_dir ${model_output_path} \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${PER_GPU_BATCH} \
    --gradient_accumulation_steps ${GRA_ACC} \
    --save_strategy "steps" \
    --save_steps ${SAVESTEP} \
    --save_total_limit 100 \
    --per_device_eval_batch_size ${PER_GPU_BATCH} \
    --log_level "info" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --weight_decay 0.1 \
    --warmup_ratio ${warmup_ratio} \
    --warmup_steps ${warmup_steps} \
    --lr_scheduler_type ${lr_scheduler} \
    --learning_rate ${LR} \
    --min_lr ${MIN_LR} \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --tf32 True \
    --model_max_length ${MAXLEN} \
    --train_data_path ${train_data_path} \
    --preprocessing_num_workers 128 \
    --dataloader_num_workers 0 \
    --gradient_checkpointing True \
    --cache_dir ${cache_path} \
    --resume_from_checkpoint ${resume_path} \
    --seed ${SEED} \
    --max_train_samples ${maxnum} \
    --skip_train_samples ${skipnum} \
    --report_to "none" \
    --deepspeed ${deepspeed_config_path}
