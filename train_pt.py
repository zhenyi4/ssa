import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import math
import copy
import json

import datasets
from datasets import load_dataset, load_from_disk, concatenate_datasets

import transformers
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
    default_data_collator,
    set_seed,
)

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

import os
import warnings

logger = logging.getLogger(__name__)

@dataclass
class SelfTrainingArguments(TrainingArguments):
    min_lr: Optional[str] = field(default=None, metadata={"help": "min lr for cosine_with_min_lr schedular"})

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "Lora attention dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha parameter for Lora scaling"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "The dropout probability for Lora layers."}
    )
    if_lora: Optional[int] = field(default=0, metadata={"help": "Whether run lora or full training."})
    cache_dir: Optional[str] = field(default="/root/cache", metadata={"help": "cache dir of datasets"})

@dataclass
class DataTrainingArguments:
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessed_path: str = field(
        default=None, metadata={"help": "Path to the preprocessed training data."}
    )
    train_data_path: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "The input evaluation data file (a jsonlines)."})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    skip_train_samples: Optional[int] = field(
        default=-1,
        metadata={"help": "The number of skip certain amount of training samples"},
    )

@dataclass
class DataCollatorWithFlattening(DefaultDataCollator):
    """
    Data collator used for padding free approach. Does the following:

    - concatate the entire mini batch into single long sequence [1, total_tokens]
    - uses `separator_id` to separate sequences within the concatenated `labels`, default value is -100
    - no padding will be added, returns `input_ids`, `labels` and `position_ids`
    """

    def __init__(self, *args, return_position_ids=True, separator_id=-100, max_len=8192, pad_token_id=128001, label_ignore_id=-100, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_position_ids = return_position_ids
        self.separator_id = separator_id
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.label_ignore_id = label_ignore_id
        warnings.warn(
            "Using `DataCollatorWithFlattening` will flatten the entire mini batch into single long sequence."
            "Make sure your attention computation is able to handle it!"
        )

    def __call__(self, features, return_tensors=None, separator_id=None):
        def padding_ret(ret):
            padding_len = self.max_len - len(ret["input_ids"])
            if self.return_position_ids:
                padded_position_ids = list(range(padding_len))
                ret["position_ids"] += padded_position_ids
            ret["input_ids"] += [self.pad_token_id] * padding_len
            ret["labels"] += [self.label_ignore_id] * padding_len
            ret["input_ids"] = ret["input_ids"][:self.max_len]
            ret["labels"] = ret["labels"][:self.max_len]
            return ret
        
        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id

        rets = []
        for idx in range(0, len(features)):
            ret = {"input_ids": [], "labels": []}
            if self.return_position_ids:
                ret.update({"position_ids": []})
            for f_input_ids in features[idx]["input_ids"]:
                ret["input_ids"] += f_input_ids
                ret["labels"] += [separator_id] + f_input_ids[1:]
                if self.return_position_ids:
                    ret["position_ids"] += list(range(len(f_input_ids)))
            rets.append(padding_ret(ret))

        return default_data_collator(rets, return_tensors)

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.cpu().clone().detach()
    return param

def get_peft_state_maybe_zero_3(state_dict, bias):
    if bias == "none":
        to_return = {
            k: state_dict[k].cpu().clone().detach() for k in state_dict if "lora_" in k
        }
    elif bias == "all":
        to_return = {
            k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in state_dict:
            if "lora_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


class CustomTrainer(Trainer):
    """Trainer that logs per-component losses (cross_entropy, alignment) using proportions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ce_sum = 0.0
        self._al_sum = 0.0
        self._count = 0
        # Enable adasplash sparsity stats if available
        try:
            from adasplash import enable_sparsity_stats, get_sparsity_stats, reset_sparsity_stats
            enable_sparsity_stats(num_layers=16)
            self._sparsity_stats_available = True
        except ImportError:
            self._sparsity_stats_available = False

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        unwrapped = self.accelerator.unwrap_model(model)
        if hasattr(unwrapped, '_last_ce_loss'):
            self._ce_sum += unwrapped._last_ce_loss
            self._al_sum += unwrapped._last_alignment_loss
            self._count += 1

        return loss

    def log(self, logs, start_time=None):
        if self._count > 0 and "loss" in logs:
            total = self._ce_sum + self._al_sum
            if total > 0:
                logs["cross_entropy"] = round(logs["loss"] * self._ce_sum / total, 4)
                logs["alignment"] = round(logs["loss"] * self._al_sum / total, 4)
            self._ce_sum = 0.0
            self._al_sum = 0.0
            self._count = 0
        # Log adasplash zero attention ratio
        if self._sparsity_stats_available:
            from adasplash import get_sparsity_stats, reset_sparsity_stats
            stats = get_sparsity_stats()
            if stats:
                total_nz = sum(s["nonzero_elements"] for s in stats.values())
                total_el = sum(s["total_elements"] for s in stats.values())
                if total_el > 0:
                    logs["attn_zero_ratio"] = round(1.0 - total_nz / total_el, 4)
                reset_sparsity_stats()
        super().log(logs, start_time)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SelfTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup seed
    set_seed(training_args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # load config and tokenziers
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.use_cache = False
    # use truncation_side='left' to preserve linking between end of prompt and target labels
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, truncation_side='left', trust_remote_code=True)

    # load dataset
    logger.info("start data preprocess")
    label_ignore_id = -100
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token

    # add pad token in tokenizer if needed
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_shards = [load_from_disk(os.path.join(data_args.train_data_path, shard_path)) for shard_path in os.listdir(data_args.train_data_path) if "shard" in shard_path]
    prepared_dataset = {}
    prepared_dataset["train"] = concatenate_datasets(all_shards)

    prepared_dataset["train"] = prepared_dataset["train"].shuffle(seed=training_args.seed)
    if data_args.skip_train_samples > -1:
        logger.info("load dataset finished, num of train: {}, num of valid: 0".format(len(prepared_dataset["train"])))
        prepared_dataset["train"] = prepared_dataset["train"].select(range(data_args.skip_train_samples, len(prepared_dataset["train"])))
        logger.info("skip num of first {} training samples, num of remained training samples: {}".format(data_args.skip_train_samples, len(prepared_dataset["train"])))
    if data_args.max_train_samples > 0:
        max_train_samples = min(len(prepared_dataset["train"]), data_args.max_train_samples)
        prepared_dataset["train"] = prepared_dataset["train"].select(range(max_train_samples))
    logger.info("load dataset finished, num of train: {}, num of valid: 0".format(len(prepared_dataset["train"])))

    # initialize modules
    if "-init" not in model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True, attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_config(config=config, trust_remote_code=True, attn_implementation="flash_attention_2")
    logger.info("load model finished, parameters of model: {}".format(model.num_parameters()))

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    if len(tokenizer) > tokenizer.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # Addin Training Arguments
    if training_args.min_lr is not None and training_args.lr_scheduler_type == "cosine_with_min_lr":
        extra_kwargs = {"min_lr": float(training_args.min_lr)}
        training_args.lr_scheduler_kwargs = extra_kwargs
    # Setup Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset["train"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithFlattening(max_len=data_args.model_max_length, pad_token_id=tokenizer.pad_token_id, label_ignore_id=label_ignore_id)
    )

    # Training
    if "steps" in training_args.resume_from_checkpoint or "checkpoint" in training_args.resume_from_checkpoint:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        train_result = trainer.train()
    trainer.save_state()

    # save fp16 model under deepspeed zero2 or zero3
    c_stage = json.load(open(training_args.deepspeed, "r"))["zero_optimization"]["stage"]
    if c_stage in [2, 3]:
        if c_stage == 2:
            w_state_dict = get_peft_state_maybe_zero_3(trainer.model.named_parameters(), "none")
        else:
            w_state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if trainer.is_world_process_zero():
            state_dict = {key: value.half().cpu() for key, value in w_state_dict.items()}
            trainer._save(training_args.output_dir, state_dict=state_dict)
    else:
        trainer.save_model()

if __name__ == "__main__":
    main()
