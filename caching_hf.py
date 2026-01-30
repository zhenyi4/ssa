import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import json

import datasets
from datasets import load_dataset, load_from_disk, concatenate_datasets

import transformers
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments
)

import os
import random
random.seed(42)

Concats = ["\n\n"]

logger = logging.getLogger(__name__)


@dataclass
class SelfTrainingArguments(TrainingArguments):
    output_dir: Optional[str] = field(
        default="/root/cache",
        metadata={"help": "output dir required by TrainingArguments, but we do not use in this script"}
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default="/root/cache",
        metadata={"help": "cache dir of datasets"}
    )


@dataclass
class DataTrainingArguments:
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # HuggingFace dataset arguments
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace dataset name (e.g., 'HuggingFaceTB/smollm-corpus')"}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset configuration/subset name (e.g., 'fineweb-edu-dedup')"}
    )
    dataset_split: Optional[str] = field(
        default="train",
        metadata={"help": "Dataset split to use (default: train)"}
    )
    # Local file path (alternative to HF dataset)
    train_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local JSONL file path (alternative to dataset_name)"}
    )
    # Data format
    text_field: Optional[str] = field(
        default=None,
        metadata={"help": "Field name containing text (for non-conversation datasets). If set, treats data as raw text."}
    )
    conversations_field: Optional[str] = field(
        default="conversations",
        metadata={"help": "Field name containing conversations (default: 'conversations')"}
    )
    # Processing arguments
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    num_shards: Optional[int] = field(
        default=32,
        metadata={"help": "The number of arrow shards to save."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to process (for debugging or subset creation)"}
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Use streaming mode for large datasets"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SelfTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Validate arguments
    if data_args.dataset_name is None and data_args.train_data_path is None:
        raise ValueError("Must provide either --dataset_name or --train_data_path")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        truncation_side='left',
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Starting data preprocessing")

    # Load dataset
    if data_args.dataset_name is not None:
        logger.info(f"Loading HuggingFace dataset: {data_args.dataset_name}")
        load_kwargs = {
            "path": data_args.dataset_name,
            "split": data_args.dataset_split,
            "trust_remote_code": True,
        }
        if data_args.dataset_config:
            load_kwargs["name"] = data_args.dataset_config
        if data_args.streaming:
            load_kwargs["streaming"] = True

        raw_dataset = load_dataset(**load_kwargs)
    else:
        logger.info(f"Loading local file: {data_args.train_data_path}")
        raw_dataset = load_dataset(
            "json",
            data_files={"train": data_args.train_data_path},
            split="train"
        )

    # Limit samples if specified
    if data_args.max_samples is not None and not data_args.streaming:
        raw_dataset = raw_dataset.select(range(min(data_args.max_samples, len(raw_dataset))))
        logger.info(f"Limited to {len(raw_dataset)} samples")

    # Define preprocessing functions (matching caching.py logic)
    def preprocess_conversations(examples):
        """Process conversation-format data (list of {"content": ...} dicts)"""
        model_inputs = {"input_ids": [[]]}
        acc_len = 0
        conversations_field = data_args.conversations_field

        for conversation in examples[conversations_field]:
            concat = random.choice(Concats)
            # Handle both list of dicts and other formats
            if isinstance(conversation, list):
                contents = []
                for turn in conversation:
                    if isinstance(turn, dict) and "content" in turn:
                        content = turn["content"]
                        if isinstance(content, str):
                            contents.append(content)
                    elif isinstance(turn, str):
                        contents.append(turn)
                message = concat.join(contents)
            elif isinstance(conversation, str):
                message = conversation
            else:
                continue

            if not message:
                continue

            message_ids = tokenizer.encode(message, add_special_tokens=False)

            # Split into chunks of model_max_length
            input_ids_list = []
            for i in range(0, len(message_ids), data_args.model_max_length - 1):
                input_ids_list.append(message_ids[i:i + data_args.model_max_length - 1] + [tokenizer.eos_token_id])

            # Pack sequences into rows respecting max_length (same logic as caching.py)
            for input_ids in input_ids_list:
                if acc_len + len(input_ids) > data_args.model_max_length:
                    model_inputs["input_ids"].append([input_ids])
                    acc_len = len(input_ids)
                else:
                    model_inputs["input_ids"][-1].append(input_ids)
                    acc_len += len(input_ids)

        return model_inputs

    def preprocess_text(examples):
        """Process raw text data (single text field)"""
        model_inputs = {"input_ids": [[]]}
        acc_len = 0
        text_field = data_args.text_field

        for text in examples[text_field]:
            if not isinstance(text, str) or not text:
                continue

            message_ids = tokenizer.encode(text, add_special_tokens=False)

            # Split into chunks of model_max_length
            input_ids_list = []
            for i in range(0, len(message_ids), data_args.model_max_length - 1):
                input_ids_list.append(message_ids[i:i + data_args.model_max_length - 1] + [tokenizer.eos_token_id])

            # Pack sequences into rows respecting max_length (same logic as caching.py)
            for input_ids in input_ids_list:
                if acc_len + len(input_ids) > data_args.model_max_length:
                    model_inputs["input_ids"].append([input_ids])
                    acc_len = len(input_ids)
                else:
                    model_inputs["input_ids"][-1].append(input_ids)
                    acc_len += len(input_ids)

        return model_inputs

    # Choose preprocessing function based on data format
    if data_args.text_field is not None:
        preprocess_fn = preprocess_text
        logger.info(f"Using text field: {data_args.text_field}")
    else:
        preprocess_fn = preprocess_conversations
        logger.info(f"Using conversations field: {data_args.conversations_field}")

    # Process dataset
    os.makedirs(model_args.cache_dir, exist_ok=True)

    if data_args.streaming:
        # Streaming mode: process in batches and save incrementally
        logger.info("Processing in streaming mode...")
        batch_size = 10000
        shard_idx = 0
        current_batch = {"input_ids": []}

        for i, example in enumerate(raw_dataset):
            # Process single example
            single_example = {k: [v] for k, v in example.items()}
            processed = preprocess_fn(single_example)
            current_batch["input_ids"].extend(processed["input_ids"])

            if len(current_batch["input_ids"]) >= batch_size:
                # Save shard
                shard_dataset = datasets.Dataset.from_dict(current_batch)
                shard_path = os.path.join(model_args.cache_dir, f"shard-{shard_idx:04d}")
                shard_dataset.save_to_disk(shard_path)
                logger.info(f"Saved shard {shard_idx} with {len(current_batch['input_ids'])} samples")
                shard_idx += 1
                current_batch = {"input_ids": []}

            if data_args.max_samples and i >= data_args.max_samples - 1:
                break

        # Save remaining data
        if current_batch["input_ids"]:
            shard_dataset = datasets.Dataset.from_dict(current_batch)
            shard_path = os.path.join(model_args.cache_dir, f"shard-{shard_idx:04d}")
            shard_dataset.save_to_disk(shard_path)
            logger.info(f"Saved final shard {shard_idx} with {len(current_batch['input_ids'])} samples")

    else:
        # Standard mode: process all at once
        logger.info("Processing dataset...")
        column_names = raw_dataset.column_names

        processed_dataset = raw_dataset.map(
            preprocess_fn,
            batched=True,
            batch_size=1024,
            remove_columns=column_names,
            num_proc=data_args.preprocessing_num_workers,
            desc="Tokenizing dataset"
        )

        # Save to disk
        logger.info(f"Saving processed dataset to {model_args.cache_dir}")
        processed_dataset.save_to_disk(model_args.cache_dir, num_shards=data_args.num_shards)

    logger.info("Done!")

    # Print loading instructions
    print("\n" + "=" * 60)
    print("To load the cached dataset:")
    print("=" * 60)
    print(f"""
from datasets import load_from_disk, concatenate_datasets
import os

cache_dir = "{model_args.cache_dir}"

# If saved with num_shards (standard mode):
dataset = load_from_disk(cache_dir)

# If saved in streaming mode (multiple shard folders):
# shard_dirs = sorted([d for d in os.listdir(cache_dir) if d.startswith("shard-")])
# all_shards = [load_from_disk(os.path.join(cache_dir, d)) for d in shard_dirs]
# dataset = concatenate_datasets(all_shards)

dataset = dataset.shuffle(seed=42)
dataset.set_format(type="pt")
print(dataset[0])
""")


if __name__ == "__main__":
    main()
