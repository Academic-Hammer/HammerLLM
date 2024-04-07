#!/usr/bin/env python
# encoding: utf-8
"""
File Description:
Author: maziao, gmftbyGMFTBY
Created Time: 2023/11/23
"""
import time
import ipdb
import math
import numpy as np
from dataclasses import dataclass, field

from llama_flash_attention_monkey_patch import *
from typing import Optional
from datasets import load_from_disk, disable_caching
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
)
from transformers.integrations import WandbCallback
from scheduler import add_lr_decay_limit_for_cosine_schedule
from llama_forward_without_labels_monkey_patch import replace_llama_for_causal_lm_forward
import logging

logging.getLogger()


# ============ only for debug ========== #
class PrinterCallback(TrainerCallback):
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        torch.save(kwargs['train_dataloader'].dataset['input_ids'], 'second_run.pt')
        ipdb.set_trace()


# ============ only for debug ========== #


# https://github.com/huggingface/transformers/blob/35551f9a0f66a22de4971b4a51b3c172d3b87f95/docs/source/ja/model_memory_anatomy.md?plain=1#L63
def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")


# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

class WandbLogCallback(WandbCallback):
    """Custom WandbCallback
    """

    def __init__(self, block_size, train_batch_size):
        super().__init__()
        self.block_size = block_size
        self.train_batch_size = train_batch_size

        self.start_time = None
        self._time = None
        self._tokens = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, model, **kwargs)
        current_time = time.time()
        self.start_time = current_time
        self._time = current_time
        self._tokens = int(state.global_step * self.block_size * self.train_batch_size)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        # https://github.com/huggingface/transformers/blob/3bc50d81e6c70d63e59d635106bac6a561b47681/src/transformers/integrations/integration_utils.py#L812
        tokens = int(state.global_step * self.block_size * self.train_batch_size)
        compute_flops = int(state.total_flos)

        current_time = time.time()
        if tokens != self._tokens:
            logs['token_per_sec'] = int((tokens - self._tokens) / (current_time - self._time))
        self._time = current_time
        self._tokens = tokens

        super().on_log(args, state, control, model, logs, **kwargs)

        if state.is_world_process_zero:
            log_d = {'tokens': tokens, 'flops': compute_flops, 'time': int(current_time - self.start_time)}
            for key in logs:
                if key.startswith('eval_') and key.endswith('_loss'):
                    mode = 'zh' if 'chinese' in key else 'en'
                    log_d[f'val/eval_{mode}_loss'] = logs[key]
                    ppl = math.exp(logs[key])
                    log_d[f'val/eval_{mode}_ppl'] = ppl
            if 'loss' in logs:
                log_d['train/loss'] = logs['loss']
            if 'learning_rate' in logs:
                log_d['train/lr'] = logs['learning_rate']
            self._wandb.log(log_d)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The config directory for model initialization. It should be the path to a model checkpoint if `load_pretrained_checkpoint` is set to be `True`"
            )
        },
    )
    load_pretrained_checkpoint: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Continue training on a pretrained checkpoint or not."
            )
        },
    )
    use_flash_attention_2: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether the flash attention v2 is active during training."
            )
        }
    )
    use_torch_compile: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether to use torch.compile before training."
            )
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_files: Optional[List[str]] = field(default_factory=lambda: [])
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )


def main():
    """
    Entrance
    """
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    torch.manual_seed(training_args.seed)
    disable_caching()

    # switch of flash attention 2
    if model_args.use_flash_attention_2:
        replace_llama_attn_with_flash_attn()
        logging.info(f'[!] Flash Attention v2 is used for speedup.')
    else:
        logging.info(f'[!] Flash Attention v2 is not used.')

    # replace the forward func of LlamaForCausalLM to calculate loss without providing labels
    replace_llama_for_causal_lm_forward()

    # replace cosine lr scheduler to the custom one
    add_lr_decay_limit_for_cosine_schedule()
    logging.info(f'[!] Add lr decay limit for cosine schedule.')

    # prepare model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    logging.info(f"[!] Tokenizer instantiation completed. (vocab_size: {tokenizer.vocab_size})")
    if not model_args.load_pretrained_checkpoint:
        model_config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        model = LlamaForCausalLM(config=model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    logging.info(f"[!] Model instantiation completed.")

    # switch of torch.compile
    if model_args.use_torch_compile:
        model = torch.compile(model)
        logging.info(f'[!] torch.compile is used for more efficient training.')
    else:
        logging.info(f'[!] torch.compile is not used.')

    # prepare dataset
    train_dataset = load_from_disk(data_args.train_file)
    eval_dataset = {val_file: load_from_disk(val_file) for val_file in data_args.validation_files}
    logging.info(f'[!] Dataset instantiation completed.')

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )
    trainer.add_callback(
        WandbLogCallback(
            block_size=data_args.block_size,
            train_batch_size=trainer.args.train_batch_size * trainer.args.gradient_accumulation_steps
        )
    )
    trainer.can_return_loss = True
    logging.info(f"[!] Trainer initiation completed.")

    try:
        training_output = trainer.train(resume_from_checkpoint=True)
    except ValueError:
        training_output = trainer.train()
    trainer.save_model()
    print_summary(training_output)


if __name__ == '__main__':
    main()
