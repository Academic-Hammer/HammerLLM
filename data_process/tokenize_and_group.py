import os.path
import argparse
from itertools import chain
from transformers import AutoTokenizer
from datasets import load_dataset, disable_caching


def encode_and_group(batch):
    input_ids = tokenizer([text for text in batch['text']])['input_ids']
    # Concatenate all texts.
    concatenated_input_ids = list(chain(*[sample + [tokenizer.eos_token_id] for sample in input_ids]))
    total_length = len(concatenated_input_ids)
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    blocked_input_ids = [concatenated_input_ids[i: i + block_size] for i in range(0, total_length, block_size)]
    return {"input_ids": blocked_input_ids}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--raw-file', type=str, required=True)
    parser.add_argument('--block-size', type=int, default=2048)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=100)
    args = parser.parse_args()

    disable_caching()

    if not os.path.exists(args.raw_file):
        print(f"[!] Fatal error: raw file {args.raw_file} does not exist.")
        exit(0)

    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"[!] Info: Making output directory {args.output_dir} successfully.")
        except OSError:
            print(f"[!] Error: Making output directory {args.output_dir} failed.")
            exit(0)
    else:
        print(f"[!] Warning: Output directory {args.output_dir} exists, will probably overwrite files in it.")

    block_size = args.block_size
    assert isinstance(block_size, int) and block_size > 1
    print(f"[!] Info: grouping block size: {block_size}")

    num_workers = args.num_workers
    assert isinstance(num_workers, int) and num_workers >= 1
    print(f"[!] Info: Dataset mapping num_workers: {num_workers}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    print(f"[!] Info: Tokenizer {args.tokenizer} has been instantiated.")

    raw_dataset = load_dataset(
        "json",
        name=os.path.basename(args.raw_file),
        cache_dir=os.path.join(args.output_dir, 'tmp'),
        data_files={
            "train": args.raw_file
        },
        split="train",
        num_proc=1
    )

    tokenized_dataset = raw_dataset.map(
        encode_and_group,
        batched=True,
        cache_file_name=os.path.join(args.output_dir, 'tmp_map', 'tokenized_and_mapped.arrow'),
        remove_columns=raw_dataset.column_names,
        num_proc=num_workers
    )
    tokenized_dataset.save_to_disk(args.output_dir)
    print(f"[!] Success: Tokenization and grouping completed.")
