import os.path
import argparse
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer

B = 1e9


def encode(example):
    input_ids = tokenizer.encode_and_group(example['text'])
    tokens = len(input_ids)
    return {
        'tokens': tokens,
        'input_ids': input_ids
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--raw-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=40)
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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    print(f"[!] Info: Tokenizer {args.tokenizer} has been instantiated.")

    raw_dataset = load_dataset(
        "json",
        name=os.path.basename(args.raw_file),
        data_files={
            "train": args.raw_file
        },
        split="train",
        num_proc=1
    ).select_columns(['text'])
    tokenized_dataset = raw_dataset.map(
        encode,
        num_proc=args.num_workers
    )
    tokenized_dataset.save_to_disk(args.output_dir)
    print(f"[!] Success: Tokenization completed.")
