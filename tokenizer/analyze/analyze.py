#!/usr/bin/env python
# encoding: utf-8
"""
File Description:
Created Time: 2023/12/20
"""
import argparse
import os
from itertools import chain
from tqdm import tqdm
from transformers import AutoTokenizer
from vocab_coverage.lexicon import load_lexicon
from vocab_coverage import coverage_analysis


def coverage_analyze(tokenizer_path):
    """
    字符覆盖率分析
    """
    granularity = 'char'  # char/token
    lexicon = load_lexicon(tokenizer_path, granularity=granularity)
    coverage_analysis(tokenizer_path,
                      lexicon=lexicon,
                      granularity=granularity,
                      folder='.')


def compression_analyze(tokenizer_path):
    """
    压缩率分析
    """

    def _load_text_file(path):
        if not os.path.exists(path):
            raise Exception(
                'please download test set from https://huggingface.co/datasets/DataHammer/Tokenizer-Test-Set first')
        with open(path) as f:
            data = []
            for line in f.readlines():
                data.append(line.strip())
        print(f'[!] load {len(data)} lines from {path}')
        return data

    dataset_names = ['chinese-valid-5000.txt', 'refinedweb-valid-4000.txt', 'pile-github-valid-1000.txt']
    datasets = [_load_text_file(name) for name in dataset_names]
    datasets = list(chain(*datasets))
    print(f'collect {len(datasets)} samples for test')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True)

    tokens, bytes = 0, 0
    for text in tqdm(datasets):
        tokens += len(tokenizer.encode(text, add_special_tokens=False))
        bytes += len(text.encode())
    print('compress ratio', bytes / tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str, default='../models/internlm-7b')
    args = parser.parse_args()
    coverage_analyze(args.tokenizer_path)
    compression_analyze(args.tokenizer_path)
