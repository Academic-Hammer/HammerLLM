import argparse
import os.path
import random
from tqdm import tqdm

if __name__ == '__main__':
    """
    Sample tokenizer train & valid corpus.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-file', type=str, required=True)
    parser.add_argument('--tgt-dir', type=str, required=True)
    parser.add_argument('--tgt-base-name', type=str, required=True)
    parser.add_argument('--num-sample-train', type=int, required=True)
    parser.add_argument('--num-sample-valid', type=int, required=True)
    args = parser.parse_args()

    with open(args.src_file, 'r+') as f:
        data = f.readlines()

    num_data = len(data)
    indices = list(range(num_data))

    random.seed(0)
    print(f"[!] Sampling indices ...")
    sample_indices = random.sample(indices, args.num_sample_train + args.num_sample_valid)
    valid_sample_indices = sample_indices[-args.num_sample_valid:]
    with open(os.path.join(args.tgt_dir, f'{args.tgt_base_name}-valid-{args.num_sample_valid}.txt'), 'w+',
              encoding='utf-8') as f:
        for index in tqdm(valid_sample_indices, total=len(valid_sample_indices), desc=f"Processing valid data"):
            f.write(data[index].strip() + '\n')

    train_sample_indices = sample_indices[:args.num_sample_train]
    with open(os.path.join(args.tgt_dir, f'{args.tgt_base_name}-train-{args.num_sample_train}.txt'), 'w+',
              encoding='utf-8') as f:
        for index in tqdm(train_sample_indices, total=len(train_sample_indices), desc=f"Processing train data"):
            f.write(data[index].strip() + '\n')
