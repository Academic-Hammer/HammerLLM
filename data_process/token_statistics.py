import json
import os.path
import argparse
from tqdm import tqdm
from datasets import load_from_disk, disable_caching

B = 1e9


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True)
    args = parser.parse_args()

    disable_caching()

    dataset = load_from_disk(
        args.dataset_dir
    ).select_columns(['tokens'])

    result = {}
    n_billion = 1
    accum_tokens = 0
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        accum_tokens += sample['tokens']
        if accum_tokens >= B:
            result[f"{n_billion}B"] = i
            tqdm.write(f"{n_billion}B: {i}")

            accum_tokens = 0
            n_billion += 1

    with open(os.path.join(args.dataset_dir, 'token_statistics.json'), 'w+', encoding='utf-8') as f:
        json.dump(result, fp=f, indent=4, ensure_ascii=False)

    print(f"[!] Success: Statistics completed.")
