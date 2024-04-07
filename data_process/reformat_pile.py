import json
import argparse
from tqdm import tqdm

GB = 1024 * 1024 * 1024


def reformat(data_file: str, target_file: str, num_gb: int = None):
    accum_size = 0

    f_src = open(data_file, 'r+')
    f_dst = open(target_file, 'w+')

    progress_bar = tqdm(total=num_gb)
    for i, line in enumerate(f_src):
        text = json.loads(line)['text']
        len_text = len(text)
        accum_size += len_text

        f_dst.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

        if accum_size >= num_gb * GB:
            break
        progress_bar.update(len_text / GB)
    f_src.close()
    f_dst.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help=f"The path to a single file of pile (xxx.jsonl)"
    )
    parser.add_argument(
        '--num-gb',
        type=int,
        default=100,
        help=f"Output file size in GB"
    )
    parser.add_argument(
        '--target-file',
        type=str,
        required=True,
        help=f"The path to the target file"
    )
    args = parser.parse_args()
    reformat(args.data_file, args.target_file, args.num_gb)
