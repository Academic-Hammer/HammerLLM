import json
from typing import List
import os.path
import argparse
from tqdm import tqdm
from json.decoder import JSONDecodeError
import logging

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


def reformat(data_files: List[str], target_file: str, num_gb: int = None):
    accum_size = 0

    f_dst = open(target_file, 'w+', encoding='utf-8')

    progress_bar = tqdm(total=num_gb)
    error_counter = 0
    for data_file in data_files:
        f_src = open(data_file, 'r+')
        progress_bar.set_description(f"Processing {data_file}")
        for i, line in enumerate(f_src):
            try:
                text = json.loads(line)['text']
            except JSONDecodeError:
                logger.error(f"JSON decode error occurred in {data_file}")
                error_counter += 1
                continue
            len_text = len(text.encode(encoding='utf-8'))

            f_dst.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
            accum_size += len_text

            if accum_size >= num_gb * GB:
                break

            progress_bar.update(len_text / GB)
        f_src.close()
    f_dst.close()
    logger.error(f"Total JSONDecodeError: {error_counter}")


if __name__ == '__main__':
    """Merge multiple dataset files into a single jsonl file with a given size (num. bytes of raw text)."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help=f"Directory containing the CASIA-LM/ChineseWebText dataset"
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
    file_names = os.listdir(args.data_dir)
    file_names.sort()
    data_file_names = []
    for file_name in file_names:
        if file_name.endswith('.jsonl'):
            data_file_names.append(os.path.join(args.data_dir, file_name))
    reformat(data_file_names, args.target_file, args.num_gb)
