import json
import os
import os.path
import argparse
import pyarrow.parquet as pq
from tqdm import tqdm

GB = 1024 * 1024 * 1024


def reformat(data_dir: str, target_file: str, num_gb: int = None):
    link_file_names = os.listdir(data_dir)
    link_file_names.sort()

    accum_size = 0

    progress_bar = tqdm(total=num_gb)
    with open(target_file, 'w+', encoding='utf-8') as f:
        flag = False
        for link in link_file_names:
            raw_file_path = os.path.join(
                data_dir,
                os.readlink(os.path.join(data_dir, link))
            )

            try:
                raw_data = pq.read_table(raw_file_path)['content']
            except Exception:
                tqdm.write(f"[!] Parquet {os.path.join(data_dir, link)} is broken, skip.")

            for single_data in tqdm(raw_data, total=len(raw_data), desc=f"Processing {link}"):
                text = str(single_data)
                f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
                len_text = len(text) + 1
                accum_size += len_text

                if accum_size >= num_gb * GB:
                    flag = True
                    break

                progress_bar.update(len_text / GB)

            if flag is True:
                break


if __name__ == '__main__':
    """For datasets with symlink."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help=f"Directory containing the symbolic links to the parquet files of tiiuae/falcon-refinedweb dataset"
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
    reformat(args.data_dir, args.target_file, args.num_gb)
