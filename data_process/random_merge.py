import random
import argparse
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    """Merge multiple line-by-line files randomly."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', required=True)
    parser.add_argument('--num-lines', type=int, nargs='+', required=True)
    parser.add_argument('--target-file', type=str, required=True)
    args = parser.parse_args()

    random.seed(0)
    assert len(args.files) >= 2, f"Please provide at least 2 files to combine."
    assert len(args.num_lines) == len(args.files), (f"Argument `num-lines` should be `None` or a list sharing the same "
                                                    f"length with `files`")

    file_list = [{
        "file_name": file_name,
        "file_iter": open(file_name, 'r+', encoding='utf-8'),
        "num_lines": num_lines,
        "num_lines_left": num_lines,
        "upper_bound": None,
        "progress_bar": tqdm(total=num_lines, desc=f"Processing {file_name}")
    } for i, (file_name, num_lines) in enumerate(zip(args.files, args.num_lines))]

    with open(args.target_file, 'w+') as f:
        complete = False
        reorganize_weights = True
        while not complete:
            if reorganize_weights:
                weights = np.array([file['num_lines'] for file in file_list])
                upper_bound = np.cumsum(weights / np.sum(weights))
                for i in range(len(file_list)):
                    file_list[i]['upper_bound'] = upper_bound[i]
                reorganize_weights = False

            rand_num = random.random()
            complete_file_id = None
            for i, file in enumerate(file_list):
                if rand_num < file['upper_bound']:
                    line = file['file_iter'].readline()
                    f.write(line)

                    file['progress_bar'].update(1)

                    file['num_lines_left'] -= 1
                    if file['num_lines_left'] == 0:
                        complete_file_id = i
                        file['file_iter'].close()
                        file['progress_bar'].set_description(f"File {file['file_name']} reached EOF")

                    break

            if complete_file_id is not None:
                file_list.pop(complete_file_id)
                reorganize_weights = True
                if len(file_list) == 0:
                    complete = True

    print(f"[!] Success: Random merging completed.")
