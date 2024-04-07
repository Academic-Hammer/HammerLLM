import argparse
import json
import os.path

if __name__ == '__main__':
    """Split pretraining test set splits from Pile test set (English and Code)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-file', type=str, default='/data1/aurora2/dev_test/tinyllm_test_set/pile_test/pile_test_100.jsonl')
    parser.add_argument('--target-dir', type=str, default='/data1/aurora2/pretraining_dataset')
    args = parser.parse_args()

    en_file = open(os.path.join(args.target_dir, 'test-en.jsonl'), 'w+', encoding='utf-8')
    code_file = open(os.path.join(args.target_dir, 'test-code.jsonl'), 'w+', encoding='utf-8')

    with open(args.raw_file, 'r+', encoding='utf-8') as f:
        for line in f:
            set_name = json.loads(line)['meta']['pile_set_name']
            if set_name in ['Github', 'Pile-CC']:
                code_file.write(line + '\n')
            else:
                en_file.write(line + '\n')
