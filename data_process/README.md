# Process Large Corpus for LLM Pretraining

## 🗂Data Statistics

| Dataset        | Size | Line      | Token (Billion) |
|----------------|------|-----------|-----------------|
| ChineseWebText | 563G | 195274620 | 142             |
| RefinedWeb     | 504G | 203780635 | 128             |
| Pile-arxiv     | 111G | 2382582   | 38              |
| Pile-wiki      | 46G  | 15815247  | 12              |
| Pile-github    | 98G  | 18080750  | 30              | 


## 📂Getting Started

### 1. Download datasets to local directory

e.g.

* Download `CASIA-LM/ChineseWebText` dataset with script:

```commandline
bash ./download_chinese.sh
```

* Download `tiiuae/falcon-refinedweb` dataset:

```commandline
pip install -U huggingface_hub
huggingface-cli download --resume-download --repo-type dataset --cache-dir <cache dir> tiiuae/falcon-refinedweb
```

### 2. Reformat raw corpus to line-by-line format (jsonl)

e.g.

```commandline
python reformat_chinese.py --data-dir <data dir> --num-gb 100 --target-file <path to target file>
```

### 3. Sample reformatted corpus for **tokenizer** training & validation

e.g.

```commandline
python sample_tokenizer_dataset.py \
    --src-file <raw corpus (line-by-line)> \
    --tgt-dir <path to sampled dataset> \
    --tgt-base-name <prefix of target file name> \
    --num-sample-train <number of samples for training> \
    --num-sample-valid <number of samples for validation>
```


### 4. Shuffle reformatted corpus for **LLM pretraining**

#### Step 1. Shuffle each subset with [terashuf](https://github.com/alexandres/terashuf)

```commandline
cat file1.jsonl file2.jsonl etc | TMPDIR=<path to tmp file> MEMORY=500 ./terashuf > shuffled.jsonl
```

#### Step 2. Combine each shuffled subset with weighted sampling

```commandline
python random_merge.py \
    --files <path to file1> <path to file2> etc \
    --num-lines <number of lines of file1> <number of lines of file2> etc \
    --target-file <path to target file>
```

#### Step 3. Shuffle the combined dataset with terashuf

```commandline
cat shuffled.jsonl | TMPDIR=<path to tmp file> MEMORY=500 ./terashuf > shuffled.jsonl
```

### [Optional] Count number of tokens of certain subset

#### Step 1. Tokenize

```commandline
python tokenize.py \
    --tokenizer <path to tokenizer> \
    --raw-file <path to jsonl file> \
    --output-dir <path to target dataset> \
    --num-workers <number of processed in dataset mapping (tokenization)>
```

#### Step 2. Counting

```commandline
python token_statistics.py --dataset-dir <path to target dataset>
```

### 5. Tokenize and group shuffled corpus
```commandline
python tokenize_and_group.py \
    --tokenizer <path to tokenizer> \
    --raw-file <path to jsonl file> \
    --block-size <number of tokens per block> \
    --output-dir <path to target dataset> \
    --num-workers <number of processed in dataset mapping (tokenization)>
```