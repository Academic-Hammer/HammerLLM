<div align="center">

</div>

# HammerLLM🔨

English | [中文](REPRODUCE_zh-CN.md)

## Table of Contents

- [HammerLLM🔨](#hammerllm)
  - [Table of Contents](#table-of-contents)
  - [✨Getting Start](#getting-start)
    - [📓Build Tokenizer](#build-tokenizer)
      - [Experimental Results](#experimental-results)
    - [⚙️Train Model](#️train-model)
      - [⚒Architecture](#architecture)
      - [⚙Training](#training)
      - [🗂Training Corpus](#training-corpus)
      - [⏩Acceleration Strategy](#acceleration-strategy)
      - [🖥Hardware Requirement](#hardware-requirement)
      - [Pretrain](#pretrain)
        - [Environment Setup](#environment-setup)
        - [Data Preparation](#data-preparation)
        - [Start Training (train.sh)](#start-training-trainsh)
      - [Convert Checkpoints](#convert-checkpoints)
      - [Inference](#inference)
      - [Example Code](#example-code)

## ✨Getting Start

This section covers the architecture and training details of the model.

### 📓Build Tokenizer

The motivation is to build a tokenizer that has higher compression rate on Chinese, English, Code data, and covers 100% Chinese characters.
Since InternLM tokenizer has the best tokenization compression rate on our test set, we modify the [InternLM tokenizer](https://huggingface.co/internlm/internlm-7b) with following improvements:
* Train our own tokenizer with 100% Chinese common characters 
* Calibrate the scores of our tokenizers based on InternLM tokenizer
* Supplement tokens in our trained tokenizer but not in InternLM tokenizer, like some Chinese common characters and user defined symbols
* Convert it into Llama tokenizer format

You can find more details in [here](./tokenizer) about how we build our tokenizer. Our tokenizer is in [here](./tokenizer/models/internlm_merged_fast), which has `105789` tokens, with `100%` Chinese character coverage.
 
#### Experimental Results

To reveal the effectiveness of our tokenizer, we compare it with some famous open-source LLMs' tokenizers by using following metrics:
1. **Compression Rate:** We compare two kinds of compression rate of tokenizer with tokenizers of some open-source LLMs:
   * [Byte per token compression rate](https://kexue.fm/archives/9752#%E6%95%88%E6%9E%9C%E6%B5%8B%E8%AF%95) 
   * [Compared compression](https://arxiv.org/pdf/2309.16609.pdf) that measures the advantage over base Llama-2-7B tokenizer. 
   
    Please refer to this [python script](./tokenizer/analyze/analyze.py) for more details. 
    We evaluate tokenizers on Chinese, English, and Code test set for computing the compression rate, and the data source are listed as follows:
    * Chinese: [Skywork/ChineseDomainModelingEval](https://huggingface.co/datasets/Skywork/ChineseDomainModelingEval)
    * English: test set of [EleutherAI/pile](https://huggingface.co/datasets/EleutherAI/pile)
    * Code: split from [Pile-GitHub](https://huggingface.co/datasets/EleutherAI/pile)
  
    These test data are publicly available at [this link](https://huggingface.co/datasets/DataHammer/Tokenizer-Test-Set/tree/main).

2. **Chinese Character Coverage:** a good Chinese LLM should cover more characters. In our work, we leverage [vocab-coverage](https://github.com/twang2218/vocab-coverage) for computing the coverage of Chinese characters, which includes:
     * **F**irst-level **C**hinese character **C**overage (FCC) contains 3500 widely-used Chinese characters.
     * **S**econd-level **C**hinese character **C**overage (SCC) contains 3000 Chinese characters.
     * **T**hird-level **C**hinese character **C**overage (TCC) contains 1605 uncommon Chinese characters.
     
The experimental results are shown as follows:

|     Tokenizer      | FCC | SCC | TCC | Byte per Token $\uparrow$ | Comparied Compression $\downarrow$ |
|:------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:----------------:|
| [Chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)     |   99.97%    |   57.47%    |    2.99%    | 4.2911      |     0.5303     | 
| [Chatglm2/3-6b](https://huggingface.co/THUDM/chatglm2-6b)    |   **100.00%**  |   77.83%    |   13.89%    | 4.0329      |     0.5642     |  
| [Baichuan2-7b/14b](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)  |   **100.00%**   |    99.8%    |   86.48%    | 4.1827      |     0.5440     | 
| [Internlm-7b/20b](https://huggingface.co/internlm/internlm-7b)   |   **100.00%**   |   65.93%    |    5.67%    | 4.3133      |     0.5276     |   
| [Qwen-7b/14b/72b](https://huggingface.co/Qwen/Qwen-7B)   |   **100.00%**   |   **100.00%**   |   **100.00%**   | 4.1326      |     0.5506     |   
| [Llama-2-7b/13b/70b](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |   17.29%    |    0.13%    |    0.00%    | 2.2755      |        1.00         |  
| [Ours](https://github.com/Academic-Hammer/HammerLLM/tree/main/merge_tokenizer/internlm_merged_fast)   |    **100.00%**     |   **100.00%**    |    **100.00%**    |   **4.3143**   |     **0.5274**             | 

The experimental results reveal the advantages of our tokenizer over existing popular LLMs' tokenizers in compression rate (on Chinese, English, and Code), and the Chinese character coverage.

### ⚙️Train Model

#### ⚒Architecture

Our model leverages an advanced architecture designed to maximize efficiency and effectiveness in processing and generating text. Here are the key specifications:

| Setting             | Description             |
| ------------------- | ----------------------- |
| parameters          | 1.4B                    |
| attention           | Grouped-query Attention (GQA) |
| num layers          | 22                      |
| num attention heads | 32                      |
| query groups        | 4                       |
| hidden size         | 2048                    |
| intermediate size   | 5632                    |
| activation func     | SiLU                    |
| max sequence length | 2048                    |

#### ⚙Training Hyper-paramter

Our training process was meticulously planned and executed to ensure the model’s robustness and agility:

| Setting                     | Description |
| --------------------------- | ----------- |
| block size                  | 2048        |
| per-device batch size       | 8           |
| gradient accumulation steps | 16          |
| num device                  | 8           |
| total batch size            | 2M tokens   |
| max learning rate           | 5e-4        |
| min learning rate           | 5e-5        |
| warmup steps                | 2000        |
| learning rate schedule      | cosine      |

We have modified the learning rate schedule, and more details can be found [here](https://github.com/huggingface/transformers/issues/28441).

#### 🗂Training Corpus

Our model was trained on a meticulously curated selection of **Chinese, English and Coding** datasets, designed to foster wide-ranging language understanding:

| Dataset                                                                   | Split   | Token (Billion) | Domain  |
| ------------------------------------------------------------------------- | ------- | --------------- | ------- | 
| [ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText) | Chinese | 142             | Chinese |
| [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)    | English | 128             | English | 
| [Pile-arXiv](https://huggingface.co/datasets/EleutherAI/pile)             | English | 38              | English | 
| [Pile-Wiki](https://huggingface.co/datasets/EleutherAI/pile)              | English | 12              | English | 
| [Pile-GitHub](https://huggingface.co/datasets/EleutherAI/pile)            | Code    | 30              | Coding  |

#### ⏩Acceleration Strategy

With the help of these two stragegies, we could achieve a high throughput of **16k tokens per GPU per second** for training:
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
- [torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)

The setting for computing the throughput is:
* ZeRO-1
* block_size: 2048
* per_device_train_batch_size: 8
* gradient_accumulation_steps: 16

| Settings                          | tokens per GPU per second |
|-----------------------------------|---------------------------|
| None                              | CUDA OOM                  |
| Flash Attention 2                 | 13k                       |
| torch.compile                     | CUDA OOM                  |
| Flash Attention 2 + torch.compile | 16k                       |

To cover more Chinese characters, our tokenizer is much larger than [TinyLlama](https://github.com/jzhang38/TinyLlama) (105789 > 32000), leading to the lower throughput than TinyLlama (16k < 24k). 

However, our throughput is comparable with TinyLlama, when the size of the tokenizer is the same (per_device_train_batch_size is set to 20 for this).

| Settings                          | tokens per GPU per second |
|-----------------------------------|---------------------------|
| Flash Attention 2 + torch.compile | 24k                       |

Unlike TinyLlama that leverages some complex the operations fusion, we achieve this throughput solely based on `torch.compile` and `flash attention 2`.

#### 🖥Hardware Requirement

Our pre-training is running on the 8x80G A100 server with 1T CPU memory. 
You could train your model with less GPU cards and memory with smaller batch size.

#### Pretrain

##### Environment Setup

We have prepared the docker environment for pretraining, which has already incorporated the `flash attention 2` and `torch.compile` for efficient pre-training. The docker file is [here](./Dockerfile).

##### Data Preparation

Please refer to our [data preparation guide](./data_process/README.md) for more details about how we pre-process the pretraining dataset, such as the dataset reformat and sufficient shuffle.

##### Start Training ([train.sh](./pretrain/train.sh))

Once you have prepared the running environment and the data, you could launch the pre-training process by:

```shell
set -ex
export WANDB_PROJECT=hammerllm
BASE_DIR="$PWD"
DATE=$(TZ=Asia/Shanghai date +'%Y%m%d%H%M%S')
CONFIG_PATH=${BASE_DIR}/configs/hammerllm
RUN_NAME=hammerllm_torch_compile_flash_attn_2
OUTPUT_DIR=${BASE_DIR}/checkpoint/${RUN_NAME}

DATA_SEED=3407
MODEL_SEED=3407

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online

if [ ! -d ${OUTPUT_DIR} ]
then
  mkdir -p ${OUTPUT_DIR}
fi
echo "Setting checkpoint directory to ${OUTPUT_DIR}"

MASTER_PORT=$(shuf -n 1 -i 60000-65535)
torchrun --nproc_per_node=8 --master_port ${MASTER_PORT} train.py \
  --model_name_or_path ${CONFIG_PATH} \
  --use_flash_attention_2 \
  --use_torch_compile \
  --train_file /path/to/your/tokenized/train/dataset \
  --validation_files /path/to/your/tokenized/validation/dataset_1 /path/to/your/tokenized/validation/dataset_2 ... \
  --preprocessing_num_workers 100 \
  --block_size 2048 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 16 \
  --logging_steps 10 \
  --max_steps 1000000 \
  --warmup_steps 2000 \
  --eval_steps 500 \
  --save_steps 500 \
  --evaluation_strategy steps \
  --save_strategy steps \
  --greater_is_better false \
  --load_best_model_at_end false \
  --ddp_find_unused_parameters false \
  --remove_unused_columns false \
  --save_total_limit 50 \
  --learning_rate 5e-4 \
  --lr_scheduler_type cosine \
  --output_dir ${OUTPUT_DIR} \
  --report wandb \
  --run_name ${RUN_NAME} \
  --bf16 \
  --seed ${MODEL_SEED} \
  --data_seed ${DATA_SEED} \
  --deepspeed ${BASE_DIR}/configs/zero_1.json
```

This code is the launcing shell script for pre-training, the details of our pre-training codebase could be found in [here](https://github.com/tiny-llm/TrainerCodeBase/blob/main/readme.md).

#### Convert Checkpoints

The saved checkpoints contains some unusual key name introduced by `torch.compile`, leading to the failed loading of model parameters. 
We have provided the scripts in [here](pretrain/convert_checkpoint.py), which calibrate these key name.

```shell
python convert_checkpoint.py --input-path <path of saved checkpoint> --output-path <path of converted transformers checkpoint>
```

#### Inference

#### Example Code
Here is a code snippet to show you how to play with our model with **HuggingFace** `transformers`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'DataHammer/hammerllm-1.4b-222k'
text = '北京理工大学是'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# if your device donot support the bfloat16, you could remove it
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

input_ids = tokenizer(text, return_tensors='pt').input_ids
output = model.generate(
    input_ids=input_ids.cuda(),
    max_length=min(int(len(input_ids) + 100), 1024),
    do_sample=True,
    top_p=0.95
).tolist()

generation = tokenizer.decode(output[0])
print(generation)
```
