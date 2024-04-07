<div align="center">
</div>

# HammerLLM🔨

中文 | [English](REPRODUCE.md)

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

### 📓Build Tokenizer

该研究的动机是构建一个分词器，该分词器在中文、英文和代码数据上具有更高的压缩率，并覆盖100%的汉字。鉴于InternLM分词器较高的压缩率，我们决定对[InternLM分词器](https://huggingface.co/internlm/internlm-7b)进行了以下改进：
* 补充100%的常见汉字以训练我们自己的分词器
* 根据InternLM分词器校准我们分词器的score
* 补充InternLM分词器中没有而我们训练的分词器中有的token，如一些常见汉字和用户定义的符号
* 转换成Llama format的分词器
您可以在[这里](./merge_tokenizer)找到更多关于我们如何构建分词器的详细信息。
我们的分词器位于[这里](https://github.com/Academic-Hammer/HammerLLM/tree/main/merge_tokenizer/internlm_merged_fast)，它包含`105789`个标记，实现了`100%`的汉字覆盖。

 
#### Experimental Results

为了揭示我们分词器的有效性，我们使用以下指标将其与一些著名开源LLM的分词器进行比较：
1. **压缩率：**我们将分词器的两种压缩率与一些开源LLM的分词器进行比较：
   * [每标记的字节压缩率](https://kexue.fm/archives/9752#%E6%95%88%E6%9E%9C%E6%B5%8B%E8%AF%95)
   * [比较压缩](https://arxiv.org/pdf/2309.16609.pdf)，测量与基础Llama-2-7B分词器相比的优势。
   
   请参考这个[pthon脚本](tokenizer/analyze_tokenizer/compression_analyze.py)获取更多细节。
   我们在中文、英文和代码测试集上评估分词器以计算压缩率，数据来源如下：
   * 中文：[Skywork/ChineseDomainModelingEval](https://huggingface.co/datasets/Skywork/ChineseDomainModelingEval)
   * 英文：[EleutherAI/pile](https://huggingface.co/datasets/EleutherAI/pile)的测试集
   * 代码：来自[Pile-GitHub](https://huggingface.co/datasets/EleutherAI/pile)的分割
   
   这些测试数据在[这个链接](https://huggingface.co/datasets/DataHammer/Tokenizer-Test-Set/tree/main)上公开可用。
2. **汉字覆盖率：**一个好的中文LLM应该覆盖更多的汉字。在我们的工作中，我们使用[vocab-coverage](https://github.com/twang2218/vocab-coverage)来计算汉字的覆盖率，包括：
     * **一级汉字覆盖率**（FCC）包含3500个广泛使用的汉字。
     * **二级汉字覆盖率**（SCC）包含3000个汉字。
     * **三级汉字覆盖率**（TCC）包含1605个不常见的汉字。
     
为了获得实验结果，请运行以下命令：

```shell
cd analyze_tokenizer
# compute the compression rate
python compression_analyze.py
# compute the Chinese character coverage
python coverage_analyze.py
```

实验结果如下所示：

|     Tokenizer      | FCC | SCC | TCC | Byte per Token $\uparrow$ | Comparied Compression $\downarrow$ |
|:------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:----------------:|
| [Chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)     |   99.97%    |   57.47%    |    2.99%    | 4.2911      |     0.5303     | 
| [Chatglm2/3-6b](https://huggingface.co/THUDM/chatglm2-6b)    |   **100.00%**  |   77.83%    |   13.89%    | 4.0329      |     0.5642     |  
| [Baichuan2-7b/14b](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)  |   **100.00%**   |    99.8%    |   86.48%    | 4.1827      |     0.5440     | 
| [Internlm-7b/20b](https://huggingface.co/internlm/internlm-7b)   |   **100.00%**   |   65.93%    |    5.67%    | 4.3133      |     0.5276     |   
| [Qwen-7b/14b/72b](https://huggingface.co/Qwen/Qwen-7B)   |   **100.00%**   |   **100.00%**   |   **100.00%**   | 4.1326      |     0.5506     |   
| [Llama-2-7b/13b/70b](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |   17.29%    |    0.13%    |    0.00%    | 2.2755      |        1.00         |  
| [Ours](https://github.com/Academic-Hammer/HammerLLM/tree/main/merge_tokenizer/internlm_merged_fast)   |    **100.00%**     |   **100.00%**    |    **100.00%**    |   **4.3143**   |     **0.5274**             | 

实验结果揭示了我们的分词器在压缩率（对中文、英文和代码数据）和汉字覆盖率方面，相较于现有流行LLM的分词器的优势。

### ⚙️Train Model

#### ⚒Architecture

我们的模型采用了基础的Llama架构，以下是关键参数：


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

#### ⚙Training Hyper-parameter

我们的训练过程经过精心规划和执行，以确保模型的健壮性和灵活性：

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

我们修改了已有的learning scheduler，更多细节可以在[这里](https://github.com/huggingface/transformers/issues/28441)找到。


#### 🗂Training Corpus

我们的模型在精心挑选的**中文、英文和编程**数据集上进行训练，旨在培养广泛的中英文语言理解能力：

| Dataset                                                                   | Split   | Token (Billion) | Domain  |
| ------------------------------------------------------------------------- | ------- | --------------- | ------- | 
| [ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText) | Chinese | 142             | Chinese |
| [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)    | English | 128             | English | 
| [Pile-arXiv](https://huggingface.co/datasets/EleutherAI/pile)             | English | 38              | English | 
| [Pile-Wiki](https://huggingface.co/datasets/EleutherAI/pile)              | English | 12              | English | 
| [Pile-GitHub](https://huggingface.co/datasets/EleutherAI/pile)            | Code    | 30              | Coding  |

#### ⏩Acceleration Strategy

在以下两种策略的帮助下，我们能够实现per second per GPU **16k token**的高吞吐量进行训练：
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
- [torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)

计算吞吐量的设置如下：
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

为了覆盖更多的汉字，我们的分词器比[TinyLlama](https://github.com/jzhang38/TinyLlama)大得多（105789 > 32000），导致吞吐量低于TinyLlama（16k < 24k）。

然而，当分词器大小相同时，我们的吞吐量与TinyLlama相当（为此将per_device_train_batch_size设置为20）。


| Settings                          | tokens per GPU per second |
|-----------------------------------|---------------------------|
| Flash Attention 2 + torch.compile | 24k                       |

与利用一些复杂操作融合的TinyLlama不同，我们只通过结合`torch.compile`和`flash attention 2`即可实现了这一吞吐量。


#### 🖥Hardware Requirement

我们的预训练是在配备1TB CPU内存的8x80G A100服务器上运行的。
您可以使用较少的GPU卡和内存，通过减小批量大小来适配您的硬件条件。

#### Pretrain

##### Environment Setup

我们已经为预训练准备了Docker环境，该环境已经集成了`flash attention 2`和`torch.compile`以实现高效的预训练。Docker文件位于[这里](./Dockerfile)。


##### Data Preparation

请参考我们的[数据准备指南](./data_process/README.md)，了解更多关于我们如何预处理预训练数据集的细节，例如数据集的重新格式化和充分的洗牌。


##### Start Training ([train.sh](./pretrain/train.sh))

一旦您准备好了运行环境和数据，您可以通过以下方式启动预训练：

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

上述shell代码是用于预训练的启动脚本，关于我们预训练代码库的详细信息可以在[这里](https://github.com/tiny-llm/TrainerCodeBase/blob/main/readme.md)找到。


#### Convert Checkpoints

checkpoint中包含由`torch.compile`引入的一些不寻常的key name，这会导致模型参数加载失败。我们已经在[这里](pretrain/convert_checkpoint.py)提供了转换脚本，对这些key name进行校准。


```shell
python convert_checkpoint.py --input-path <path of saved checkpoint> --output-path <path of converted transformers checkpoint>
```

#### Inference

##### Example Code
如下代码展示了如何使用**HuggingFace** `transformers`与我们的模型进行交互：


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
