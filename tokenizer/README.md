# Llama-2 Compatible Tokenizer based on InternLM Tokenizer

Our motivation is to build a tokenizer with high compression rate on Chinese, English, and Code data, and cover 100%  Chinese characters.

## Analyze Exist Tokenizers

First, we should analyze existing open-source tokenizers for our judgments from compression rate and Chinese character
coverage. More details could be found in [README.md](../README.md).

|     Tokenizer      | FCC | SCC | TCC | Byte per Token $\uparrow$ | Comparied Compression $\downarrow$ |
|:------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:----------------:|
| [Chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)     |   99.97%    |   57.47%    |    2.99%    | 4.2911      |     0.5303     | 
| [Chatglm2/3-6b](https://huggingface.co/THUDM/chatglm2-6b)    |   **100.00%**  |   77.83%    |   13.89%    | 4.0329      |     0.5642     |  
| [Baichuan2-7b/14b](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)  |   **100.00%**   |    99.8%    |   86.48%    | 4.1827      |     0.5440     |   
| [Qwen-7b/14b/72b](https://huggingface.co/Qwen/Qwen-7B)   |   **100.00%**   |   **100.00%**   |   **100.00%**   | 4.1326      |     0.5506     |   
| [Llama-2-7b/13b/70b](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |   17.29%    |    0.13%    |    0.00%    | 2.2755      |        1.00         |  
| [Internlm-7b/20b](https://huggingface.co/internlm/internlm-7b)   |   **100.00%**   |   65.93%    |    5.67%    | **4.3133**      |     **0.5276**     | 

It could be found that although InternLM tokenizer doesn't cover 100% Chinese characters, its performance on compression
rate is very impressive. Thus, we decide to supplement other Chinese characters into it.

## Train Our Tokenizer

First, we want to train our tokenizers that cover 100% Chinese characters for the subsequent merging process.
Here, we train our tokenizer on the subset of following datasets.
More details about how to train the tokenizer could be found in [this repo](https://github.com/tiny-llm/Train-Tokenizer/tree/analyze).

| Dataset                                                                   | Split   | Domain  |
| ------------------------------------------------------------------------- | ------- | --------------- | 
| [ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText) | Chinese        | Chinese |
| [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)    | English              | English | 
| [Pile-arXiv](https://huggingface.co/datasets/EleutherAI/pile)             | English               | English | 
| [Pile-Wiki](https://huggingface.co/datasets/EleutherAI/pile)              | English               | English | 
| [Pile-GitHub](https://huggingface.co/datasets/EleutherAI/pile)            | Code                | Coding  |

The training corpus are uploaded in [this link](https://huggingface.co/datasets/DataHammer/HammerLLM-Tokenizer-Train-Set), 
and the parameters of sentencepiece training are shown as follows:

|          Parameters           |                Value                 | 
|:-----------------------------:|:------------------------------------:| 
|          vocab size           |                 128k                 |
|      character_coverage       |                0.9995                | 
|         byte_fallback         |                 true                 | 
|         split_digits          |                 true                 |
|          model_type           |                 BPE                  |
|            bos_id             |                  1                   | 
|            eos_id             |                  2                   | 
|            pad_id             |                  3                   |
|      max_sentence_length      |                 8384                 | 
|   max_sentencepiece_length    |                  32                  |
|          num_threads          |                 120                  |
| allow_whitespace_only_pieces  |                 true                 | 
|    shuffle_input_sentence     |                 true                 |
|         input_format          |                 text                 | 
|   user_defined_symbols_file   | ./data/user_defined_symbol_files.txt |
|      required_chars_file      |     ./data/common_char_file.txt      |

After training, the tokenizer [llama_zh_1b_128000_1g_0.9995](./models/llama_zh_1b_128000_1g_0.9995) could be obtained.

## Calibration

In this step, we calibrate tokens in our trained tokenizer that not appear in original InternLM tokenizer.
Please refer to the `calibrate()` function in [merge.py](merge.py).

It can be found that the calibration results are stable, and the ratio could be chosen as the average
0.9388603563642239.

```
一 llama 1b score -119447.0 internlm score -111752.0 [ratio]internlm score/llama 1b score 0.935578122514588
乙 llama 1b score -121046.0 internlm score -113612.0 [ratio]internlm score/llama 1b score 0.9385853311964046
二 llama 1b score -119646.0 internlm score -111956.0 [ratio]internlm score/llama 1b score 0.935727061498086
十 llama 1b score -119607.0 internlm score -112005.0 [ratio]internlm score/llama 1b score 0.9364418470490857
丁 llama 1b score -120906.0 internlm score -112939.0 [ratio]internlm score/llama 1b score 0.9341058342844855
厂 llama 1b score -120285.0 internlm score -112929.0 [ratio]internlm score/llama 1b score 0.9388452425489463
...
镶 llama 1b score -122607.0 internlm score -115045.0 [ratio]internlm score/llama 1b score 0.9383232604989927
瓤 llama 1b score -124426.0 internlm score -116774.0 [ratio]internlm score/llama 1b score 0.9385015993441885
罐 llama 1b score -121588.0 internlm score -114131.0 [ratio]internlm score/llama 1b score 0.9386699345330132
矗 llama 1b score -123712.0 internlm score -117550.0 [ratio]internlm score/llama 1b score 0.9501907656492499
ratio max 0.9607368812802382 min 0.9241164921465969 avg 0.9388603563642239
```

User defined symbols don't need to be calibrated, but should be set as `score=0.0, type=4`

## Merge

Now, we want to merge **tokens and user defined symbols** in our
trained [llama_zh_1b_128000_1g_0.9995](./models/llama_zh_1b_128000_1g_0.9995)
tokenizer that are not covered by InternLM tokenizer into InternLM tokenizer.
Besides, the final format of tokenizer should be in Llama-2 format.

Please refer to function `merge_llama1b_to_internlm()` in [merge.py](merge.py) for more details.

## Test

The test results are shown as follows and you can find more details in to function `tokenizer_test()`
in [merge.py](merge.py).

```
pad None None
bos <s> 1
eos </s> 2
unk <unk> 0
internlm+llama1b ['白', '日', '依', '山', '尽', '，', '黄河', '入', '海', '流', '。', '\n', '欲', '穷', '千里', '目', '，', '更', '上一层', '楼', '。']
internlm ['白', '日', '依', '山', '尽', '，', '黄河', '入', '海', '流', '。', '\n', '欲', '穷', '千里', '目', '，', '更', '上一层', '楼', '。']
llama1b ['▁白', '日', '依山', '尽', ',', '黄河', '入海', '流', '。', '▁欲', '穷', '千里', '目', ',', '更上一层楼', '。']
llama ['▁', '白', '日', '<0xE4>', '<0xBE>', '<0x9D>', '山', '<0xE5>', '<0xB0>', '<0xBD>', '，', '黄', '河', '入', '海', '流', '。', '<0x0A>', '<0xE6>', '<0xAC>', '<0xB2>', '<0xE7>', '<0xA9>', '<0xB7>', '千', '里', '目', '，', '更', '上', '一', '<0xE5>', '<0xB1>', '<0x82>', '<0xE6>', '<0xA5>', '<0xBC>', '。']
internlm+llama1b ['伢', '子']
internlm ['<0xE4>', '<0xBC>', '<0xA2>', '子']
llama1b ['▁', '伢', '子']
llama ['▁', '<0xE4>', '<0xBC>', '<0xA2>', '子']
internlm+llama1b ['<h5>', '1', '2', '3', '4', '5', '</h5>']
internlm ['<h', '5', '>', '123', '45', '</', 'h', '5', '>']
llama1b ['▁', '<h5>', '1', '2', '3', '4', '5', '</h5>']
llama ['▁<', 'h', '5', '>', '1', '2', '3', '4', '5', '</', 'h', '5', '>']
```

From these results, we can summarize following conclusions:

1. If the tokens are in original InternLM tokenizer, its tokenization results are the same as the InternLM tokenizer.
2. If the tokens are not in original InternLM tokenizer, the merged tokenizer could tokenize it in the expected way.
3. The user defined symbols, like the numbers, could be split separately, which is the same as our trained tokenizer.

The final merged tokenizer is in [here](./models/internlm_merged_fast).
