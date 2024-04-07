import os
import torch
import argparse
import safetensors
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(args.input_path, use_fast=False)
    tokenizer.pad_token_id = 0
    state_dict = safetensors.torch.load_file(os.path.join(args.input_path, "model.safetensors"), device="cpu")
    config = LlamaConfig.from_pretrained(args.input_path)
    model = LlamaForCausalLM(config=config).cuda().to(torch.bfloat16)

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = '.'.join(key.split('.')[1:])
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, True)

    prompt = '罗马尼亚尚存古老民族 生活与世隔绝落后几百年\n在欧洲大陆上，有这样一个与世隔绝的地方：没有上网冲浪，没有太空探索，更没有什么遗传工程；'
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    output = model.generate(
        input_ids=input_ids.cuda(),
        max_length=min(int(len(input_ids) + 100), 2048),
        early_stopping=False,
        do_sample=True,
        top_p=0.95
    ).tolist()

    generation = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generation)

    tokenizer.save_pretrained(save_directory=args.output_path)
    model.save_pretrained(save_directory=args.output_path)
    print(f"[!] converted checkpoint saved at {args.output_path}")
