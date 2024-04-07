#!/usr/bin/env python
# encoding: utf-8
"""
File Description:
Created Time: 2024/1/2
"""
import json
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2
from transformers import LlamaTokenizer, AutoTokenizer


def load_sp_model(sp_model_file):
    """
    :param sp_model_file:
    :return:
    """
    p = spm.SentencePieceProcessor()
    p.Load(sp_model_file)
    # https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
    model = sentencepiece_model_pb2.ModelProto()
    model.ParseFromString(p.serialized_model_proto())
    return model


def calibrate():
    """
    标定
    :return:
    """
    calibrate_tokens = json.load(open('./analyze/vocab_coverage/charsets_char.json'))['《通用规范汉字表》一级汉字']['texts']
    llama_1b_sp = load_sp_model('./models/llama_zh_1b_128000_1g_0.9995/tokenizer.model')
    llama_1b_calibrate = {
        piece.piece: piece.score
        for piece in llama_1b_sp.pieces
        if piece.piece in calibrate_tokens
    }
    internlm_sp = load_sp_model('./models/internlm-7b/tokenizer.model')
    internlm_calibrate = {
        piece.piece: piece.score
        for piece in internlm_sp.pieces
        if piece.piece in calibrate_tokens
    }
    ratios = []
    for token in calibrate_tokens:
        print(token,
              'llama 1b score', llama_1b_calibrate[token],
              'internlm score', internlm_calibrate[token],
              '[ratio]internlm score/llama 1b score', internlm_calibrate[token] / llama_1b_calibrate[token],
              )
        ratios.append(internlm_calibrate[token] / llama_1b_calibrate[token])
    print('ratio', 'max', max(ratios), 'min', min(ratios), 'avg', sum(ratios) / len(ratios))


def merge_llama1b_to_internlm():
    """
    合并llama1b和internlm
    """
    # https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_tokenizer/merge_tokenizers.py
    ratio = 0.9388603563642239
    chars = json.load(open('./analyze/vocab_coverage/charsets_char.json'))
    required_chars = []
    required_chars.extend(chars['《通用规范汉字表》一级汉字']['texts'])
    required_chars.extend(chars['《通用规范汉字表》二级汉字']['texts'])
    required_chars.extend(chars['《通用规范汉字表》三级汉字']['texts'])
    # 从llama1b中提取char和score
    required_char_set = set(required_chars)
    llama_1b_token_to_score = dict()
    llama_1b_sp = load_sp_model('./models/llama_zh_1b_128000_1g_0.9995/tokenizer.model')
    for piece in llama_1b_sp.pieces:
        if piece.piece in required_char_set:
            llama_1b_token_to_score[piece.piece] = piece.score
    # 补充internlm没有覆盖的字符
    internlm_sp = load_sp_model('./models/internlm-7b/tokenizer.model')
    internlm_token_set = set(p.piece for p in internlm_sp.pieces)
    print(f"internlm origin tokens,{len(internlm_sp.pieces)}")
    for token in required_chars:
        if token not in internlm_token_set:
            new_p = sentencepiece_model_pb2.ModelProto().SentencePiece()
            new_p.piece = token
            new_p.score = llama_1b_token_to_score[token] * ratio
            internlm_sp.pieces.append(new_p)
            print(f'add token {token}')
    # 补充用户自定义符号
    with open('./data/user_defined_symbol_file.txt', 'rt', encoding='utf-8') as f:
        user_defined_tokens = f.read().splitlines()
    for token in user_defined_tokens:
        if token not in internlm_token_set:
            new_p = sentencepiece_model_pb2.ModelProto().SentencePiece()
            new_p.piece = token
            new_p.score = 0.0
            new_p.type = 4
            internlm_sp.pieces.append(new_p)
            print(f'add token {token}')
        else:
            # 遍历并找到设置socre和type
            for piece in internlm_sp.pieces:
                if piece.piece == token:
                    piece.score = 0.0
                    piece.type = 4
                    break
    print(f"internlm new tokens: {len(internlm_sp.pieces)}")
    output_sp_model = 'merged.model'
    with open(output_sp_model, 'wb') as f:
        f.write(internlm_sp.SerializeToString())
    tokenizer = LlamaTokenizer(vocab_file=output_sp_model)
    tokenizer.save_pretrained('./models/internlm_merged')


def tokenizer_test():
    new_tokenizer = LlamaTokenizer.from_pretrained('./models/internlm_merged')
    print('pad', new_tokenizer.pad_token, new_tokenizer.pad_token_id)
    print('bos', new_tokenizer.bos_token, new_tokenizer.bos_token_id)
    print('eos', new_tokenizer.eos_token, new_tokenizer.eos_token_id)
    print('unk', new_tokenizer.unk_token, new_tokenizer.unk_token_id)
    llama_1b_tokenizer = LlamaTokenizer.from_pretrained('./models/llama_zh_1b_128000_1g_0.9995')
    llama_tokenizer = AutoTokenizer.from_pretrained('./models/llama2-7b')
    internlm_tokenizer = AutoTokenizer.from_pretrained('./models/internlm-7b', trust_remote_code=True)
    text = '''白日依山尽，黄河入海流。\n欲穷千里目，更上一层楼。'''
    print('internlm+llama1b', new_tokenizer.tokenize(text))
    print('internlm', internlm_tokenizer.tokenize(text))
    print('llama1b', llama_1b_tokenizer.tokenize(text))
    print('llama', llama_tokenizer.tokenize(text))
    text = '伢子'
    print('internlm+llama1b', new_tokenizer.tokenize(text))
    print('internlm', internlm_tokenizer.tokenize(text))
    print('llama1b', llama_1b_tokenizer.tokenize(text))
    print('llama', llama_tokenizer.tokenize(text))
    text = '<h5>12345</h5>'
    print('internlm+llama1b', new_tokenizer.tokenize(text))
    print('internlm', internlm_tokenizer.tokenize(text))
    print('llama1b', llama_1b_tokenizer.tokenize(text))
    print('llama', llama_tokenizer.tokenize(text))


if __name__ == '__main__':
    # merge_llama1b_to_internlm()
    tokenizer_test()
