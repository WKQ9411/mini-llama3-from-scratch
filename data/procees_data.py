# 使用 https://github.com/DLLXW/baby-llama2-chinese 的经过预处理的语料
# 其预训练语料进行了提前分词，对一个样本做完分词后在末尾加上一个结束符号<eos>
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 添加上一级目录以导入tokenizer和config
from config import get_config


cfg = get_config()

# 将data文件夹下所有.bin数合并
def find_bin_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".bin")]

# 将所有.bin文件进行拼接, 在 ./data/pretrain_data 下生成 pretrain_data.bin 文件
def merge_bin_files(data_path_list):
    data_lst=[]
    for data_path in tqdm(data_path_list):
        with open(data_path,'rb') as f:
            data = np.fromfile(f, dtype=np.uint16)
            data_lst.append(data)
    arr = np.concatenate(data_lst)
    print(arr.shape)
    with open('./pretrain_data/pretrain_data.bin','wb') as f:
        f.write(arr.tobytes())

# sft数据处理
def sft_process():
    with open('./sft_data/alpaca_gpt4_data_zh.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    q_lst = []
    a_lst = []
    for per in data:
        # sft数据包括instruction、input、output三个部分
        q = per['instruction']
        i = per['input']
        a = per['output']
        q = q + i  # 将instruction和input拼接起来，一起作为prompt

        # 过滤长度不合适的数据
        if len(q) < 10 or len(a) < 5:
            continue
        if len(q)+10 > cfg.MODEL.MAX_SEQ_LEN//2 or len(a)+10 > cfg.MODEL.MAX_SEQ_LEN//2:
            # sft输入的数据构造为[prompt]+[bos]+[answer]+[eos]，这个序列不能超过max_seq_len
            # 如果max_seq_len为256，则输入和输出分别不超过128
            # 其中输入:[prompt]+[bos]，输出:[answer]+[eos]
            # 如果修改了max_seq_len，则需要重新处理以获得合适的数据
            # 这里q, a是文本长度，tokenizer后，token的长度有可能变得更长，因此这里留一定的余量，使q和a加10后不大于max_seq_len//2
            continue

        q_lst.append(q)
        a_lst.append(a)

    f = open('./sft_data/Belle_open_source_1M.json', 'r', encoding='utf-8')
    while True:
        line = f.readline()
        if not line:
            break
        per = json.loads(line)
        q = per['instruction']
        i = per['input']
        a = per['output']
        q = q + i
        if len(q) < 10 or len(a) < 5:
            continue
        if len(q)+10 > cfg.MODEL.MAX_SEQ_LEN//2 or len(a)+10 > cfg.MODEL.MAX_SEQ_LEN//2:
            continue
        q_lst.append(q)
        a_lst.append(a)
    df = pd.DataFrame(columns=['prompt', 'answer'])
    df['prompt'] = q_lst
    df['answer'] = a_lst
    df.to_csv('./sft_data/sft_data.csv', index=False)


if __name__ == "__main__":
    # 预训练数据处理，根据需要执行
    # ---------------------------------------------------------------------
    # # 预训练语料使用baidubaike、medical、wiki三种数据集，总共约7GB
    # # 找出当前文件夹下所有.bin文件
    # data_directory = "."
    # data_path_list = find_bin_files(data_directory)
    # print(data_path_list)

    # # 将所有.bin文件进行拼接, 在 ./data/pretrain_data 下生成 pretrain_data.bin 文件
    # # 已有或无需更新 pretrain_data.bin 文件, 则无需执行此代码块
    # merge_bin_files(data_path_list)

    # # 查看数据
    # # 导入 tokenizer
    # from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer  # transformers==4.33.2
    # tokenizer=ChatGLMTokenizer(vocab_file='./../chatglm_tokenizer/tokenizer.model')

    # # 读取 pretrain_data.bin 数据集
    # with open('./pretrain_data/pretrain_data.bin','rb') as f:
    #     data = np.fromfile(f, dtype=np.uint16)

    # # 数据集大小
    # print('data shape:', data.shape)
    # print(f'约{(data.shape[0]/1e8):.2f}亿tokens')

    # # 取前10个token_ids
    # prompt_ids = data[:10]
    # print('token_ids:', prompt_ids)

    # # 将token_ids解码为文本
    # decoded_text = tokenizer.decode(prompt_ids)
    # print('text:', decoded_text)

    # # 获取词典大小
    # vocab_size = tokenizer.vocab_size
    # print("vocab_size:", vocab_size)

    # # 特殊字符id
    # print('special tokens:', tokenizer.special_tokens)

    # sft数据处理，根据需要执行
    # ---------------------------------------------------------------------
    # sft_process()

    # 查看数据
    sft_data = pd.read_csv('./sft_data/sft_data.csv')
    print(sft_data.shape)
    print(sft_data.head())

    # 检查有无超过长度的样本
    max_len = 0
    for i in tqdm(range(len(sft_data))):
        prompt = sft_data.iloc[i]['prompt']
        answer = sft_data.iloc[i]['answer']
        seq_len = len(prompt) + len(answer) + 2
        if seq_len > max_len:
            max_len = seq_len
        if len(prompt)+1 > cfg.MODEL.MAX_SEQ_LEN//2 or len(answer)+1 > cfg.MODEL.MAX_SEQ_LEN//2:
            print(i, prompt, answer)
    print('max_len:', max_len)
