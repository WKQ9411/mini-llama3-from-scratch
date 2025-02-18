import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


# 定义预训练数据集
class PretrainDataset(Dataset):
    def __init__(self, data_path, max_length, memmap=False):
        super().__init__()

        # 由于后续要构造的输入和预测下一个字符的输出长度均为 max_length，这里需要截取 max_length + 1 长度的数据来构造
        max_length = max_length + 1

        # memmap定义是否使用内存映射加载数据
        # 通过内存映射，程序可以像访问内存一样访问文件，而不需要将整个文件加载到内存中。
        # 这种方式特别适合处理大文件，因为它可以避免一次性加载大量数据导致的内存不足问题。
        if memmap:
            with open(data_path, 'r') as f:  # 'r' 模式：用于获取文件大小，不涉及文件内容的读取。
                # seek 是文件对象的方法，用于移动文件指针到指定位置， 这里将文件指针移动到文件末尾
                nbytes = f.seek(0,2)

                # tell 是文件对象的方法，用于返回当前文件指针的位置（以字节为单位）
                # np.dtype('uint16').itemsize 表示一个 uint16 类型的数据占用的字节数
                # 因此 flen 表示存储文件中 uint16 类型元素的总数，即 token_ids 的总数
                flen = f.tell() // np.dtype('uint16').itemsize

            # 创建内存映射数组，flen // max_length：样本的数量，max_length：每个样本的长度
            self.data = np.memmap(data_path,
                                  dtype=np.dtype('uint16'),
                                  shape=(flen//max_length,max_length))
            
        else:
            with open(data_path,'rb') as f:
                data = np.fromfile(f, dtype=np.uint16)
            data = data[:max_length*int(len(data)/max_length)]
            self.data = data.reshape(-1, max_length)
        
        # 创建形状为(num_samples, max_length)的self.data数据集
        print(f"pretrain data shape: {self.data.shape}")
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index: int):
        sample = self.data[index]
        # pytorch 默认使用 int64 或 float32 类型进行计算
        # 这里取X和Y用于预测下一个字符
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)
        
        return torch.from_numpy(X), torch.from_numpy(Y)  # 转换为tensor

# 定义sft数据集
class SFTDataset(Dataset):
    def __init__(self, df_data, tokenizer, max_length):
        # max_length为prompt和answer的长度之和
        super().__init__()
        self.df = df_data
        self.max_length = max_length
        self.prompt_max_len = max_length // 2  # 最大序列长度的一半用于容纳prompt
        self.answer_max_len = max_length // 2  # 最大序列长度的一半用于容纳answer

        self.tokenizer = tokenizer
        self.bos = self.tokenizer.special_tokens['<bos>']
        self.eos = self.tokenizer.special_tokens['<eos>']
        self.pad = self.tokenizer.special_tokens['<pad>']
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        # prompt和answer之间一定要有一个开始符<bos>隔开，然后answer后需要一个结束符<eos>
        # 【sft训练时，相当于让模型对prompt+<bos>+answer+<eos>这个序列执行预测下一个字符，这个序列长度短于max_seq_len】
        # 【计算loss的时候，对prompt+<bos>部分的loss进行mask，只计算answer部分的loss】
        # <bos>的主要作用是划分输入的结构边界，而非需要生成的内容，它的存在为模型提供了明确的上下文分段信号
        # 模型需基于完整输入（包括 <bos>）预测后续 token，间接学习到 <bos> 的引导作用
        # <bos>只是定义方式的一种，这里可以命名为任何特殊字符，例如<sep>等
        sample = self.df.iloc[index]  # 取出第index个样本
        prompt = self.tokenizer.encode(sample['prompt'], add_special_tokens=False)
        answer = self.tokenizer.encode(sample['answer'], add_special_tokens=False)
        
        input_id = prompt + [self.bos] + answer + [self.eos]  # 构造序列
        input_len = len(prompt + [self.bos])  # prompt+<bos>的长度
        output_len = len(answer + [self.eos])  # answer+<eos>的长度
        pad_len = self.max_length + 1 - len(input_id)  # 填充pad的长度，+1是为了确保后续[:-1]和[1:]截取出max_seq_len长度
        input_id = input_id + [self.pad] * pad_len
        # loss_mask用来计算loss，1表示需要计算loss，0表示不需要计算loss，[prompt]、[bos]和[pad]部分不计算loss
        # [answer]+[eos]需要计算loss
        loss_mask = [0] * input_len + [1] * output_len + [0] * pad_len
        
        input_id = np.array(input_id)
        # 如果input_id的长度大于了max_seq_len+1，则截取后max_seq_len+1的长度
        if len(input_id) > self.max_length + 1:
            input_id = input_id[-(self.max_length + 1):]
            loss_mask = loss_mask[-(self.max_length + 1):]
        assert len(input_id) == len(loss_mask) and len(input_id) == self.max_length + 1

        X = np.array(input_id[:-1]).astype(np.int64)  # 构造输入
        Y = np.array(input_id[1:]).astype(np.int64)  # 构造输出标签
        loss_mask = np.array(loss_mask[1:])
        
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
