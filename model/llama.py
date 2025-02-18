import torch
from torch import nn
from torch.nn import functional as F

import math
import inspect
import numpy as np
from yacs.config import CfgNode
from typing import Optional, Tuple


# ----------------------------------------------------------------------------
# RMSNorm
# ----------------------------------------------------------------------------
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 输入维度为(batch_size, seq_len, dim),对最后一个维度进行归一化
        # x.pow(2)用于对每个元素进行平方运算
        # torch.rsqrt()是计算倒数平方根
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# ----------------------------------------------------------------------------
# ROPE
# ----------------------------------------------------------------------------
# 预先计算旋转矩阵的各个角度
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """计算频率矩阵, 并将其表示为复数的极坐标表示, 函数名中的cis指cos(θ)+i·sin(θ), 表示一个复数位于单位圆上的位置

    Args:
        dim (int): Embedding的维度
        end (int): 序列长度
        theta (float, optional): 计算θ的底数值【θ=10000^(-2i/d)】. Defaults to 10000.0.

    Returns:
        代表各个位置m旋转角度的复数矩阵, 形状为(end, dim//2), 每两个维度对应一个旋转角度
    """
    # 计算旋转矩阵中的θ值, 原文中θ=10000^(-2i/d)【这里源代码[: (dim // 2)]的操作似乎是冗余的？】
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 计算位置信息m的序列
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)

    # torch.outer用于计算外积, 就得到不同位置m和不同θ值的所有组合m*θ
    # 得到的freqs矩阵形状为(end, dim//2), 索引含义为freqs[mi][θi]=mi*θi
    freqs = torch.outer(t, freqs)

    # 生成一个模长为1, 幅角为freqs的复数矩阵
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


# 调整freqs_cis以方便其与x进行广播计算
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """调整freqs_cis以方便其与x进行广播计算

    Args:
        freqs_cis (torch.Tensor): 旋转矩阵, 初始形状为(end, head_dim//2)
        x (torch.Tensor): query, 初始形状为(batch_size, seq_len, n_heads, head_dim//2)

    Returns:
        调整形状后的旋转矩阵, 形状为(1, seq_len, 1, head_dim//2)
    """
    ndim = x.ndim  # 获取x的维度数
    assert 0 <= 1 < ndim  # 确保x至少为2维【这里0<=1似乎也是冗余】

    # x形状一般为(batch_size, seq_len, n_heads, head_dim//2)
    # 这里确保freqs_cis与x的seq_len, head_dim//2维度一致, RoPE是对每个头分别进行的
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # 将第二维度和最后一维度分别变为seq_len和head_dim//2, 其余维度均为1，即(1, seq_len, 1, head_dim//2)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# 应用RoPE
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用RoPE, llama3是通过转换成复数形式来旋转角度的

    Args:
        xq (torch.Tensor): query
        xk (torch.Tensor): key
        freqs_cis (torch.Tensor): 旋转矩阵

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: query和key的旋转结果
    """
    # 将xq和xk由(batch_size, seq_len, n_(kv)_heads, head_dim)转换为(batch_size, seq_len, n_(kv)_heads, head_dim//2, 2)
    # 即每个头的维度两两一组, 以此作为复数的实部和虚部, 转换为复数
    # xq_和xk_的形状为(batch_size, seq_len, n_(kv)_heads, head_dim//2), 里面保存的是复数, 这样转换后最后一维就与freqs_cis的最后一维一致了
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # (batch_size, seq_len, n_heads, head_dim//2)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # (batch_size, seq_len, n_kv_heads, head_dim//2)

    # 按照xq_将freqs_cis的维度变为(1, seq_len, 1, head_dim//2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 通过复数乘法实现角度旋转
    # 复数张量转换为实数张量后, 通常为(..., 2)的形状, 即最后一维代表实部与虚部
    # 因此使用flatten将索引为3的维度展平, 形状由(batch_size, seq_len, n_(kv)_heads, head_dim//2, 2)变为(batch_size, seq_len, n_(kv)_heads, head_dim)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # (batch_size, seq_len, n_heads, head_dim)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)  # (batch_size, seq_len, n_kv_heads, head_dim)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ----------------------------------------------------------------------------
# Attention (GQA/KV Cache)
# ----------------------------------------------------------------------------
# 复制kv heads
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """当key和value的头数量n_kv_heads小于查询头(query heads)数量时, 需要将key和value进行重复, 以匹配查询头的数量

    Args:
        x (torch.Tensor): key/value: (batch_size, seq_len, n_kv_heads, head_dim)
        n_rep (int): 重复的次数

    Returns:
        key/value: (batch_size, seq_len, n_kv_heads*n_rep, head_dim)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # x[:, :, :, None, :]用于插入一个维度, 使得形状变为: (batch_size, seq_len, n_kv_heads, 1, head_dim)
    # expand()用于扩展张量的维度, 使得形状变为: (batch_size, seq_len, n_kv_heads, n_rep, head_dim)
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# 源码仅用于推理, 且使用了分布式训练方法, 这里进行了部分修改
class Attention(nn.Module):
    def __init__(self, args: CfgNode):
        super().__init__()
    
        # 此处考虑单卡, 使用常用的简单方式进行初始化
        self.args = args
        self.n_heads = args.MODEL.N_HEADS  # query的头数
        self.n_kv_heads = args.MODEL.N_HEADS if args.MODEL.N_KV_HEADS is None else args.MODEL.N_KV_HEADS  # key/value的头数, 未设置kv头数时, 默认与n_heads一致, 即MHA
        self.head_dim = args.MODEL.DIM // args.MODEL.N_HEADS
        self.n_rep = args.MODEL.N_HEADS // self.n_kv_heads  # query heads必须是kv heads的整数倍

        # 初始化权重矩阵
        self.wq = nn.Linear(args.MODEL.DIM, args.MODEL.N_HEADS * self.head_dim, bias=False, device=args.DEVICE)
        self.wk = nn.Linear(args.MODEL.DIM, self.n_kv_heads * self.head_dim, bias=False, device=args.DEVICE)
        self.wv = nn.Linear(args.MODEL.DIM, self.n_kv_heads * self.head_dim, bias=False, device=args.DEVICE)
        self.wo = nn.Linear(args.MODEL.N_HEADS * self.head_dim, args.MODEL.DIM, bias=False, device=args.DEVICE)  # GQA也产生n_heads个头的attention

        # 实现KV Cache, 用于存储KV矩阵, 包括prompt部分和生成部分的KV, 因此形状为(max_batch_size, max_seq_len*2, n_kv_heads, head_dim)
        self.cache_k = torch.zeros((args.TRAIN.BATCH_SIZE, args.MODEL.MAX_SEQ_LEN*2, self.n_kv_heads, self.head_dim), device=args.DEVICE)
        self.cache_v = torch.zeros((args.TRAIN.BATCH_SIZE, args.MODEL.MAX_SEQ_LEN*2, self.n_kv_heads, self.head_dim), device=args.DEVICE)

    # 源代码仅有推理模式, 这里区分训练与推理
    def forward(self, x: torch.Tensor, start_pos, inference, freqs_cis):
        # 输入维度为(batch_size, seq_len, dim)
        bsz, seq_len, _ = x.shape
        # mask只在训练时使用, 由于使用了KV Cache, 因此在推理模式下不需要使用mask
        mask = None
        
        # 由于只对线性层只对dim做变换，因此实际上跟seq_len无关，可以接受任意长度的seq_len
        xq = self.wq(x)  # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_heads * head_dim)
        xk = self.wk(x)  # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_kv_heads * head_dim)
        xv = self.wv(x)  # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_kv_heads * head_dim)

        # 转换形状
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)      # (batch_size, seq_len, n_heads, head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   # (batch_size, seq_len, n_kv_heads, head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   # (batch_size, seq_len, n_kv_heads, head_dim)

        # 推理模式, KV Cache仅在推理模式下使用
        if inference:
            # 【推理模式中使用max_seq_len*2是为了同时容纳prompt和生成内容, 因此需要乘以2】
            # 【推理时只考虑当前位置token在序列长度范围内的旋转矩阵】
            freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
            
            # xq:(batch_size, seq_len, n_heads, head_dim), xk:(batch_size, seq_len, n_kv_heads, head_dim)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            # 将当前位置新产生的key和value存入KV Cache
            self.cache_k[:bsz, start_pos:start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos:start_pos + seq_len] = xv

            # 取出所有的历史key和value
            keys = self.cache_k[:bsz, :start_pos + seq_len]
            values = self.cache_v[:bsz, :start_pos + seq_len]

            # 使用repeat_kv函数将key/value的维度变为与query一致
            keys = repeat_kv(keys, self.n_rep)  # (batch_size, seq_len, n_heads, head_dim)
            values = repeat_kv(values, self.n_rep)  # (batch_size, seq_len, n_heads, head_dim)

        # 训练模式, 无需使用KV Cache
        else:
            # xq:(batch_size, seq_len, n_heads, head_dim), xk:(batch_size, seq_len, n_kv_heads, head_dim)
            # 预训练时，这里使训练的输入序列和freq_cis都按照max_seq_len进行计算，因此预训练的输入长度必须为max_seq_len
            # 而推理时，进行了freqs_cis = freqs_cis[start_pos : start_pos + seq_len]截取，因此可以接受任意长度的输入序列
            # 类比到transformer的绝对位置编码，实际也是可以计算更大的freqs_cis，然后根据序列长度来截取的
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            # 使用repeat_kv函数将key/value的维度变为与query一致
            keys = repeat_kv(xk, self.n_rep)  # (batch_size, seq_len, n_heads, head_dim)
            values = repeat_kv(xv, self.n_rep)  # (batch_size, seq_len, n_heads, head_dim)

            # 生成因果掩码(causal mask / sequence mask)
            mask = torch.full((seq_len, seq_len), float("-inf"), device=self.args.DEVICE)  # (seq_len, seq_len)的全为负无穷的张量
            mask = torch.triu(mask, diagonal=1).to(self.args.DEVICE)  # 生成上三角矩阵, 对角线上方不变, 对角线及下方全为0

        # 调整形状进行注意力计算
        xq = xq.transpose(1,2)  # (batch_size, n_heads, seq_len, head_dim)
        keys = keys.transpose(1,2)  # (batch_size, n_heads, seq_len, head_dim)
        values = values.transpose(1,2)  # (batch_size, n_heads, seq_len, head_dim)

        # 计算注意力分数
        scores = torch.matmul(xq, keys.transpose(2,3)).to(self.args.DEVICE)/math.sqrt(self.head_dim)  # (batch_size, n_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores + mask

        # 应用softmax
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 乘value
        output = torch.matmul(scores, values).to(self.args.DEVICE)  # (batch_size, n_heads, seq_len, head_dim)

        # (batch_size, n_heads, seq_len, head_dim) -> (batch_size, seq_len, n_heads * head_dim)
        output = output.transpose(1,2).contiguous().view(bsz, seq_len, -1)

        return self.wo(output)  # (batch_size, seq_len, n_heads * head_dim) -> (batch_size, seq_len, dim)

# ----------------------------------------------------------------------------
# FFN
# ----------------------------------------------------------------------------
# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim:int, hidden_dim:int, multiple_of:int, ffn_dim_multiplier: Optional[float], args: CfgNode):
        super().__init__()
        self.dim = dim

        # 以下hidden dim计算方式源于源码, 用于保证hidden dim是256的倍数
        # 其中传入的初始hidden dim为4 * dim, multiple_of为256
        hidden_dim = int(2 * hidden_dim/3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 定义线性层
        self.w1 = nn.Linear(self.dim, hidden_dim, bias=False, device=args.DEVICE)
        self.w2 = nn.Linear(hidden_dim, self.dim, bias=False, device=args.DEVICE)
        self.w3 = nn.Linear(self.dim, hidden_dim, bias=False, device=args.DEVICE)

    def forward(self, x):
        # (batch_size, seq_len, dim)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))  # silu是beta=1的Swish

# ----------------------------------------------------------------------------
# Transformer Block
# ----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: CfgNode):
        super().__init__()
    
        # 定义参数
        self.args = args
        self.n_heads = args.MODEL.N_HEADS
        self.dim = args.MODEL.DIM
        self.head_dim = args.MODEL.DIM // args.MODEL.N_HEADS
        self.layer_id = layer_id
    
        # 定义attention部分
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.MODEL.DIM, eps=args.MODEL.NORM_EPS)

        # 定义feedforward部分
        self.feed_forward = FeedForward(
            dim=args.MODEL.DIM,
            hidden_dim=4 * args.MODEL.DIM,
            multiple_of=args.MODEL.MULTIPLE_OF,
            ffn_dim_multiplier=args.MODEL.FFN_DIM_MULTIPLER,
            args=args,
            )
        self.ffn_norm = RMSNorm(args.MODEL.DIM, eps=args.MODEL.NORM_EPS)

    def forward(self, x, start_pos, inference, freqs_cis):
        # (batch_size, seq_len, dim)
        h = x + self.attention(self.attention_norm(x), start_pos, inference, freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# ----------------------------------------------------------------------------
# Llama
# ----------------------------------------------------------------------------
class Llama(nn.Module):
    def __init__(self, params: CfgNode):
        super().__init__()
        
        # 定义参数
        self.params = params

        # 定义embedding层
        self.tok_embeddings = nn.Embedding(params.MODEL.VOCAB_SIZE, params.MODEL.DIM)

        # 定义transformer模块
        self.layers = nn.ModuleList()
        for layer_id in range(params.MODEL.N_LAYERS):
            self.layers.append(TransformerBlock(layer_id=layer_id, args=params))

        # 定义输出模块的RMSNorm及线性层
        self.norm = RMSNorm(params.MODEL.DIM, eps = params.MODEL.NORM_EPS)
        self.output = nn.Linear(params.MODEL.DIM, params.MODEL.VOCAB_SIZE, bias=False)

        # 在模型初始化时，预先计算好旋转矩阵，区分训练时使用的旋转矩阵和推理时使用的旋转矩阵
        self.head_dim = params.MODEL.DIM // params.MODEL.N_HEADS
        freqs_cis_for_train = precompute_freqs_cis(
            dim=self.head_dim, 
            end=self.params.MODEL.MAX_SEQ_LEN, 
            theta=self.params.MODEL.ROPE_THETA
            )  # (max_seq_len, head_dim//2)
        freqs_cis_for_inference = precompute_freqs_cis(
            dim=self.head_dim, 
            end=self.params.MODEL.MAX_SEQ_LEN*2, 
            theta=self.params.MODEL.ROPE_THETA
            )  # (max_seq_len*2, head_dim//2)
        self.register_buffer('freqs_cis_for_train', freqs_cis_for_train.to(params.DEVICE))
        self.register_buffer('freqs_cis_for_inference', freqs_cis_for_inference.to(params.DEVICE))
        self.freqs_cis = None

    def forward(self, x, targets=None, start_pos=0):

        # start_pos: 推理模式下, 当前token的位置索引
        # x:(batch_size, seq_len) -> h:(batch_size, seq_len, dim)
        h = self.tok_embeddings(x)

        # 根据是否传入targets，确定是否是推理模式
        if targets is None:
            inference = True
            self.freqs_cis = self.freqs_cis_for_inference
        else:
            inference = False
            self.freqs_cis = self.freqs_cis_for_train

        # 依次传入各个transformer block
        for layer in self.layers:
            h = layer(h, start_pos, inference, self.freqs_cis)

        # 传入输出模块
        h = self.norm(h)
        # h:(batch_size, seq_len, dim) -> logits:(batch_size, seq_len, vocab_size)
        logits = self.output(h).float()
        loss = None

        # 如果是训练模式, 就计算loss
        if targets is None:
            loss = None
        else:
            # logits:(batch_size, seq_len, vocab_size)
            # targets:(batch_size, seq_len)
            loss = F.cross_entropy(logits.view(-1, self.params.MODEL.VOCAB_SIZE), targets.view(-1))

        return logits, loss  # 如果是推理模式, logits后续还需使用softmax产生概率分布

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 参考 https://github.com/DLLXW/baby-llama2-chinese 中 model.configure_optimizers()方法
        
        # 获取模型参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤不需要梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 维度大于等于 2 的参数（如权重矩阵、嵌入层参数），这些参数会应用权重衰减
        # 这些参数通常是模型的主要可学习参数，直接影响模型的表达能力
        # 维度小于 2 的参数（如偏置、LayerNorm 参数），这些参数不会应用权重衰减
        # 这些参数通常用于调整模型的输出分布，而不是直接参与特征变换
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        # 创建优化器参数组
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # 统计参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # 检查是否支持融合 AdamW
        # 融合 AdamW（Fused AdamW） 是 PyTorch 提供的一种优化 AdamW 实现的高性能版本，通过将多个操作融合为一个内核（kernel）来加速计算
        # 它特别适用于 GPU 上的大规模深度学习训练任务
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def generate(self, prompt, tokenizer, max_gen_length=None, stream=False, temperature=1.0, top_k=None, model_type='sft'):
        """执行模型推理，返回生成的文本

        Args:
            prompt (str): 提示词
            max_gen_length (int, optional): 最大生成长度, 这里给prompt的长度为max_seq_len, 给max_gen_len也为max_seq_len
            stream (bool, optional): 是否流式输出, 默认为False
            temperature (float, optional): 温度参数, 默认为1.0, 值越大, 输出文本越随机, 为0.0时, 使用贪婪采样
            top_k (int, optional): 采样的top-k值, 默认为None, 不进行采样
            model_type (str, optional): 模型类型, 默认为'sft', 可选值为'sft'和'pretrain'
        
        Returns:
            str: 生成的文本
        """
        if max_gen_length is None:
            max_gen_length = self.params.MODEL.MAX_SEQ_LEN
        # 将prompt编码成token id
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if model_type == 'sft':
            prompt_ids = prompt_ids + [tokenizer.special_tokens['<bos>']]  # sft模型需添加<bos>引导标记
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.params.DEVICE)[None, ...]  # (1, seq_len)
        prompt_len = prompt_ids.size(1)  # prompt的token id长度
        prompt_last_pos = prompt_len - 1  # prompt最后一个id的位置索引
        # 确保prompt的token id不超过模型允许的最大序列长度max_seq_len，如果超过，只保留最后max_seq_len的token id
        prompt_ids = prompt_ids if prompt_len <= self.params.MODEL.MAX_SEQ_LEN else prompt_ids[:, -self.params.MODEL.MAX_SEQ_LEN:]

        answer = []
        with torch.no_grad():
            # 由于使用了KV Cache，每次给模型输入新的Q即可，同时逐次缓存新产生的KV
            # 分为两个阶段，prefill和decoding
            # 1) prefill阶段相当于不采用模型的输出作为下一次输入，而是将prompt的真实下一个字符作为下一次输入，这一阶段的目的是把prompt的KV Cache缓存起来
            # 【注意！】需要理清以下几点：
            # a. 在预训练中，模型的训练方式是【预测下一个字符】，prompt实际可以当做是已经生成的部分，因此接下的任务是紧接着继续生成
            # b. 那么，prompt所产生过的KV Cache，也应当是由预测下一个字符而产生的一系列KV Cache
            # c. 在当前的模型实现下，我们仅在训练中应用了因果mask，由于推理过程使用KV Cache，因此没有应用因果mask
            # d. 因此，prompt部分我们必须逐次填充，来模拟prompt的预测下一个字符的过程，并累计这个过程的KV Cache，只不过我们不关心prompt部分的预测结果，而是将真实的prompt中的token作为下一次输入
            # e. 如果将prompt的前prompt_len-1部分一次性填充，由于我们没有在推理过程中使用因果mask，这就会导致prompt前面的字符会看到后面的字符，因而在这个过程中产生的KV Cache是不合理的
            # f. 因为每一层的KV Cache计算依赖于上一层的输出，而上一层的输出由于没有使用因果mask，会将本不该看到的字符看到，这样产生的KV Cache与模型的训练逻辑不符
            # g. 因此，如果想要一次性填充，也可以在模型实现上，为推理部分单独加上prefill，并为其应用因果mask
            for start_pos in range(prompt_last_pos):
                logits, _ = self(prompt_ids[:, [start_pos]], start_pos=start_pos)  # logits: (1, seq_len, vocab_size)
                
            # 2) decoding阶段，从prompt的最后一个token开始逐个预测，直到达到最大长度或遇到<eos>
            prev_id = prompt_ids[:, [-1]]
            for start_pos in range(prompt_last_pos, prompt_last_pos+max_gen_length):  # 从prompt的最后一个id的Q开始输入，产生生成部分的第一个输出
                logits, _ = self(prev_id, start_pos=start_pos)  # 从prompt_len后开始生成, logits: (1, seq_len, vocab_size)
                logits = logits[:, -1, :]  # 取出最后一步的输出 (1, vocab_size)
                
                if temperature == 0.0:
                    _, next_token = torch.topk(logits, k=1, dim=-1)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)  # (1, seq_len, top_k)
                        logits[logits < v[:, [-1]]] = -float('Inf')  # v[:, [-1]]表示从v中提取最后一个维度的值作为阈值
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                answer.append(next_token.item())
                prev_id = next_token  # 【由于使用了KV Cache，下次输入只使用本次产生的next_token即可】
                
                if stream:
                    text = tokenizer.decode(np.array([next_token.item()]))
                    print(text, end='', flush=True)

                if next_token.item() == 2:  # 2为<eos>的id
                    break
    
        answer = tokenizer.decode(np.array(answer))
        # answer = answer.replace(prompt,'')
        return answer
    
    def count_parameters(self):
        """计算模型可训练参数量

        Returns:
            _type_: trainable_params:精确参数量, approx_params:大致参数量
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        approx_params = f"{trainable_params / 1_000_000 if trainable_params < 1_000_000_000 else trainable_params / 1_000_000_000:.1f}{'M' if trainable_params < 1_000_000_000 else 'B'}"
        return trainable_params, approx_params

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 添加上一级目录以导入config
    from config import get_config

    cfg = get_config()
    model = Llama(cfg).to(cfg.DEVICE)
    print(model)

    # 统计可训练参数总量
    trainable_params, approx_params = model.count_parameters()
    print(f"模型参数量: {trainable_params}，约：{approx_params}")