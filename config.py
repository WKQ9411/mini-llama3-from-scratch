from yacs.config import CfgNode
import torch
import os

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available.")
    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")

# -----------------------------------------------------------------------------
# 基本配置
# -----------------------------------------------------------------------------
CN = CfgNode()
CN.DEVICE = None  # 在运行代码时，根据实际用的机器来临时配置
CN.SAVE_PATH = './results'

# -----------------------------------------------------------------------------
# 模型配置
# -----------------------------------------------------------------------------
CN.MODEL = CfgNode()
CN.MODEL.DIM = 1024                 # 嵌入维度
CN.MODEL.N_LAYERS = 12             # 解码器模块层数
CN.MODEL.N_HEADS = 8               # Q的头数
CN.MODEL.N_KV_HEADS = 4            # KV的头数(如果为None，则与N_HEADS一致)
CN.MODEL.VOCAB_SIZE = 64794        # 词表大小
CN.MODEL.MULTIPLE_OF = 256         # 用于计算前馈网络长度
CN.MODEL.FFN_DIM_MULTIPLER = None  # 用于计算前馈网络长度
CN.MODEL.NORM_EPS = 1e-5           # RMSNorm计算的默认Epsilon值
CN.MODEL.ROPE_THETA = 500000       # RoPE计算的默认theta值
CN.MODEL.MAX_SEQ_LEN = 256         # 最大序列长度

# -----------------------------------------------------------------------------
# 数据配置
# -----------------------------------------------------------------------------
CN.DATA = CfgNode()
CN.DATA.PRETRAIN_DATA_PATH = './data/pretrain_data/pretrain_data.bin'
CN.DATA.SFT_DATA_PATH = './data/sft_data/sft_data.csv'
CN.DATA.MEMMAP = True  # 是否使用memmap

# -----------------------------------------------------------------------------
# 训练配置
# -----------------------------------------------------------------------------
CN.TRAIN = CfgNode()
# 预训练部分
CN.TRAIN.BATCH_SIZE = 32
CN.TRAIN.EPOCHS = 1
CN.TRAIN.LR = 3e-4  # 最大学习率
CN.TRAIN.MIN_LR = 1e-5  # 最小学习率，约为最大学习率的1/10
CN.TRAIN.WEIGHT_DECAY = 1e-1  # 权重衰减系数
CN.TRAIN.BETA_1 = 0.9  # AdamW的beta1参数
CN.TRAIN.BETA_2 = 0.95  # AdamW的beta2参数
CN.TRAIN.WARMUP_ITERS = 5000  # warm up的迭代次数，在训练的初期阶段，学习率从0线性增长到预设的初始学习率lr
CN.TRAIN.LR_DECAY_ITERS = 300000  # 约与最大训步数相等，epoch*iter_per_epoch
CN.TRAIN.LOG_INTERVAL = 50  # 打印log间隔数

# SFT部分
CN.TRAIN.SFT_LR = 1e-6  # SFT的学习率
CN.TRAIN.SFT_EPOCHS = 1  # SFT的训练轮数
CN.TRAIN.SFT_BATCH_SIZE = 16  # SFT的batch size

# -----------------------------------------------------------------------------
# 函数
# -----------------------------------------------------------------------------

def get_config() -> CfgNode:
    """获取 yacs CfgNode 配置对象"""
    return CN

def save_to_yaml(yaml_path):
    """将配置保存为 yaml 文件"""
    with open(yaml_path, 'w') as f:
        f.write(CN.dump())
    print(f'Save config to: {yaml_path}')

def get_config_from_yaml(yaml_path):
    """需确保 yaml 与 CfgNode 配置结构完全一致"""
    CN.merge_from_file(yaml_path)
    return CN


if __name__ == '__main__':
    print(get_config())
    # save_to_yaml('./results/config.yaml')