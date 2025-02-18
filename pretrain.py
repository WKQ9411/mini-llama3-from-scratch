from config import get_config, save_to_yaml
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer  # transformers==4.33.2
from model.llama import Llama
from dataset import PretrainDataset

import numpy as np
import math
import time
import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import os


# 训练准备
# ------------------------------------------------------------------------
# 获取配置
cfg = get_config()
cfg.defrost()  # 解冻配置
cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 根据实际用的机器修改配置
cfg.freeze()  # 冻结配置
# 加载tokenizer
tokenizer=ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')

# 创建保存结果的文件夹
if not os.path.exists(cfg.SAVE_PATH):
    os.makedirs(cfg.SAVE_PATH)

# 预训练数据集
train_data = PretrainDataset(data_path=cfg.DATA.PRETRAIN_DATA_PATH, max_length=cfg.MODEL.MAX_SEQ_LEN, memmap=cfg.DATA.MEMMAP)
# 构造数据加载器
train_loader = DataLoader(
    train_data, 
    batch_size=cfg.TRAIN.BATCH_SIZE,
    drop_last=False,
    shuffle=True, 
    )
iter_per_epoch = len(train_loader)  # 每个epoch的迭代次数
# 初始化模型
model = Llama(cfg).to(cfg.DEVICE)
# 构建优化器
optimizer = model.configure_optimizers(
    weight_decay=cfg.TRAIN.WEIGHT_DECAY, 
    learning_rate=cfg.TRAIN.LR, 
    betas=(cfg.TRAIN.BETA_1, cfg.TRAIN.BETA_2),
    device_type=cfg.DEVICE
    )

# 定义函数
# ------------------------------------------------------------------------
# 获取学习率
def get_lr(it):
    # 根据迭代次数返回学习率，it为总迭代次数

    # 1) warmup 阶段
    if it < cfg.TRAIN.WARMUP_ITERS:
        return cfg.TRAIN.LR * it / cfg.TRAIN.WARMUP_ITERS
    # 2) 衰减结束，使用最小学习率
    if it > cfg.TRAIN.LR_DECAY_ITERS:
        return cfg.TRAIN.MIN_LR
    # 3) 余弦衰减阶段
    decay_ratio = (it - cfg.TRAIN.WARMUP_ITERS) / (cfg.TRAIN.LR_DECAY_ITERS - cfg.TRAIN.WARMUP_ITERS)  # 衰减阶段中，当前迭代相对于剩余迭代的比例
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff 是一个从0到1之间变化的系数，控制学习率的衰减
    return cfg.TRAIN.MIN_LR + coeff * (cfg.TRAIN.LR - cfg.TRAIN.MIN_LR)

total_loss = []
# 训练循环
def train_epoch(epoch):
    start_time = time.time()
    for step, (X, Y) in enumerate(train_loader):
        
        X = X.to(cfg.DEVICE)
        Y = Y.to(cfg.DEVICE)

        # 清零梯度
        optimizer.zero_grad(set_to_none=True)

        lr = get_lr(epoch*iter_per_epoch+step)
        for param_group in optimizer.param_groups:  # 将新的学习率值应用到优化器中
            param_group['lr'] = lr
        
        logits, loss = model(X, Y)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        # 打印日志
        if step != 0 and step % cfg.TRAIN.LOG_INTERVAL == 0:
            spend_time = time.time() - start_time
            # 计算剩余时间
            rest_time = spend_time / (step+1) * iter_per_epoch - spend_time
            rest_time = str(datetime.timedelta(seconds=rest_time))
            print(f"Epoch: {epoch+1}/{cfg.TRAIN.EPOCHS} | Step: {step+1}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {lr:.6f} | Rest Time of Epoch {epoch+1}: {rest_time}")

def create_folder(base_path):
    folder_name = base_path
    counter = 1
    # 如果文件夹存在，尝试添加尾号
    while os.path.exists(folder_name):
        folder_name = f"{base_path}_{counter}"
        counter += 1
    # 创建文件夹
    os.makedirs(folder_name)
        
# 训练
# ------------------------------------------------------------------------
print('Start Training..')
_, approx_params = model.count_parameters()
print(f"Model parameters: {approx_params}")

for epoch in range(cfg.TRAIN.EPOCHS):
    train_epoch(epoch)

# 保存结果
model_name = f'pretrain_model_max_seq_{cfg.MODEL.MAX_SEQ_LEN}_params_{approx_params}'
current_train_path = os.path.join(cfg.SAVE_PATH, model_name)
create_folder(current_train_path)

model_path = os.path.join(current_train_path, f'{model_name}.pth')
torch.save(model.state_dict(), model_path)

# 绘制loss曲线
plt.plot(total_loss)
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig(os.path.join(current_train_path, f'{model_name}_loss.png'))

# 保存配置
save_to_yaml(os.path.join(current_train_path, f'{model_name}_config.yaml'))