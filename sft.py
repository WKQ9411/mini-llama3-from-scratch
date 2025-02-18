from config import save_to_yaml, get_config_from_yaml
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer  # transformers==4.33.2
from model.llama import Llama
from dataset import SFTDataset

import numpy as np
import pandas as pd
import math
import time
import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import os


# 训练准备
# ------------------------------------------------------------------------
# 选择pretrain model
pretrain_model_dict = './results/model.pth'
pretrain_model_config_path = './results/config512.yaml'
# 获取配置
cfg = get_config_from_yaml(pretrain_model_config_path)
cfg.defrost()  # 解冻配置
cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 根据实际用的机器修改配置
cfg.freeze()  # 冻结配置
# 加载tokenizer
tokenizer=ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')

# 创建保存结果的文件夹
if not os.path.exists(cfg.SAVE_PATH):
    os.makedirs(cfg.SAVE_PATH)

# 加载sft数据
df_data = pd.read_csv(cfg.DATA.SFT_DATA_PATH)
train_data = SFTDataset(df_data=df_data, tokenizer=tokenizer, max_length=cfg.MODEL.MAX_SEQ_LEN)
# 构造数据加载器
train_loader = DataLoader(
    train_data, 
    batch_size=cfg.TRAIN.SFT_BATCH_SIZE,
    drop_last=False,
    shuffle=True, 
    )
iter_per_epoch = len(train_loader)  # 每个epoch的迭代次数
# 初始化模型
model = Llama(cfg).to(cfg.DEVICE)
model.load_state_dict(torch.load(pretrain_model_dict, map_location=cfg.DEVICE))
model.train()
# 构建优化器
optimizer = model.configure_optimizers(
    weight_decay=cfg.TRAIN.WEIGHT_DECAY, 
    learning_rate=cfg.TRAIN.SFT_LR,  # 使用sft的学习率
    betas=(cfg.TRAIN.BETA_1, cfg.TRAIN.BETA_2),
    device_type=cfg.DEVICE
    )

# 定义函数
# ------------------------------------------------------------------------
total_loss = []
# 训练循环
def train_epoch(epoch):
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        
        X = X.to(cfg.DEVICE)
        Y = Y.to(cfg.DEVICE)

        # 清零梯度
        optimizer.zero_grad(set_to_none=True)
        
        logits, _ = model(X, Y)  # 内置的计算loss方式直接计算了全部的loss，不能用，因此要用logits进一步计算loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='none')  # reduction用于控制是否对损失进行汇总，若有版本问题，尝试reduce=False
        loss_mask = loss_mask.view(-1).to(cfg.DEVICE)
        loss = torch.sum(loss*loss_mask)/loss_mask.sum()

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        # 打印日志
        if step != 0 and step % cfg.TRAIN.LOG_INTERVAL == 0:
            spend_time = time.time() - start_time
            # 计算剩余时间
            rest_time = spend_time / (step+1) * iter_per_epoch - spend_time
            rest_time = str(datetime.timedelta(seconds=rest_time))
            print(f"Epoch: {epoch+1}/{cfg.TRAIN.SFT_EPOCHS} | Step: {step+1}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {cfg.TRAIN.SFT_LR:.6f} | Rest Time of Epoch {epoch+1}: {rest_time}")

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

for epoch in range(cfg.TRAIN.SFT_EPOCHS):
    train_epoch(epoch)

# 保存结果
model_name = f'sft_model_max_seq_{cfg.MODEL.MAX_SEQ_LEN}_params_{approx_params}'
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