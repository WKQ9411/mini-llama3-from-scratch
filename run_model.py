from config import get_config_from_yaml
import torch
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer  # transformers==4.33.2
from model.llama import Llama


# 加载模型
model_name = 'sft_model_max_seq_256_params_274.3M'  # 填写模型文件夹名称，下面的模型路径、配置路径、模型种类会自动匹配

model_path = f'./results/{model_name}/{model_name}.pth'
config_path = f'./results/{model_name}/{model_name}_config.yaml'
model_type = model_name.split('_')[0]

# 获取配置
cfg = get_config_from_yaml(config_path)
cfg.defrost()  # 解冻配置
cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 根据实际用的机器修改配置
cfg.freeze()  # 冻结配置

# 初始化模型
model = Llama(cfg)
# 加载tokenizer
tokenizer=ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')

# 根据device加载模型
model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
model.eval()
model.to(cfg.DEVICE)

_, approx_params = model.count_parameters()
print(f'模型参数量：{approx_params}')
print(f'最大输入长度：{cfg.MODEL.MAX_SEQ_LEN}')

# 对话循环
while True:
    prompt = input("input:\n")
    if prompt == "q":
        break
    print('output:')
    # 根据要测试的模型类型，model_type选择：sft 或 pretrain
    answer = model.generate(prompt=prompt, tokenizer=tokenizer, stream=True, temperature=1.5, top_k=5, model_type=model_type)
    print('\n')
    

