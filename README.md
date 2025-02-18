# mini-llama3-from-scratch
本项目旨在从零实现一个迷你版llama3，并进行预训练和微调。

## 1. 配置环境
`pip install -r requirements.txt`

```
D:\CODE\手撕LLAMA3
│  config.py
│  dataset.py
│  example.mp4
│  pretrain.py
│  requirements.txt
│  run_model.py
│  sft.py
│
├─chatglm_tokenizer
│      tokenization_chatglm.py
│      tokenizer.model
│      tokenizer_config.json
│
├─data
│  │  baidubaike_563w_1.bin
│  │  baidubaike_563w_2.bin
│  │  baidubaike_563w_3.bin
│  │  baidubaike_563w_4.bin
│  │  baidubaike_563w_5.bin
│  │  medical_book.bin
│  │  medical_encyclopedia.bin
│  │  procees_data.py
│  │  wiki.bin
│  │
│  ├─pretrain_data
│  │      pretrain_data.bin
│  │
│  └─sft_data
│          alpaca_gpt4_data_zh.json
│          Belle_open_source_1M.json
│          sft_data.csv
│
├─model
│  │
│  └─llama.py
│
└─results
    ├─pretrain_model_max_seq_256_params_274.3M
    │      pretrain_model_max_seq_256_params_274.3M.pth
    │      pretrain_model_max_seq_256_params_274.3M_config.yaml
    │      pretrain_model_max_seq_256_params_274.3M_loss.png
    │
    └─sft_model_max_seq_256_params_274.3M
            sft_model_max_seq_256_params_274.3M.pth
            sft_model_max_seq_256_params_274.3M_config.yaml
            sft_model_max_seq_256_params_274.3M_loss.png
```
