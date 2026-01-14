# -*- coding: utf-8 -*-
# 步骤3：Transformer模型定义+校验有效性（独立可运行，前置：步骤2）
# 运行成功后：验证模型结构，下一步运行4_transformer_trainer.py
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
# 【仅新增】引入绝对路径变量
from _1_config import INTERIM_DIR

# ---------------------- 前置校验+加载参数 ----------------------
def check_prerequisite():
    """校验步骤2是否完成"""
    # 【最小修改】使用从 config 导入的绝对路径
    test_tensor_path = os.path.join(INTERIM_DIR, "step2_X_train_tensor.pt")
    if not os.path.exists(test_tensor_path):
        print(f"错误！请先运行2_data_processor.py\n预期路径: {test_tensor_path}")
        exit(1)
    return joblib.load(os.path.join(INTERIM_DIR, "step1_params.joblib"))

# ---------------------- Transformer模型定义 ----------------------
# PositionalEncoding 类显式注入了时序位置信息，既保留了自注意力的并行优势，又弥补了位置感知的缺失
class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, seq_len, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout_rate)
        
        # 生成位置编码矩阵
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x

class AsphaltTransformer(nn.Module):
    """沥青混合料疲劳寿命预测Transformer模型"""
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, seq_len, output_dim, dropout_rate=0.1):
        super().__init__()
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码层
        self.pos_enc = PositionalEncoding(d_model, seq_len, dropout_rate)
        
        # Transformer编码器层
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Transformer编码器
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # 全连接层
        self.fc1 = nn.Linear(d_model * seq_len, 128)
        self.fc2 = nn.Linear(128, output_dim)
        
        # 激活函数+Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x_flatten = x.reshape(x.shape[0], -1)
        out = self.fc1(x_flatten)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ---------------------- 核心功能：校验模型 ----------------------
def validate_model(params_dict):
    """初始化模型并校验前向传播"""
    # 提取参数
    INPUT_DIM = params_dict["INPUT_DIM"]
    D_MODEL = params_dict["D_MODEL"]
    NHEAD = params_dict["NHEAD"]
    NUM_ENCODER_LAYERS = params_dict["NUM_ENCODER_LAYERS"]
    SEQ_LEN = params_dict["SEQ_LEN"]
    OUTPUT_DIM = params_dict["OUTPUT_DIM"]
    DROPOUT_RATE = params_dict["DROPOUT_RATE"]
    BATCH_SIZE = params_dict["BATCH_SIZE"]
    
    # 初始化模型
    model = AsphaltTransformer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        seq_len=SEQ_LEN,
        output_dim=OUTPUT_DIM,
        dropout_rate=DROPOUT_RATE
    )
    
    # 打印结构+校验前向传播
    print("="*30)
    print("Transformer模型结构：")
    print(model)
    print("="*30)
    test_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    try:
        test_output = model(test_input)
        print(f"前向传播成功！输入形状：{test_input.shape}，输出形状：{test_output.shape}")
        print(f"模型参数总数：{sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"前向传播失败：{e}")
        exit(1)

# ---------------------- 独立运行入口 ----------------------
if __name__ == "__main__":
    print("="*50)
    print("步骤3：Transformer模型定义+校验有效性")
    print("="*50)
    params = check_prerequisite()
    validate_model(params)
    print("="*50)
    print("步骤3运行成功！下一步运行：_4_transformer_hyperparam_tuning.py")
    print("="*50)