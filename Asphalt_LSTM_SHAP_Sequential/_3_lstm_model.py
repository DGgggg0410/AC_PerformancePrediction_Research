# -*- coding: utf-8 -*-
# 步骤3：LSTM模型定义+校验有效性（独立可运行，前置：步骤2）
# 运行成功后：验证模型结构，下一步运行4_lstm_trainer.py
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from _1_config import INTERIM_DIR, DATA_DIR, OUTPUT_DIR
# ---------------------- 前置校验+加载参数 ----------------------
def check_prerequisite():
    """校验步骤2是否完成"""
    # 【关键修改2】：这里会自动使用 _1_config 传过来的绝对路径
    test_tensor_path = os.path.join(INTERIM_DIR, "step2_X_train_tensor.pt")
    
    if not os.path.exists(test_tensor_path):
        print(f"错误！未找到文件：{test_tensor_path}")
        print("请确认已经运行了 _2_data_processor.py")
        exit(1)
    
    return joblib.load(os.path.join(INTERIM_DIR, "step1_params.joblib"))


# ---------------------- LSTM模型定义 ----------------------
# LSTM 类通过循环神经网络结构捕捉时序特征，相比 Transformer，它在小样本或序列依赖较简单的场景下更具鲁棒性
class AsphaltLSTM(nn.Module):
    """沥青混合料疲劳寿命预测LSTM模型"""
    def __init__(self, input_dim, hidden_dim, num_layers, seq_len, output_dim, dropout_rate=0.1):
        super().__init__()
        # LSTM层：核心时序特征提取层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 全连接层：将LSTM最后时刻或全序列输出映射为疲劳寿命预测值
        self.fc1 = nn.Linear(hidden_dim * seq_len, 128)
        self.fc2 = nn.Linear(128, output_dim)
        
        # 激活函数+Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # LSTM前向传播：提取时序特征
        # lstm_out 形状: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        
        # 展平输出：适配全连接层
        x_flatten = lstm_out.reshape(lstm_out.shape[0], -1)
        
        # 全连接层前向传播：输出疲劳寿命预测值
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
    HIDDEN_DIM = params_dict["HIDDEN_DIM"]
    NUM_LAYERS = params_dict["NUM_LAYERS"]
    SEQ_LEN = params_dict["SEQ_LEN"]
    OUTPUT_DIM = params_dict["OUTPUT_DIM"]
    DROPOUT_RATE = params_dict["DROPOUT_RATE"]
    BATCH_SIZE = params_dict["BATCH_SIZE"]
    
    # 初始化模型
    model = AsphaltLSTM(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN,
        output_dim=OUTPUT_DIM,
        dropout_rate=DROPOUT_RATE
    )
    
    # 打印结构+校验前向传播
    print("="*30)
    print("LSTM模型结构：")
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
    print("步骤3：LSTM模型定义+校验有效性")
    print("="*50)
    params = check_prerequisite()
    validate_model(params)
    print("="*50)
    print("步骤3运行成功！下一步运行：_4_lstm_hyperparam_tuning.py")
    print("="*50)