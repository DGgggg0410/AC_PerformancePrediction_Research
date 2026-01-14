# -*- coding: utf-8 -*-
# 步骤6：LSTM模型训练+保存最优模型（独立可运行，前置：步骤3）
# 运行成功后：生成模型权重+损失历史，下一步运行7_lstm_evaluator.py
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
from _3_lstm_model import AsphaltLSTM
# 【仅新增】引入绝对路径变量
from _1_config import INTERIM_DIR

# ---------------------- 前置校验+加载数据 ----------------------
def check_prerequisite():
    """校验步骤2是否完成"""
    required_files = ["step2_X_train_tensor.pt", "step2_y_train_tensor.pt", 
                     "step2_X_val_tensor.pt", "step2_y_val_tensor.pt"]
    for file in required_files:
        # 【最小修改】使用绝对路径进行校验
        if not os.path.exists(os.path.join(INTERIM_DIR, file)):
            print(f"错误！缺少中间产物：{file}，请先运行2_data_processor.py")
            exit(1)
    # 加载参数+数据
    params = joblib.load(os.path.join(INTERIM_DIR, "step1_params.joblib"))
    tensor_data = {
        "X_train": torch.load(os.path.join(INTERIM_DIR, "step2_X_train_tensor.pt")),
        "y_train": torch.load(os.path.join(INTERIM_DIR, "step2_y_train_tensor.pt")),
        "X_val": torch.load(os.path.join(INTERIM_DIR, "step2_X_val_tensor.pt")),
        "y_val": torch.load(os.path.join(INTERIM_DIR, "step2_y_val_tensor.pt"))
    }
    return params, tensor_data

# ---------------------- 核心功能：模型训练 ----------------------
def train_model(params_dict, tensor_data):
    """训练LSTM模型并保存最优权重+损失历史"""
    # --- 原有参数提取 ---
    INPUT_DIM = params_dict["INPUT_DIM"]
    SEQ_LEN = params_dict["SEQ_LEN"]
    OUTPUT_DIM = params_dict["OUTPUT_DIM"]
    EPOCHS = params_dict["EPOCHS"]
    
    # --- ！！！手动覆盖为最优参数 (根据你的调优结果) ！！！ ---
    HIDDEN_DIM = 64        # 调优后的最优 hidden_dim
    NUM_LAYERS = 2         # 调优后的最优 num_layers
    BATCH_SIZE = 8        # 调优后的最优 batch_size
    LEARNING_RATE = 0.0001 # 调优后的最优 learning_rate
    DROPOUT_RATE = 0.2     # 调优后的最优 dropout
    # ---------------------------------------------------
    
    # 提取数据
    X_train = tensor_data["X_train"]
    y_train = tensor_data["y_train"]
    X_val = tensor_data["X_val"]
    y_val = tensor_data["y_val"]
    
    # 初始化模型+优化器+损失函数
    model = AsphaltLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, SEQ_LEN, OUTPUT_DIM, DROPOUT_RATE)
    # 保持与Transformer一致，使用AdamW或Adam（根据习惯，此处对齐调优习惯）
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 训练过程
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    # 【最小修改】使用绝对路径
    model_path = os.path.join(INTERIM_DIR, "step4_best_lstm_model.pth")
    
    print(f"开始训练（{EPOCHS}轮，批次大小：{BATCH_SIZE}）")
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_X = X_train[i:i+BATCH_SIZE]
            batch_y = y_train[i:i+BATCH_SIZE]
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.shape[0]
        avg_train_loss = train_loss / len(X_train)
        train_loss_history.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(X_val), BATCH_SIZE):
                batch_X = X_val[i:i+BATCH_SIZE]
                batch_y = y_val[i:i+BATCH_SIZE]
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_X.shape[0]
        avg_val_loss = val_loss / len(X_val)
        val_loss_history.append(avg_val_loss)
        
        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"第{epoch+1}轮：验证损失最优，已保存模型")
        
        # 打印进度
        if (epoch+1) % 10 == 0:
            print(f"第{epoch+1}轮 | 训练损失：{avg_train_loss:.6f} | 验证损失：{avg_val_loss:.6f}")
    
    # 保存损失历史到绝对路径
    joblib.dump(train_loss_history, os.path.join(INTERIM_DIR, "step4_train_loss.joblib"))
    joblib.dump(val_loss_history, os.path.join(INTERIM_DIR, "step4_val_loss.joblib"))
    print("训练完成，模型权重+损失历史已保存")

# ---------------------- 独立运行入口 ----------------------
if __name__ == "__main__":
    print("="*50)
    print("步骤6：LSTM模型训练+保存最优模型")
    print("="*50)
    params, tensor_data = check_prerequisite()
    train_model(params, tensor_data)
    print("="*50)
    print("步骤6运行成功！下一步运行：7_lstm_evaluator.py")
    print("="*50)