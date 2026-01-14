# -*- coding: utf-8 -*-
# 镜像修复版：LSTM 超参数自动化调优
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import joblib
from itertools import product
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# 导入你的数据处理函数
from _2_data_processor import check_prerequisite, load_params, generate_and_process_data
# 【仅新增】引入绝对路径变量
from _1_config import INTERIM_DIR, OUTPUT_DIR

# ---------------------- 1. 全局参数定义 ----------------------
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 2. 核心训练函数 (逻辑完全镜像) ----------------------
def train_model(model, X_train, y_train, X_val, y_val, batch_size, lr, epochs):
    # 转换为tensor并移到设备
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    early_stop_patience = 5
    early_stop_counter = 0
    min_val_loss = float('inf')
    
    # 【最小修改】临时模型路径指向 interim
    temp_tuning_path = os.path.join(INTERIM_DIR, "temp_best_model.pth")
    
    model.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(X_train_tensor.size(0))
        for i in range(0, len(X_train_tensor), batch_size):
            idx = indices[i:i+batch_size]
            batch_X, batch_y = X_train_tensor[idx], y_train_tensor[idx]
            
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), temp_tuning_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience: break
    
    model.load_state_dict(torch.load(temp_tuning_path))
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor).cpu().numpy()
    
    return val_pred, min_val_loss

# ---------------------- 3. 数据准备 ----------------------
params_dict = check_prerequisite()
SAMPLE_NUM, SEQ_LEN, INPUT_DIM = load_params(params_dict)

# 【最小修改】使用绝对路径变量加载，不再手动定义 interim_dir = "./interim"
X_train = torch.load(os.path.join(INTERIM_DIR, "step2_X_train_tensor.pt")).numpy()
y_train = torch.load(os.path.join(INTERIM_DIR, "step2_y_train_tensor.pt")).numpy()
X_val = torch.load(os.path.join(INTERIM_DIR, "step2_X_val_tensor.pt")).numpy()
y_val = torch.load(os.path.join(INTERIM_DIR, "step2_y_val_tensor.pt")).numpy()

# ---------------------- 4. 参数网格 ----------------------
param_grid = {
    "num_layers": [1, 2],
    "hidden_dim": [32, 64],
    "lr": [1e-4],
    "batch_size": [8],
    "dropout": [0.2]
}

param_names = list(param_grid.keys())
param_combinations = list(product(*param_grid.values()))
best_score = -float('inf')
best_params = None
results = []

# ---------------------- 5. 遍历调优 (这里完全保留你要求的 UI 镜像) ----------------------
print(f"开始超参数调优，共 {len(param_combinations)} 组参数组合...")

for idx, params in enumerate(param_combinations):
    param_dict = dict(zip(param_names, params))
    print(f"\n=== 第 {idx+1}/{len(param_combinations)} 组参数 ===")
    print(param_dict)
    
    NUM_LAYERS = param_dict["num_layers"]
    HIDDEN_DIM = param_dict["hidden_dim"]
    LR = param_dict["lr"]
    BATCH_SIZE = param_dict["batch_size"]
    DROPOUT = param_dict["dropout"]
    
    class AsphaltLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True, dropout=DROPOUT if NUM_LAYERS > 1 else 0)
            self.fc_out = nn.Linear(HIDDEN_DIM * SEQ_LEN, 1)
        def forward(self, x):
            x, _ = self.lstm(x)
            x = x.reshape(x.shape[0], -1)
            return self.fc_out(x)

    model = AsphaltLSTM()
    
    try:
        val_pred, val_loss = train_model(model, X_train, y_train, X_val, y_val, BATCH_SIZE, LR, EPOCHS)
        
        val_r2 = r2_score(y_val.flatten(), val_pred.flatten())
        val_rmse = np.sqrt(mean_squared_error(y_val.flatten(), val_pred.flatten()))
        val_mae = mean_absolute_error(y_val.flatten(), val_pred.flatten())
        
        result = param_dict.copy()
        result.update({"val_r2": val_r2, "val_rmse": val_rmse, "val_mae": val_mae, "val_loss": val_loss, "notes": "Tuning_Phase"})
        results.append(result)
        
        if val_r2 > best_score:
            best_score = val_r2
            best_params = param_dict
            print(f"更新最优参数！当前最优R²: {best_score:.4f}")
    except Exception as e:
        print(f"该参数组合训练失败：{str(e)}")

# ---------------------- 6. 总结与保存 (完全保留你原来的 Retrain 逻辑和打印) ----------------------
print("\n=== 调优完成 ===")
if best_params:
    print(f"最优参数：{best_params}")
    print(f"最优验证集R²：{best_score:.4f}")

    print("\n=== 用最优参数训练最终模型 ===")
    final_nl = best_params["num_layers"]
    final_hd = best_params["hidden_dim"]
    final_dr = best_params["dropout"]

    class FinalAsphaltLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(INPUT_DIM, final_hd, final_nl, batch_first=True, dropout=final_dr if final_nl > 1 else 0)
            self.fc_out = nn.Linear(final_hd * SEQ_LEN, 1)
        def forward(self, x):
            x, _ = self.lstm(x)
            x = x.reshape(x.shape[0], -1)
            return self.fc_out(x)

    final_model = FinalAsphaltLSTM()
    final_val_pred, final_val_loss = train_model(final_model, X_train, y_train, X_val, y_val, best_params["batch_size"], best_params["lr"], EPOCHS)
    
    final_r2 = r2_score(y_val.flatten(), final_val_pred.flatten())
    print(f"最终模型验证 R²: {final_r2:.4f}, 损失: {final_val_loss:.6f}")

    final_entry = best_params.copy()
    final_entry.update({
        "val_r2": final_r2, 
        "val_loss": final_val_loss, 
        "val_rmse": np.sqrt(mean_squared_error(y_val.flatten(), final_val_pred.flatten())),
        "val_mae": mean_absolute_error(y_val.flatten(), final_val_pred.flatten()),
        "notes": "FINAL_RETRAINED_MODEL" 
    })
    results.append(final_entry)

    # 【最小修改】保存路径指向 OUTPUT_DIR
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="val_r2", ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "lstm_hyperparam_results.csv"), index=False, encoding="utf-8-sig")
    print(f"所有记录（含最终定型结果）已同步至: {os.path.join(OUTPUT_DIR, 'lstm_hyperparam_results.csv')}")

    # 【最小修改】权重和参数保存指向 INTERIM_DIR
    torch.save(final_model.state_dict(), os.path.join(INTERIM_DIR, "step4_best_lstm_model.pth"))
    
    best_config_to_save = {**best_params, 'input_dim': INPUT_DIM, 'seq_len': SEQ_LEN, 'output_dim': 1, 'dropout_rate': best_params["dropout"]}
    joblib.dump(best_config_to_save, os.path.join(INTERIM_DIR, "step4_best_lstm_params.joblib"))
    
    print("✅ 最优模型与参数包已成功保存。")