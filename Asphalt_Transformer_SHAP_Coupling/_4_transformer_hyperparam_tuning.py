# -*- coding: utf-8 -*-
# 修复了键名不匹配（RuntimeError）的Transformer超参数调优脚本
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import joblib
from itertools import product
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# 导入数据处理函数
from _2_data_processor import check_prerequisite, load_params, generate_and_process_data
# 【仅新增】引入绝对路径变量
from _1_config import INTERIM_DIR, OUTPUT_DIR

# 【关键修改 1】从 _3 直接导入原始模型结构
try:
    from _3_transformer_model import AsphaltTransformer
    print("✅ 已成功链接原始模型定义，确保权重名称匹配")
except ImportError:
    print("❌ 无法导入 _3_transformer_model.py，请确保文件在同级目录下")
    exit(1)

# ---------------------- 1. 全局参数定义 ----------------------
EPOCHS = 100  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义临时模型存放路径（放入 interim 避免根目录混乱）
TEMP_MODEL_PATH = os.path.join(INTERIM_DIR, "temp_tuning_model.pth")

# ---------------------- 2. 核心训练函数 ----------------------
def train_model(model, X_train, y_train, X_val, y_val, batch_size, lr, epochs):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    early_stop_patience = 5
    early_stop_counter = 0
    min_val_loss = float('inf')
    
    model.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stop_counter = 0
            # 【路径修改】使用临时路径变量
            torch.save(model.state_dict(), TEMP_MODEL_PATH)
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                break
    
    # 【路径修改】加载临时保存的最优模型
    model.load_state_dict(torch.load(TEMP_MODEL_PATH))
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor).cpu().numpy()
    
    return val_pred, min_val_loss

# ---------------------- 3. 参数网格定义 ----------------------
param_grid = {
    "num_encoder_layers": [1],      
    "nhead": [2],                   
    "d_model": [64],                
    "lr": [1e-4],                   
    "batch_size": [8, 16],          
    "dropout": [0.1]                
}

param_names = list(param_grid.keys())
param_combinations = list(product(*param_grid.values()))
best_score = -float('inf')
best_params = None
results = []

# ---------------------- 4. 数据准备 ----------------------
params_dict = check_prerequisite()
SAMPLE_NUM, SEQ_LEN, INPUT_DIM = load_params(params_dict)

# 【路径修改】检查 step2 产物，使用 INTERIM_DIR 绝对路径
if not os.path.exists(os.path.join(INTERIM_DIR, "step2_X_train_tensor.pt")):
    generate_and_process_data(SAMPLE_NUM, SEQ_LEN, INPUT_DIM)

X_train = torch.load(os.path.join(INTERIM_DIR, "step2_X_train_tensor.pt"), weights_only=True).numpy()
y_train = torch.load(os.path.join(INTERIM_DIR, "step2_y_train_tensor.pt"), weights_only=True).numpy()
X_val = torch.load(os.path.join(INTERIM_DIR, "step2_X_val_tensor.pt"), weights_only=True).numpy()
y_val = torch.load(os.path.join(INTERIM_DIR, "step2_y_val_tensor.pt"), weights_only=True).numpy()

# ---------------------- 5. 遍历调优 ----------------------
print(f"开始超参数调优，共 {len(param_combinations)} 组参数...")
for idx, params in enumerate(param_combinations):
    p_dict = dict(zip(param_names, params))
    print(f"\n=== 第 {idx+1}/{len(param_combinations)} 组: {p_dict} ===")
    
    model = AsphaltTransformer(
        input_dim=INPUT_DIM,
        d_model=p_dict["d_model"],
        nhead=p_dict["nhead"],
        num_encoder_layers=p_dict["num_encoder_layers"],
        seq_len=SEQ_LEN,
        output_dim=1,
        dropout_rate=p_dict["dropout"]
    )
    
    try:
        val_pred, val_loss = train_model(model, X_train, y_train, X_val, y_val, p_dict["batch_size"], p_dict["lr"], EPOCHS)
        val_r2 = r2_score(y_val.flatten(), val_pred.flatten())
        val_rmse = np.sqrt(mean_squared_error(y_val.flatten(), val_pred.flatten()))
        val_mae = mean_absolute_error(y_val.flatten(), val_pred.flatten())
        
        result = p_dict.copy()
        result.update({"val_r2": val_r2, "val_loss": val_loss,"val_rmse":val_rmse ,"val_mae":val_mae})
        results.append(result)
        
        if val_r2 > best_score:
            best_score = val_r2
            best_params = p_dict
            print(f"🌟 更新最优！当前R²: {best_score:.4f}")
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")

# ---------------------- 6. 最终训练与保存 ----------------------
if best_params:
    print(f"\n=== 最终最优参数重新定型: {best_params} ===")
    final_model = AsphaltTransformer(
        input_dim=INPUT_DIM,
        d_model=best_params["d_model"],
        nhead=best_params["nhead"],
        num_encoder_layers=best_params["num_encoder_layers"],
        seq_len=SEQ_LEN,
        output_dim=1,
        dropout_rate=best_params["dropout"]
    ).to(DEVICE)
    
    final_val_pred, final_val_loss = train_model(final_model, X_train, y_train, X_val, y_val, best_params["batch_size"], best_params["lr"], EPOCHS)
    
    # 【路径修改】统一保存路径
    torch.save(final_model.state_dict(), os.path.join(INTERIM_DIR, "step4_best_transformer_model.pth"))
    # 额外保存一份到 interim 供特定步骤读取（兼容你的习惯）
    torch.save(final_model.state_dict(), os.path.join(INTERIM_DIR, "transformer_best_model.pth"))
    # 保存参数包
    joblib.dump(best_params, os.path.join(INTERIM_DIR, "step4_best_transformer_params.joblib"))
    
    print(f"✅ 最优权重已保存至 interim 目录，最终验证集R²: {r2_score(y_val.flatten(), final_val_pred.flatten()):.4f}")

    # 【路径修改】CSV 结果保存到 output 目录
    results_df = pd.DataFrame(results).sort_values(by="val_r2", ascending=False)
    results_csv_path = os.path.join(OUTPUT_DIR, "transformer_hyperparam_results.csv")
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 调优详情已保存至: {results_csv_path}")

# 清理调优产生的临时文件
if os.path.exists(TEMP_MODEL_PATH):
    os.remove(TEMP_MODEL_PATH)