# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from _2_data_processor import check_prerequisite, load_params
import platform
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 【核心修改】引入标准绝对路径
from _1_config import INTERIM_DIR, OUTPUT_DIR

# 解决中文乱码问题
def set_ch_font():
    system = platform.system()
    if system == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei']
    elif system == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

set_ch_font()

# ---------------------- 1. 加载前置条件+数据 ----------------------
params_dict = check_prerequisite()
SAMPLE_NUM, SEQ_LEN, INPUT_DIM = load_params(params_dict)

# 加载测试集 (【最小修改】使用 INTERIM_DIR)
X_test = torch.load(os.path.join(INTERIM_DIR, "step2_X_test_tensor.pt")).numpy()
y_test = torch.load(os.path.join(INTERIM_DIR, "step2_y_test_tensor.pt")).numpy()
scaler_y = joblib.load(os.path.join(INTERIM_DIR, "step2_scaler_y.joblib"))

# ---------------------- 2. 定义最优模型 ----------------------
# 保持你的调优结果参数
BEST_NUM_LAYERS = 1    
BEST_NHEAD = 2         
BEST_D_MODEL = 64      
BEST_DROPOUT = 0.1     
OUTPUT_DIM = 1          

try:
    from _3_transformer_model import AsphaltTransformer
    model = AsphaltTransformer(
        input_dim=INPUT_DIM,
        d_model=BEST_D_MODEL,
        nhead=BEST_NHEAD,
        num_encoder_layers=BEST_NUM_LAYERS,
        seq_len=SEQ_LEN,
        output_dim=OUTPUT_DIM,
        dropout_rate=BEST_DROPOUT
    )
    print("✅ 已成功按照原始定义初始化 Transformer 模型")
except ImportError:
    print("❌ 错误：找不到 _3_transformer_model.py")
    exit(1)

# ---------------------- 3. 加载模型+预测 ----------------------
# 【最小修改】统一从 INTERIM_DIR 获取模型权重
model_path = os.path.join(INTERIM_DIR, "step4_best_transformer_model.pth")

if not os.path.exists(model_path):
    print(f"❌ 错误：找不到模型权重文件 {model_path}")
    exit(1)

print(f"正在加载 Transformer 模型: {model_path}")
model.load_state_dict(torch.load(model_path))
model.eval()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

y_test_original = scaler_y.inverse_transform(y_test)
y_pred_original = scaler_y.inverse_transform(y_pred)

# ---------------------- 4. 验证与保存 ----------------------
test_r2 = r2_score(y_test_original.flatten(), y_pred_original.flatten())
print(f"✅ Transformer 最优模型测试集R2: {test_r2:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_original, alpha=0.6, label="预测值vs真实值", c='red')
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 
         'k--', label="完美预测线")
plt.xlabel("真实疲劳寿命")
plt.ylabel("预测疲劳寿命")
plt.title(f"Transformer最优模型预测效果（R2={test_r2:.4f}）")
plt.legend()
plt.tight_layout()

# 【路径修改】图片保存到 output 目录
plot_save_path = os.path.join(OUTPUT_DIR, "transformer_prediction_result.png")
plt.savefig(plot_save_path)
print(f"✅ 预测图表已保存至: {plot_save_path}")

# 【关键修改】强制保存预测结果到 interim，确保后续分析脚本能找到
save_path = os.path.join(INTERIM_DIR, "y_pred_tensor.pt")
torch.save(torch.tensor(y_pred), save_path)

val_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
val_mae = mean_absolute_error(y_test_original, y_pred_original)

print(f"✅ Transformer 预测值张量已保存至: {save_path}")