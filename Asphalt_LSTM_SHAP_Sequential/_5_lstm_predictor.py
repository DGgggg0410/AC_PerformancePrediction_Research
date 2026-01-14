# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from _2_data_processor import check_prerequisite, load_params
# 【仅新增】引入绝对路径变量
from _1_config import INTERIM_DIR, OUTPUT_DIR

# ---------------------- 1. 加载前置条件+数据 ----------------------
params_dict = check_prerequisite()
SAMPLE_NUM, SEQ_LEN, INPUT_DIM = load_params(params_dict)

# 【最小修改】使用从 config 导入的绝对路径加载测试集
X_test = torch.load(os.path.join(INTERIM_DIR, "step2_X_test_tensor.pt")).numpy()
y_test = torch.load(os.path.join(INTERIM_DIR, "step2_y_test_tensor.pt")).numpy()
scaler_y = joblib.load(os.path.join(INTERIM_DIR, "step2_scaler_y.joblib"))

# ---------------------- 2. 定义最优模型 ----------------------
# ！！！请确认这里填写的参数与您 CSV 文件中的最优参数一致！！！
BEST_NUM_LAYERS = 2     
BEST_HIDDEN_DIM = 64    
BEST_DROPOUT = 0.2     
OUTPUT_DIM = 1          

class BestAsphaltLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_DIM,
            hidden_size=BEST_HIDDEN_DIM,
            num_layers=BEST_NUM_LAYERS,
            batch_first=True,
            dropout=BEST_DROPOUT if BEST_NUM_LAYERS > 1 else 0
        )
        self.fc_out = nn.Linear(BEST_HIDDEN_DIM * SEQ_LEN, OUTPUT_DIM)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out.reshape(lstm_out.shape[0], -1)
        out = self.fc_out(x)
        return out

# ---------------------- 3. 加载模型+预测 ----------------------
model = BestAsphaltLSTM()

# 【最小修改】直接指向步骤4保存的绝对路径权重文件
model_path = os.path.join(INTERIM_DIR, "step4_best_lstm_model.pth")

print(f"正在加载模型: {model_path}") # 保持你的打印习惯不变
model.load_state_dict(torch.load(model_path))
model.eval()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

y_test_original = scaler_y.inverse_transform(y_test)
y_pred_original = scaler_y.inverse_transform(y_pred)

# ---------------------- 4. 验证与保存 ----------------------
from sklearn.metrics import r2_score
test_r2 = r2_score(y_test_original.flatten(), y_pred_original.flatten())
print(f"✅ 最优模型测试集R2: {test_r2:.4f}")

plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
plt.scatter(y_test_original, y_pred_original, alpha=0.6, label="预测值vs真实值")
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 
         'r--', label="完美预测线")
plt.xlabel("真实疲劳寿命")
plt.ylabel("预测疲劳寿命")
plt.title(f"LSTM最优模型预测效果（R2={test_r2:.4f}）")
plt.legend()
plt.tight_layout()

# 【最小修改】图片保存到 OUTPUT_DIR
plt.savefig(os.path.join(OUTPUT_DIR, "lstm_prediction_result.png"))

# 【最小修改】预测结果保存到绝对路径 INTERIM_DIR
save_path = os.path.join(INTERIM_DIR, "y_pred_tensor.pt")
torch.save(torch.tensor(y_pred), save_path)
print(f"✅ LSTM 预测值已保存至: {save_path}")