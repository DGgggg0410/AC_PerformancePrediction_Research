# -*- coding: utf-8 -*-
# Post 2：双模型残差分布对比分析
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib

# ---------------------- 路径自动化定位 ----------------------
# 获取当前脚本所在文件夹 (Model_Comparison_Analysis)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目总根目录 (AC_PerformancePrediction_Research)
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# 定位子项目目录
LSTM_INTERIM = os.path.join(ROOT_DIR, "Asphalt_LSTM_SHAP_Sequential", "interim")
TRANS_INTERIM = os.path.join(ROOT_DIR, "Asphalt_Transformer_SHAP_Coupling", "interim")

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 1. 加载真实数据与预测数据 ----------------------
print("💡 正在读取数据并计算残差...")

try:
    # 1. 加载测试集真实值 (通常两个模型共用一套 Step 2 的测试集)
    y_test_tensor = torch.load(os.path.join(TRANS_INTERIM, "step2_y_test_tensor.pt"))
    scaler_y = joblib.load(os.path.join(TRANS_INTERIM, "step2_scaler_y.joblib"))
    
    # 将真实值转为原始量级 (万次)
    y_true = scaler_y.inverse_transform(y_test_tensor.numpy().reshape(-1, 1)).flatten()

    # 2. 加载各模型的预测值 (需确保各自 Step 5 已保存 y_pred_tensor.pt)
    y_pred_lstm_tensor = torch.load(os.path.join(LSTM_INTERIM, "y_pred_tensor.pt"))
    y_pred_trans_tensor = torch.load(os.path.join(TRANS_INTERIM, "y_pred_tensor.pt"))

    # 反归一化预测值
    y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_tensor.reshape(-1, 1)).flatten()
    y_pred_trans = scaler_y.inverse_transform(y_pred_trans_tensor.reshape(-1, 1)).flatten()

    # 3. 计算残差 (Residuals = True - Predicted)
    res_lstm = y_true - y_pred_lstm
    res_trans = y_true - y_pred_trans

except FileNotFoundError as e:
    print(f"❌ 错误：找不到必要的数据文件，请确保两个模型的 Step 2 和 Step 5 均运行成功。")
    print(f"缺失路径: {e.filename}")
    exit(1)

# ---------------------- 2. 绘制残差密度图 ----------------------

plt.figure(figsize=(10, 6))

# 绘制 LSTM 的残差分布
sns.kdeplot(res_lstm, label='LSTM 残差分布', fill=True, color='#1f77b4', alpha=0.3)
# 绘制 Transformer 的残差分布
sns.kdeplot(res_trans, label='Transformer 残差分布', fill=True, color='#d62728', alpha=0.3)

# 绘制 0 误差基准线
plt.axvline(x=0, color='black', linestyle='--', alpha=0.6, label='零误差基准线')

plt.title("LSTM 与 Transformer 预测残差密度分布对比", fontsize=14)
plt.xlabel("残差值 (真实寿命 - 预测寿命)", fontsize=12)
plt.ylabel("概率密度 (Probability Density)", fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.5)

# ---------------------- 3. 保存结果 ----------------------
plt.tight_layout()
save_path = os.path.join(CURRENT_DIR, "total_residual_comparison.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 残差对比图已生成并保存至:\n   {save_path}")

# 如果需要，可以打印简单的统计信息
print(f"LSTM 残差均值: {res_lstm.mean():.4f}")
print(f"Transformer 残差均值: {res_trans.mean():.4f}")

plt.show()