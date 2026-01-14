# -*- coding: utf-8 -*-
# Post 3：高级模型对比分析（性能PK + SHAP特征关注度对比）
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- 0. 全局设置 & 绘图字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# --- 1. [关键点] 路径自动化定位 ---
# 获取当前脚本所在文件夹 (Model_Comparison_Analysis)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目总根目录 (AC_PerformancePrediction_Research)
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# 动态定位两个子项目的 interim 目录
LSTM_INTERIM = os.path.join(ROOT_DIR, "Asphalt_LSTM_SHAP_Sequential", "interim")
TRANS_INTERIM = os.path.join(ROOT_DIR, "Asphalt_Transformer_SHAP_Coupling", "interim")

# 定义最终产物存放路径 (放在当前分析文件夹内)
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output_final_comparison")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("🚀 开始进行跨模型高级对比分析...")

# --- 2. 加载数据的函数 ---
def load_model_results(model_name, interim_path):
    print(f"\n正在加载 {model_name} 的数据...")
    try:
        # 加载真实值 (两者通常共用 y_test)
        y_true_path = os.path.join(interim_path, "step2_y_test_tensor.pt")
        y_true = torch.load(y_true_path, weights_only=False).numpy()

        # 加载预测值
        y_pred_path = os.path.join(interim_path, "y_pred_tensor.pt")
        y_pred = torch.load(y_pred_path, weights_only=False).numpy()

        # 加载反归一化工具 (使用各自项目内的 scaler 以防万一)
        scaler = joblib.load(os.path.join(interim_path, "step2_scaler_y.joblib"))

        # 反归一化并压平
        y_true_org = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_org = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # 加载 SHAP 值
        shap_path = os.path.join(interim_path, "shap_values.npy")
        if os.path.exists(shap_path):
            shap_values = np.load(shap_path)
            # 维度预处理：(Samples, Seq, Features) -> (Samples, Features)
            if shap_values.ndim == 3:
                shap_values = np.mean(shap_values, axis=1)
            elif shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)
        else:
            shap_values = None

        return y_true_org, y_pred_org, shap_values

    except Exception as e:
        print(f"❌ 加载 {model_name} 失败: {str(e)}")
        return None, None, None

# --- 3. 执行数据加载 ---
lstm_true, lstm_pred, lstm_shap = load_model_results("LSTM", LSTM_INTERIM)
trans_true, trans_pred, trans_shap = load_model_results("Transformer", TRANS_INTERIM)

# --- 4. 性能指标 PK 逻辑 ---
if lstm_true is not None and trans_true is not None:
    def calc_metrics(y_t, y_p):
        return {
            "R2": r2_score(y_t, y_p),
            "RMSE": np.sqrt(mean_squared_error(y_t, y_p)),
            "MAE": mean_absolute_error(y_t, y_p)
        }

    lstm_metrics = calc_metrics(lstm_true, lstm_pred)
    trans_metrics = calc_metrics(trans_true, trans_pred)

    print("\n=== 🏆 最终模型 PK 结果汇总 ===")
    print(f"{'指标':<12} {'LSTM':<18} {'Transformer':<18}")
    print("-" * 50)
    print(f"{'R2':<12} {lstm_metrics['R2']:.4f}             {trans_metrics['R2']:.4f}")
    print(f"{'RMSE':<12} {lstm_metrics['RMSE']:.4f}             {trans_metrics['RMSE']:.4f}")
    print(f"{'MAE':<12} {lstm_metrics['MAE']:.4f}             {trans_metrics['MAE']:.4f}")
    
    # 绘图：准确度 & 残差分布对比
    
    plt.figure(figsize=(14, 6))
    
    # 子图 1: 散点对比
    plt.subplot(1, 2, 1)
    plt.scatter(lstm_true, lstm_pred, alpha=0.5, label=f'LSTM (R2={lstm_metrics["R2"]:.3f})', c='blue', edgecolors='white')
    plt.scatter(trans_true, trans_pred, alpha=0.5, label=f'Transformer (R2={trans_metrics["R2"]:.3f})', c='red', marker='x')
    plt.plot([lstm_true.min(), lstm_true.max()], [lstm_true.min(), lstm_true.max()], 'k--', lw=1.5)
    plt.xlabel("真实疲劳寿命 (万次)"); plt.ylabel("预测疲劳寿命 (万次)"); plt.title("预测准确度 PK"); plt.legend()

    # 子图 2: 残差分布
    plt.subplot(1, 2, 2)
    plt.hist(lstm_true - lstm_pred, bins=20, alpha=0.4, label='LSTM 误差', color='blue', density=True)
    plt.hist(trans_true - trans_pred, bins=20, alpha=0.4, label='Transformer 误差', color='red', density=True)
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel("误差 (真值 - 预测值)"); plt.ylabel("概率密度"); plt.title("残差分布对比"); plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_performance_pk.png"), dpi=300)
    print(f"✅ 性能对比图已保存至: {OUTPUT_DIR}")

# --- 5. SHAP 特征重要性对比 ---
if lstm_shap is not None and trans_shap is not None:
    feature_names = ["油石比", "空隙率", "矿粉用量", "沥青型号", "骨料级配", "温度", "应力水平", "劲度模量"]
    
    # 计算平均绝对贡献 (Global Importance)
    lstm_imp = np.mean(np.abs(lstm_shap), axis=0).flatten()
    trans_imp = np.mean(np.abs(trans_shap), axis=0).flatten()
    
    # 归一化为百分比
    l_norm = list(lstm_imp / (np.sum(lstm_imp) + 1e-9))
    t_norm = list(trans_imp / (np.sum(trans_imp) + 1e-9))

    # 对齐特征名长度
    min_len = min(len(feature_names), len(l_norm), len(t_norm))
    f_names = feature_names[:min_len]
    l_norm = l_norm[:min_len]
    t_norm = t_norm[:min_len]

    
    plt.figure(figsize=(12, 7))
    x = np.arange(len(f_names))
    width = 0.35
    
    plt.bar(x - width/2, l_norm, width, label='LSTM 特征关注点', color='#4E79A7', edgecolor='black', alpha=0.8)
    plt.bar(x + width/2, t_norm, width, label='Transformer 特征关注点', color='#E15759', edgecolor='black', alpha=0.8)
    
    plt.ylabel('归一化相对重要性 (SHAP权重)')
    plt.title('模型决策机制对比：LSTM vs Transformer')
    plt.xticks(x, f_names, rotation=20)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_feature_attention_comparison.png"), dpi=300)
    print(f"✅ 特征关注度对比图已保存。")

print(f"\n🎉 全部分析任务圆满结束！所有对比图表见: \n{OUTPUT_DIR}")