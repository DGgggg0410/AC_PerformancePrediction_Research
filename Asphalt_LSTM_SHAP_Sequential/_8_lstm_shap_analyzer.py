# -*- coding: utf-8 -*-
# 步骤8：LSTM模型SHAP分析 (SCI 1区视觉美化-中文版)
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import shap
from matplotlib import gridspec 

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False
from _3_lstm_model import AsphaltLSTM
from _1_config import INTERIM_DIR, OUTPUT_DIR

# ---------------------- 前置校验+加载数据 ----------------------
def check_prerequisite():
    """校验步骤6是否完成"""
    required_files = ["step4_best_lstm_model.pth", "step2_X_test_tensor.pt"]
    for file in required_files:
        file_path = os.path.join(INTERIM_DIR, file)
        if not os.path.exists(file_path):
            print(f"错误！缺少中间产物：{file_path}，请先运行对应步骤")
            exit(1)
    params = joblib.load(os.path.join(INTERIM_DIR, "step1_params.joblib"))
    interim_data = {
        "model_weights": torch.load(os.path.join(INTERIM_DIR, "step4_best_lstm_model.pth")),
        "X_test": torch.load(os.path.join(INTERIM_DIR, "step2_X_test_tensor.pt"))
    }
    return params, interim_data

def shap_analysis(params_dict, interim_data):
    """进行SHAP分析并绘图"""
    # --- 1. 提取基础参数 ---
    INPUT_DIM, SEQ_LEN, OUTPUT_DIM = params_dict["INPUT_DIM"], params_dict["SEQ_LEN"], params_dict["OUTPUT_DIM"]
    
    # --- 2. 手动覆盖为最优参数 ---
    HIDDEN_DIM, NUM_LAYERS, DROPOUT_RATE = 64, 2, 0.2

    # --- 3. 初始化模型结构 ---
    model = AsphaltLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, SEQ_LEN, OUTPUT_DIM, DROPOUT_RATE)
    model.load_state_dict(interim_data["model_weights"])
    model.eval()
    
    # 准备数据
    X_test = interim_data["X_test"]
    if X_test.dim() == 2: X_test = X_test.unsqueeze(1)
    
    num_samples = 30 
    X_sample = X_test[:100].detach()
    background = X_sample[:20]
    test_samples = X_sample[:num_samples]
    
    # 计算 SHAP 值
    explainer = shap.GradientExplainer(model, background)
    shap_values_raw = explainer.shap_values(test_samples)
    shap_values = shap_values_raw[0] if isinstance(shap_values_raw, list) else shap_values_raw

    np.save(os.path.join(INTERIM_DIR, "shap_values.npy"), shap_values)
    print("✅ LSTM SHAP值已保存")
    
    # 维度压缩
    shap_values_2d = np.mean(shap_values, axis=1) 
    X_display = test_samples.numpy().mean(axis=1) 
    if shap_values_2d.ndim == 3:
        shap_values_2d = shap_values_2d.squeeze(-1)

    # 修改为中文标签
    full_feature_names = ["油石比", "空隙率", "矿粉用量", "沥青型号", "骨料级配", "温度", "应力水平", "劲度模量"]

    # ------------------ 5. 绘图：全局贡献图 ------------------
    plt.close('all')
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_2d, X_display, feature_names=full_feature_names, plot_type="dot", show=False)
    plt.title("LSTM特征重要性全局分析", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lstm_shap_summary.png"), dpi=600)
    plt.close()
    
    # ------------------ 6. 2x4 高级依赖图矩阵 (中文SCI风格) ------------------
    print("正在生成 2x4 高级特征效应矩阵图...")
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 4, wspace=0.35, hspace=0.4)
    
    # 核心特征统一坐标轴范围
    core_idx = [0, 1, 5, 6] # 油石比, 空隙率, 温度, 应力
    y_limit = np.max(np.abs(shap_values_2d[:, core_idx])) * 1.1

    for i in range(len(full_feature_names)):
        ax = fig.add_subplot(gs[i])
        x_data, y_data = X_display[:, i], shap_values_2d[:, i]
        
        # 散点：深蓝色调
        ax.scatter(x_data, y_data, color='#4C72B0', alpha=0.7, s=45, edgecolors='white', linewidth=0.6)
        
        # 趋势线：深红色二次拟合
        try:
            z = np.polyfit(x_data, y_data, 2)
            p = np.poly1d(z)
            x_range = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x_range, p(x_range), color='#C44E52', lw=2.5, alpha=0.9)
        except: pass
            
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
        
        # 子图编号 + 中文标题
        ax.set_title(f"({chr(97+i)}) {full_feature_names[i]}", loc='left', fontsize=13, fontweight='bold')
        ax.set_xlabel("特征量值", fontsize=11)
        ax.set_ylabel("SHAP Value", fontsize=11)
        
        # 关键特征统一轴限
        if i in core_idx: ax.set_ylim(-y_limit, y_limit)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    plt.suptitle("LSTM-SHAP 沥青疲劳寿命边际效应影响分析", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, "lstm_feature_dependence_matrix_2x4.png"), dpi=600, bbox_inches='tight')
    plt.close()

    print(f"SHAP分析圆满完成！\n图片已保存至: {OUTPUT_DIR}\n包含：summary图与2x4矩阵图 (600 DPI)")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("正在初始化环境并加载模型...")
    params, interim_data = check_prerequisite()
    
    print("正在开始 SHAP 分析...")
    shap_analysis(params, interim_data)
    
    print("程序运行结束！请检查 output 目录。")