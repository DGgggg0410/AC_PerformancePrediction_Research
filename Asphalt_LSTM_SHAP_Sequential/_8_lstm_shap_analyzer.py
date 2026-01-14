# -*- coding: utf-8 -*-
# 步骤8：LSTM模型SHAP分析+特征贡献度（独立可运行，前置：步骤7）
# 运行成功后：全流程完成，生成SHAP可解释性结果
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import shap
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
from _3_lstm_model import AsphaltLSTM
# 【仅新增】引入绝对路径变量
from _1_config import INTERIM_DIR, OUTPUT_DIR

# ---------------------- 前置校验+加载数据 ----------------------
def check_prerequisite():
    """校验步骤6是否完成"""
    required_files = ["step4_best_lstm_model.pth", "step2_X_test_tensor.pt"]
    for file in required_files:
        # 【最小修改】使用从 config 导入的绝对路径变量
        file_path = os.path.join(INTERIM_DIR, file)
        if not os.path.exists(file_path):
            print(f"错误！缺少中间产物：{file_path}，请先运行对应步骤")
            exit(1)
    # 加载参数+数据
    params = joblib.load(os.path.join(INTERIM_DIR, "step1_params.joblib"))
    interim_data = {
        "model_weights": torch.load(os.path.join(INTERIM_DIR, "step4_best_lstm_model.pth")),
        "X_test": torch.load(os.path.join(INTERIM_DIR, "step2_X_test_tensor.pt"))
    }
    return params, interim_data

def shap_analysis(params_dict, interim_data):
    """进行SHAP分析并绘图"""
    # --- 1. 提取基础参数 ---
    INPUT_DIM = params_dict["INPUT_DIM"]
    SEQ_LEN = params_dict["SEQ_LEN"]
    OUTPUT_DIM = params_dict["OUTPUT_DIM"]
    
    # --- 2. 手动覆盖为最优参数 (必须与你第6、7步改的一模一样) ---
    HIDDEN_DIM = 64         # 对标Transformer的D_MODEL
    NUM_LAYERS = 2          # 对标Transformer的NUM_ENCODER_LAYERS
    DROPOUT_RATE = 0.2      # 保持与重训一致
    # -------------------------------------------------------------

    # --- 3. 初始化模型结构 ---
    model = AsphaltLSTM(
        INPUT_DIM,           
        HIDDEN_DIM,          
        NUM_LAYERS,          
        SEQ_LEN,             
        OUTPUT_DIM,          
        DROPOUT_RATE         
    )
    
    # --- 4. 加载权重 ---
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

    # 【最小修改】保存到绝对路径
    np.save(os.path.join(INTERIM_DIR, "shap_values.npy"), shap_values)
    print("✅ LSTM SHAP值已保存")
    
    # 维度压缩
    shap_values_2d = np.mean(shap_values, axis=1) 
    X_display = test_samples.numpy().mean(axis=1) 
    
    # 暴力放大
    scale_factor = 1000
    shap_values_2d = shap_values_2d * scale_factor
    
    full_feature_names = ["油石比(%)", "空隙率(%)", "矿粉用量(%)", "沥青型号", "骨料级配", "温度(℃)", "应力水平(MPa)", "劲度模量"]
    
    if shap_values_2d.ndim == 3:
        shap_values_2d = shap_values_2d.squeeze(-1)
    
    # ------------------ 5. 绘图：全局贡献图 (Summary Plot) ------------------
    plt.close('all')
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values_2d, 
        X_display,
        feature_names=full_feature_names,
        plot_type="dot",
        show=False,
    )
    plt.title(f"LSTM模型特征贡献全局分析 (数值已放大 {scale_factor} 倍)", fontsize=14)
    plt.xlabel("SHAP值 (对疲劳寿命的影响)", fontsize=12)
    plt.tight_layout()
    # 【最小修改】保存到 OUTPUT_DIR 绝对路径
    plt.savefig(os.path.join(OUTPUT_DIR, "lstm_shap_summary.png"), dpi=300)
    plt.close()
    
    # ------------------ 6. 绘图：油石比依赖图 (Dependence Plot) ------------------
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        0, # 第0个特征：油石比
        shap_values_2d, 
        X_display, 
        feature_names=full_feature_names, 
        interaction_index=None, 
        show=False,
    )
    plt.title("油石比与疲劳寿命的边际效应分析")
    plt.tight_layout()
    # 【最小修改】保存到 OUTPUT_DIR 绝对路径
    plt.savefig(os.path.join(OUTPUT_DIR, "lstm_shap_dependence_oil_stone.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP分析圆满完成！请查看 output 目录下的 summary 和 dependence 图片。")

if __name__ == "__main__":
    # 【最小修改】使用 config 导入的绝对路径进行目录检查
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("正在初始化环境并加载模型...")
    params, interim_data = check_prerequisite()
    
    print("正在开始 SHAP 分析，这可能需要 1-2 分钟...")
    shap_analysis(params, interim_data)
    
    print("程序运行结束！请检查 output 目录。")