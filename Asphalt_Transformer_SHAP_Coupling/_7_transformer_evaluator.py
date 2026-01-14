# -*- coding: utf-8 -*-
# 步骤7：Transformer模型评估+绘制结果图表
# 运行成功后：生成最终评估结果，下一步运行6_transformer_shap_analyzer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
# 【仅新增】引入标准绝对路径
from _1_config import INTERIM_DIR, OUTPUT_DIR

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from _3_transformer_model import AsphaltTransformer

# ---------------------- 前置校验+加载数据 ----------------------
def check_prerequisite():
    """校验步骤4是否完成"""
    required_files = ["step4_best_transformer_model.pth", "step4_train_loss.joblib", "step4_val_loss.joblib",
                      "step2_X_test_tensor.pt", "step2_y_test_tensor.pt", "step2_scaler_y.joblib"]
    for file in required_files:
        # 【最小修改】使用标准路径校验
        file_path = os.path.join(INTERIM_DIR, file)
        if not os.path.exists(file_path):
            print(f"错误！缺少中间产物：{file_path}，请先运行对应步骤")
            exit(1)
    # 加载参数+数据
    params = joblib.load(os.path.join(INTERIM_DIR, "step1_params.joblib"))
    interim_data = {
        "model_weights": torch.load(os.path.join(INTERIM_DIR, "step4_best_transformer_model.pth")),
        "train_loss": joblib.load(os.path.join(INTERIM_DIR, "step4_train_loss.joblib")),
        "val_loss": joblib.load(os.path.join(INTERIM_DIR, "step4_val_loss.joblib")),
        "X_test": torch.load(os.path.join(INTERIM_DIR, "step2_X_test_tensor.pt")),
        "y_test": torch.load(os.path.join(INTERIM_DIR, "step2_y_test_tensor.pt")),
        "scaler_y": joblib.load(os.path.join(INTERIM_DIR, "step2_scaler_y.joblib"))
    }
    return params, interim_data

# ---------------------- 核心功能：模型评估 ----------------------
def evaluate_model(params_dict, interim_data):
    """评估Transformer模型并保存结果图表+指标"""
    # --- 原有参数提取 ---
    INPUT_DIM = params_dict["INPUT_DIM"]
    SEQ_LEN = params_dict["SEQ_LEN"]
    OUTPUT_DIM = params_dict["OUTPUT_DIM"]
    
    # --- ！！！手动覆盖为最优参数 (必须与训练脚本一致) ！！！ ---
    D_MODEL = 64          
    NHEAD = 2             
    NUM_ENCODER_LAYERS = 1 
    DROPOUT_RATE = 0.1    
    # -------------------------------------------------------
    
    # 提取数据
    model_weights = interim_data["model_weights"]
    train_loss = interim_data["train_loss"]
    val_loss = interim_data["val_loss"]
    X_test = interim_data["X_test"]
    y_test = interim_data["y_test"]
    scaler_y = interim_data["scaler_y"]
    
    # 初始化模型+加载权重
    model = AsphaltTransformer(INPUT_DIM, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, SEQ_LEN, OUTPUT_DIM, DROPOUT_RATE)
    model.load_state_dict(model_weights)
    model.eval()
    
    # 测试集预测+反归一化
    with torch.no_grad():
        y_pred = model(X_test).numpy()
    y_test_original = scaler_y.inverse_transform(y_test.numpy())
    y_pred_original = scaler_y.inverse_transform(y_pred)
    
    # 计算评估指标
    r2 = r2_score(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    
    # 打印指标
    print("="*30)
    print("Transformer模型评估指标")
    print(f"R2：{r2:.4f} | RMSE：{rmse:.4f} | MAE：{mae:.4f}")
    print("="*30)
    
    # 绘制并保存图表
    # 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="训练损失", color="blue")
    plt.plot(val_loss, label="验证损失", color="red")
    plt.xlabel("训练轮数")
    plt.ylabel("MSE损失")
    plt.title("Transformer模型训练/验证损失曲线")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 【最小修改】使用 OUTPUT_DIR
    plt.savefig(os.path.join(OUTPUT_DIR, "transformer_loss_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 预测散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_pred_original, alpha=0.6, color="blue", s=20)
    plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], "r--", linewidth=2)
    plt.xlabel("真实疲劳寿命（万次）")
    plt.ylabel("预测疲劳寿命（万次）")
    plt.title(f"Transformer模型预测结果（R2={r2:.4f}）")
    plt.grid(True, alpha=0.3)
    # 【最小修改】使用 OUTPUT_DIR
    plt.savefig(os.path.join(OUTPUT_DIR, "transformer_pred_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存指标文本 - 【最小修改】使用 OUTPUT_DIR
    metrics_path = os.path.join(OUTPUT_DIR, "transformer_evaluation_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Transformer模型评估指标\n")
        f.write(f"R2 score: {r2:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
    print(f"评估结果已保存至: {OUTPUT_DIR}")

# ---------------------- 独立运行入口 ----------------------
if __name__ == "__main__":
    print("="*50)
    print("步骤7：Transformer模型评估+绘制结果图表")
    print("="*50)
    params, interim_data = check_prerequisite()
    evaluate_model(params, interim_data)
    print("="*50)
    print("步骤7运行成功！下一步运行：8_transformer_shap_analyzer.py")
    print("="*50)