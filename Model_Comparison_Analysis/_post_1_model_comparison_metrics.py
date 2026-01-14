# -*- coding: utf-8 -*-
"""
Post 1: 多模型性能指标自动提取与汇总
功能：自动跨文件夹读取 LSTM 和 Transformer 的调优结果，生成对比报表。
"""
import pandas as pd
import os
import sys

# ---------------------- [关键修改] 路径自动化定位 ----------------------
# 1. 获取当前脚本所在文件夹的绝对路径 (Analysis_Post_Processing)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 获取项目总根目录 (AC_PerformancePrediction_Research)
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# 3. 将 Transformer 子文件夹加入 Python 搜索路径，以便导入 _1_config
TRANS_SUBDIR = os.path.join(ROOT_DIR, "Asphalt_Transformer_SHAP_Coupling")
if TRANS_SUBDIR not in sys.path:
    sys.path.append(TRANS_SUBDIR)

try:
    # 借用 Transformer 的配置来获取标准的路径定义
    from _1_config import OUTPUT_DIR as TRANS_OUTPUT_DIR
    print("✅ 成功链接项目配置文件并识别路径结构")
except ImportError:
    print(f"❌ 严重错误：无法在 {TRANS_SUBDIR} 中找到 _1_config.py")
    exit(1)

# ---------------------- 1. 路径定义 ----------------------
# 定位 LSTM 的输出目录
LSTM_OUTPUT_DIR = os.path.join(ROOT_DIR, "Asphalt_LSTM_SHAP_Sequential", "output")

# 【核心修改】：将输出路径修改为 CURRENT_DIR (即 Model_Comparison_Analysis 文件夹)
FINAL_REPORT_PATH = os.path.join(CURRENT_DIR, "final_model_comparison_report.csv")

def get_best_metrics(file_path, model_name):
    """从 CSV 文件中提取 R2 最高的一组评估指标"""
    if not os.path.exists(file_path):
        print(f"⚠️ 警告：未找到 {model_name} 的结果文件: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"⚠️ 警告：{model_name} 的结果文件内容为空")
            return None
            
        # 按 R2 降序排列，取第一行（性能最优的模型组合）
        best_row = df.sort_values(by="val_r2", ascending=False).iloc[0].copy()
        
        # 封装为统一的对比字典
        metrics = {
            "Model": model_name,
            "Best_R2": round(best_row["val_r2"], 4),
            "Best_RMSE": round(best_row["val_rmse"], 4),
            "Best_MAE": round(best_row["val_mae"], 4),
            "Best_Loss": round(best_row["val_loss"], 6)
        }
        return metrics
    except Exception as e:
        print(f"❌ 读取 {model_name} 数据时发生错误: {e}")
        return None

# ---------------------- 2. 执行汇总与计算 ----------------------
print("\n" + "="*50)
print("正在执行：跨模型性能指标汇总分析")
print("="*50)

# 拼接具体的 CSV 文件路径
lstm_csv_path = os.path.join(LSTM_OUTPUT_DIR, "lstm_hyperparam_results.csv")
trans_csv_path = os.path.join(TRANS_OUTPUT_DIR, "transformer_hyperparam_results.csv")

# 提取数据
lstm_metrics = get_best_metrics(lstm_csv_path, "LSTM")
trans_metrics = get_best_metrics(trans_csv_path, "Transformer")

results = []
if lstm_metrics: results.append(lstm_metrics)
if trans_metrics: results.append(trans_metrics)

if results:
    comparison_df = pd.DataFrame(results)
    
    # 如果两个模型的数据都有，则计算性能差值 (Gap)
    if len(results) == 2:
        gap = {
            "Model": "Difference (L - T)",
            "Best_R2": round(results[0]["Best_R2"] - results[1]["Best_R2"], 4),
            "Best_RMSE": round(results[0]["Best_RMSE"] - results[1]["Best_RMSE"], 4),
            "Best_MAE": round(results[0]["Best_MAE"] - results[1]["Best_MAE"], 4),
            "Best_Loss": round(results[0]["Best_Loss"] - results[1]["Best_Loss"], 6)
        }
        comparison_df = pd.concat([comparison_df, pd.DataFrame([gap])], ignore_index=True)

    # 结果展示
    print("\n[模型性能对比预览]")
    print(comparison_df.to_string(index=False))
    
    # 保存结果
    comparison_df.to_csv(FINAL_REPORT_PATH, index=False, encoding="utf-8-sig")
    print("-" * 50)
    print(f"✅ 汇总报表已成功保存至:\n   {FINAL_REPORT_PATH}")
else:
    print("\n❌ 提取失败：请确保各个子项目的 Step 4 (超参数调优) 已经运行并生成了结果 CSV 文件。")

print("="*50 + "\n")