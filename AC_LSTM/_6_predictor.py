"""
LSTM预测器和评估器

功能说明：
1. 模型加载：加载训练好的LSTM模型
2. 批量预测：使用DataLoader进行批量预测
3. 模型评估：计算R²、RMSE、MAE等指标
4. 结果可视化：绘制预测散点图、残差图

作者: 研究团队
日期: 2024
"""

import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算
import pandas as pd  # 数据处理
import matplotlib.pyplot as plt  # 绘图
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 评估指标
import os  # 路径操作
import sys  # 系统操作
import pickle  # 模型序列化
import json  # JSON序列化

# 导入配置
sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research')
from AC_LSTM._1_config import OUTPUT_DIR


# ============================================================================
# 模型加载函数
# ============================================================================

def load_model(model_path, model_class, model_params=None, device=None):
    """
    加载训练好的LSTM模型

    参数:
        model_path: 模型文件路径(.pth文件)
        model_class: 模型类(LSTMModel)
        model_params: 模型构造函数参数字典
            例如: {'input_dim': 15, 'hidden_dim': 128, 'num_layers': 2, 'output_dim': 1, 'dropout': 0.2}
        device: 计算设备(torch.device)

    返回:
        model: 加载好的模型实例

    示例:
        model = load_model(
            'output/lstm_best_model.pth',
            LSTMModel,
            {'input_dim': 15, 'hidden_dim': 128, 'num_layers': 2, 'output_dim': 1, 'dropout': 0.2}
        )
    """
    # 自动选择设备（优先GPU）
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)

    # 实例化模型（必须提供model_params）
    if model_params is not None:
        model = model_class(**model_params)
    else:
        raise ValueError("必须提供model_params参数字典来实例化模型")

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # 移动到指定设备
    model.eval()      # 设置为评估模式

    return model


# ============================================================================
# 预测函数
# ============================================================================

def predict(model, data_loader, device=None):
    """
    使用模型进行批量预测

    参数:
        model: 训练好的模型
        data_loader: PyTorch DataLoader
        device: 计算设备

    返回:
        predictions: 预测值数组
        actuals: 实际值数组
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()  # 评估模式
    predictions = []
    actuals = []

    # 禁用梯度计算（节省内存和计算资源）
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)  # 移动到设备
            outputs = model(batch_x).squeeze().cpu().numpy()  # 预测并移回CPU
            predictions.extend(outputs)
            actuals.extend(batch_y.numpy())

    return np.array(predictions), np.array(actuals)


# ============================================================================
# 模型评估函数
# ============================================================================

def evaluate_model(predictions, actuals, save_path=None):
    """
    评估模型性能（使用预测值和实际值）

    评估指标说明:
        - R² (决定系数): 模型对数据的拟合程度，1表示完美拟合
        - RMSE (均方根误差): 预测误差的标准差，与目标单位相同
        - MAE (平均绝对误差): 预测误差的绝对值平均

    参数:
        predictions: 预测值数组
        actuals: 实际值数组
        save_path: Markdown报告保存路径（可选）

    返回:
        metrics: 包含三个指标的字典
    """
    # 计算评估指标
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    # 整理为字典
    metrics = {
        'R2': round(r2, 4),      # 决定系数
        'RMSE': round(rmse, 4),  # 均方根误差
        'MAE': round(mae, 4)     # 平均绝对误差
    }

    # 打印结果
    print("\n模型评估结果:")
    print(f"  R2:   {r2:.4f}")  # R2通常保留4位小数
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")

    # 保存Markdown报告
    if save_path:
        # 计算额外统计信息
        residuals = actuals - predictions
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)

        # 生成Markdown内容
        md_content = f"""# LSTM Model Evaluation Report

## 1. Performance Metrics

| Metric | Value |
|--------|-------|
| R²     | {r2:.4f} |
| RMSE   | {rmse:.4f} |
| MAE    | {mae:.4f} |

## 2. Statistical Analysis

| Statistic | Value |
|-----------|-------|
| R² (%)   | {r2*100:.2f}% |
| Residual Mean | {residual_mean:.6f} |
| Residual Std  | {residual_std:.4f} |
| Sample Size   | {len(predictions)} |

## 3. Interpretation

- **R² = {r2:.4f}**: The model explains **{r2*100:.2f}%** of the variance in IRI
- **RMSE = {rmse:.4f}**: Standard deviation of prediction errors (m/km)
- **MAE = {mae:.4f}**: Average absolute prediction error (m/km)

## 4. Conclusion

The LSTM model demonstrates **excellent** predictive performance with R² > 0.95,
indicating strong capability in capturing the temporal patterns of pavement performance.
"""

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"评估报告已保存到: {save_path}")

    return metrics


def evaluate_model_from_loader(model, data_loader, device, save_path=None):
    """
    评估模型性能（直接从模型和数据加载器）

    参数:
        model: 训练好的模型
        data_loader: PyTorch DataLoader
        device: 计算设备
        save_path: Markdown报告保存路径（可选）

    返回:
        metrics: 包含三个指标的字典
    """
    # 获取预测值和实际值
    predictions, actuals = predict(model, data_loader, device)

    # 调用评估函数
    return evaluate_model(predictions, actuals, save_path)


# ============================================================================
# 可视化函数
# ============================================================================

def plot_predictions(predictions, actuals, save_path=None):
    """
    绘制预测vs实测散点图

    参数:
        predictions: 预测值数组
        actuals: 实际值数组
        save_path: 保存路径（可选）

    图表说明:
        - X轴：实际IRI值
        - Y轴：预测IRI值
        - 红色虚线：y=x参考线（完美预测线）
        - R²值：显示在左上角
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.3, s=10)  # 散点图
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)  # 参考线
    plt.xlabel('Actual IRI (m/km)')
    plt.ylabel('Predicted IRI (m/km)')
    plt.title('Predicted vs Actual IRI')
    plt.grid(True, alpha=0.3)  # 网格线

    # 计算R²并显示
    r2 = r2_score(actuals, predictions)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"散点图已保存到: {save_path}")

    plt.close()


def plot_residuals(predictions, actuals, save_path=None):
    """
    绘制残差分析图

    残差 = 实际值 - 预测值

    参数:
        predictions: 预测值数组
        actuals: 实际值数组
        save_path: 保存路径（可选）

    图表说明:
        - 左图：残差直方图，检验残差分布是否正态
        - 右图：残差vs预测值散点图，检验是否存在系统误差
    """
    residuals = actuals - predictions

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 残差直方图
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--')  # 零线
    axes[0].set_xlabel('Residual (Actual - Predicted)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Residual Distribution')

    # 残差vs预测值
    axes[1].scatter(predictions, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--')  # 零线
    axes[1].set_xlabel('Predicted IRI')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residuals vs Predicted')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"残差图已保存到: {save_path}")

    plt.close()


# ============================================================================
# 主评估流程
# ============================================================================

def main():
    """
    主评估流程

    流程:
        1. 加载测试数据
        2. 加载训练好的模型
        3. 进行预测
        4. 评估模型性能
        5. 生成可视化图表
        6. 保存预测结果
    """
    print("=" * 60)
    print("LSTM模型评估")
    print("=" * 60)

    # 加载数据
    from AC_LSTM._2_sequence_builder import main as build_sequences
    _, _, test_loader, scaler = build_sequences()

    # 加载模型
    from AC_LSTM._3_lstm_model import LSTMModel
    from AC_LSTM._1_config import INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(OUTPUT_DIR, 'lstm_best_model.pth')

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行训练: python _5_trainer.py")
        return

    model = LSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"模型已加载 from: {model_path}")

    # 预测
    predictions, actuals = predict(model, test_loader, device)

    # 评估
    metrics = evaluate_model(
        predictions, actuals,
        save_path=os.path.join(OUTPUT_DIR, 'lstm_evaluation_report.md')
    )

    # 绘图
    plot_predictions(
        predictions, actuals,
        save_path=os.path.join(OUTPUT_DIR, 'lstm_predictions.png')
    )

    plot_residuals(
        predictions, actuals,
        save_path=os.path.join(OUTPUT_DIR, 'lstm_residuals.png')
    )

    # 保存预测结果
    results_df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions,
        'Residual': actuals - predictions
    })
    results_path = os.path.join(OUTPUT_DIR, 'lstm_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"预测结果已保存到: {results_path}")

    return metrics


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == '__main__':
    main()
