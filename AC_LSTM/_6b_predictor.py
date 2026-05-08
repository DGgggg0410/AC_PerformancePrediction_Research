"""
LSTM 5年预测器和评估器
扩展实验：用前5年数据预测未来第5年

与基础预测器的差异：
- 使用 _1b_config.py 配置
- 使用 _2b_sequence_builder.py 构建数据
- 加载 lstm_5yr_best_model.pth
- 输出文件添加 _5yr 后缀

作者: 研究团队
日期: 2024
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import sys
import pickle

sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research')
from AC_LSTM._1b_config import OUTPUT_DIR, PREDICT_HORIZON


def predict(model, data_loader, device=None):
    """使用模型进行批量预测"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).squeeze().cpu().numpy()
            predictions.extend(outputs)
            actuals.extend(batch_y.numpy())

    return np.array(predictions), np.array(actuals)


def evaluate_model(predictions, actuals, save_path=None):
    """评估模型性能"""
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    metrics = {
        'R2': round(r2, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4)
    }

    print(f"\n[5年预测] 模型评估结果 (预测未来{PREDICT_HORIZON}年):")
    print(f"  R²:  {r2:.4f}")
    print(f"  RMSE: {rmse:.4f} (m/km)")
    print(f"  MAE:  {mae:.4f} (m/km)")

    if save_path:
        residuals = actuals - predictions
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)

        md_content = f"""# LSTM 5年预测模型评估报告

## 实验配置

| 参数 | 值 |
|------|-----|
| 预测期限 | 未来{PREDICT_HORIZON}年 |
| 序列长度 | 5年 |
| 输入特征 | 19维 |

---

## 评估指标

| 指标 | 值 | 说明 |
|------|-----|------|
| **R²** | {r2:.4f} | 决定系数，越接近1越好 |
| **RMSE** | {rmse:.4f} | 均方根误差 (m/km) |
| **MAE** | {mae:.4f} | 平均绝对误差 (m/km) |

### 指标解读

- **R² = {r2:.4f}**：模型解释了 **{r2*100:.2f}%** 的{PREDICT_HORIZON}年后IRI方差
- **RMSE = {rmse:.4f} m/km**：预测误差的标准差
- **MAE = {mae:.4f} m/km**：平均绝对误差

### 统计信息

| 统计量 | 值 |
|--------|-----|
| 残差均值 | {residual_mean:.6f} |
| 残差标准差 | {residual_std:.4f} |
| 样本数量 | {len(predictions)} |

---

## 与1年预测对比

| 指标 | 1年预测 | {PREDICT_HORIZON}年预测 | 变化 |
|------|---------|------------------------|------|
| R² | ~0.95+ | {r2:.4f} | 预期下降 |
| RMSE | ~0.1 | {rmse:.4f} | 预期上升 |

---

*报告由 _6b_predictor.py 自动生成*
"""

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"[5年预测] 评估报告已保存到: {save_path}")

    return metrics


def evaluate_model_from_loader(model, data_loader, device, save_path=None):
    """直接从模型和DataLoader评估"""
    predictions, actuals = predict(model, data_loader, device)
    return evaluate_model(predictions, actuals, save_path)


def plot_predictions(predictions, actuals, save_path=None):
    """绘制预测vs实测散点图"""
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.3, s=10)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual IRI (m/km)')
    plt.ylabel(f'Predicted IRI (m/km) - {PREDICT_HORIZON} Year Ahead')
    plt.title(f'LSTM {PREDICT_HORIZON}-Year Ahead Prediction: Predicted vs Actual')
    plt.grid(True, alpha=0.3)

    r2 = r2_score(actuals, predictions)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[5年预测] 散点图已保存到: {save_path}")

    plt.close()


def plot_residuals(predictions, actuals, save_path=None):
    """绘制残差分析图"""
    residuals = actuals - predictions

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--')
    axes[0].set_xlabel('Residual (Actual - Predicted)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'LSTM {PREDICT_HORIZON}-Year Residual Distribution')

    axes[1].scatter(predictions, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel(f'Predicted IRI ({PREDICT_HORIZON}-Year Ahead)')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residuals vs Predicted')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[5年预测] 残差图已保存到: {save_path}")

    plt.close()


def main():
    """主评估流程"""
    print("=" * 60)
    print(f"LSTM {PREDICT_HORIZON}年预测模型评估")
    print("=" * 60)

    # 加载数据
    from AC_LSTM._2b_sequence_builder import main as build_sequences
    _, _, test_loader, _ = build_sequences()

    # 加载模型
    from AC_LSTM._3_lstm_model import LSTMModel
    from AC_LSTM._1b_config import INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(OUTPUT_DIR, 'lstm_5yr_best_model.pth')

    if not os.path.exists(model_path):
        print(f"[5年预测] 模型文件不存在: {model_path}")
        print("[5年预测] 请先运行训练: python _5b_trainer.py")
        return

    model = LSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"[5年预测] 模型已加载 from: {model_path}")

    # 预测
    predictions, actuals = predict(model, test_loader, device)

    # 评估
    metrics = evaluate_model(
        predictions, actuals,
        save_path=os.path.join(OUTPUT_DIR, 'lstm_5yr_evaluation_report.md')
    )

    # 绘图
    plot_predictions(
        predictions, actuals,
        save_path=os.path.join(OUTPUT_DIR, 'lstm_5yr_predictions.png')
    )

    plot_residuals(
        predictions, actuals,
        save_path=os.path.join(OUTPUT_DIR, 'lstm_5yr_residuals.png')
    )

    # 保存预测结果
    results_df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions,
        'Residual': actuals - predictions
    })
    results_path = os.path.join(OUTPUT_DIR, 'lstm_5yr_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"[5年预测] 预测结果已保存到: {results_path}")

    return metrics


if __name__ == '__main__':
    main()
