"""
Transformer预测器和评估器
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import sys

sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research')
from AC_Transformer._1_config import OUTPUT_DIR


def predict(model, data_loader, device=None):
    """使用模型进行预测"""
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


def evaluate_model(predictions, actuals):
    """评估模型性能"""
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }

    print("\n模型评估结果:")
    print(f"  R²:  {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")

    return metrics


def plot_predictions(predictions, actuals, save_path=None):
    """绘制预测vs实测散点图"""
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.3, s=10)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual IRI (m/km)')
    plt.ylabel('Predicted IRI (m/km)')
    plt.title('Predicted vs Actual IRI (Transformer)')
    plt.grid(True, alpha=0.3)

    r2 = r2_score(actuals, predictions)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"散点图已保存到: {save_path}")

    plt.close()


def plot_residuals(predictions, actuals, save_path=None):
    """绘制残差分布图"""
    residuals = actuals - predictions

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--')
    axes[0].set_xlabel('Residual (Actual - Predicted)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Residual Distribution (Transformer)')

    axes[1].scatter(predictions, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted IRI')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residuals vs Predicted')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"残差图已保存到: {save_path}")

    plt.close()


def main():
    """主评估流程"""
    print("=" * 60)
    print("Transformer模型评估")
    print("=" * 60)

    # 加载数据
    from AC_Transformer._2_sequence_builder import main as build_sequences
    _, _, test_loader, scaler, _ = build_sequences()

    # 加载模型
    from AC_Transformer._3_transformer_model import TransformerModel
    from AC_Transformer._1_config import INPUT_DIM, TRANSFORMER_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, DROPOUT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(OUTPUT_DIR, 'transformer_best_model.pth')

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行训练: python _6_trainer.py")
        return

    model = TransformerModel(INPUT_DIM, TRANSFORMER_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, 1, DROPOUT)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"模型已加载 from: {model_path}")

    # 预测
    predictions, actuals = predict(model, test_loader, device)

    # 评估
    metrics = evaluate_model(predictions, actuals)

    # 绘图
    plot_predictions(
        predictions, actuals,
        save_path=os.path.join(OUTPUT_DIR, 'transformer_predictions.png')
    )

    plot_residuals(
        predictions, actuals,
        save_path=os.path.join(OUTPUT_DIR, 'transformer_residuals.png')
    )

    # 保存预测结果
    results_df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions,
        'Residual': actuals - predictions
    })
    results_path = os.path.join(OUTPUT_DIR, 'transformer_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"预测结果已保存到: {results_path}")

    # 生成Markdown报告
    report_path = os.path.join(OUTPUT_DIR, 'transformer_evaluation_report.md')
    generate_markdown_report(model, metrics, checkpoint, report_path)
    print(f"评估报告已保存到: {report_path}")

    return metrics


def generate_markdown_report(model, metrics, checkpoint, save_path):
    """生成Markdown格式的评估报告"""
    from datetime import datetime

    # 获取模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 获取训练配置
    from AC_Transformer._1_config import (
        INPUT_DIM, TRANSFORMER_DIM, NUM_HEADS, NUM_LAYERS,
        FF_DIM, DROPOUT, SEQ_LEN, LEARNING_RATE, BATCH_SIZE
    )

    report_content = f"""# Transformer模型评估报告

## 📊 评估时间

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📋 模型配置

| 参数 | 值 |
|------|-----|
| INPUT_DIM | {INPUT_DIM} |
| TRANSFORMER_DIM | {TRANSFORMER_DIM} |
| NUM_HEADS | {NUM_HEADS} |
| NUM_LAYERS | {NUM_LAYERS} |
| FF_DIM | {FF_DIM} |
| DROPOUT | {DROPOUT} |
| SEQ_LEN | {SEQ_LEN} |
| LEARNING_RATE | {LEARNING_RATE} |
| BATCH_SIZE | {BATCH_SIZE} |

---

## 📈 模型规模

| 指标 | 值 |
|------|-----|
| 总参数量 | {total_params:,} |
| 可训练参数量 | {trainable_params:,} |

---

## 🎯 评估指标

| 指标 | 值 | 说明 |
|------|-----|------|
| **R²** | {metrics['R2']:.6f} | 决定系数，越接近1越好 |
| **RMSE** | {metrics['RMSE']:.6f} | 均方根误差 (m/km) |
| **MAE** | {metrics['MAE']:.6f} | 平均绝对误差 (m/km) |

### 指标解读

- **R² = {metrics['R2']:.4f}**：模型解释了 **{metrics['R2']*100:.2f}%** 的IRI方差
- **RMSE = {metrics['RMSE']:.4f} m/km**：预测误差的标准差
- **MAE = {metrics['MAE']:.4f} m/km**：平均绝对误差

---

## 🏋️ 训练信息

| 项目 | 值 |
|------|-----|
| 最佳Epoch | {checkpoint.get('epoch', 'N/A') + 1 if isinstance(checkpoint.get('epoch'), int) else 'N/A'} |
| 最佳验证损失 | {checkpoint.get('val_loss', 'N/A'):.6f} |

---

## 📁 输出文件

- `transformer_best_model.pth` - 最佳模型权重
- `transformer_predictions.csv` - 预测结果数据
- `transformer_predictions.png` - 预测vs实测散点图
- `transformer_residuals.png` - 残差分析图
- `transformer_training_history.png` - 训练历史曲线
- `transformer_evaluation_report.md` - 本报告

---

*报告由 _6_predictor.py 自动生成*
"""

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"Markdown报告已生成: {save_path}")


if __name__ == '__main__':
    main()