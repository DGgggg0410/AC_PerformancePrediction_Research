"""
LSTM消融实验 - 模型训练
使用消融后的特征配置进行训练

作者: 研究团队
日期: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from datetime import datetime

# 导入消融实验配置
sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/AC_LSTM/ablation')
from _1_lstm_ablation_no_climate_config import (
    OUTPUT_DIR, EXPERIMENT_NAME, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS,
    DROPOUT, OUTPUT_DIM, EPOCHS, LEARNING_RATE, PATIENCE, RANDOM_SEED
)
from _2_lstm_ablation_sequence_builder import main as build_data

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class LSTMModel(nn.Module):
    """LSTM模型"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(batch_y)
    return total_loss / len(train_loader.dataset)


def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * len(batch_y)
    return total_loss / len(val_loader.dataset)


def evaluate(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).squeeze()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))
    r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)

    return r2, rmse, mae, predictions, actuals


def plot_results(train_losses, val_losses, r2, rmse, mae, experiment_name):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='训练损失', color='blue')
    axes[0].plot(val_losses, label='验证损失', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title(f'{experiment_name} - 训练曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    metrics = ['R^2', 'RMSE\n(m/km)', 'MAE\n(m/km)']
    values = [r2, rmse, mae]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = axes[1].bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('值')
    axes[1].set_title(f'{experiment_name} - 测试集性能指标')
    axes[1].set_ylim(0, max(values) * 1.2)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{experiment_name}_performance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"性能图表已保存: {experiment_name}_performance.png")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print(f"消融实验 [{EXPERIMENT_NAME}] - LSTM模型训练")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("\n构建数据集...")
    train_loader, val_loader, test_loader, scaler, _ = build_data()

    print(f"\n创建LSTM模型 (INPUT_DIM={INPUT_DIM}, HIDDEN_DIM={HIDDEN_DIM})...")
    model = LSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, OUTPUT_DIM).to(device)
    print(f"模型结构:\n{model}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    print("\n开始训练...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | "
                  f"训练损失: {train_loss:.6f} | "
                  f"验证损失: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{EXPERIMENT_NAME}_best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n早停触发于 Epoch {epoch+1}")
                break

    print("\n加载最佳模型进行评估...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'{EXPERIMENT_NAME}_best_model.pth')))

    r2, rmse, mae, predictions, actuals = evaluate(model, test_loader, device)

    print(f"\n{'='*60}")
    print(f"消融实验 [{EXPERIMENT_NAME}] - 测试集性能")
    print(f"{'='*60}")
    print(f"R^2  (决定系数): {r2:.4f}")
    print(f"RMSE (均方根误差): {rmse:.4f} m/km")
    print(f"MAE  (平均绝对误差): {mae:.4f} m/km")
    print(f"{'='*60}")

    plot_results(train_losses, val_losses, r2, rmse, mae, EXPERIMENT_NAME)

    results = {
        'experiment_name': EXPERIMENT_NAME,
        'description': '去掉气候因素 - 保留时序(3) + 结构(5) + 地理(3) = 11个特征',
        'features_count': INPUT_DIM,
        'metrics': {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae)
        },
        'training_info': {
            'epochs_trained': len(train_losses),
            'best_val_loss': float(best_val_loss),
            'final_train_loss': float(train_losses[-1])
        }
    }

    results_path = os.path.join(OUTPUT_DIR, f'{EXPERIMENT_NAME}_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {results_path}")

    return results


if __name__ == '__main__':
    results = main()
    print("\n消融实验训练完成!")
