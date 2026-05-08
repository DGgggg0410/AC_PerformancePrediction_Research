"""
LSTM 5年预测训练器
扩展实验：用前5年数据预测未来第5年

与基础训练器的差异：
- 使用 _1b_config.py 配置
- 使用 _2b_sequence_builder.py 构建序列
- 输出到 output_5yr 目录

作者: 研究团队
日期: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys
import time

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_chinese_font():
    fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'PingFang SC', 'STHeiti']
    for font in fonts:
        if font in [f.name for f in fm.fontManager.ttflist]:
            return font
    return fm.FontProperties(family='sans-serif').get_name()

CHINESE_FONT = get_chinese_font()

sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research')
from AC_LSTM._1b_config import (
    INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT,
    LEARNING_RATE, EPOCHS, PATIENCE, OUTPUT_DIR, RANDOM_SEED, PREDICT_HORIZON
)
from AC_LSTM._3_lstm_model import LSTMModel


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, val_loader, criterion, device):
    """在验证集上评估"""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def plot_training_history(history, save_path):
    """绘制并保存训练曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    plt.xlabel('Epoch', fontproperties=fm.FontProperties(family=CHINESE_FONT))
    plt.ylabel('Loss (MSE)', fontproperties=fm.FontProperties(family=CHINESE_FONT))
    plt.title(f'LSTM 5年预测训练过程 - 损失曲线', fontproperties=fm.FontProperties(family=CHINESE_FONT))
    plt.legend(prop=fm.FontProperties(family=CHINESE_FONT))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到: {save_path}")


def train_model(train_loader, val_loader, device=None):
    """训练LSTM 5年预测模型"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[5年预测] 使用设备: {device}")

    set_seed(RANDOM_SEED)

    model = LSTMModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT
    ).to(device)

    print(f"[5年预测] 模型参数总数: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"\n[5年预测] 开始训练 (预测未来{PREDICT_HORIZON}年)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        epoch_time = time.time() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(OUTPUT_DIR, 'lstm_5yr_best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Time: {epoch_time:.1f}s")

        if patience_counter >= PATIENCE:
            print(f"\n[5年预测] Early stopping! Epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    print(f"\n[5年预测] 训练完成! 总时间: {total_time:.1f}s")
    print(f"[5年预测] 最佳验证损失: {best_val_loss:.6f}")

    # 加载最佳模型
    best_model_path = os.path.join(OUTPUT_DIR, 'lstm_5yr_best_model.pth')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 绘制训练曲线
    plot_path = os.path.join(OUTPUT_DIR, 'lstm_5yr_training_history.png')
    plot_training_history(history, plot_path)

    return model, history, best_model_path


if __name__ == '__main__':
    from AC_LSTM._2b_sequence_builder import load_and_build_sequences, split_data, create_data_loaders

    print("=" * 60)
    print(f"LSTM 5年预测训练 (PREDICT_HORIZON={PREDICT_HORIZON})")
    print("=" * 60)

    sequences, targets, scaler, df = load_and_build_sequences()
    train_data, val_data, test_data = split_data(sequences, targets, df)
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, history, best_path = train_model(train_loader, val_loader, device)
    print(f"[5年预测] 模型已保存到: {best_path}")

    # 在测试集上评估
    print("\n" + "=" * 60)
    print("[5年预测] 在测试集上评估最终性能")
    print("=" * 60)

    from AC_LSTM._6b_predictor import evaluate_model_from_loader
    test_metrics = evaluate_model_from_loader(
        model, test_loader, device,
        save_path=os.path.join(OUTPUT_DIR, 'lstm_5yr_test_evaluation_report.md')
    )
    print(f"\n[5年预测] 测试集最终评估结果:")
    print(f"  R²: {test_metrics['R2']:.4f}")
    print(f"  RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  MAE: {test_metrics['MAE']:.4f}")
