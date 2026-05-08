"""
Transformer训练器
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_chinese_font():
    """获取可用的中文字体"""
    fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'PingFang SC', 'STHeiti']
    for font in fonts:
        if font in [f.name for f in fm.fontManager.ttflist]:
            return font
    return fm.FontProperties(family='sans-serif').get_name()

CHINESE_FONT = get_chinese_font()

sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research')
from AC_Transformer._1_config import (
    INPUT_DIM, TRANSFORMER_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, DROPOUT,
    LEARNING_RATE, EPOCHS, PATIENCE, OUTPUT_DIR, RANDOM_SEED, WEIGHT_DECAY
)
from AC_Transformer._3_transformer_model import TransformerModel


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, val_loader, criterion, device):
    """验证模型"""
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
    plt.title('Transformer训练过程 - 损失曲线', fontproperties=fm.FontProperties(family=CHINESE_FONT))
    plt.legend(prop=fm.FontProperties(family=CHINESE_FONT))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到: {save_path}")


def train_model(train_loader, val_loader, device=None):
    """训练Transformer模型"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    set_seed(RANDOM_SEED)

    # 初始化模型
    model = TransformerModel(
        input_dim=INPUT_DIM,
        d_model=TRANSFORMER_DIM,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        output_dim=1,
        dropout=DROPOUT
    ).to(device)

    print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 使用ReduceLROnPlateau：监控验证损失，损失停滞时降低学习率
    # 这是最稳定的学习率调度策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,       # 每次降低50%
        patience=10,       # 10个epoch没有改善才降低
        min_lr=1e-6       # 最低学习率
    )

    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print("\n开始训练...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        # ReduceLROnPlateau需要传入验证损失
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        epoch_time = time.time() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(OUTPUT_DIR, 'transformer_best_model.pth')
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
            print(f"\nEarly stopping! Epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    print(f"\n训练完成! 总时间: {total_time:.1f}s")
    print(f"最佳验证损失: {best_val_loss:.6f}")

    # 加载最佳模型
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 绘制并保存训练曲线
    plot_path = os.path.join(OUTPUT_DIR, 'transformer_training_history.png')
    plot_training_history(history, plot_path)

    return model, history, best_model_path


if __name__ == '__main__':
    from AC_Transformer._2_sequence_builder import load_and_build_sequences, split_data, create_data_loaders
    
    print("=" * 60)
    print("Transformer时序序列构建")
    print("=" * 60)
    
    sequences, targets, scaler, df = load_and_build_sequences()
    train_data, val_data, test_data = split_data(sequences, targets, df)
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, history, best_path = train_model(train_loader, val_loader, device)
    print(f"模型已保存到: {best_path}")