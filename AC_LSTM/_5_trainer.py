"""
LSTM训练器

功能说明：
1. 模型训练：使用训练集数据训练LSTM模型
2. 验证调参：使用验证集监控模型性能
3. 早停机制：防止过拟合
4. 学习率调度：动态调整学习率
5. 模型保存：保存最佳模型检查点

训练策略说明：
- 使用MSE损失函数（均方误差）
- 使用Adam优化器（自适应学习率）
- 使用ReduceLROnPlateau调度器（验证损失不降时降低学习率）
- 使用EarlyStopping（早停）防止过拟合

作者: 研究团队
日期: 2024
"""

import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torch.utils.data import DataLoader  # 数据加载器
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图
import matplotlib.font_manager as fm  # 字体管理
import os  # 路径操作
import sys  # 系统操作
import time  # 时间测量

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

# 导入配置和模型
sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research')
from AC_LSTM._1_config import (
    INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT,
    LEARNING_RATE, EPOCHS, PATIENCE, OUTPUT_DIR, RANDOM_SEED
)
from AC_LSTM._3_lstm_model import LSTMModel


# ============================================================================
# 随机种子设置
# ============================================================================

def set_seed(seed):
    """
    设置所有随机种子以确保实验可复现

    参数:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ============================================================================
# 训练函数
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个epoch

    参数:
        model: LSTM模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备

    返回:
        avg_loss: 平均训练损失
    """
    model.train()  # 训练模式
    total_loss = 0
    n_batches = 0

    for batch_x, batch_y in train_loader:
        # 移动数据到设备
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # 前向传播
        optimizer.zero_grad()  # 梯度清零
        outputs = model(batch_x).squeeze()  # 前向传播
        loss = criterion(outputs, batch_y)  # 计算损失

        # 反向传播
        loss.backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()  # 更新参数

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, val_loader, criterion, device):
    """
    在验证集上评估模型

    参数:
        model: LSTM模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备

    返回:
        avg_loss: 平均验证损失
    """
    model.eval()  # 评估模式
    total_loss = 0
    n_batches = 0

    # 禁用梯度计算
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


# ============================================================================
# 训练曲线可视化
# ============================================================================

def plot_training_history(history, save_path):
    """
    绘制并保存训练曲线

    参数:
        history: 训练历史，{'train_loss': [], 'val_loss': []}
        save_path: 保存路径
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    plt.xlabel('Epoch', fontproperties=fm.FontProperties(family=CHINESE_FONT))
    plt.ylabel('Loss (MSE)', fontproperties=fm.FontProperties(family=CHINESE_FONT))
    plt.title('LSTM训练过程 - 损失曲线', fontproperties=fm.FontProperties(family=CHINESE_FONT))
    plt.legend(prop=fm.FontProperties(family=CHINESE_FONT))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到: {save_path}")


# ============================================================================
# 模型训练主函数
# ============================================================================

def train_model(train_loader, val_loader, device=None):
    """
    训练LSTM模型

    参数:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备（默认自动选择）

    返回:
        model: 训练好的模型
        history: 训练历史{'train_loss': [], 'val_loss': []}
        best_model_path: 最佳模型保存路径
    """
    # 自动选择设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 设置随机种子
    set_seed(RANDOM_SEED)

    # 初始化模型
    model = LSTMModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT
    ).to(device)

    print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")

    # 损失函数：均方误差（MSE）
    criterion = nn.MSELoss()

    # 优化器：Adam（自适应学习率优化算法）+ L2正则化防止过拟合
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # 学习率调度器：当验证损失不再下降时，降低学习率
    # factor=0.5: 学习率减半
    # patience=5: 连续5个epoch不改善则触发
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',       # 监控验证损失（最小化）
        factor=0.5,        # 学习率衰减因子
        patience=10         # 增大耐心epoch数以更稳定地调整学习率
    )

    # 训练循环
    best_val_loss = float('inf')  # 初始最佳损失为无穷大
    patience_counter = 0           # 早停计数器
    history = {'train_loss': [], 'val_loss': []}

    print("\n开始训练...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # 训练和验证
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step(val_loss)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        epoch_time = time.time() - epoch_start

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            best_model_path = os.path.join(OUTPUT_DIR, 'lstm_best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)
        else:
            patience_counter += 1

        # 打印训练进度（每5个epoch或第1个epoch）
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Time: {epoch_time:.1f}s")

        # 早停判断
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
    plot_path = os.path.join(OUTPUT_DIR, 'lstm_training_history.png')
    plot_training_history(history, plot_path)

    return model, history, best_model_path


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == '__main__':
    # 构建数据
    from AC_LSTM._2_sequence_builder import load_and_build_sequences, split_data, create_data_loaders
    
    print("=" * 60)
    print("LSTM时序序列构建")
    print("=" * 60)
    
    sequences, targets, scaler, df = load_and_build_sequences()
    train_data, val_data, test_data = split_data(sequences, targets, df)
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data)

    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, history, best_path = train_model(train_loader, val_loader, device)
    print(f"模型已保存到: {best_path}")

    # 【修复】在测试集上评估最终性能
    print("\n" + "=" * 60)
    print("在测试集上评估最终性能")
    print("=" * 60)

    from AC_LSTM._6_predictor import evaluate_model_from_loader
    test_metrics = evaluate_model_from_loader(
        model, test_loader, device,
        save_path=os.path.join(OUTPUT_DIR, 'lstm_test_evaluation_report.md')
    )
    print(f"\n测试集最终评估结果:")
    print(f"  R²: {test_metrics['R2']:.4f}")
    print(f"  RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  MAE: {test_metrics['MAE']:.4f}")
