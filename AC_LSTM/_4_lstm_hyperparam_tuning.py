"""
LSTM超参数调优

功能说明：
1. 使用随机搜索(Random Search)或网格搜索(Grid Search)寻找最佳超参数
2. 评估指标：验证集MSE损失
3. 可调超参数范围：
   - HIDDEN_DIM: [64, 128, 256]
   - NUM_LAYERS: [1, 2, 3]
   - DROPOUT: [0.1, 0.2, 0.3]
   - LEARNING_RATE: [0.0005, 0.001, 0.005]
   - BATCH_SIZE: [128, 256]

调优策略说明：
- 使用随机搜索而非网格搜索（效率更高，效果相近）
- 每次试验使用早停机制避免过拟合
- 保存最佳超参数配置到配置文件

作者: 研究团队
日期: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys
import time
import json
import re
from itertools import product
from random import sample, choice

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
    INPUT_DIM, OUTPUT_DIM, SEQ_LEN, OUTPUT_DIR, RANDOM_SEED,
    TRAIN_RATIO, VAL_RATIO
)
from AC_LSTM._3_lstm_model import LSTMModel


# ============================================================================
# 随机种子设置
# ============================================================================

def set_seed(seed):
    """设置所有随机种子以确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ============================================================================
# 超参数搜索空间
# ============================================================================

# 超参数搜索空间定义
PARAM_SPACE = {
    'HIDDEN_DIM': [64, 128, 256],
    'NUM_LAYERS': [1, 2, 3],
    'DROPOUT': [0.1, 0.2, 0.3],
    'LEARNING_RATE': [0.0005, 0.001, 0.005],
    'BATCH_SIZE': [128, 256],
}

# 每个超参数组合的训练轮数（快速搜索用较少轮数）
QUICK_EPOCHS = 30  # 快速搜索时的epoch数
FULL_EPOCHS = 100  # 最终训练时的epoch数
PATIENCE = 10      # 早停耐心值


# ============================================================================
# 数据加载（复用sequence_builder的逻辑）
# ============================================================================

def load_data(sample_ratio=0.1):
    """加载并准备数据

    参数:
        sample_ratio: 采样比例，默认0.1表示使用10%的数据
    """
    from AC_LSTM._2_sequence_builder import load_and_build_sequences, split_data, create_data_loaders

    print("加载数据...")
    sequences, targets, scaler, df = load_and_build_sequences()
    train_data, val_data, test_data = split_data(sequences, targets, df)

    # 采样加速调优
    if sample_ratio < 1.0:
        print(f"采样数据 (比例: {sample_ratio})...")
        train_seq, train_target = train_data
        val_seq, val_target = val_data
        test_seq, test_target = test_data

        n_train = int(len(train_seq) * sample_ratio)
        n_val = int(len(val_seq) * sample_ratio)
        n_test = int(len(test_seq) * sample_ratio)

        np.random.seed(42)
        train_idx = np.random.choice(len(train_seq), n_train, replace=False)
        val_idx = np.random.choice(len(val_seq), n_val, replace=False)
        test_idx = np.random.choice(len(test_seq), n_test, replace=False)

        train_data = (train_seq[train_idx], train_target[train_idx])
        val_data = (val_seq[val_idx], val_target[val_idx])
        test_data = (test_seq[test_idx], test_target[test_idx])

        print(f"  训练集: {n_train} 样本 | 验证集: {n_val} 样本 | 测试集: {n_test} 样本")

    return train_data, val_data, test_data, scaler


# ============================================================================
# 模型训练与评估（单次试验）
# ============================================================================

def train_single_trial(model, train_loader, val_loader, epochs, device, verbose=False):
    """
    训练单个模型试验

    参数:
        model: LSTM模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        device: 计算设备
        verbose: 是否打印详细信息

    返回:
        best_val_loss: 最佳验证损失
        best_epoch: 最佳epoch数
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model.lr)  # 使用模型的lr属性
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / n_batches

        # 验证阶段
        model.eval()
        val_loss = 0
        n_val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / n_val_batches
        scheduler.step(avg_val_loss)

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        if patience_counter >= PATIENCE:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_val_loss, best_epoch


# ============================================================================
# 超参数搜索策略
# ============================================================================

def random_search(train_data, val_data, n_trials=20, quick=True):
    """
    随机搜索超参数

    参数:
        train_data: 训练数据元组
        val_data: 验证数据元组
        n_trials: 试验次数
        quick: 是否使用快速模式（少epoch）

    返回:
        best_params: 最佳超参数组合
        best_val_loss: 最佳验证损失
        results: 所有试验结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"随机搜索: {n_trials} 次试验")

    results = []
    best_val_loss = float('inf')
    best_params = None
    best_model_state = None

    # 生成随机超参数组合
    all_combinations = []
    hidden_dims = PARAM_SPACE['HIDDEN_DIM']
    num_layers_list = PARAM_SPACE['NUM_LAYERS']
    dropouts = PARAM_SPACE['DROPOUT']
    learning_rates = PARAM_SPACE['LEARNING_RATE']
    batch_sizes = PARAM_SPACE['BATCH_SIZE']

    for _ in range(n_trials):
        params = {
            'HIDDEN_DIM': choice(hidden_dims),
            'NUM_LAYERS': choice(num_layers_list),
            'DROPOUT': choice(dropouts),
            'LEARNING_RATE': choice(learning_rates),
            'BATCH_SIZE': choice(batch_sizes),
        }
        all_combinations.append(params)

    # 执行搜索
    for i, params in enumerate(all_combinations):
        print(f"\n试验 {i+1}/{n_trials}: {params}")
        start_time = time.time()

        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(train_data[0]),
            torch.FloatTensor(train_data[1])
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(val_data[0]),
            torch.FloatTensor(val_data[1])
        )

        train_loader = DataLoader(train_dataset, batch_size=params['BATCH_SIZE'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)

        # 创建模型
        set_seed(RANDOM_SEED + i)
        model = LSTMModel(
            input_dim=INPUT_DIM,
            hidden_dim=params['HIDDEN_DIM'],
            num_layers=params['NUM_LAYERS'],
            output_dim=OUTPUT_DIM,
            dropout=params['DROPOUT']
        ).to(device)
        model.lr = params['LEARNING_RATE']  # 临时添加lr属性

        # 训练
        epochs = QUICK_EPOCHS if quick else FULL_EPOCHS
        val_loss, best_epoch = train_single_trial(
            model, train_loader, val_loader, epochs, device, verbose=False
        )

        elapsed = time.time() - start_time

        # 记录结果
        result = {
            'trial': i + 1,
            'params': params,
            'val_loss': val_loss,
            'best_epoch': best_epoch,
            'time': elapsed
        }
        results.append(result)

        print(f"  Val Loss: {val_loss:.6f} | Best Epoch: {best_epoch} | Time: {elapsed:.1f}s")

        # 更新最佳参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params.copy()
            best_model_state = model.state_dict().copy()
            print(f"  *** 新最佳参数! ***")

    return best_params, best_val_loss, results, best_model_state


def grid_search(train_data, val_data, quick=True):
    """
    网格搜索超参数（穷举搜索）

    参数:
        train_data: 训练数据元组
        val_data: 验证数据元组
        quick: 是否使用快速模式

    返回:
        best_params: 最佳超参数组合
        best_val_loss: 最佳验证损失
        results: 所有试验结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print("网格搜索: 穷举所有组合")

    results = []
    best_val_loss = float('inf')
    best_params = None
    best_model_state = None

    # 生成网格
    hidden_dims = PARAM_SPACE['HIDDEN_DIM']
    num_layers_list = PARAM_SPACE['NUM_LAYERS']
    dropouts = PARAM_SPACE['DROPOUT']
    learning_rates = PARAM_SPACE['LEARNING_RATE']
    batch_sizes = PARAM_SPACE['BATCH_SIZE']

    total_trials = (
        len(hidden_dims) * len(num_layers_list) * len(dropouts) *
        len(learning_rates) * len(batch_sizes)
    )
    print(f"总共 {total_trials} 个组合")

    trial = 0
    for hidden_dim in hidden_dims:
        for num_layers in num_layers_list:
            for dropout in dropouts:
                for lr in learning_rates:
                    for batch_size in batch_sizes:
                        trial += 1
                        params = {
                            'HIDDEN_DIM': hidden_dim,
                            'NUM_LAYERS': num_layers,
                            'DROPOUT': dropout,
                            'LEARNING_RATE': lr,
                            'BATCH_SIZE': batch_size,
                        }
                        print(f"\n试验 {trial}/{total_trials}: {params}")
                        start_time = time.time()

                        # 创建数据加载器
                        train_dataset = torch.utils.data.TensorDataset(
                            torch.FloatTensor(train_data[0]),
                            torch.FloatTensor(train_data[1])
                        )
                        val_dataset = torch.utils.data.TensorDataset(
                            torch.FloatTensor(val_data[0]),
                            torch.FloatTensor(val_data[1])
                        )

                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                        # 创建模型
                        set_seed(RANDOM_SEED)
                        model = LSTMModel(
                            input_dim=INPUT_DIM,
                            hidden_dim=hidden_dim,
                            num_layers=num_layers,
                            output_dim=OUTPUT_DIM,
                            dropout=dropout
                        ).to(device)
                        model.lr = lr

                        # 训练
                        epochs = QUICK_EPOCHS if quick else FULL_EPOCHS
                        val_loss, best_epoch = train_single_trial(
                            model, train_loader, val_loader, epochs, device, verbose=False
                        )

                        elapsed = time.time() - start_time

                        # 记录结果
                        result = {
                            'trial': trial,
                            'params': params,
                            'val_loss': val_loss,
                            'best_epoch': best_epoch,
                            'time': elapsed
                        }
                        results.append(result)

                        print(f"  Val Loss: {val_loss:.6f} | Best Epoch: {best_epoch} | Time: {elapsed:.1f}s")

                        # 更新最佳参数
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_params = params.copy()
                            best_model_state = model.state_dict().copy()
                            print(f"  *** 新最佳参数! ***")

    return best_params, best_val_loss, results, best_model_state


# ============================================================================
# 结果可视化
# ============================================================================

def plot_tuning_results(results, save_path):
    """绘制超参数调优结果"""

    # 按验证损失排序
    sorted_results = sorted(results, key=lambda x: x['val_loss'])

    # 图1：所有试验的验证损失
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1.1 试验损失分布
    ax1 = axes[0, 0]
    val_losses = [r['val_loss'] for r in sorted_results]
    ax1.bar(range(len(sorted_results)), val_losses, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Trial (sorted by val loss)')
    ax1.set_ylabel('Validation Loss (MSE)')
    ax1.set_title('All Trials - Validation Loss')
    ax1.grid(True, alpha=0.3)

    # 1.2 损失箱线图
    ax2 = axes[0, 1]
    ax2.boxplot(val_losses, vert=True)
    ax2.set_ylabel('Validation Loss (MSE)')
    ax2.set_title('Validation Loss Distribution')
    ax2.grid(True, alpha=0.3)

    # 1.3 各超参数与损失的关系
    param_names = ['HIDDEN_DIM', 'NUM_LAYERS', 'DROPOUT', 'LEARNING_RATE', 'BATCH_SIZE']
    for idx, param_name in enumerate(param_names):
        # 前3个参数放第二行，后2个参数放第一行
        if idx < 3:
            ax = axes[1, idx]
        else:
            ax = axes[0, idx - 2]  # idx=3 -> axes[0,1], idx=4 -> axes[0,2]

        param_values = [r['params'][param_name] for r in sorted_results]
        colors = [plt.cm.viridis((v - min(param_values)) / (max(param_values) - min(param_values) + 1e-10))
                                for v in param_values]
        ax.scatter(param_values, val_losses, c=colors, alpha=0.6, s=50)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Val Loss')
        ax.set_title(f'{param_name} vs Val Loss')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"调优结果图已保存到: {save_path}")


def print_tuning_summary(results, best_params):
    """打印调优摘要"""

    print("\n" + "=" * 70)
    print("超参数调优摘要")
    print("=" * 70)

    # 最佳参数
    print("\n最佳超参数组合:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Top 5 结果
    sorted_results = sorted(results, key=lambda x: x['val_loss'])
    print("\nTop 5 试验结果:")
    print("-" * 70)
    for i, r in enumerate(sorted_results[:5]):
        print(f"{i+1}. Val Loss: {r['val_loss']:.6f} | {r['params']}")

    # 平均性能
    avg_loss = np.mean([r['val_loss'] for r in results])
    std_loss = np.std([r['val_loss'] for r in results])
    print(f"\n平均验证损失: {avg_loss:.6f} ± {std_loss:.6f}")

    # 总时间
    total_time = sum([r['time'] for r in results])
    print(f"总调优时间: {total_time:.1f}s ({total_time/60:.1f}min)")


# ============================================================================
# 保存最佳超参数
# ============================================================================

def save_best_params(best_params, best_val_loss, results, output_dir):
    """保存最佳超参数到配置文件"""

    # 保存超参数到JSON
    params_path = os.path.join(output_dir, 'best_hyperparams.json')
    tuning_result = {
        'best_params': best_params,
        'best_val_loss': best_val_loss,
        'n_trials': len(results),
        'param_space': PARAM_SPACE
    }
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(tuning_result, f, indent=2, ensure_ascii=False)
    print(f"最佳超参数已保存到: {params_path}")

    # 保存所有试验结果
    results_path = os.path.join(output_dir, 'tuning_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"所有试验结果已保存到: {results_path}")

    # 【修复】直接自动更新配置文件，无需手动运行脚本
    config_path = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/AC_LSTM/_1_config.py'
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换各超参数
    replacements = [
        ('HIDDEN_DIM = \\d+', f"HIDDEN_DIM = {best_params['HIDDEN_DIM']}"),
        ('NUM_LAYERS = \\d+', f"NUM_LAYERS = {best_params['NUM_LAYERS']}"),
        ('DROPOUT = 0\\.\\d+', f"DROPOUT = {best_params['DROPOUT']}"),
        ('LEARNING_RATE = 0\\.\\d+', f"LEARNING_RATE = {best_params['LEARNING_RATE']}"),
        ('BATCH_SIZE = \\d+', f"BATCH_SIZE = {best_params['BATCH_SIZE']}"),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n【自动更新】配置文件已更新: {config_path}")
    print("最佳超参数:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # 同时在output目录保存一个备份更新脚本（供查阅）
    update_script_path = os.path.join(output_dir, 'update_config.py')
    best_params_repr = repr(best_params)
    update_script = f'''
"""自动生成的配置更新脚本（备份）
将最佳超参数应用到 _1_config.py

注意：此脚本已自动执行，如需手动更新可运行此脚本
"""
import re

CONFIG_PATH = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/AC_LSTM/_1_config.py'

# 最佳超参数
BEST_PARAMS = {best_params_repr}

def update_config():
    """更新配置文件"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换各超参数
    replacements = [
        ('HIDDEN_DIM = \\\\d+', 'HIDDEN_DIM = {best_params["HIDDEN_DIM"]}'),
        ('NUM_LAYERS = \\\\d+', 'NUM_LAYERS = {best_params["NUM_LAYERS"]}'),
        ('DROPOUT = 0\\\\.\\\\d+', 'DROPOUT = {best_params["DROPOUT"]}'),
        ('LEARNING_RATE = 0\\\\.\\\\d+', 'LEARNING_RATE = {best_params["LEARNING_RATE"]}'),
        ('BATCH_SIZE = \\\\d+', 'BATCH_SIZE = {best_params["BATCH_SIZE"]}'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(content)

    print("配置已更新!")
    print("最佳超参数:")
    for k, v in BEST_PARAMS.items():
        print(f"  {{k}}: {{v}}")

if __name__ == '__main__':
    update_config()
'''
    with open(update_script_path, 'w', encoding='utf-8') as f:
        f.write(update_script)
    print(f"备份更新脚本已保存到: {update_script_path}")


# ============================================================================
# 主函数
# ============================================================================

def main(search_method='random', n_trials=20):
    """
    超参数调优主函数

    参数:
        search_method: 'random' 或 'grid'
        n_trials: 随机搜索的试验次数
    """
    print("=" * 70)
    print("LSTM超参数调优")
    print("=" * 70)
    print(f"搜索方法: {search_method}")
    if search_method == 'random':
        print(f"试验次数: {n_trials}")
    print(f"超参数空间: {PARAM_SPACE}")
    print("=" * 70)

    # 设置随机种子
    set_seed(RANDOM_SEED)

    # 加载数据
    train_data, val_data, test_data, scaler = load_data()

    # 执行搜索
    start_time = time.time()

    if search_method == 'random':
        best_params, best_val_loss, results, best_model_state = random_search(
            train_data, val_data, n_trials=n_trials, quick=True
        )
    else:
        best_params, best_val_loss, results, best_model_state = grid_search(
            train_data, val_data, quick=True
        )

    total_time = time.time() - start_time

    # 打印摘要
    print_tuning_summary(results, best_params)

    # 绘制结果图
    plot_path = os.path.join(OUTPUT_DIR, 'lstm_hyperparam_tuning.png')
    plot_tuning_results(results, plot_path)

    # 保存结果
    save_best_params(best_params, best_val_loss, results, OUTPUT_DIR)

    # 保存最佳模型
    if best_model_state is not None:
        best_model_path = os.path.join(OUTPUT_DIR, 'lstm_tuned_model.pth')
        torch.save({
            'model_state_dict': best_model_state,
            'hyperparams': best_params,
            'val_loss': best_val_loss,
        }, best_model_path)
        print(f"最佳模型已保存到: {best_model_path}")

    print(f"\n总调优时间: {total_time:.1f}s ({total_time/60:.1f}min)")
    print("\n下一步: 运行 python _5_trainer.py 使用最佳超参数训练最终模型")
    print("或运行 python update_config.py 自动更新配置文件")

    return best_params, results


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LSTM超参数调优')
    parser.add_argument('--method', type=str, default='random',
                        choices=['random', 'grid'],
                        help='搜索方法: random 或 grid')
    parser.add_argument('--trials', type=int, default=20,
                        help='随机搜索的试验次数')

    args = parser.parse_args()

    best_params, results = main(search_method=args.method, n_trials=args.trials)