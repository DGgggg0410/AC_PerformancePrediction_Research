"""
Transformer消融实验 - 统一运行器
运行所有Transformer消融实验并生成对比报告

【重要】数据加载方式：
使用主模型的 _2_sequence_builder.py 加载并划分数据，确保所有消融实验
使用与主模型完全相同的训练/验证/测试集划分。每个实验仅通过特征维度
切片来改变输入特征，数据集划分完全相同。

实验列表：
1. ablation_baseline - 全特征基线（19个特征）
2. ablation_no_climate - 去掉气候因素（11个特征）
3. ablation_no_structure - 去掉结构因素（14个特征）
4. ablation_no_geographic - 去掉地理因素（16个特征）
5. ablation_only_temporal - 只保留时序特征（3个特征）
6. ablation_no_climate_structure - 去掉气候和结构因素（6个特征）

作者: 研究团队
日期: 2024
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import math


class PositionalEncoding(nn.Module):
    """位置编码（与主模型一致）"""
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer模型（与主模型结构完全一致）"""
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=1, ff_dim=512, dropout=0.3, output_dim=1):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 与主模型一致：中间加 256→128 映射层 + LayerNorm
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 256 → 128
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)  # 128 → 1
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = self.input_projection(x)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, 0, :]  # 取CLS token的输出
        x = self.fc(x)
        return x


class LTPPSequenceDataset(Dataset):
    """数据集"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def train_and_evaluate(config_name, device, train_loader, val_loader, test_loader, input_dim):
    """训练并评估模型"""
    print(f"\n{'='*60}")
    print(f"运行实验: {config_name}")
    print(f"{'='*60}")

    project_dir = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research'
    output_dir = os.path.join(project_dir, 'AC_Transformer', 'output', config_name)
    os.makedirs(output_dir, exist_ok=True)

    # 创建模型（与主模型完全一致的超参数）
    model = TransformerModel(
        input_dim=input_dim, d_model=256, nhead=8, num_layers=2,
        ff_dim=256, dropout=0.2, output_dim=1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    # 学习率调度器（与主模型一致）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    patience_counter = 0

    for epoch in range(100):
        # 训练
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            # 【修复】与主模型一致：max_norm=0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * len(batch_y)
        val_loss = total_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f'{config_name}_best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= 60:
                print(f"  早停于 Epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: 训练损失={train_loss:.6f}, 验证损失={val_loss:.6f}")

    # 评估
    model.load_state_dict(torch.load(os.path.join(output_dir, f'{config_name}_best_model.pth')))
    model.eval()

    predictions, actuals = [], []
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

    print(f"\n  结果: R^2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    results = {
        'experiment_name': config_name,
        'features_count': input_dim,
        'metrics': {'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae)},
        'training': {'epochs': len(train_losses), 'best_val_loss': float(best_val_loss)}
    }

    with open(os.path.join(output_dir, f'{config_name}_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(val_losses, label='验证损失', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{config_name} - 训练曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{config_name}_training.png'), dpi=150)
    plt.close()

    return r2, rmse, mae, input_dim


def main():
    """主函数 - 运行所有消融实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # ========================================================================
    # 【修复】使用主模型的序列构建器加载数据，确保所有消融实验使用
    # 与主模型完全相同的训练/验证/测试集划分
    # ========================================================================
    sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research')
    from AC_Transformer._2_sequence_builder import load_and_build_sequences, split_data

    print("\n" + "="*60)
    print("加载数据（使用主模型的序列构建器，与主模型完全一致）")
    print("="*60)

    # 一次性加载数据并划分（19个特征全部参与）
    sequences, targets, scaler, df = load_and_build_sequences()
    (train_seq_all, train_target_all), (val_seq_all, val_target_all), (test_seq_all, test_target_all) = \
        split_data(sequences, targets, df)

    # ========================================================================
    # 特征索引映射
    # 与 AC_Transformer._1_config.FEATURE_COLS 顺序一致：
    #   0:PAVEMENT_AGE     1:IRI_LAG_1         2:IRI_LAG_2
    #   3:PAVEMENT_FAMILY_ENC
    #   4:LATITUDE          5:LONGITUDE         6:ELEVATION
    #   7:DEGREE_DAYS_OVER_10C_YR  8:COLDEST_AIR_TEMP  9:HIGH_TEMP_7DAYS
    #   10:MIN_SURFACE_50_TEMP
    #   11:FREEZE_INDEX     12:FREEZE_THAW      13:PRECIPITATION   14:EVAPORATION
    #   15:TOTAL_THICKNESS  16:AC_THICKNESS     17:BASE_THICKNESS  18:NUM_LAYERS
    # ========================================================================
    FEATURE_INDICES = {
        'ablation_baseline':               list(range(19)),                                  # 全19特征
        'ablation_no_climate':             [0,1,2,3,4,5,6,15,16,17,18],                      # 去掉7-14（气候）
        'ablation_no_structure':           [0,1,2,4,5,6,7,8,9,10,11,12,13,14],               # 去掉3,15-18（结构）
        'ablation_no_geographic':          [0,1,2,3,7,8,9,10,11,12,13,14,15,16,17,18],       # 去掉4-6（地理）
        'ablation_only_temporal':          [0,1,2],                                          # 只保留时序
        'ablation_no_climate_structure':   [0,1,2,4,5,6],                                    # 去掉气候+结构
    }
    FEATURE_COUNTS = {k: len(v) for k, v in FEATURE_INDICES.items()}

    # 消融实验列表（第一条为全特征基线，后续 ΔR² 均基于内部基线计算）
    experiments = [
        ('ablation_baseline', '全特征基线\n(19特征)'),
        ('ablation_no_climate', '去掉气候因素\n(11特征)'),
        ('ablation_no_structure', '去掉结构因素\n(14特征)'),
        ('ablation_no_geographic', '去掉地理因素\n(16特征)'),
        ('ablation_only_temporal', '只保留时序\n(3特征)'),
        ('ablation_no_climate_structure', '去掉气候+结构\n(6特征)'),
    ]

    all_results = []

    for exp_name, description in experiments:
        indices = FEATURE_INDICES[exp_name]
        n_features = FEATURE_COUNTS[exp_name]

        # 对特征维度（最后一维）进行切片
        train_seq = train_seq_all[:, :, indices]
        val_seq = val_seq_all[:, :, indices]
        test_seq = test_seq_all[:, :, indices]

        # 创建DataLoader（与主模型相同的batch_size）
        train_loader = DataLoader(
            LTPPSequenceDataset(train_seq, train_target_all), batch_size=256, shuffle=True
        )
        val_loader = DataLoader(
            LTPPSequenceDataset(val_seq, val_target_all), batch_size=256, shuffle=False
        )
        test_loader = DataLoader(
            LTPPSequenceDataset(test_seq, test_target_all), batch_size=256, shuffle=False
        )

        r2, rmse, mae, _ = train_and_evaluate(exp_name, device, train_loader, val_loader, test_loader, n_features)
        all_results.append({
            'name': exp_name,
            'description': description,
            'features': n_features,
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae)
        })

    # 从内部基线实验获取参考值（与主模型完全一致）
    baseline = all_results[0]
    baseline_r2 = baseline['r2']
    baseline_rmse = baseline['rmse']
    baseline_mae = baseline['mae']
    print(f"\n内部基线（与主模型测试集完全一致）: R²={baseline_r2:.4f}, RMSE={baseline_rmse:.4f}, MAE={baseline_mae:.4f}")

    # 创建对比图表（仅对比消融实验，基线用横线表示）
    print(f"\n\n{'='*60}")
    print("Transformer 消融实验对比汇总")
    print(f"{'='*60}")

    # 排除基线本身，只显示消融实验的对比
    ablation_results = all_results[1:]

    print(f"\n{'实验名称':<30} {'特征数':<8} {'R^2':<10} {'RMSE':<10} {'MAE':<10} {'ΔR²':<10}")
    print("-" * 80)
    for r in ablation_results:
        r2_drop = baseline_r2 - r['r2']
        print(f"{r['description']:<30} {r['features']:<8} {r['r2']:<10.4f} {r['rmse']:<10.4f} {r['mae']:<10.4f} {r2_drop:<10.4f}")

    # 对比图
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    names = [r['description'].replace('\n', ' ') for r in ablation_results]
    r2_values = [r['r2'] for r in ablation_results]
    rmse_values = [r['rmse'] for r in ablation_results]
    mae_values = [r['mae'] for r in ablation_results]

    bars1 = axes[0].bar(names, r2_values, color='#9b59b6', edgecolor='black', width=0.6)
    axes[0].axhline(y=baseline_r2, color='red', linestyle='--', label=f'基线 ({baseline_r2:.4f})')
    axes[0].set_ylabel('R^2', fontsize=12)
    axes[0].set_title('R^2 对比 (越高越好)', fontsize=14)
    axes[0].set_ylim(min(r2_values) - 0.02, max(r2_values) + 0.02)
    axes[0].legend(loc='lower right')
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)
    axes[0].tick_params(axis='y', labelsize=10)
    for bar, val in zip(bars1, r2_values):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    max_rmse = max(rmse_values) * 1.15
    bars2 = axes[1].bar(names, rmse_values, color='#1abc9c', edgecolor='black', width=0.6)
    axes[1].axhline(y=baseline_rmse, color='red', linestyle='--', label=f'基线 ({baseline_rmse:.4f})')
    axes[1].set_ylabel('RMSE (m/km)', fontsize=12)
    axes[1].set_title('RMSE 对比 (越低越好)', fontsize=14)
    axes[1].set_ylim(0, max_rmse)
    axes[1].legend(loc='upper right')
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)
    axes[1].tick_params(axis='y', labelsize=10)
    for bar, val in zip(bars2, rmse_values):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_rmse*0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    max_mae = max(mae_values) * 1.15
    bars3 = axes[2].bar(names, mae_values, color='#e67e22', edgecolor='black', width=0.6)
    axes[2].axhline(y=baseline_mae, color='red', linestyle='--', label=f'基线 ({baseline_mae:.4f})')
    axes[2].set_ylabel('MAE (m/km)', fontsize=12)
    axes[2].set_title('MAE 对比 (越低越好)', fontsize=14)
    axes[2].set_ylim(0, max_mae)
    axes[2].legend(loc='upper right')
    axes[2].tick_params(axis='x', rotation=45, labelsize=10)
    axes[2].tick_params(axis='y', labelsize=10)
    for bar, val in zip(bars3, mae_values):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_mae*0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_dir = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/AC_Transformer/output'
    plt.savefig(os.path.join(output_dir, 'transformer_ablation_comparison.png'), dpi=150)
    plt.close()

    print(f"\n对比图表已保存: {os.path.join(output_dir, 'transformer_ablation_comparison.png')}")

    summary = {
        'baseline': {'r2': baseline_r2, 'rmse': baseline_rmse, 'mae': baseline_mae, 'features': 19},
        'ablations': ablation_results
    }
    with open(os.path.join(output_dir, 'transformer_ablation_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n所有Transformer消融实验完成!")


if __name__ == '__main__':
    main()
