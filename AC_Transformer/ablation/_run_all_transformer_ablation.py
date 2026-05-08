"""
Transformer消融实验 - 统一运行器
运行所有Transformer消融实验并生成对比报告

实验列表：
1. ablation_no_climate - 去掉气候因素（11个特征）
2. ablation_no_structure - 去掉结构因素（16个特征）
3. ablation_no_geographic - 去掉地理因素（17个特征）
4. ablation_only_temporal - 只保留时序特征（3个特征）
5. ablation_no_climate_structure - 去掉气候和结构因素（6个特征）

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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import math
import pickle

np.random.seed(42)
torch.manual_seed(42)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
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
    """Transformer模型"""
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
        self.fc = nn.Linear(d_model, output_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = self.input_projection(x)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, 0, :]
        x = self.fc(x)
        return x


class LTPPSequenceDataset:
    """数据集"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def load_data(config_name):
    """加载消融实验数据"""
    from sklearn.preprocessing import StandardScaler

    project_dir = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research'
    data_path = os.path.join(project_dir, 'processed_data', 'ltpp_processed_data.csv')
    output_dir = os.path.join(project_dir, 'AC_Transformer', 'output', config_name)

    # 根据配置获取特征列表
    if 'no_climate' in config_name:
        feature_cols = [
            'PAVEMENT_AGE', 'IRI_LAG_1', 'IRI_LAG_2',
            'PAVEMENT_FAMILY_ENC', 'TOTAL_THICKNESS', 'AC_THICKNESS',
            'BASE_THICKNESS', 'NUM_LAYERS',
            'LATITUDE', 'LONGITUDE', 'ELEVATION'
        ]
    elif 'no_structure' in config_name:
        feature_cols = [
            'PAVEMENT_AGE', 'IRI_LAG_1', 'IRI_LAG_2',
            'LATITUDE', 'LONGITUDE', 'ELEVATION',
            'DEGREE_DAYS_OVER_10C_YR', 'COLDEST_AIR_TEMP', 'HIGH_TEMP_7DAYS',
            'MIN_SURFACE_50_TEMP', 'FREEZE_INDEX', 'FREEZE_THAW',
            'PRECIPITATION', 'PRECIP_DAYS', 'EVAPORATION'
        ]
    elif 'no_geographic' in config_name:
        feature_cols = [
            'PAVEMENT_AGE', 'IRI_LAG_1', 'IRI_LAG_2',
            'PAVEMENT_FAMILY_ENC', 'TOTAL_THICKNESS', 'AC_THICKNESS',
            'BASE_THICKNESS', 'NUM_LAYERS',
            'DEGREE_DAYS_OVER_10C_YR', 'COLDEST_AIR_TEMP', 'HIGH_TEMP_7DAYS',
            'MIN_SURFACE_50_TEMP', 'FREEZE_INDEX', 'FREEZE_THAW',
            'PRECIPITATION', 'PRECIP_DAYS', 'EVAPORATION'
        ]
    elif 'only_temporal' in config_name:
        feature_cols = ['PAVEMENT_AGE', 'IRI_LAG_1', 'IRI_LAG_2']
    elif 'no_climate_structure' in config_name:
        feature_cols = [
            'PAVEMENT_AGE', 'IRI_LAG_1', 'IRI_LAG_2',
            'LATITUDE', 'LONGITUDE', 'ELEVATION'
        ]
    else:
        raise ValueError(f"Unknown config: {config_name}")

    # 加载数据
    df = pd.read_csv(data_path, low_memory=False)
    df['SHRP_ID'] = df['SHRP_ID'].astype(str)
    df = df.sort_values(['SHRP_ID', 'VISIT_DATE']).reset_index(drop=True)

    X = df[feature_cols].values
    y = df['MRI'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 构建序列
    seq_len = 5
    sequences, targets = [], []
    for shrp_id, group_indices in df.groupby('SHRP_ID').groups.items():
        group_indices = list(group_indices)
        group_X = X_scaled[group_indices]
        group_y = y[group_indices]

        if len(group_X) >= seq_len + 1:
            for i in range(len(group_X) - seq_len):
                sequences.append(group_X[i:i + seq_len])
                targets.append(group_y[i + seq_len])

    sequences = np.array(sequences)
    targets = np.array(targets)

    # 划分数据集
    section_sample_counts = df.groupby('SHRP_ID').size().reset_index(name='count')
    section_sample_counts = section_sample_counts.sample(frac=1, random_state=42).reset_index(drop=True)

    total_sections = len(section_sample_counts)
    train_end = int(total_sections * 0.7)
    val_end = int(total_sections * 0.85)

    train_shrp_ids = set(section_sample_counts['SHRP_ID'][:train_end])
    val_shrp_ids = set(section_sample_counts['SHRP_ID'][train_end:val_end])

    # 构建索引
    train_idx, val_idx, test_idx = [], [], []
    current_idx = 0

    for shrp_id in section_sample_counts['SHRP_ID']:
        section_count = section_sample_counts[section_sample_counts['SHRP_ID'] == shrp_id]['count'].values[0]
        seq_count = max(0, section_count - seq_len)

        if shrp_id in train_shrp_ids:
            train_idx.extend(range(current_idx, current_idx + seq_count))
        elif shrp_id in val_shrp_ids:
            val_idx.extend(range(current_idx, current_idx + seq_count))
        else:
            test_idx.extend(range(current_idx, current_idx + seq_count))

        current_idx += seq_count

    # 划分
    train_seq = np.array([sequences[i] for i in train_idx])
    train_target = np.array([targets[i] for i in train_idx])
    val_seq = np.array([sequences[i] for i in val_idx])
    val_target = np.array([targets[i] for i in val_idx])
    test_seq = np.array([sequences[i] for i in test_idx])
    test_target = np.array([targets[i] for i in test_idx])

    print(f"  训练集: {len(train_seq)} | 验证集: {len(val_seq)} | 测试集: {len(test_seq)}")

    train_loader = DataLoader(LTPPSequenceDataset(train_seq, train_target), batch_size=256, shuffle=True)
    val_loader = DataLoader(LTPPSequenceDataset(val_seq, val_target), batch_size=256, shuffle=False)
    test_loader = DataLoader(LTPPSequenceDataset(test_seq, test_target), batch_size=256, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    return train_loader, val_loader, test_loader, len(feature_cols), output_dir


def train_and_evaluate(config_name, device):
    """训练并评估模型"""
    print(f"\n{'='*60}")
    print(f"运行实验: {config_name}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, input_dim, output_dir = load_data(config_name)

    # 创建模型
    model = TransformerModel(
        input_dim=input_dim, d_model=256, nhead=8, num_layers=1,
        ff_dim=512, dropout=0.3, output_dim=1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    patience_counter = 0

    for epoch in range(100):
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
        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

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
    """主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    experiments = [
        ('ablation_no_climate', '去掉气候因素\n(11特征)'),
        ('ablation_no_structure', '去掉结构因素\n(16特征)'),
        ('ablation_no_geographic', '去掉地理因素\n(17特征)'),
        ('ablation_only_temporal', '只保留时序\n(3特征)'),
        ('ablation_no_climate_structure', '去掉气候+结构\n(6特征)'),
    ]

    all_results = []

    for exp_name, description in experiments:
        r2, rmse, mae, n_features = train_and_evaluate(exp_name, device)
        all_results.append({
            'name': exp_name,
            'description': description,
            'features': n_features,
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae)
        })

    print(f"\n\n{'='*60}")
    print("Transformer 消融实验对比汇总")
    print(f"{'='*60}")

    baseline_r2 = 0.9582

    print(f"\n{'实验名称':<30} {'特征数':<8} {'R^2':<10} {'RMSE':<10} {'MAE':<10} {'R^2下降':<10}")
    print("-" * 80)
    for r in all_results:
        r2_drop = baseline_r2 - r['r2']
        print(f"{r['description']:<30} {r['features']:<8} {r['r2']:<10.4f} {r['rmse']:<10.4f} {r['mae']:<10.4f} {r2_drop:<10.4f}")

    # 对比图
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    names = [r['description'].replace('\n', ' ') for r in all_results]
    r2_values = [r['r2'] for r in all_results]
    rmse_values = [r['rmse'] for r in all_results]
    mae_values = [r['mae'] for r in all_results]

    bars1 = axes[0].bar(names, r2_values, color='#9b59b6', edgecolor='black', width=0.6)
    axes[0].axhline(y=baseline_r2, color='red', linestyle='--', label=f'基准 ({baseline_r2:.4f})')
    axes[0].set_ylabel('R^2', fontsize=12)
    axes[0].set_title('R^2 对比 (越高越好)', fontsize=14)
    axes[0].set_ylim(0.88, 1.02)
    axes[0].legend(loc='lower right')
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)
    axes[0].tick_params(axis='y', labelsize=10)
    for bar, val in zip(bars1, r2_values):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    max_rmse = max(rmse_values) * 1.15
    bars2 = axes[1].bar(names, rmse_values, color='#1abc9c', edgecolor='black', width=0.6)
    axes[1].axhline(y=0.1427, color='red', linestyle='--', label=f'基准 (0.1427)')
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
    axes[2].axhline(y=0.0410, color='red', linestyle='--', label=f'基准 (0.0410)')
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
        'baseline': {'r2': baseline_r2, 'rmse': 0.1427, 'mae': 0.0410, 'features': 19},
        'ablations': all_results
    }
    with open(os.path.join(output_dir, 'transformer_ablation_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n所有Transformer消融实验完成!")


if __name__ == '__main__':
    main()
