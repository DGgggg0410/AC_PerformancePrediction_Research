"""
消融实验配置修复脚本
修复内容：
1. 标准化：从全量数据 fit_transform → 仅训练集 fit，再 transform 全部
2. 更新超参数：与主模型配置保持一致
3. 移除不在主模型特征列表中的 PRECIP_DAYS
4. 更新图表中的基准参考线为当前实际值

运行方式：在消融实验之前运行
    python fix_ablation_configs.py

运行此脚本后，_run_all_lstm_ablation.py 和 _run_all_transformer_ablation.py
将使用与主模型一致的配置进行消融实验。
"""

import re
import os

LSTM_RUNNER = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/AC_LSTM/ablation/_run_all_lstm_ablation.py'
TRANSFORMER_RUNNER = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/AC_Transformer/ablation/_run_all_transformer_ablation.py'


def patch_file(filepath, replacements):
    """
    对文件进行多重文本替换
    replacements: list of (old_string, new_string) 元组
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  ✅ 替换成功: {old[:50]}...")
        else:
            print(f"  ⚠️  未找到: {old[:50]}...")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✅ 文件已保存: {filepath}")


def fix_lstm_ablation():
    """修复 LSTM 消融运行器"""
    print("=" * 60)
    print("修复 LSTM 消融实验配置")
    print("=" * 60)

    replacements = [
        # ========== 1. 修复标准化：仅训练集拟合 ==========
        (
            '''    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)''',
            '''    # 【修复】仅基于训练集拟合标准化参数
    section_counts_tmp = df.groupby('SHRP_ID').size().reset_index(name='count')
    section_counts_tmp = section_counts_tmp.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    train_section_end = int(len(section_counts_tmp) * 0.7)
    train_shrp_ids_tmp = set(section_counts_tmp['SHRP_ID'][:train_section_end])
    train_mask = df['SHRP_ID'].isin(train_shrp_ids_tmp)

    scaler = StandardScaler()
    scaler.fit(X[train_mask])
    X_scaled = scaler.transform(X)'''
        ),

        # ========== 2. 修复超参数 ==========
        # batch_size: 128 → 256
        ('batch_size=128, shuffle=True)',
         'batch_size=256, shuffle=True)'),
        ('batch_size=128, shuffle=False)',
         'batch_size=256, shuffle=False)'),

        # 模型定义 hidden_dim=128 ✅ 不变, dropout=0.1 → 0.2
        ('model = LSTMModel(input_dim, hidden_dim=128, num_layers=2, dropout=0.1, output_dim=1)',
         'model = LSTMModel(input_dim, hidden_dim=128, num_layers=2, dropout=0.2, output_dim=1)'),

        # optimizer: lr=0.001 → 0.005, 添加 weight_decay=1e-3
        ("optimizer = optim.Adam(model.parameters(), lr=0.001)",
         "optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)"),

        # ========== 3. 移除 PRECIP_DAYS ==========
        # no_structure 特征列表
        ("'PRECIPITATION', 'PRECIP_DAYS', 'EVAPORATION'",
         "'PRECIPITATION', 'EVAPORATION'"),

        # ========== 4. 更新图表基准参考线为当前实际值 ==========
        # baseline
        ('baseline_r2 = 0.9562',
         'baseline_r2 = 0.7279'),
        # RMSE baseline
        ("axes[1].axhline(y=0.1461, color='red', linestyle='--', label=f'基准 (0.1461)')",
         "axes[1].axhline(y=0.3724, color='red', linestyle='--', label=f'基准 (0.3724)')"),
        # MAE baseline
        ("axes[2].axhline(y=0.0358, color='red', linestyle='--', label=f'基准 (0.0358)')",
         "axes[2].axhline(y=0.2056, color='red', linestyle='--', label=f'基准 (0.2056)')"),

        # 保存 summary 中的 baseline 值
        ("'baseline': {'r2': baseline_r2, 'rmse': 0.1461, 'mae': 0.0358, 'features': 19}",
         "'baseline': {'r2': baseline_r2, 'rmse': 0.3724, 'mae': 0.2056, 'features': 19}"),

        # 图表 y 轴范围（R² 从 0.88~1.02 改为适合新基线的范围）
        ("axes[0].set_ylim(0.88, 1.02)",
         "axes[0].set_ylim(0.60, 0.85)"),
    ]

    patch_file(LSTM_RUNNER, replacements)


def fix_transformer_ablation():
    """修复 Transformer 消融运行器"""
    print("=" * 60)
    print("修复 Transformer 消融实验配置")
    print("=" * 60)

    replacements = [
        # ========== 1. 修复标准化 ==========
        (
            '''    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)''',
            '''    # 【修复】仅基于训练集拟合标准化参数
    section_counts_tmp = df.groupby('SHRP_ID').size().reset_index(name='count')
    section_counts_tmp = section_counts_tmp.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    train_section_end = int(len(section_counts_tmp) * 0.7)
    train_shrp_ids_tmp = set(section_counts_tmp['SHRP_ID'][:train_section_end])
    train_mask = df['SHRP_ID'].isin(train_shrp_ids_tmp)

    scaler = StandardScaler()
    scaler.fit(X[train_mask])
    X_scaled = scaler.transform(X)'''
        ),

        # ========== 2. 修复 Transformer 模型参数 ==========
        # num_layers=1 → 2, ff_dim=512 → 256, dropout=0.3 → 0.2
        ("model = TransformerModel(\n        input_dim=input_dim, d_model=256, nhead=8, num_layers=1,\n        ff_dim=512, dropout=0.3, output_dim=1\n    )",
         "model = TransformerModel(\n        input_dim=input_dim, d_model=256, nhead=8, num_layers=2,\n        ff_dim=256, dropout=0.2, output_dim=1\n    )"),

        # optimizer: lr=0.001 → 0.0005
        ("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)",
         "optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)"),

        # ========== 3. 移除 PRECIP_DAYS ==========
        ("'PRECIPITATION', 'PRECIP_DAYS', 'EVAPORATION'",
         "'PRECIPITATION', 'EVAPORATION'"),

        # ========== 4. 更新图表基准参考线 ==========
        # baseline
        ('baseline_r2 = 0.9582',
         'baseline_r2 = 0.7472'),
        # RMSE baseline
        ("axes[1].axhline(y=0.1427, color='red', linestyle='--', label=f'基准 (0.1427)')",
         "axes[1].axhline(y=0.3590, color='red', linestyle='--', label=f'基准 (0.3590)')"),
        # MAE baseline
        ("axes[2].axhline(y=0.0410, color='red', linestyle='--', label=f'基准 (0.0410)')",
         "axes[2].axhline(y=0.1889, color='red', linestyle='--', label=f'基准 (0.1889)')"),

        # 保存 summary 中的 baseline 值
        ("'baseline': {'r2': baseline_r2, 'rmse': 0.1427, 'mae': 0.0410, 'features': 19}",
         "'baseline': {'r2': baseline_r2, 'rmse': 0.3590, 'mae': 0.1889, 'features': 19}"),

        # 图表 y 轴范围
        ("axes[0].set_ylim(0.88, 1.02)",
         "axes[0].set_ylim(0.55, 0.85)"),
    ]

    patch_file(TRANSFORMER_RUNNER, replacements)


def main():
    print("=" * 60)
    print("消融实验配置修复工具")
    print("=" * 60)

    fix_lstm_ablation()
    print()
    fix_transformer_ablation()

    print("\n" + "=" * 60)
    print("修复完成！现在可以运行消融实验了：")
    print("  python AC_LSTM/ablation/_run_all_lstm_ablation.py")
    print("  python AC_Transformer/ablation/_run_all_transformer_ablation.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
