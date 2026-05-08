"""
LSTM 5年预测序列构建器
扩展实验：用前5年数据预测未来第5年

与基础版本的差异：
- 目标索引变为 y[i + seq_len + PREDICT_HORIZON - 1]
- 需要路段至少有 seq_len + PREDICT_HORIZON 年的数据

示例（seq_len=5, PREDICT_HORIZON=5）:
    时间步:  0    1    2    3    4    5    6    7    8    9
    数据:   x0   x1   x2   x3   x4   x5   x6   x7   x8   x9
                   ↓
    序列1:  x0   x1   x2   x3   x4  →  预测y9 (第9年，5年后)
                   ↓
    序列2:  x1   x2   x3   x4   x5  →  预测y10 (第10年)

作者: 研究团队
日期: 2024
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import sys
import pickle

sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research')
from AC_LSTM._1b_config import (
    DATA_PATH, FEATURE_COLS, TARGET_COL, SEQ_LEN, PREDICT_HORIZON,
    BATCH_SIZE, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, OUTPUT_DIR
)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class LTPPSequenceDataset(Dataset):
    """LTPP时序数据集（5年预测版）"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def load_and_build_sequences():
    """加载数据并构建5年预测序列"""
    print(f"[5年预测] 加载处理后的数据...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"[5年预测] 原始数据: {len(df)} 行")

    df['SHRP_ID'] = df['SHRP_ID'].astype(str)
    df = df.sort_values(['SHRP_ID', 'VISIT_DATE']).reset_index(drop=True)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[5年预测] 构建时序序列 (SEQ_LEN={SEQ_LEN}, PREDICT_HORIZON={PREDICT_HORIZON})")
    sequences, targets = build_sequences(X_scaled, y, SEQ_LEN, PREDICT_HORIZON, df)
    print(f"[5年预测] 序列数量: {len(sequences)}")

    return sequences, targets, scaler, df


def build_sequences(X, y, seq_len, predict_horizon, df=None):
    """
    构建5年预测滑动窗口序列

    关键差异：目标索引 = i + seq_len + predict_horizon - 1

    对于 seq_len=5, predict_horizon=5:
        - 使用 x0-x4 预测 y9（第5年后的IRI）
        - 使用 x1-x5 预测 y10（第5年后的IRI）
    """
    sequences = []
    targets = []

    if df is not None:
        # 按路段边界构建序列
        for shrp_id, group_indices in df.groupby('SHRP_ID').groups.items():
            group_indices = list(group_indices)
            group_X = X[group_indices]
            group_y = y[group_indices]

            # 【关键修改】需要 seq_len + predict_horizon 年的数据
            min_required = seq_len + predict_horizon
            if len(group_X) >= min_required:
                for i in range(len(group_X) - seq_len - predict_horizon + 1):
                    seq = group_X[i:i + seq_len]
                    # 【关键修改】目标索引向前跳 predict_horizon 年
                    target = group_y[i + seq_len + predict_horizon - 1]
                    sequences.append(seq)
                    targets.append(target)
    else:
        for i in range(len(X) - seq_len - predict_horizon + 1):
            seq = X[i:i + seq_len]
            target = y[i + seq_len + predict_horizon - 1]
            sequences.append(seq)
            targets.append(target)

    return np.array(sequences), np.array(targets)


def split_data(sequences, targets, df=None):
    """按路段划分训练/验证/测试集"""
    if df is not None:
        section_sample_counts = df.groupby('SHRP_ID').size().reset_index(name='count')
        section_sample_counts = section_sample_counts.sort_values('SHRP_ID').reset_index(drop=True)

        total_sections = len(section_sample_counts)
        train_section_end = int(total_sections * TRAIN_RATIO)
        val_section_end = int(total_sections * (TRAIN_RATIO + VAL_RATIO))

        train_shrp_ids = set(section_sample_counts['SHRP_ID'][:train_section_end])
        val_shrp_ids = set(section_sample_counts['SHRP_ID'][train_section_end:val_section_end])
        test_shrp_ids = set(section_sample_counts['SHRP_ID'][val_section_end:])

        train_idx = []
        val_idx = []
        test_idx = []

        current_idx = 0
        for shrp_id in section_sample_counts['SHRP_ID']:
            section_count = section_sample_counts[section_sample_counts['SHRP_ID'] == shrp_id]['count'].values[0]
            # 【关键修改】序列数量计算考虑predict_horizon
            seq_count = max(0, section_count - SEQ_LEN - PREDICT_HORIZON + 1)

            if shrp_id in train_shrp_ids:
                train_idx.extend(range(current_idx, current_idx + seq_count))
            elif shrp_id in val_shrp_ids:
                val_idx.extend(range(current_idx, current_idx + seq_count))
            else:
                test_idx.extend(range(current_idx, current_idx + seq_count))

            current_idx += seq_count

        train_seq = np.array([sequences[i] for i in train_idx]) if train_idx else np.array([])
        train_target = np.array([targets[i] for i in train_idx]) if train_idx else np.array([])
        val_seq = np.array([sequences[i] for i in val_idx]) if val_idx else np.array([])
        val_target = np.array([targets[i] for i in val_idx]) if val_idx else np.array([])
        test_seq = np.array([sequences[i] for i in test_idx]) if test_idx else np.array([])
        test_target = np.array([targets[i] for i in test_idx]) if test_idx else np.array([])

        print(f"[5年预测] 数据集划分（按路段）:")
        print(f"  训练集: {len(train_seq)} 样本 ({len(train_shrp_ids)} 路段)")
        print(f"  验证集: {len(val_seq)} 样本 ({len(val_shrp_ids)} 路段)")
        print(f"  测试集: {len(test_seq)} 样本 ({len(test_shrp_ids)} 路段)")

        return (train_seq, train_target), (val_seq, val_target), (test_seq, test_target)
    else:
        n = len(sequences)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_seq = sequences[:train_end]
        train_target = targets[:train_end]
        val_seq = sequences[train_end:val_end]
        val_target = targets[train_end:val_end]
        test_seq = sequences[val_end:]
        test_target = targets[val_end:]

        print(f"[5年预测] 数据集划分:")
        print(f"  训练集: {len(train_seq)} 样本")
        print(f"  验证集: {len(val_seq)} 样本")
        print(f"  测试集: {len(test_seq)} 样本")

        return (train_seq, train_target), (val_seq, val_target), (test_seq, test_target)


def create_data_loaders(train_data, val_data, test_data):
    """创建DataLoader"""
    train_dataset = LTPPSequenceDataset(train_data[0], train_data[1])
    val_dataset = LTPPSequenceDataset(val_data[0], val_data[1])
    test_dataset = LTPPSequenceDataset(test_data[0], test_data[1])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    """主函数"""
    print("=" * 60)
    print(f"LSTM 5年预测时序序列构建 (PREDICT_HORIZON={PREDICT_HORIZON})")
    print("=" * 60)

    sequences, targets, scaler, df = load_and_build_sequences()
    train_data, val_data, test_data = split_data(sequences, targets, df)
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data)

    # 保存scaler
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler_5yr.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n[5年预测] Scaler已保存到: {scaler_path}")

    return train_loader, val_loader, test_loader, scaler


if __name__ == '__main__':
    train_loader, val_loader, test_loader, scaler = main()
    print("\n[5年预测] 序列构建完成!")
