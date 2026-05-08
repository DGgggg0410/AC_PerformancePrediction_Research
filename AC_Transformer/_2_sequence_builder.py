"""
Transformer时序序列构建器
与LSTM共用相同的滑动窗口逻辑
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import sys

# 添加项目路径
sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research')
from AC_Transformer._1_config import (
    DATA_PATH, FEATURE_COLS, TARGET_COL, SEQ_LEN,
    BATCH_SIZE, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)

# 设置随机种子
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class LTPPSequenceDataset(Dataset):
    """LTPP时序数据集"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def load_and_build_sequences():
    """加载数据并构建时序序列"""
    print("加载处理后的数据...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"原始数据: {len(df)} 行")

    # 按路段和时间排序
    df = df.sort_values(['SHRP_ID', 'VISIT_DATE']).reset_index(drop=True)

    # 提取特征和目标
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 构建滑动窗口序列
    print(f"构建时序序列 (SEQ_LEN={SEQ_LEN})...")
    # 【修复】传入df以确保按路段边界构建序列（与LSTM一致）
    sequences, targets = build_sequences(X_scaled, y, SEQ_LEN, df)
    print(f"序列数量: {len(sequences)}")

    return sequences, targets, scaler, df


def build_sequences(X, y, seq_len, df=None):
    """
    构建滑动窗口序列
    
    【修复】按路段边界分割，确保时序数据不跨越路段边界
    与LSTM版本保持一致

    参数:
        X: 标准化后的特征数组
        y: 目标值数组
        seq_len: 序列长度
        df: 原始DataFrame（需要包含SHRP_ID列）

    返回:
        sequences: 序列数组
        targets: 目标数组
    """
    sequences = []
    targets = []

    if df is not None:
        # 按路段边界构建序列，确保不跨越路段边界
        for shrp_id, group_indices in df.groupby('SHRP_ID').groups.items():
            group_indices = list(group_indices)
            group_X = X[group_indices]
            group_y = y[group_indices]

            # 只在有足够数据的路段构建序列
            if len(group_X) >= seq_len + 1:
                for i in range(len(group_X) - seq_len):
                    seq = group_X[i:i + seq_len]
                    target = group_y[i + seq_len]
                    sequences.append(seq)
                    targets.append(target)
    else:
        # 降级为全局滑动窗口
        for i in range(len(X) - seq_len):
            seq = X[i:i + seq_len]
            target = y[i + seq_len]
            sequences.append(seq)
            targets.append(target)

    return np.array(sequences), np.array(targets)


def split_data(sequences, targets, df=None):
    """
    划分训练集、验证集、测试集

    【修正】随机打乱SHRP_ID后再划分，确保三个数据集的分布相似
    - 之前按SHRP_ID字母顺序划分，导致训练集和测试集地理分布差异大
    - 现在随机打乱后再划分，保证各数据集特征分布一致

    参数:
        sequences: 时序序列
        targets: 目标值
        df: 原始DataFrame（包含SHRP_ID列）

    返回:
        train_data: (训练序列, 训练目标) 元组
        val_data: (验证序列, 验证目标) 元组
        test_data: (测试序列, 测试目标) 元组
    """
    if df is not None:
        # 【修正】获取每个路段的样本数量
        section_sample_counts = df.groupby('SHRP_ID').size().reset_index(name='count')

        # 【修正】随机打乱路段顺序，确保数据分布相似（使用固定种子保证可复现）
        section_sample_counts = section_sample_counts.sample(
            frac=1, random_state=RANDOM_SEED
        ).reset_index(drop=True)

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
            seq_count = max(0, section_count - SEQ_LEN)

            if shrp_id in train_shrp_ids:
                train_idx.extend(range(current_idx, current_idx + seq_count))
            elif shrp_id in val_shrp_ids:
                val_idx.extend(range(current_idx, current_idx + seq_count))
            else:
                test_idx.extend(range(current_idx, current_idx + seq_count))

            current_idx += seq_count

        train_seq = sequences[train_idx]
        train_target = targets[train_idx]
        val_seq = sequences[val_idx]
        val_target = targets[val_idx]
        test_seq = sequences[test_idx]
        test_target = targets[test_idx]

        print(f"数据集划分（随机打乱SHRP_ID后分层）:")
        print(f"  训练集: {len(train_seq)} 样本 ({len(train_shrp_ids)} 路段)")
        print(f"  验证集: {len(val_seq)} 样本 ({len(val_shrp_ids)} 路段)")
        print(f"  测试集: {len(test_seq)} 样本 ({len(test_shrp_ids)} 路段)")

        return (train_seq, train_target), (val_seq, val_target), (test_seq, test_target)
    else:
        # 降级为全局顺序划分
        n = len(sequences)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_seq = sequences[:train_end]
        train_target = targets[:train_end]
        val_seq = sequences[train_end:val_end]
        val_target = targets[train_end:val_end]
        test_seq = sequences[val_end:]
        test_target = targets[val_end:]

        print(f"数据集划分（全局顺序）:")
        print(f"  训练集: {len(train_seq)} 样本")
        print(f"  验证集: {len(val_seq)} 样本")
        print(f"  测试集: {len(test_seq)} 样本")

        return (train_seq, train_target), (val_seq, val_target), (test_seq, test_target)


def create_data_loaders(train_data, val_data, test_data):
    """创建PyTorch DataLoader"""
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
    print("Transformer时序序列构建")
    print("=" * 60)

    sequences, targets, scaler, df = load_and_build_sequences()
    train_data, val_data, test_data = split_data(sequences, targets, df)
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data)

    # 保存scaler
    import pickle
    scaler_path = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/AC_Transformer/output/scaler.pkl'
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\nScaler已保存到: {scaler_path}")

    return train_loader, val_loader, test_loader, scaler, df


if __name__ == '__main__':
    train_loader, val_loader, test_loader, scaler, df = main()
    print("\n序列构建完成!")