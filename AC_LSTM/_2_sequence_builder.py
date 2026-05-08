"""
时序序列构建器
将处理后的LTPP数据构建为滑动窗口序列
与Transformer共用相同的逻辑

功能说明：
1. 从CSV文件加载处理后的LTPP数据
2. 对特征进行标准化（Z-score标准化）
3. 使用滑动窗口构建时序序列
4. 按时间顺序划分训练集、验证集、测试集
5. 创建PyTorch DataLoader用于批量训练

作者: 研究团队
日期: 2024
"""

import numpy as np  # 数值计算
import pandas as pd  # 数据处理
import torch  # PyTorch深度学习框架
from torch.utils.data import Dataset, DataLoader  # PyTorch数据集和DataLoader
from sklearn.preprocessing import StandardScaler  # 特征标准化
import os  # 路径操作
import sys  # 系统操作

# 添加项目路径以导入配置
sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research')
from AC_LSTM._1_config import (
    DATA_PATH, FEATURE_COLS, TARGET_COL, SEQ_LEN,
    BATCH_SIZE, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)

# 设置随机种子，确保实验可复现
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================================
# PyTorch数据集类
# ============================================================================

class LTPPSequenceDataset(Dataset):
    """
    LTPP时序数据集类

    继承自PyTorch的Dataset类，用于创建可迭代的数据集
    每个样本包含一个时序序列和对应的目标值

    属性:
        sequences: 时序特征数组，形状为 (n_samples, seq_len, n_features)
        targets: 目标值数组，形状为 (n_samples,)
    """

    def __init__(self, sequences, targets):
        """
        初始化数据集

        参数:
            sequences: 时序特征数组
            targets: 目标值数组
        """
        # 转换为PyTorch张量
        self.sequences = torch.FloatTensor(sequences)  # 浮点型张量
        self.targets = torch.FloatTensor(targets)      # 浮点型张量

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.targets)

    def __getitem__(self, idx):
        """
        获取指定索引的样本

        参数:
            idx: 样本索引

        返回:
            (序列, 目标值) 元组
        """
        return self.sequences[idx], self.targets[idx]


# ============================================================================
# 数据加载和序列构建函数
# ============================================================================

def load_and_build_sequences():
    """
    加载数据并构建时序序列

    完整流程:
        1. 从CSV文件加载处理后的数据
        2. 按路段和时间排序
        3. 提取特征和目标值
        4. 对特征进行标准化
        5. 构建滑动窗口序列（按路段边界）

    返回:
        sequences: 时序序列数组
        targets: 目标值数组
        scaler: 标准化器（用于后续预测）
        df: 排序后的DataFrame（用于按路段划分）
    """
    print("加载处理后的数据...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"原始数据: {len(df)} 行")

    # 确保SHRP_ID为字符串类型（避免排序时混合类型错误）
    df['SHRP_ID'] = df['SHRP_ID'].astype(str)

    # 按路段和时间排序，确保时序数据的连贯性
    df = df.sort_values(['SHRP_ID', 'VISIT_DATE']).reset_index(drop=True)

    # 提取特征和目标
    X = df[FEATURE_COLS].values  # 特征矩阵
    y = df[TARGET_COL].values    # 目标向量

    # 标准化特征（Z-score标准化）
    # 公式: z = (x - mean) / std
    # 使每个特征变为均值为0，标准差为1的分布
    # 这有助于神经网络的训练稳定性
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 构建滑动窗口序列
    print(f"构建时序序列 (SEQ_LEN={SEQ_LEN})...")
    # 【修复】传入df以确保按路段边界构建序列
    sequences, targets = build_sequences(X_scaled, y, SEQ_LEN, df)
    print(f"序列数量: {len(sequences)}")

    return sequences, targets, scaler, df


def build_sequences(X, y, seq_len, df=None):
    """
    构建滑动窗口序列

    【修复】现在按路段边界分割，确保时序数据不跨越路段边界

    滑动窗口机制:
        对于每个时间点i，使用X[i:i+seq_len]作为输入序列
        预测y[i+seq_len]作为目标

    示例（seq_len=3）:
        时间步:  0    1    2    3    4    5
        数据:   x0   x1   x2   x3   x4   x5
                       ↓
        序列1:  x0   x1   x2  →  预测y3
                       ↓
        序列2:  x1   x2   x3  →  预测y4
                       ↓
        序列3:  x2   x3   x4  →  预测y5

    参数:
        X: 标准化后的特征数组，形状为 (n_samples, n_features)
        y: 目标值数组，形状为 (n_samples,)
        seq_len: 序列长度
        df: 原始DataFrame（需要包含SHRP_ID列以识别路段边界）

    返回:
        sequences: 序列数组，形状为 (n_sequences, seq_len, n_features)
        targets: 目标数组，形状为 (n_sequences,)
    """
    sequences = []
    targets = []

    if df is not None:
        # 【修复】按路段边界构建序列，确保不跨越路段边界
        # 这解决了时序数据泄漏问题（IRI_LAG特征不会引用相邻路段的数据）
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
        # 降级为全局滑动窗口（不推荐，可能跨越路段边界）
        for i in range(len(X) - seq_len):
            seq = X[i:i + seq_len]
            target = y[i + seq_len]
            sequences.append(seq)
            targets.append(target)

    return np.array(sequences), np.array(targets)


# ============================================================================
# 数据集划分函数
# ============================================================================

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

        # 获取各数据集的路段ID
        train_shrp_ids = set(section_sample_counts['SHRP_ID'][:train_section_end])
        val_shrp_ids = set(section_sample_counts['SHRP_ID'][train_section_end:val_section_end])
        test_shrp_ids = set(section_sample_counts['SHRP_ID'][val_section_end:])

        # 划分数据
        train_idx = []
        val_idx = []
        test_idx = []

        current_idx = 0
        for shrp_id in section_sample_counts['SHRP_ID']:
            section_count = section_sample_counts[section_sample_counts['SHRP_ID'] == shrp_id]['count'].values[0]
            # 每个路段有 section_count - SEQ_LEN 个序列（因为构建序列时损失了SEQ_LEN个样本）
            seq_count = max(0, section_count - SEQ_LEN)

            if shrp_id in train_shrp_ids:
                train_idx.extend(range(current_idx, current_idx + seq_count))
            elif shrp_id in val_shrp_ids:
                val_idx.extend(range(current_idx, current_idx + seq_count))
            else:  # test_shrp_ids
                test_idx.extend(range(current_idx, current_idx + seq_count))

            # 无论该路段是否有序列，都要更新索引
            current_idx += seq_count

        # 划分数据
        train_seq = np.array([sequences[i] for i in train_idx]) if train_idx else np.array([])
        train_target = np.array([targets[i] for i in train_idx]) if train_idx else np.array([])
        val_seq = np.array([sequences[i] for i in val_idx]) if val_idx else np.array([])
        val_target = np.array([targets[i] for i in val_idx]) if val_idx else np.array([])
        test_seq = np.array([sequences[i] for i in test_idx]) if test_idx else np.array([])
        test_target = np.array([targets[i] for i in test_idx]) if test_idx else np.array([])

        print(f"数据集划分（随机打乱SHRP_ID后分层）:")
        print(f"  训练集: {len(train_seq)} 样本 ({len(train_shrp_ids)} 路段)")
        print(f"  验证集: {len(val_seq)} 样本 ({len(val_shrp_ids)} 路段)")
        print(f"  测试集: {len(test_seq)} 样本 ({len(test_shrp_ids)} 路段)")

        return (train_seq, train_target), (val_seq, val_target), (test_seq, test_target)
    else:
        # 降级：全局顺序划分（不推荐）
        n = len(sequences)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_seq = sequences[:train_end]
        train_target = targets[:train_end]
        val_seq = sequences[train_end:val_end]
        val_target = targets[train_end:val_end]
        test_seq = sequences[val_end:]
        test_target = targets[val_end:]

        print(f"数据集划分:")
        print(f"  训练集: {len(train_seq)} 样本")
        print(f"  验证集: {len(val_seq)} 样本")
        print(f"  测试集: {len(test_seq)} 样本")

        return (train_seq, train_target), (val_seq, val_target), (test_seq, test_target)


# ============================================================================
# DataLoader创建函数
# ============================================================================

def create_data_loaders(train_data, val_data, test_data):
    """
    创建PyTorch DataLoader

    DataLoader提供以下功能:
        - 自动批量加载数据
        - 可选的随机打乱（训练集）
        - 多进程数据加载（加速）

    参数:
        train_data: 训练数据元组 (sequences, targets)
        val_data: 验证数据元组
        test_data: 测试数据元组

    返回:
        train_loader: 训练DataLoader
        val_loader: 验证DataLoader
        test_loader: 测试DataLoader
    """
    # 创建Dataset对象
    train_dataset = LTPPSequenceDataset(train_data[0], train_data[1])
    val_dataset = LTPPSequenceDataset(val_data[0], val_data[1])
    test_dataset = LTPPSequenceDataset(test_data[0], test_data[1])

    # 创建DataLoader
    # shuffle=True: 训练时打乱数据，增加泛化能力
    # shuffle=False: 验证和测试时保持顺序，便于分析
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：构建完整的时序数据处理流程

    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        scaler: 特征标准化器（用于后续预测）
    """
    print("=" * 60)
    print("时序序列构建")
    print("=" * 60)

    # 加载并构建序列
    sequences, targets, scaler, df = load_and_build_sequences()

    # 划分数据集（按路段划分以避免数据泄漏）
    train_data, val_data, test_data = split_data(sequences, targets, df)

    # 创建DataLoader
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data
    )

    # 保存scaler供预测使用
    # scaler保存了特征的均值和标准差，用于将预测结果反标准化
    import pickle
    scaler_path = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/AC_LSTM/output/scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\nScaler已保存到: {scaler_path}")

    return train_loader, val_loader, test_loader, scaler


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == '__main__':
    train_loader, val_loader, test_loader, scaler = main()
    print("\n序列构建完成!")
