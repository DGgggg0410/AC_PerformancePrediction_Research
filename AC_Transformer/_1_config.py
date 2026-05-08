"""
Transformer模型配置
与LSTM保持统一的SEQ_LEN和INPUT_DIM

配置说明：
本配置文件定义Transformer模型的所有超参数。
与LSTM配置保持一致，确保模型比较的公平性：
- 相同的输入特征
- 相同的序列长度
- 相同的数据划分比例

Transformer模型特点：
- 使用自注意力机制（Self-Attention）
- 可以并行处理序列数据
- 能够捕捉任意位置的依赖关系

作者: 研究团队
日期: 2024
"""

import os  # 操作系统模块

# ============================================================================
# 项目路径配置
# ============================================================================

# 项目根目录
PROJECT_DIR = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research'

# 处理后数据目录
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'processed_data')

# Transformer模型输出目录
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'AC_Transformer', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据文件路径
DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'ltpp_processed_data.csv')


# ============================================================================
# 特征配置（与LSTM完全一致）
# ============================================================================

# 目标变量：MRI (路面平整度指数)
TARGET_COL = 'MRI'

# 输入特征列表（与LSTM配置完全一致）
# 注意：数据中实际存在的列
FEATURE_COLS = [
    'PAVEMENT_AGE',              # 路面龄期
    'IRI_LAG_1',                  # 过去1年IRI
    'IRI_LAG_2',                  # 过去2年IRI
    'PAVEMENT_FAMILY_ENC',        # 路面结构类型编码
    'LATITUDE',                   # 纬度
    'LONGITUDE',                  # 经度
    'ELEVATION',                  # 海拔
    # 基本气候特征
    'DEGREE_DAYS_OVER_10C_YR',   # 年度度日数
    'COLDEST_AIR_TEMP',           # 最冷气温
    'HIGH_TEMP_7DAYS',            # 最高7日气温
    'MIN_SURFACE_50_TEMP',        # 最低地表温度
    # 文献补充气候特征
    'FREEZE_INDEX',              # 年冷冻指数
    'FREEZE_THAW',               # 年冻融天数
    'PRECIPITATION',             # 年降水量
    'EVAPORATION',               # 年蒸发量
    # 结构特征
    'TOTAL_THICKNESS',            # 总路面厚度
    'AC_THICKNESS',               # 沥青层厚度
    'BASE_THICKNESS',             # 基层厚度【新增】
    'NUM_LAYERS',                 # 结构层数量
]

# 输入特征维度（自动计算）
INPUT_DIM = len(FEATURE_COLS)


# ============================================================================
# 时序配置（与LSTM一致）
# ============================================================================

# 序列长度：用前5年数据预测第6年
SEQ_LEN = 5


# ============================================================================
# Transformer模型配置
# ============================================================================

# Transformer模型维度（决定模型容量）
# 增加维度以匹配数据量（200万样本需要更大容量）
TRANSFORMER_DIM = 256

# 注意力头数
# 多头注意力可以学习不同类型的依赖关系
NUM_HEADS = 8

# Transformer编码器层数
# 调优后最佳值：1层即可捕捉时序依赖，过多层反而过拟合
NUM_LAYERS = 1

# 前馈网络维度（Feed-Forward Network）
# 通常为TRANSFORMER_DIM的2-4倍
FF_DIM = 512

# Dropout比例：调优后最佳值0.3，防止过拟合
DROPOUT = 0.3

# 输出维度
OUTPUT_DIM = 1


# ============================================================================
# 训练配置
# ============================================================================

# Batch Size
BATCH_SIZE = 256

# 训练轮数
EPOCHS = 100

# 学习率：超参数调优最佳值
LEARNING_RATE = 0.001

# 权重衰减（正则化）
WEIGHT_DECAY = 1e-4

# 早停轮数
PATIENCE = 60


# ============================================================================
# 数据集划分配置（与LSTM一致）
# ============================================================================

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================================
# 随机种子
# ============================================================================

RANDOM_SEED = 42


# ============================================================================
# 特征名称映射（与LSTM一致）
# ============================================================================

FEATURE_NAMES_CN = {
    'PAVEMENT_AGE': '路面龄期',
    'IRI_LAG_1': '去年IRI',
    'IRI_LAG_2': '前年IRI',
    'PAVEMENT_FAMILY_ENC': '路面结构类型',
    'LATITUDE': '纬度',
    'LONGITUDE': '经度',
    'ELEVATION': '海拔',
    # 基本气候特征
    'DEGREE_DAYS_OVER_10C_YR': '年度度日数',
    'COLDEST_AIR_TEMP': '最冷气温',
    'HIGH_TEMP_7DAYS': '最高7日气温',
    'MIN_SURFACE_50_TEMP': '最低地表温度',
    # 文献补充气候特征
    'FREEZE_INDEX': '年冷冻指数',
    'FREEZE_THAW': '年冻融天数',
    'PRECIPITATION': '年降水量',
    'EVAPORATION': '年蒸发量',
    # 结构特征
    'TOTAL_THICKNESS': '总路面厚度',
    'AC_THICKNESS': '沥青层厚度',
    'BASE_THICKNESS': '基层厚度',
    'NUM_LAYERS': '结构层数量',
}
