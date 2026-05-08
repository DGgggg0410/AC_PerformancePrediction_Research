"""
Transformer 5年预测配置文件
扩展实验：用前5年数据预测未来第5年

与基础配置的差异：
- PREDICT_HORIZON = 5（预测5年后，而非1年后）
- 输出目录独立：output_5yr

作者: 研究团队
日期: 2024
"""

import os

# ============================================================================
# 项目路径配置
# ============================================================================

PROJECT_DIR = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research'
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'processed_data')

# 5年预测模型输出目录（与1年预测分开）
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'AC_Transformer', 'output_5yr')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'ltpp_processed_data.csv')


# ============================================================================
# 特征配置（与基础Transformer完全一致，19个特征）
# ============================================================================

TARGET_COL = 'MRI'

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
    'BASE_THICKNESS',             # 基层厚度
    'NUM_LAYERS',                 # 结构层数量
]

INPUT_DIM = len(FEATURE_COLS)


# ============================================================================
# 时序配置
# ============================================================================

SEQ_LEN = 5

# 【扩展实验关键参数】预测时间跨度
PREDICT_HORIZON = 5


# ============================================================================
# Transformer模型配置（增大容量以提升5年预测效果）
# ============================================================================

TRANSFORMER_DIM = 512   # 增大：256 → 512
NUM_HEADS = 4           # 调整：8 → 4（每个head更丰富，512/4=128）
NUM_LAYERS = 4          # 增大：3 → 4
FF_DIM = 1024           # 增大：512 → 1024
DROPOUT = 0.1           # 减小：0.2 → 0.1（增大模型后降低正则化）
OUTPUT_DIM = 1


# ============================================================================
# 训练配置
# ============================================================================

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3     # 增大：5e-4 → 1e-3（配合更大模型）
WEIGHT_DECAY = 1e-4
PATIENCE = 60


# ============================================================================
# 数据集划分配置
# ============================================================================

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================================
# 特征名称映射
# ============================================================================

FEATURE_NAMES_CN = {
    'PAVEMENT_AGE': '路面龄期',
    'IRI_LAG_1': '去年IRI',
    'IRI_LAG_2': '前年IRI',
    'PAVEMENT_FAMILY_ENC': '路面结构类型',
    'LATITUDE': '纬度',
    'LONGITUDE': '经度',
    'ELEVATION': '海拔',
    'DEGREE_DAYS_OVER_10C_YR': '年度度日数',
    'COLDEST_AIR_TEMP': '最冷气温',
    'HIGH_TEMP_7DAYS': '最高7日气温',
    'MIN_SURFACE_50_TEMP': '最低地表温度',
    'FREEZE_INDEX': '年冷冻指数',
    'FREEZE_THAW': '年冻融天数',
    'PRECIPITATION': '年降水量',
    'EVAPORATION': '年蒸发量',
    'TOTAL_THICKNESS': '总路面厚度',
    'AC_THICKNESS': '沥青层厚度',
    'BASE_THICKNESS': '基层厚度',
    'NUM_LAYERS': '结构层数量',
}
