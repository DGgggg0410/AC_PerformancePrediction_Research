"""
LSTM消融实验配置 - 去掉气候因素
用于验证气候因素对路面性能预测的贡献

实验设计：
- 原始特征：19个（时序3 + 结构5 + 地理3 + 气候10）
- 消融后特征：11个（时序3 + 结构5 + 地理3）- 去掉10个气候因素

作者: 研究团队
日期: 2024
"""

import os

# ============================================================================
# 项目路径配置
# ============================================================================

PROJECT_DIR = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research'
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'processed_data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'AC_LSTM', 'output', 'ablation_no_climate')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'ltpp_processed_data.csv')


# ============================================================================
# 特征配置 - 去掉所有气候因素
# ============================================================================

TARGET_COL = 'MRI'

# 去掉气候因素后的特征列表（共11个）
FEATURE_COLS = [
    # 时序特征（3个）- 保留
    'PAVEMENT_AGE',              # 路面龄期
    'IRI_LAG_1',                 # 去年IRI
    'IRI_LAG_2',                 # 前年IRI

    # 结构特征（5个）- 保留
    'PAVEMENT_FAMILY_ENC',      # 路面结构类型编码
    'TOTAL_THICKNESS',           # 总路面厚度
    'AC_THICKNESS',              # 沥青层厚度
    'BASE_THICKNESS',            # 基层厚度
    'NUM_LAYERS',                # 结构层数量

    # 地理特征（3个）- 保留
    'LATITUDE',                  # 纬度
    'LONGITUDE',                 # 经度
    'ELEVATION',                 # 海拔

    # ❌ 去掉气候因素（10个）
    # 'DEGREE_DAYS_OVER_10C_YR',  # 年度度日数
    # 'COLDEST_AIR_TEMP',         # 最冷气温
    # 'HIGH_TEMP_7DAYS',          # 最高7日气温
    # 'MIN_SURFACE_50_TEMP',     # 最低地表温度
    # 'FREEZE_INDEX',             # 年冷冻指数
    # 'FREEZE_THAW',              # 年冻融天数
    # 'PRECIPITATION',            # 年降水量
    # 'PRECIP_DAYS',              # 年降水天数
    # 'EVAPORATION',              # 年蒸发量
]

INPUT_DIM = len(FEATURE_COLS)

# 实验名称（用于保存结果）
EXPERIMENT_NAME = "ablation_no_climate"


# ============================================================================
# 时序配置
# ============================================================================

SEQ_LEN = 5


# ============================================================================
# 模型架构配置
# ============================================================================

HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.1
OUTPUT_DIM = 1


# ============================================================================
# 训练配置
# ============================================================================

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 30


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
    'TOTAL_THICKNESS': '总路面厚度',
    'AC_THICKNESS': '沥青层厚度',
    'BASE_THICKNESS': '基层厚度',
    'NUM_LAYERS': '结构层数量',
    'LATITUDE': '纬度',
    'LONGITUDE': '经度',
    'ELEVATION': '海拔',
}
