# -*- coding: utf-8 -*-
# 步骤1：初始化目录结构+定义Transformer全局参数（独立可运行，无前置条件）
# 运行成功后：生成interim/目录，下一步运行2_data_processor.py
import os
import joblib

# ---------------------- 【核心修改：绝对路径锁定】 ----------------------
# 获取当前脚本所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------- 全局参数（与LSTM版本完全一致） ----------------------
# 数据参数
SAMPLE_NUM = 1000  # 样本数量（与LSTM一致）
SEQ_LEN = 10       # 时序序列长度（与LSTM一致）
INPUT_DIM = 8      # 输入特征维度（与LSTM一致）
OUTPUT_DIM = 1     # 输出维度（疲劳寿命）（与LSTM一致）

# Transformer模型参数
D_MODEL = 64       # 模型维度（对应LSTM的HIDDEN_DIM）
NHEAD = 8          # 多头注意力头数
NUM_ENCODER_LAYERS = 3  # 编码器层数
DROPOUT_RATE = 0.1 # Dropout失活率（与LSTM一致）

# 训练参数（与LSTM完全一致）
BATCH_SIZE = 8    # 批次大小（与LSTM一致）
EPOCHS = 100       # 训练轮数（与LSTM一致）
LEARNING_RATE = 1e-4  # 学习率（与LSTM一致）

# ---------------------- 文件路径定义（关键手术点） ----------------------
# 使用 os.path.join 将相对路径转换为基于 BASE_DIR 的绝对路径
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERIM_DIR = os.path.join(BASE_DIR, "interim")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 中间产物路径（完全保留原有逻辑，仅变量已变为绝对路径）
PARAMS_SAVE_PATH = os.path.join(INTERIM_DIR, "step1_params.joblib")
TRANSFORMER_MODEL_SAVE_PATH = os.path.join(INTERIM_DIR, "step4_best_transformer_model.pth")
TRAIN_LOSS_PATH = os.path.join(INTERIM_DIR, "step4_train_loss.joblib")
VAL_LOSS_PATH = os.path.join(INTERIM_DIR, "step4_val_loss.joblib")

# ---------------------- 核心功能 ----------------------
def init_directories():
    """创建必要目录"""
    for dir_path in [DATA_DIR, INTERIM_DIR, OUTPUT_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建目录：{dir_path}")
        else:
            print(f"目录已存在：{dir_path}")

def save_params():
    """保存全局参数到interim目录"""
    params_dict = {
        "SAMPLE_NUM": SAMPLE_NUM, "SEQ_LEN": SEQ_LEN, "INPUT_DIM": INPUT_DIM,
        "OUTPUT_DIM": OUTPUT_DIM, "D_MODEL": D_MODEL, "NHEAD": NHEAD,
        "NUM_ENCODER_LAYERS": NUM_ENCODER_LAYERS, "DROPOUT_RATE": DROPOUT_RATE,
        "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS, "LEARNING_RATE": LEARNING_RATE
    }
    joblib.dump(params_dict, PARAMS_SAVE_PATH)
    print(f"参数保存至：{PARAMS_SAVE_PATH}")

# ---------------------- 独立运行入口 ----------------------
if __name__ == "__main__":
    print("="*50)
    print("步骤1：初始化目录+定义Transformer参数")
    print("="*50)
    init_directories()
    save_params()
    print("="*50)
    print("步骤1运行成功！下一步运行：2_data_processor.py")
    print("="*50)