# -*- coding: utf-8 -*-
# 步骤1：初始化目录结构+定义LSTM全局参数（独立可运行，无前置条件）
# 运行成功后：生成interim/目录，下一步运行2_data_processor.py
import os
import joblib

# ---------------------- 全局参数（与Transformer版本完全一致） ----------------------
# 数据参数
SAMPLE_NUM = 1000  # 样本数量（与Transformer一致）
SEQ_LEN = 10       # 时序序列长度（与Transformer一致）
INPUT_DIM = 8      # 输入特征维度（与Transformer一致）
OUTPUT_DIM = 1     # 输出维度（疲劳寿命）（与Transformer一致）

# LSTM模型参数
HIDDEN_DIM = 64    # LSTM隐藏层维度（对应Transformer的D_MODEL）
NUM_LAYERS = 3     # LSTM网络层数（对应Transformer的NUM_ENCODER_LAYERS）
DROPOUT_RATE = 0.1 # Dropout失活率（与Transformer一致）

# 训练参数（与Transformer完全一致）
BATCH_SIZE = 8    # 批次大小（与Transformer一致）
EPOCHS = 100       # 训练轮数（与Transformer一致）
LEARNING_RATE = 1e-4  # 学习率（与Transformer一致）

# 自动锁定当前脚本（_1_config.py）所在的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 强制路径指向子文件夹内部
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERIM_DIR = os.path.join(BASE_DIR, "interim")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 中间产物路径（与Transformer版本对应）
PARAMS_SAVE_PATH = os.path.join(INTERIM_DIR, "step1_params.joblib")
LSTM_MODEL_SAVE_PATH = os.path.join(INTERIM_DIR, "step4_best_lstm_model.pth")
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
        "OUTPUT_DIM": OUTPUT_DIM, "HIDDEN_DIM": HIDDEN_DIM, 
        "NUM_LAYERS": NUM_LAYERS, "DROPOUT_RATE": DROPOUT_RATE,
        "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS, "LEARNING_RATE": LEARNING_RATE
    }
    joblib.dump(params_dict, PARAMS_SAVE_PATH)
    print(f"参数保存至：{PARAMS_SAVE_PATH}")

# ---------------------- 独立运行入口 ----------------------
if __name__ == "__main__":
    print("="*50)
    print("步骤1：初始化目录+定义LSTM参数")
    print("="*50)
    init_directories()
    save_params()
    print("="*50)
    print("步骤1运行成功！下一步运行：2_data_processor.py")
    print("="*50)