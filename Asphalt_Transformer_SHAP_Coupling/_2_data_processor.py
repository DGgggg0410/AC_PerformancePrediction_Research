# -*- coding: utf-8 -*-
# 步骤2：数据生成+预处理+划分数据集（独立可运行，前置：步骤1，与LSTM完全一致）
# 运行成功后：生成中间产物（归一化器、数据集），下一步运行3_transformer_model.py
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
# 【仅新增】引入绝对路径变量
from _1_config import INTERIM_DIR, DATA_DIR, PARAMS_SAVE_PATH

# ---------------------- 加载步骤1的全局参数+校验前置条件 ----------------------
def check_prerequisite():
    """校验步骤1是否运行成功，缺少中间产物则退出"""
    # 【最小修改】使用从 config 导入的绝对路径
    if not os.path.exists(PARAMS_SAVE_PATH):
        print(f"错误！未找到步骤1的中间产物，请先运行1_config.py\n预期路径: {PARAMS_SAVE_PATH}")
        exit(1)
    # 加载全局参数
    params_dict = joblib.load(PARAMS_SAVE_PATH)
    return params_dict

def load_params(params_dict):
    """提取全局参数"""
    SAMPLE_NUM = params_dict["SAMPLE_NUM"]
    SEQ_LEN = params_dict["SEQ_LEN"]
    INPUT_DIM = params_dict["INPUT_DIM"]
    return SAMPLE_NUM, SEQ_LEN, INPUT_DIM

# ---------------------- 数据处理核心功能 ----------------------
def generate_and_process_data(SAMPLE_NUM, SEQ_LEN, INPUT_DIM):
    """生成数据+预处理+划分数据集（与LSTM版本完全一致）"""
    # 1. 生成静态配方/工况特征（逻辑不变）
    oil_stone_ratio = np.random.uniform(3.5, 5.5, SAMPLE_NUM)
    void_ratio = np.random.uniform(2.0, 6.0, SAMPLE_NUM)
    mineral_powder = np.random.uniform(3.0, 8.0, SAMPLE_NUM)
    asphalt_type = np.random.randint(0, 3, SAMPLE_NUM)
    aggregate_grade = np.random.randint(0, 3, SAMPLE_NUM)
    temperature = np.random.uniform(-10.0, 40.0, SAMPLE_NUM)
    stress = np.random.uniform(0.1, 1.0, SAMPLE_NUM)
    
    # 2. 生成时序劲度模量特征（逻辑不变）
    stiffness_data = []
    for i in range(SAMPLE_NUM):
        base_stiffness = 12000 - (temperature[i] * 100) - (stress[i] * 1000)
        init_stiffness = base_stiffness + np.random.uniform(-500, 500)
        decay = np.random.uniform(30, 120, SEQ_LEN)
        stiffness_seq = init_stiffness - np.cumsum(decay)
        stiffness_data.append(stiffness_seq)
    stiffness_data = np.array(stiffness_data)
    
    # 3. 生成疲劳寿命标签（逻辑不变）
    fatigue_life = (
        15 + 
        1.2 * oil_stone_ratio - 
        0.8 * void_ratio - 
        0.1 * temperature - 
        5.0 * stress + 
        0.00015 * stiffness_data[:, -1]
    ) + np.random.normal(0, 0.3, SAMPLE_NUM)
    
    # 4. 整合特征（静态特征+时序特征）
    static_features = np.column_stack([
        oil_stone_ratio, void_ratio, mineral_powder,
        asphalt_type, aggregate_grade, temperature, stress
    ])
    X_static = np.repeat(static_features[:, np.newaxis, :], SEQ_LEN, axis=1)
    X_stiffness = stiffness_data[:, :, np.newaxis]
    X = np.concatenate([X_static[:, :, :7], X_stiffness], axis=2)
    
    # 5. 数据归一化
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_flatten = X.reshape(-1, X.shape[2])
    X_flatten_scaled = scaler_X.fit_transform(X_flatten)
    X_scaled = X_flatten_scaled.reshape(X.shape)
    
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(fatigue_life.reshape(-1, 1))
    
    # 6. 划分训练/验证/测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 7. 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # 8. 保存原始数据到data目录 - 【最小修改】使用绝对路径
    df_columns = [
        "oil_stone_ratio", "void_ratio", "mineral_powder",
        "asphalt_type", "aggregate_grade", "temperature", "stress"
    ] + [f"stiffness_{(i+1)*1000}" for i in range(SEQ_LEN)] + ["fatigue_life"]
    df_data = np.column_stack([static_features, stiffness_data, fatigue_life.reshape(-1, 1)])
    df = pd.DataFrame(df_data, columns=df_columns)
    
    raw_data_path = os.path.join(DATA_DIR, "asphalt_sample_data.csv")
    df.to_csv(raw_data_path, index=False)
    print(f"成功保存原始数据到：{raw_data_path}")
    
    # 9. 保存中间产物到interim目录 - 【最小修改】使用绝对路径
    interim_files = {
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "X_train_tensor": X_train_tensor,
        "y_train_tensor": y_train_tensor,
        "X_val_tensor": X_val_tensor,
        "y_val_tensor": y_val_tensor,
        "X_test_tensor": X_test_tensor,
        "y_test_tensor": y_test_tensor
    }
    
    for key, value in interim_files.items():
        save_path = os.path.join(INTERIM_DIR, f"step2_{key}.{'joblib' if 'scaler' in key else 'pt'}")
        if "scaler" in key:
            joblib.dump(value, save_path)
        else:
            torch.save(value, save_path)
        print(f"成功保存中间产物：{save_path}")
    
    return scaler_X, scaler_y, X_test_tensor, y_test_tensor

# ---------------------- 独立运行入口 ----------------------
if __name__ == "__main__":
    print("="*50)
    print("开始执行步骤2：数据生成+预处理+划分数据集")
    print("="*50)
    
    # 1. 校验前置条件+加载参数
    params_dict = check_prerequisite()
    SAMPLE_NUM, SEQ_LEN, INPUT_DIM = load_params(params_dict)
    
    # 2. 执行数据处理核心功能
    generate_and_process_data(SAMPLE_NUM, SEQ_LEN, INPUT_DIM)
    
    # 3. 运行成功提示
    print("="*50)
    print("步骤2运行成功！所有中间产物已保存至interim目录")
    print("下一步请运行：3_transformer_model.py")
    print("="*50)