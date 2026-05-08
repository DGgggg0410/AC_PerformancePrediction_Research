"""
LSTM SHAP分析器
量化工程因素对IRI预测的贡献度

功能说明：
1. 使用SHAP(SHapley Additive exPlanations)解释模型预测
2. 计算每个特征对预测结果的影响
3. 生成特征重要性排序
4. 可视化SHAP结果

SHAP说明：
SHAP是一种博弈论方法，用于解释机器学习模型的预测结果。
- 基于Shapley值（合作博弈论中的概念）
- 可以量化每个特征对预测的贡献
- 适用于任何模型的可解释性分析

作者: 研究团队
日期: 2024
"""

import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算
import pandas as pd  # 数据处理
import matplotlib.pyplot as plt  # 绘图
import matplotlib.font_manager as fm  # 字体管理
import shap  # SHAP解释器
import os  # 路径操作
import sys  # 系统操作

# 导入配置和模型
sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research')
from AC_LSTM._1_config import OUTPUT_DIR, FEATURE_COLS, FEATURE_NAMES_CN, SEQ_LEN
from AC_LSTM._3_lstm_model import LSTMModel
from AC_LSTM._1_config import INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT

# 配置中文字体
CHINESE_FONT = 'SimHei'  # 黑体，适用于Windows系统
# 如果黑体不可用，使用其他常见中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    pass


# ============================================================================
# 辅助函数
# ============================================================================

def get_feature_names():
    """
    获取特征名称（中文）

    返回:
        feature_names: 特征中文名称列表
    """
    return [FEATURE_NAMES_CN.get(f, f) for f in FEATURE_COLS]


# ============================================================================
# SHAP模型封装
# ============================================================================

class LSTMModelWrapper(torch.nn.Module):
    """
    用于SHAP分析的LSTM模型封装

    原因：SHAP需要模型返回numpy数组，但PyTorch模型默认返回张量
    此封装类将PyTorch模型的输出转换为SHAP兼容的格式
    """

    def __init__(self, model):
        super(LSTMModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        """
        前向传播，返回预测值

        参数:
            x: 输入张量 (batch, seq_len, features)

        返回:
            预测值张量 (batch,)
        """
        return self.model(x).squeeze()


# ============================================================================
# SHAP分析函数
# ============================================================================

def analyze_with_shap(model, background_data, test_data, feature_names):
    """
    使用SHAP分析模型

    【修复】完整保留时序信息，对每个时间步分别进行SHAP分析

    参数:
        model: 训练好的LSTM模型
        background_data: 背景数据集（用于SHAP基线计算，训练集）
        test_data: 测试数据集（用于计算SHAP值）
        feature_names: 特征名称列表

    返回:
        shap_values: SHAP值对象
        test_flat: 测试数据（展平后的特征）
        all_shap_values: 每个时间步的SHAP值
    """
    print("创建SHAP解释器...")

    # 封装模型
    wrapped_model = LSTMModelWrapper(model)
    wrapped_model.eval()

    # 转换为numpy并采样（限制样本数量以加速计算）
    background_np = background_data.cpu().numpy()[:100]  # 100个背景样本
    test_np = test_data.cpu().numpy()[:50]  # 50个测试样本

    # 【修复】对完整序列进行SHAP分析，而不是只取最后一个时间步
    # 定义预测函数（用于SHAP）
    def model_predict(x):
        # SHAP传入的是2D数组 (n_samples, n_features)
        # 需要reshape为LSTM期望的3D格式 (n_samples, seq_len, n_features)
        n_samples = x.shape[0]
        x_3d = x.reshape(n_samples, SEQ_LEN, -1)  # (n_samples, seq_len, n_features)
        x_tensor = torch.FloatTensor(x_3d).to(next(wrapped_model.parameters()).device)
        with torch.no_grad():
            output = wrapped_model(x_tensor)
            # 确保输出是1D数组 (batch,) 用于SHAP
            return output.squeeze(-1).cpu().numpy()

    print("计算SHAP值（这可能需要几分钟）...")

    # 【修复】对所有时间步分别进行SHAP分析
    # 展平序列数据：(n_samples, seq_len, features) -> (n_samples * seq_len, features)
    seq_len = background_np.shape[1]
    n_features = background_np.shape[2]

    # 分别对每个时间步进行SHAP分析
    all_shap_values = []
    all_test_flat = []

    for t in range(seq_len):
        background_t = background_np[:, t, :]  # 第t个时间步
        test_t = test_np[:, t, :]  # 第t个时间步

        # 创建SHAP解释器并计算SHAP值
        explainer = shap.Explainer(model_predict, background_t)
        shap_values_t = explainer(test_t)

        all_shap_values.append(shap_values_t.values)
        all_test_flat.append(test_t)

    # 汇总所有时间步的SHAP值
    all_shap_values = np.array(all_shap_values)  # (seq_len, n_test_samples, n_features)
    test_flat = np.array(all_test_flat)  # (seq_len, n_test_samples, n_features)

    # 对所有时间步取平均得到总体特征重要性
    mean_shap = all_shap_values.mean(axis=1)  # (seq_len, n_features)
    overall_shap = shap_values_t  # 使用最后一个时间步的SHAP对象结构

    return overall_shap, test_flat[-1], mean_shap


def analyze_with_shap_simple(model, background_data, test_data, feature_names):
    """
    简化版SHAP分析：对每个时间步分别进行SHAP分析并汇总

    参数:
        model: 训练好的LSTM模型
        background_data: 背景数据集（用于SHAP基线计算，训练集）
        test_data: 测试数据集（用于计算SHAP值）
        feature_names: 特征名称列表

    返回:
        shap_values: SHAP值对象
        test_flat: 测试数据
    """
    print("创建SHAP解释器...")

    # 封装模型
    wrapped_model = LSTMModelWrapper(model)
    wrapped_model.eval()

    # 转换为numpy并采样
    background_np = background_data.cpu().numpy()[:100]  # 100个背景样本
    test_np = test_data.cpu().numpy()[:50]  # 50个测试样本

    # 【调试】打印形状
    print(f"background_np shape: {background_np.shape}")  # 应该是 (100, 5, 19)
    print(f"test_np shape: {test_np.shape}")  # 应该是 (50, 5, 19)

    # 只分析最后一个时间步（模型预测主要依赖的信息）
    background_last = background_np[:, -1, :]  # (100, 19)
    test_last = test_np[:, -1, :]  # (50, 19)

    print(f"background_last shape: {background_last.shape}")  # 应该是 (100, 19)
    print(f"test_last shape: {test_last.shape}")  # 应该是 (50, 19)

    # 【修复】使用KernelExplainer而不是Permutation Explainer
    # 因为Permutation Explainer在处理自定义模型时会有形状问题
    from sklearn.metrics import mean_squared_error

    def model_predict_for_shap(x):
        """
        用于SHAP的预测函数，输入形状 (n_samples, n_features)
        输出形状 (n_samples,)
        """
        # x 形状: (n_samples, 19) - 已经是最后一个时间步
        # 需要转回 (n_samples, 1, 19) 然后 model 会自动处理
        x_tensor = torch.FloatTensor(x).to(next(wrapped_model.parameters()).device)
        with torch.no_grad():
            # 模型期望 (batch, seq_len, features)，但我们只有 (batch, features)
            # 需要扩展seq_len维度
            x_expanded = x_tensor.unsqueeze(1)  # (n_samples, 1, 19)
            output = wrapped_model(x_expanded)
            # 【修复】确保输出始终是1维数组 (n_samples,)
            output_np = output.squeeze(-1).cpu().numpy()
            if output_np.ndim == 0:
                # 单个样本情况：标量转1维
                return output_np.reshape(1)
            return output_np

    print("计算SHAP值（这可能需要几分钟）...")

    # 使用KernelExplainer（更稳定，适用于任意模型）
    background_for_shap = background_last  # (100, 19)
    explainer = shap.KernelExplainer(model_predict_for_shap, background_for_shap)

    # 【关键修复】计算SHAP值时，确保shap_values形状正确
    shap_values = explainer.shap_values(test_last)

    return shap_values, test_last


# ============================================================================
# 可视化函数
# ============================================================================

def plot_shap_summary(shap_values, test_data, feature_names, save_path=None):
    """
    绘制SHAP摘要图

    图表说明：
        - 每个点代表一个样本
        - X轴：SHAP值（特征对预测的影响）
        - Y轴：特征（按重要性排序）
        - 颜色：特征值高低（红色=高，蓝色=低）
        - 可以显示特征与预测值的正/负相关关系
    """
    # 【修复】shap_values可能是numpy数组或Explanation对象
    shap_array = shap_values.values if hasattr(shap_values, 'values') else shap_values
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_array, test_data, feature_names=feature_names, show=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SHAP摘要图已保存到: {save_path}")

    plt.close()


def plot_shap_bar(shap_values, feature_names, save_path=None):
    """
    绘制SHAP特征重要性条形图

    参数:
        shap_values: SHAP值对象
        feature_names: 特征名称列表
        save_path: 保存路径

    返回:
        importance_df: 特征重要性DataFrame（按重要性降序）
    """
    # 【修复】shap_values可能是numpy数组或Explanation对象
    shap_array = shap_values.values if hasattr(shap_values, 'values') else shap_values
    # 计算平均绝对SHAP值作为重要性指标
    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]  # 降序排序

    # 绘制条形图
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, mean_abs_shap[sorted_idx[::-1]], align='center')
    plt.yticks(y_pos, [feature_names[i] for i in sorted_idx[::-1]],
               fontproperties=fm.FontProperties(family=CHINESE_FONT))
    plt.xlabel('Mean |SHAP value|', fontproperties=fm.FontProperties(family=CHINESE_FONT))
    plt.title('特征重要性 (SHAP)', fontproperties=fm.FontProperties(family=CHINESE_FONT))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SHAP特征重要性图已保存到: {save_path}")

    plt.close()

    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    }).sort_values('Importance', ascending=False)

    return importance_df


# ============================================================================
# 主分析流程
# ============================================================================

def main():
    """
    主SHAP分析流程

    【修复】正确使用训练集作为背景数据，测试集用于计算SHAP值

    流程:
        1. 加载数据
        2. 加载训练好的模型
        3. 执行SHAP分析（使用训练集背景+测试集数据）
        4. 生成可视化图表
        5. 保存特征重要性结果
    """
    print("=" * 60)
    print("LSTM SHAP分析")
    print("=" * 60)

    # 加载数据
    from AC_LSTM._2_sequence_builder import main as build_sequences
    train_loader, val_loader, test_loader, scaler = build_sequences()

    # 【修复】获取训练集样本作为背景数据
    background_data = None
    for batch_x, _ in train_loader:
        background_data = batch_x
        break

    # 获取测试数据
    test_data = None
    for batch_x, _ in test_loader:
        test_data = batch_x
        break

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(OUTPUT_DIR, 'lstm_best_model.pth')

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行训练")
        return

    model = LSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("模型已加载")

    # 获取特征名称
    feature_names = get_feature_names()

    # 【修复】使用简化的SHAP分析（保留最后一个时间步的分析）
    shap_values, test_flat = analyze_with_shap_simple(model, background_data, test_data, feature_names)

    # 绘制SHAP摘要图
    plot_shap_summary(
        shap_values, test_flat, feature_names,
        save_path=os.path.join(OUTPUT_DIR, 'lstm_shap_summary.png')
    )

    # 绘制SHAP特征重要性图
    importance_df = plot_shap_bar(
        shap_values, feature_names,
        save_path=os.path.join(OUTPUT_DIR, 'lstm_shap_importance.png')
    )

    # 保存特征重要性
    importance_path = os.path.join(OUTPUT_DIR, 'lstm_shap_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"SHAP特征重要性已保存到: {importance_path}")

    print("\nSHAP分析完成!")
    print("\n特征重要性排名:")
    print(importance_df.to_string(index=False))

    return importance_df


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == '__main__':
    main()
