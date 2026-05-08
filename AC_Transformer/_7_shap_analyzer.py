"""
Transformer SHAP分析器
量化工程因素对IRI预测的贡献度

功能说明：
与LSTM SHAP分析器功能相同，专门用于Transformer模型

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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_chinese_font():
    """获取可用的中文字体"""
    fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'PingFang SC', 'STHeiti']
    for font in fonts:
        if font in [f.name for f in fm.fontManager.ttflist]:
            return font
    return fm.FontProperties(family='sans-serif').get_name()

CHINESE_FONT = get_chinese_font()

# 导入配置和模型
sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research')
from AC_Transformer._1_config import OUTPUT_DIR, FEATURE_COLS, FEATURE_NAMES_CN
from AC_Transformer._3_transformer_model import TransformerModel
from AC_Transformer._1_config import INPUT_DIM, TRANSFORMER_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, DROPOUT


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

class TransformerModelWrapper(torch.nn.Module):
    """
    用于SHAP分析的Transformer模型封装
    """

    def __init__(self, model):
        super(TransformerModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        """
        前向传播
        """
        return self.model(x).squeeze()


# ============================================================================
# SHAP分析函数
# ============================================================================

def analyze_with_shap(model, background_data, test_data, feature_names):
    """
    使用SHAP分析Transformer模型

    参数:
        model: 训练好的Transformer模型
        background_data: 背景数据集
        test_data: 测试数据集
        feature_names: 特征名称列表

    返回:
        shap_values: SHAP值对象
        test_flat: 测试数据
    """
    print("创建SHAP解释器...")

    # 封装模型
    wrapped_model = TransformerModelWrapper(model)
    wrapped_model.eval()

    # 转换为numpy并采样
    background_np = background_data.cpu().numpy()[:100]
    test_np = test_data.cpu().numpy()[:50]

    # 定义预测函数
    def model_predict(x):
        # SHAP传入的是2D数据 (batch, features)
        # 需要转换为3D (batch, 1, features) 模拟单个时间步输入
        x_tensor = torch.FloatTensor(x).unsqueeze(1).to(next(wrapped_model.parameters()).device)
        with torch.no_grad():
            return wrapped_model(x_tensor).cpu().numpy()

    print("计算SHAP值（这可能需要几分钟）...")

    # 取最后一个时间步的特征进行SHAP分析
    background_flat = background_np[:, -1, :]  # (100, features)
    test_flat = test_np[:, -1, :]  # (50, features)

    # 创建SHAP解释器并计算SHAP值
    explainer = shap.Explainer(model_predict, background_flat)
    shap_values = explainer(test_flat)

    return shap_values, test_flat


# ============================================================================
# 可视化函数
# ============================================================================

def plot_shap_summary(shap_values, test_data, feature_names, save_path=None):
    """
    绘制SHAP摘要图
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values.values, test_data, feature_names=feature_names, show=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SHAP摘要图已保存到: {save_path}")

    plt.close()


def plot_shap_bar(shap_values, feature_names, save_path=None):
    """
    绘制SHAP特征重要性条形图
    """
    # 计算平均绝对SHAP值
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    # 绘制条形图
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, mean_abs_shap[sorted_idx[::-1]], align='center')
    plt.yticks(y_pos, [feature_names[i] for i in sorted_idx[::-1]],
               fontproperties=fm.FontProperties(family=CHINESE_FONT))
    plt.xlabel('Mean |SHAP value|', fontproperties=fm.FontProperties(family=CHINESE_FONT))
    plt.title('特征重要性 (SHAP) - Transformer', fontproperties=fm.FontProperties(family=CHINESE_FONT))

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
    """
    print("=" * 60)
    print("Transformer SHAP分析")
    print("=" * 60)

    # 加载数据
    from AC_Transformer._2_sequence_builder import main as build_sequences
    train_loader, _, test_loader, scaler, _ = build_sequences()

    # 获取背景和测试数据
    background_data = None
    for batch_x, _ in train_loader:
        background_data = batch_x
        break

    test_data = None
    for batch_x, _ in test_loader:
        test_data = batch_x
        break

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(OUTPUT_DIR, 'transformer_best_model.pth')

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行训练")
        return

    model = TransformerModel(INPUT_DIM, TRANSFORMER_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, 1, DROPOUT)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("模型已加载")

    # 获取特征名称
    feature_names = get_feature_names()

    # SHAP分析
    shap_values, test_flat = analyze_with_shap(model, background_data, test_data, feature_names)

    # 绘制SHAP摘要图
    plot_shap_summary(
        shap_values, test_flat, feature_names,
        save_path=os.path.join(OUTPUT_DIR, 'transformer_shap_summary.png')
    )

    # 绘制SHAP特征重要性图
    importance_df = plot_shap_bar(
        shap_values, feature_names,
        save_path=os.path.join(OUTPUT_DIR, 'transformer_shap_importance.png')
    )

    # 保存特征重要性
    importance_path = os.path.join(OUTPUT_DIR, 'transformer_shap_importance.csv')
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
