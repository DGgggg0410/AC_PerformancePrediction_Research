"""
LSTM模型定义
双层LSTM用于IRI时序预测

模型架构说明：
- 输入层：接收(seq_len, input_dim)的序列
- LSTM层：两层堆叠，提取时序特征
- 全连接层：将LSTM输出转换为最终预测

LSTM (Long Short-Term Memory) 是一种特殊的循环神经网络：
- 擅长处理长序列依赖问题
- 通过门控机制避免梯度消失
- 适合IRI这种具有时序相关性的预测任务

作者: 研究团队
日期: 2024
"""

import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块


# ============================================================================
# LSTM模型定义
# ============================================================================

class LSTMModel(nn.Module):
    """
    LSTM模型用于IRI路面平整度预测

    模型结构:
        1. LSTM层：双层LSTM，提取序列特征
        2. 全连接层：128→64→1，将LSTM输出转换为IRI预测值

    前向传播:
        输入x → LSTM → 取最后时间步 → FC层 → 输出预测IRI
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        初始化LSTM模型

        参数:
            input_dim: 输入特征维度（每个时间步的特征数）
            hidden_dim: LSTM隐藏层维度（决定模型容量）
            num_layers: LSTM层数（增加可学习更深层的特征）
            output_dim: 输出维度（预测目标的数量）
            dropout: Dropout比例（防止过拟合）
        """
        super(LSTMModel, self).__init__()  # 调用父类构造函数

        # 保存模型参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM层
        # batch_first=True: 输入/输出张量的第一个维度是batch size
        # dropout: 除最后一层外，在LSTM层之间应用Dropout
        self.lstm = nn.LSTM(
            input_dim,              # 输入特征维度
            hidden_dim,             # 隐藏层维度
            num_layers,             # LSTM层数
            batch_first=True,       # (batch, seq, feature) 格式
            dropout=dropout if num_layers > 1 else 0  # 多层时使用Dropout
        )

        # 全连接层（输出层）
        # 两层结构：hidden_dim → hidden_dim//2 → output_dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 128 → 64
            nn.ReLU(),                                 # 激活函数
            nn.Dropout(dropout),                       # Dropout
            nn.Linear(hidden_dim // 2, output_dim)      # 64 → 1
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)

        返回:
            输出张量，形状为 (batch_size, output_dim)
        """
        # LSTM前向传播
        # lstm_out: 所有时间步的输出 (batch, seq_len, hidden_dim)
        # hidden: 最后时间步的隐藏状态 (num_layers, batch, hidden_dim)
        # cell: 最后时间步的细胞状态 (num_layers, batch, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(x)

        # 取最后一个时间步的隐藏状态作为序列表示
        # hidden[-1] 是最后一层LSTM的隐藏状态
        out = self.fc(hidden[-1])  # (batch_size, output_dim)

        return out


# ============================================================================
# 模型测试函数
# ============================================================================

def test_model():
    """
    测试模型结构和输出形状

    用于验证模型是否正确定义
    """
    # 测试参数
    batch_size = 32
    seq_len = 5
    # 动态获取input_dim，避免硬编码fallback值
    try:
        import sys
        sys.path.insert(0, r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research')
        from AC_LSTM._1_config import INPUT_DIM
        input_dim = INPUT_DIM
    except (ImportError, ModuleNotFoundError):
        input_dim = 19  # 默认19个特征（FALLBACK）

    # 创建模型实例
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        output_dim=1
    )

    # 创建测试输入
    x = torch.randn(batch_size, seq_len, input_dim)

    # 前向传播
    output = model(x)

    # 打印信息
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == '__main__':
    test_model()
