"""
Transformer模型定义（简化版）
使用标准TransformerEncoder进行IRI时序预测

模型架构说明：
- 位置编码：添加序列位置信息
- 输入投影：将输入特征映射到模型维度
- Transformer编码器：使用自注意力提取序列特征
- 输出层：将编码器输出转换为IRI预测

Transformer vs LSTM：
- Transformer使用自注意力机制，可并行计算
- LSTM使用循环结构，串行计算但参数效率高
- Transformer能更好地捕捉长距离依赖

作者: 研究团队
日期: 2024
"""

import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import math  # 数学函数

# 动态获取输入维度，与训练配置保持一致
from AC_Transformer._1_config import INPUT_DIM


# ============================================================================
# 位置编码
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    位置编码层

    作用：为序列中的每个位置添加位置信息
    Transformer本身不包含位置信息，需要显式添加

    实现：使用正弦和余弦函数的不同频率组合
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    这种编码方式可以让模型学习相对位置关系
    """

    def __init__(self, d_model, max_len=100, dropout=0.1):
        """
        初始化位置编码

        参数:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比例
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 位置索引
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 频率项
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数维度使用sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度使用cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加batch维度
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # 注册为buffer（不参与训练但会保存）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播：添加位置编码

        参数:
            x: 输入张量 (batch, seq_len, d_model)

        返回:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# Transformer模型
# ============================================================================

class TransformerModel(nn.Module):
    """
    Transformer模型用于IRI预测

    模型结构:
        1. 输入投影层：feature_dim → d_model
        2. [CLS]标记：添加可学习的分类标记用于聚合序列信息
        3. 位置编码层：添加序列位置信息
        4. Transformer编码器：提取序列特征
        5. [CLS]输出层：使用[CLS]标记的输出进行预测

    改进说明：
        - 原始版本使用平均池化会丢失时序信息
        - 使用[CLS]标记可以让模型学习如何聚合序列信息
        - [CLS]标记在所有位置都会参与注意力计算
    """

    def __init__(self, input_dim, d_model, nhead, num_layers, ff_dim, output_dim, dropout=0.2):
        """
        初始化Transformer模型

        参数:
            input_dim: 输入特征维度
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_layers: Transformer编码器层数
            ff_dim: 前馈网络维度
            output_dim: 输出维度
            dropout: Dropout比例
        """
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 输入投影层：将输入特征映射到模型维度
        self.input_projection = nn.Linear(input_dim, d_model)

        # [CLS]标记：一个可学习的向量，用于聚合序列信息
        # 类似于BERT中的[CLS]标记
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,      # 模型维度
            nhead=nhead,          # 注意力头数
            dim_feedforward=ff_dim,  # 前馈网络维度
            dropout=dropout,      # Dropout比例
            batch_first=True      # (batch, seq, feature) 格式
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 输出层：添加LayerNorm和更强的正则化
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # d_model → d_model/2
            nn.LayerNorm(d_model // 2),         # 层归一化
            nn.ReLU(),                          # 激活函数
            nn.Dropout(dropout),                # Dropout
            nn.Linear(d_model // 2, output_dim)  # d_model/2 → output_dim
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量 (batch_size, seq_len, input_dim)

        返回:
            输出张量 (batch_size, output_dim)
        """
        batch_size = x.size(0)

        # 输入投影
        x = self.input_projection(x)  # (batch, seq, d_model)

        # 添加[CLS]标记到序列开头
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq+1, d_model)

        # 添加位置编码
        x = self.pos_encoder(x)

        # Transformer编码器
        x = self.transformer_encoder(x)  # (batch, seq+1, d_model)

        # 取[CLS]标记的输出（第一个位置）
        x = x[:, 0, :]  # (batch, d_model)

        # 输出层
        x = self.fc(x)  # (batch, output_dim)

        return x


# ============================================================================
# 模型测试函数
# ============================================================================

def test_model():
    """
    测试模型结构和输出形状
    """
    # 测试参数（从配置动态获取输入维度）
    batch_size = 32
    seq_len = 5
    input_dim = INPUT_DIM

    # 创建模型
    model = TransformerModel(
        input_dim=input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        ff_dim=128,
        output_dim=1
    )

    # 测试输入
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
