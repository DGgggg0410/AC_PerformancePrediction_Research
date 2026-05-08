# 消融实验说明文档

## 实验概述

本目录包含LSTM和Transformer模型的消融实验（Ablation Study），用于验证不同特征类别对路面性能预测的贡献。

## 消融实验列表

### 实验1: 去掉气候因素 (ablation_no_climate)
- **移除特征**: 10个气候相关特征
  - DEGREE_DAYS_OVER_10C_YR (年度度日数)
  - COLDEST_AIR_TEMP (最冷气温)
  - HIGH_TEMP_7DAYS (最高7日气温)
  - MIN_SURFACE_50_TEMP (最低地表温度)
  - FREEZE_INDEX (年冷冻指数)
  - FREEZE_THAW (年冻融天数)
  - PRECIPITATION (年降水量)
  - PRECIP_DAYS (年降水天数)
  - EVAPORATION (年蒸发量)
- **保留特征**: 11个 (时序3 + 结构5 + 地理3)
- **目的**: 验证气候因素对路面性能预测的重要性

### 实验2: 去掉结构因素 (ablation_no_structure)
- **移除特征**: 5个结构相关特征
  - PAVEMENT_FAMILY_ENC (路面结构类型)
  - TOTAL_THICKNESS (总路面厚度)
  - AC_THICKNESS (沥青层厚度)
  - BASE_THICKNESS (基层厚度)
  - NUM_LAYERS (结构层数量)
- **保留特征**: 14个 (时序3 + 气候10 + 地理3)
- **目的**: 验证路面结构设计对性能预测的影响

### 实验3: 去掉地理因素 (ablation_no_geographic)
- **移除特征**: 3个地理相关特征
  - LATITUDE (纬度)
  - LONGITUDE (经度)
  - ELEVATION (海拔)
- **保留特征**: 16个 (时序3 + 结构5 + 气候10)
- **目的**: 验证地理位置对路面性能的影响

### 实验4: 只保留时序特征 (ablation_only_temporal)
- **保留特征**: 仅3个
  - PAVEMENT_AGE (路面龄期)
  - IRI_LAG_1 (去年IRI)
  - IRI_LAG_2 (前年IRI)
- **移除特征**: 16个 (结构5 + 气候10 + 地理3)
- **目的**: 验证纯时序自回归模型的预测能力

### 实验5: 去掉气候+结构因素 (ablation_no_climate_structure)
- **移除特征**: 15个 (气候10 + 结构5)
- **保留特征**: 6个 (时序3 + 地理3)
- **目的**: 验证在非气候、非结构因素下的模型表现

## 运行方式

### LSTM 消融实验

```bash
cd e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/AC_LSTM/ablation

python _run_all_lstm_ablation.py
```

### Transformer 消融实验

```bash
cd e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/AC_Transformer/ablation

python _run_all_transformer_ablation.py
```

### 综合对比分析

```bash
cd e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research

python ablation_analysis.py
```

## 输出文件

每个实验会在对应输出目录生成：
- `{experiment_name}_best_model.pth` - 最佳模型权重
- `{experiment_name}_results.json` - 实验结果指标
- `{experiment_name}_training.png` - 训练曲线
- `scaler.pkl` - 数据标准化器

综合分析会生成：
- `ablation_experiment_comparison.png` - 消融实验对比图表
- 终端打印详细的对比分析报告

## 预期结果解读

### 性能下降分析方法

如果某个消融实验的R²显著下降，说明被移除的特征类别对预测非常重要：

| 实验 | R²下降幅度 | 结论 |
|------|-----------|------|
| 去掉气候 | > 2% | 气候因素对路面性能有显著影响 |
| 去掉结构 | > 1% | 路面结构设计影响性能预测 |
| 去掉地理 | < 0.5% | 地理位置影响较小 |
| 只留时序 | > 5% | 时序自相关性是主要预测来源 |

### 论文写作建议

1. **重点讨论**: 对性能影响最大的特征类别
2. **工程意义**: 解释为什么某些因素更重要
3. **对比分析**: LSTM vs Transformer在不同因素上的敏感性差异
4. **局限性**: 消融实验的局限性（特征可能存在相关性）

## 基准性能

| 模型 | R² | RMSE (m/km) | MAE (m/km) | 特征数 |
|------|-----|-------------|------------|--------|
| LSTM (基准) | 0.9562 | 0.1461 | 0.0358 | 19 |
| Transformer (基准) | 0.9582 | 0.1427 | 0.0410 | 19 |

## 注意事项

1. 每个消融实验使用相同的随机种子(42)和数据划分比例(70/15/15)
2. 模型超参数保持与基准实验一致
3. 训练过程中使用了早停机制(LSTM: 30 epochs, Transformer: 60 epochs)
4. 建议在运行完整实验前先运行一个测试实验验证代码正确性

## 作者

研究团队
日期: 2024
