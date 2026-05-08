---
name: 基于LTPP的路面IRI预测研究计划
overview: 利用LTPP真实路面监测数据(154,763条记录)，通过LSTM和Transformer两个深度学习模型预测IRI，并使用SHAP量化分析交通、气候、结构等工程因素对IRI的影响程度，发表土木类核心期刊文章。
todos:
  - id: create-folder-structure
    content: 创建新文件夹结构：AC_LSTM、AC_Transformer、AC_RandomForest、AC_LinearRegression、AC_Model_Comparison
    status: completed
  - id: ltpp-data-loader
    content: 实现LTPP数据加载器：从Access数据库提取IRI+气候+坐标+结构层数据
    status: completed
  - id: lstm-config
    content: LSTM配置：定义特征维度、SEQ_LEN、模型参数
    status: completed
    dependencies:
      - ltpp-data-loader
  - id: transformer-config
    content: Transformer配置：与LSTM统一的特征维度和SEQ_LEN
    status: completed
    dependencies:
      - ltpp-data-loader
  - id: sequence-builder
    content: 统一时序序列构建：滑动窗口逻辑
    status: completed
    dependencies:
      - lstm-config
      - transformer-config
  - id: rf-config
    content: RandomForest配置：特征列表、参数设置
    status: completed
    dependencies:
      - ltpp-data-loader
  - id: lr-config
    content: LinearRegression配置：特征列表
    status: completed
    dependencies:
      - ltpp-data-loader
  - id: lstm-model-training
    content: LSTM模型训练与超参数调优
    status: completed
    dependencies:
      - sequence-builder
  - id: transformer-model-training
    content: Transformer模型训练与超参数调优
    status: completed
    dependencies:
      - sequence-builder
  - id: rf-training
    content: RandomForest模型训练与评估
    status: completed
    dependencies:
      - rf-config
  - id: lr-training
    content: LinearRegression模型训练与评估
    status: completed
    dependencies:
      - lr-config
  - id: lstm-shap
    content: LSTM SHAP分析：量化工程因素贡献度
    status: completed
    dependencies:
      - lstm-model-training
  - id: transformer-shap
    content: Transformer SHAP分析：量化工程因素贡献度
    status: completed
    dependencies:
      - transformer-model-training
  - id: rf-shap
    content: RandomForest SHAP分析
    status: completed
    dependencies:
      - rf-training
  - id: lr-shap
    content: LinearRegression SHAP分析
    status: completed
    dependencies:
      - lr-training
  - id: metrics-comparison
    content: 四模型指标对比：R²、RMSE、MAE表格
    status: completed
    dependencies:
      - lstm-model-training
      - transformer-model-training
      - rf-training
      - lr-training
  - id: residual-comparison
    content: 残差分布对比分析
    status: completed
    dependencies:
      - metrics-comparison
  - id: scatter-plots
    content: 预测vs实测散点图对比
    status: completed
    dependencies:
      - metrics-comparison
  - id: shap-comparison
    content: SHAP特征重要性综合对比
    status: completed
    dependencies:
      - lstm-shap
      - transformer-shap
      - rf-shap
      - lr-shap
  - id: final-report
    content: 生成论文图表和综合报告
    status: completed
    dependencies:
      - metrics-comparison
      - residual-comparison
      - shap-comparison
  - id: 15c71db7
    content: 将其余跟此项目无关的文件都删掉
    status: completed
---

## 研究目标

利用LTPP真实路面监测数据（154,763条记录，1,026个路段，1989-2024年）预测路面IRI（国际平整度指数），并通过SHAP量化分析影响IRI的工程因素，发表土木类核心期刊文章。

## 核心研究问题

如何利用LTPP真实数据预测路面IRI，并量化交通量、气候、路面结构等工程因素的影响程度？

## 研究贡献

1. 利用LTPP大规模真实数据验证深度学习在路面性能预测中的适用性
2. 量化分析交通量、气候、路面结构对IRI的影响程度
3. 通过SHAP为养护决策提供数据支撑

## 研究框架

| 要素 | 内容 |
| --- | --- |
| 目标变量 | MRI（Mean Roughness Index，平均IRI值） |
| 预测模型 | LSTM + Transformer + RandomForest + LinearRegression |
| 基准对比 | RF和LR作为轻量级基准，在实验表格中对比 |
| SHAP分析 | 量化工程因素对IRI的贡献程度 |
| 期刊定位 | 核心/三区四区，目标冲二区 |


## 目标变量

- **MRI**：Mean Roughness Index，国际平整度指数平均值

## 输入特征（已验证存在于数据库中）

| 特征类别 | 具体特征 | 数据来源 | 说明 |
| --- | --- | --- | --- |
| **时间特征** | PAVEMENT_AGE | VISIT_DATE - START_DATE | 路面龄期 |
| **IRI历史** | IRI_LAG_1, IRI_LAG_2 | ANALYSIS_IRI | 过去1/2年IRI值 |
| **路面结构** | PAVEMENT_FAMILY | ANALYSIS_IRI | 9种结构类型(ACUB、ACATB等) |
| **交通特征** | FUNC_CLASS | SHRP_INFO | 道路功能分类(1-19) |
| **地理特征** | LATITUDE, LONGITUDE, ELEVATION | SECTION_COORDINATES | 地理位置 |
| **气候特征** | DEGREE_DAYS_OVER_10C_YR | MERRA气候表 | 年度度日数 |
|  | COLDEST_AIR_TEMP | MERRA气候表 | 最冷气温 |
|  | HIGH_TEMP_7DAYS | MERRA气候表 | 最高7日气温 |
|  | MIN_SURFACE_50_TEMP | MERRA气候表 | 最低地表温度 |
| **结构层特征** | TOTAL_THICKNESS | TST_L05B汇总 | 总路面厚度 |
|  | AC_THICKNESS | TST_L05B汇总 | 沥青层厚度 |
|  | BASE_THICKNESS | TST_L05B汇总 | 基层厚度 |
|  | NUM_LAYERS | TST_L05B汇总 | 结构层数量 |


## 数据关联方式

- 主要通过 **SHRP_ID** 关联各表
- 气候数据通过 **SHRP_ID + STATE_CODE** 关联
- 结构层数据通过 **SHRP_ID + CONSTRUCTION_NO** 关联

## 技术栈

- **编程语言**：Python 3.12- **深度学习框架**：PyTorch
- **数据处理**：pandas, numpy, pyodbc
- **机器学习**：scikit-learn (RF, LR)
- **可解释性分析**：SHAP- **可视化**：matplotlib, seaborn

## 目录结构

```
AC_LSTM/                    # LSTM深度学习模型
  ├── _0_ltpp_data_loader.py      # LTPP数据加载器（核心模块）
  ├── _1_config.py                 # 配置参数
  ├── _2_sequence_builder.py       # 时序序列构建
  ├── _3_lstm_model.py             # LSTM模型定义
  ├── _4_hyperparam_tuning.py     # 超参数调优
  ├── _5_predictor.py              # 预测模块
  ├── _6_trainer.py                # 训练器
  ├── _7_evaluator.py              # 评估器
  └── _8_shap_analyzer.py          # SHAP分析

AC_Transformer/              # Transformer深度学习模型
  ├── _0_ltpp_data_loader.py       # 复用LSTM数据加载器
  ├── _1_config.py                 # 配置（与LSTM统一SEQ_LEN和特征维度）
  ├── _2_sequence_builder.py       # 时序序列构建（与LSTM一致）
  ├── _3_transformer_model.py      # Transformer模型（简化版）
  ├── _4_hyperparam_tuning.py      # 超参数调优
  ├── _5_predictor.py              # 预测模块
  ├── _6_trainer.py                # 训练器
  ├── _7_evaluator.py              # 评估器
  └── _8_shap_analyzer.py          # SHAP分析

AC_RandomForest/             # 随机森林基准模型
  ├── _0_data_loader.py            # 数据加载（复用LSTM预处理数据）
  ├── _1_config.py                 # RF配置
  ├── _2_rf_model.py               # RF模型
  ├── _3_train_and_evaluate.py     # 训练与评估
  └── _4_shap_analyzer.py          # SHAP分析

AC_LinearRegression/         # 线性回归基准模型
  ├── _0_data_loader.py            # 数据加载
  ├── _1_config.py                 # LR配置
  ├── _2_lr_model.py               # LR模型
  ├── _3_train_and_evaluate.py     # 训练与评估
  └── _4_shap_analyzer.py          # SHAP分析

AC_Model_Comparison/          # 模型对比分析
  ├── _1_metrics_comparison.py     # 四模型指标对比表格（R², RMSE, MAE）
  ├── _2_residual_analysis.py      # 残差分布对比
  ├── _3_scatter_plots.py          # 预测vs实测散点图
  ├── _4_shap_comparison.py        # SHAP特征重要性对比
  └── _5_final_report.py           # 综合报告生成
```

## 数据处理流程

```
LTPP Access数据库
├── Bucket_141347.mdb (主数据库)
│   ├── ANALYSIS_IRI (IRI监测数据)
│   ├── SHRP_INFO (路段信息)
│   ├── SECTION_COORDINATES (地理坐标)
│   ├── TST_L05B (结构层信息)
│   └── EXPERIMENT_SECTION (试验段信息)
└── Bucket_141348_1.accdb (气候数据)
    └── VW_MERRA_BIND_CLIMATE_DATA

          ↓ 数据关联 (SHRP_ID)

ltpp_processed_data.csv
          ↓ 滑动窗口

训练序列 → 模型训练 → SHAP分析 → 对比报告
```

## 评估指标

- R²（决定系数）
- RMSE（均方根误差）
- MAE（平均绝对误差）
- 预测vs实测散点图
- 残差分布图
- SHAP特征重要性图

## 实现要点

1. **数据加载器**：统一从Access数据库提取数据，避免各模型使用不一致的数据
2. **特征维度统一**：LSTM和Transformer使用相同的特征和SEQ_LEN
3. **简化Transformer**：去掉复杂的Feature Coupling层，使用标准TransformerEncoder
4. **RF/LR基准**：直接使用特征矩阵，不需要时序处理��理