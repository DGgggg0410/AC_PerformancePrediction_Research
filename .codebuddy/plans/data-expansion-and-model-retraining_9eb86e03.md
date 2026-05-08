---
name: data-expansion-and-model-retraining
overview: 扩展沥青疲劳数据集并重新生成序列文件，用于模型训练
todos:
  - id: find-datasets
    content: 搜索并确定可用的公开数据集URL（MEPDG、其他GitHub源）
    status: completed
  - id: expand-download-script
    content: 扩展 download_real_data.py 支持下载 MEPDG_Dataset.csv
    status: completed
    dependencies:
      - find-datasets
  - id: implement-merge-logic
    content: 实现数据合并逻辑，统一特征空间
    status: completed
    dependencies:
      - expand-download-script
  - id: regenerate-npy
    content: 重新生成 .npy 序列文件（SEQ_LEN=5）
    status: completed
    dependencies:
      - implement-merge-logic
  - id: update-config
    content: 更新三个项目的 _1_config.py 中的 SAMPLE_NUM
    status: completed
    dependencies:
      - regenerate-npy
  - id: verify-data-stats
    content: 验证合并后数据的统计信息
    status: completed
    dependencies:
      - regenerate-npy
---

## 用户需求

1. 下载并分析 AsphaltFatigueANN 数据集（已有128序列）
2. 寻找并下载更多公开数据集扩充数据量
3. 合并多个数据源
4. 重新生成适合 LSTM/Transformer/Mamba 模型的 .npy 序列文件
5. 确保数据能支撑SCI论文的数据量要求

## 数据源分析

- **当前数据**：来自 VaclavNezerka/AsphaltFatigueANN 的 asphalt_20_deg.xlsx，约189条原始样本，生成128序列
- **MEPDG_Dataset.csv**：GitHub开源数据，200条记录，包含15个路面性能特征（厚度、空隙率、沥青含量、IRI等）
- **数据缺口**：128序列对于深度学习仍偏少，需要扩充至500+序列

## 关键问题

- 当前R²仅0.31，RMSE高达百万级别
- 样本量不足（128序列）
- 需要数据合并策略

## 技术方案

### 1. 数据源整合

- **源1**：继续使用 AsphaltFatigueANN 的 asphalt_20_deg.xlsx（189样本）
- **源2**：下载 MEPDG_Dataset.csv（200样本）进行数据扩充
- **合并策略**：统一特征空间后合并，总计约400样本

### 2. 特征对齐

两个数据集的特征需要对齐：

```
公共特征：binder_content, air_voids, strain_level
可选特征：temperature, frequency（MEPDG中没有，需填充默认值或删除）
```

### 3. 数据增强

- 对数变换目标变量：`log10(fatigue_life)` 使分布更平稳
- 滑动窗口SEQ_LEN=5，STRIDE=1保持不变
- 预期生成约350-400个序列

### 4. 实现步骤

1. 扩展 download_real_data.py 支持多数据源下载
2. 实现数据合并逻辑
3. 重新生成 .npy 文件
4. 验证数据统计信息

## 目录结构

```
e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/
├── data/
│   ├── real_asphalt_fatigue_data.csv      # [MODIFY] AsphaltFatigueANN原始数据
│   ├── mepdg_data.csv                      # [NEW] MEPDG路面性能数据
│   └── combined_data.csv                   # [NEW] 合并后的数据
├── Asphalt_LSTM_SHAP_Sequential/data/
│   ├── X_train.npy                         # [MODIFY] 更新后的序列数据
│   └── y_train.npy                         # [MODIFY] 更新后的目标数据
└── download_real_data.py                   # [MODIFY] 扩展支持多数据源
```