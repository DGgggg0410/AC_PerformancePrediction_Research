---
name: LTPP路面性能预测研究方案
overview: 将现有深度学习模型对比研究框架迁移至LTPP路面性能预测方向，调整为土木类研究定位，以IRI预测为核心目标。
todos:
  - id: define-research-framework
    content: 确定研究框架：IRI预测+工程因素分析，发表土木类期刊文章
    status: pending
  - id: ltpp-data-acquisition
    content: 获取LTPP数据：通过FHWA官方数据库或LTPP DataMart申请数据
    status: pending
  - id: redesign-data-pipeline
    content: 改造数据下载模块：将download_real_data.py改为LTPP数据加载器
    status: pending
    dependencies:
      - ltpp-data-acquisition
  - id: feature-engineering
    content: 特征工程设计：定义交通、气候、结构、材料、养护等工程特征
    status: pending
    dependencies:
      - ltpp-data-acquisition
  - id: modify-data-processor
    content: 调整数据处理器：适配LTPP字段，构建有物理意义的时序数据
    status: pending
    dependencies:
      - feature-engineering
  - id: update-model-config
    content: 更新模型配置：调整特征维度、序列长度等参数
    status: pending
    dependencies:
      - modify-data-processor
  - id: adjust-shap-analysis
    content: 改造SHAP分析：将模型决策解释调整为工程因素影响程度分析
    status: pending
    dependencies:
      - modify-data-processor
  - id: add-engineering-analysis
    content: 增加工程分析模块：量化各因素对IRI的贡献度
    status: pending
    dependencies:
      - adjust-shap-analysis
---

## 研究方向调整方案

### 目标

将研究从"沥青混合料疲劳寿命预测"调整为"基于LTPP数据的路面性能预测"，发表土木类期刊文章。

### 核心研究问题

**如何利用LTPP真实路面监测数据预测路面IRI（国际平整度指数），并量化分析影响IRI的工程因素？**

### 研究贡献

1. 利用LTPP大规模真实数据验证深度学习在路面性能预测中的适用性
2. 量化分析交通量、气候、路面结构对IRI的影响程度
3. 为养护决策提供数据支撑，指导道路养护时机优化

### 技术路线调整

| 要素 | 当前状态 | 调整后 |
| --- | --- | --- |
| 数据源 | GitHub小样本实验数据(361条) | LTPP大数据（真实路面多年监测） |
| 目标变量 | fatigue_life | IRI（国际平整度指数） |
| 预测问题类型 | 静态实验预测 | 真实时序预测 |
| 序列来源 | 滑动窗口构造的"假序列" | LTPP多年观测的真实时序数据 |
| 特征工程 | 5个实验参数 | 交通量+气候区+结构层厚度+材料特性+养护历史等 |
| SHAP解读 | 模型决策机制 | 工程因素对IRI的影响程度分析 |
| 研究定位 | 计算机类（模型对比） | 土木类（工程应用+因素分析） |


### LTPP数据特征类别

基于土木工程知识，预测IRI的特征应包括：

- **交通特征**：累计轴载（ESAL）、年平均日交通量（AADT）
- **气候特征**：气候区、年均温度、降水、冻融循环次数
- **结构特征**：路面总厚度、各层材料厚度、基层类型
- **材料特征**：沥青层厚度、沥青等级、混合料类型
- **养护历史**：是否进行过养护维修
- **初始状态**：初始IRI值、修建年份

### 代码框架改造

保留现有三模型对比结构（LSTM/Transformer/Mamba），但需调整：

- 数据获取模块：从GitHub改为LTPP数据接口
- 数据处理模块：适配LTPP字段结构
- 特征工程：工程特征选择而非纯数据驱动
- SHAP解读：面向工程师的因素重要性分析

## 技术实现方案

### 数据获取策略

LTPP数据获取有两个主要途径：

**途径1：FHWA官方数据库（推荐）**

- 访问：https://www.fhwa.dot.gov/research/tfhrc/programs/infrastructure/pavements/ltpp/
- 注册账号后申请数据下载
- 核心数据表：PMS_DATA（路面性能数据）、TRF_DATA（交通数据）、CLM_DATA（气候数据）

**途径2：LTPP数据网关**

- LTPP DataMart：https://ltpp-research.com/
- 提供在线查询和数据导出功能

### 数据处理架构

```
LTPP原始数据 → 数据清洗 → 特征工程 → 时序构建 → 模型训练
                    ↓
            工程特征选择（结合领域知识）
```

### 特征工程设计

基于土木工程原理，选择以下特征类别：

1. **结构性因素**：路面层厚度、基层类型
2. **材料性因素**：沥青含量、空隙率
3. **功能性因素**：交通量、轴载
4. **环境性因素**：气候区、温度变化
5. **时序性因素**：修建年龄、历史IRI变化率

### 代码改造文件清单

| 文件 | 改造内容 |
| --- | --- |
| `download_real_data.py` | 改为LTPP数据下载/加载模块 |
| `Asphalt_LSTM_SHAP_Sequential/_2_data_processor.py` | 适配LTPP字段，构建真正时序数据 |
| `Asphalt_LSTM_SHAP_Sequential/_1_config.py` | 调整特征数量和序列长度参数 |
| 三个项目的 `_8_xxx_shap_analyzer.py` | 调整SHAP解读为工程因素分析 |
| `Model_Comparison_Analysis/` | 增加工程因素对比分析 |


### 研究创新点

1. **数据创新**：首次将LTPP数据与深度学习结合用于IRI预测
2. **方法创新**：对比LSTM/Transformer/Mamba三种时序模型在路面预测上的表现
3. **应用创新**：通过SHAP量化揭示交通量、气候、结构对IRI的影响程度

# Agent Extensions

本方案不需要使用任何扩展。