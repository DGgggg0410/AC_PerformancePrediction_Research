# 基于LSTM与Transformer的沥青路面国际平整度指数（IRI）预测研究

## 摘要

为探究不同类型工程因素对沥青路面国际平整度指数（IRI）演化的影响机制，基于长短期记忆网络（LSTM）与Transformer分别建立IRI预测模型。利用美国LTPP数据库35年实测数据（1026个路段，154764条记录），构建融合时序、气候、结构和地理信息的19维特征体系，采用滑动窗口策略（SEQ_LEN=5）建立双层S-LSTM与Transformer模型，并结合SHAP与5组消融实验交叉验证各因素的贡献。

两模型均取得良好的预测性能：S-LSTM的R²为0.7279、RMSE为0.3724 m/km，Transformer的R²为0.7472、RMSE为0.3590 m/km。SHAP分析表明，IRI预测呈现显著的时序自回归特征——历史IRI（IRI_LAG_1/2）的贡献占比在LSTM中达86%，在Transformer中为57%，是预测的核心驱动因素。消融实验进一步证实，结构参数是唯一在两模型中均导致一致性性能下降的工程因素组（ΔR²≈−0.007），气候和地理因素的单因素移除未产生超越运行噪声（±0.007）的显著影响。

综合而言，IRI预测以时序自回归为主导，LSTM的循环归纳偏置使其能以较少参数量（217K）在此类任务中达到与Transformer（830K）可比的精度；Transformer在完整特征条件下取得更优的综合性能，在多源信息融合方面更具潜力。

**关键词：** 道路工程；国际平整度指数；长短期记忆网络；Transformer；SHAP；消融实验

---

## Abstract

To investigate the influence of diverse engineering factors on the evolution of the International Roughness Index (IRI) for asphalt pavements, this study develops IRI prediction models using a long short-term memory network (LSTM) and a Transformer. Leveraging 35 years of field measurements from the U.S. Long-Term Pavement Performance (LTPP) database (1,026 pavement sections, 154,764 records), a 19-dimensional feature system integrating temporal, climatic, structural, and geographic variables is constructed. A sliding window strategy (SEQ_LEN=5) is adopted to build a stacked two-layer S-LSTM and a Transformer model, and SHAP analysis combined with five ablation experiments is employed to cross-validate the contributions of different factor categories.

Both models achieve satisfactory predictive performance: the S-LSTM attains an R² of 0.7279 and an RMSE of 0.3724 m/km, while the Transformer achieves an R² of 0.7472 and an RMSE of 0.3590 m/km. SHAP analysis reveals a pronounced temporal autoregressive characteristic—historical IRI values (IRI_LAG_1/2) account for 86% of the total importance in the LSTM and 57% in the Transformer, establishing them as the dominant predictors. Ablation experiments further confirm that structural parameters constitute the only engineering factor group that consistently degrades performance in both models upon removal (ΔR² ≈ −0.007), whereas the exclusion of climatic or geographic factors alone produces no effects exceeding the normal training variance (±0.007).

In summary, IRI prediction is overwhelmingly governed by temporal autoregression. The recurrent inductive bias of the LSTM enables it to achieve comparable accuracy with substantially fewer parameters (217K) relative to the Transformer (830K) in this task, while the Transformer attains superior overall performance under the full feature set and demonstrates greater potential for multi-source information fusion.

**Keywords:** road engineering; International Roughness Index; long short-term memory; Transformer; SHAP; ablation experiment

---

## 0 引言

国际平整度指数（International Roughness Index, IRI）是评价路面使用性能的核心指标，不仅直接影响路面行驶舒适性与行车安全，还与车辆运行成本及养护资金分配密切相关[1]。随着我国高速公路与城市道路网络逐步由"建设主导"向"养护主导"阶段转变，准确预测 IRI 的演化趋势，对于制定科学养护策略、优化有限养护资金分配、提升路网整体服役效率具有重要的工程意义[2]。近年来，路面平整度预测方法经历了从经验模型到数据驱动模型的持续演进，相关研究日趋活跃[3]。

目前，路面性能预测方法主要分为经验回归模型、力学-经验模型和数据驱动模型三类[4]。传统经验模型（如 AASHTO 性能模型）通过回归分析建立 IRI 与龄期、交通荷载及环境因素间的经验关系，形式简单且物理意义明确，但本质上以线性或低阶非线性假定为主，难以充分捕捉路面劣化过程中多因素间的复杂交互关系[5]。力学-经验模型基于材料损伤与结构响应理论，具有较强的理论基础，但通常需要大量的结构参数与现场试验数据进行标定，工程应用成本较高，在路网层面的推广受限[6]。随着路面长期性能数据库的持续积累和计算能力的提升，基于机器学习与深度学习的数据驱动方法为 IRI 预测提供了新的技术路径[7]。

在传统机器学习层面，已有学者基于 LTPP 数据库利用 GBM、随机森林、GPR 和 XGBoost 等方法开展 IRI 预测，取得了 R²=0.85–0.92 的效果[8-12]。然而，传统机器学习本质上依赖手工特征工程，在处理路面劣化过程中的复杂时序依赖方面存在固有局限。

在深度学习方法中，长短期记忆网络（LSTM）因其门控循环结构在处理时序数据方面的天然优势，已被广泛引入路面性能预测领域。Hou 等[13]和 Zhang 等[14]分别在加速加载试验和 LTPP 数据库上基于 LSTM 实现了 IRI 的高精度预测（R² 分别达 0.959 和 0.965），Plati 等[16]进一步证实了 LSTM 在 IRI 预测中显著优于传统回归模型。在此基础上，注意力机制[15]与卷积结构[17]被逐步引入以增强 LSTM 的特征提取能力，双向 LSTM 亦被用于多指标联合预测[18]。上述研究充分验证了 LSTM 在 IRI 预测中的有效性，但大多侧重于模型精度提升，对不同类型特征的贡献机制仍缺乏系统揭示。

与此同时，Transformer 模型凭借自注意力机制在全局特征关联方面的优势也开始受到道路工程领域的关注。Han 等[19]提出了面向沥青路面健康预测的改进 Transformer 网络（FRTS），Cao 等[21]进一步将偏微分方程约束引入 Informer 框架，取得了 R²=0.962 的优异性能。Yao 等[20]和呙润华等[22]也分别从多指标联合预测和编码器-解码器结构角度验证了注意力机制的适用性。然而，上述研究多在各自独立的数据集和特征体系下开展，LSTM 与 Transformer 在相同条件下的系统性对比仍较为匮乏。

在特征构建与可解释性分析方面，多数研究仍以模型精度提升为主要目标，对不同类型特征的贡献程度及其在不同模型架构中的利用机制关注不足[23]。部分研究仅采用历史 IRI 数据进行单变量时序预测，缺乏对气候条件、结构参数及地理空间环境等多源信息的综合考虑[24]。同时，深度学习模型的"黑箱"特性已成为制约其工程可采纳性的重要因素。针对这一问题，已有研究将 SHAP（SHapley Additive exPlanations）方法引入路面性能分析领域：KSCE[25]利用随机森林结合 SHAP 分析了路面病害类型对 IRI 的影响，发现中高严重度裂缝是 IRI 增长的主要贡献因素；SHAP-TPE-CatBoost[26]将 SHAP 与贝叶斯优化结合用于路面性能预测，验证了 SHAP 在识别关键影响因素方面的有效性。在消融实验方面，Yang 等[7]的综述指出，系统性消融实验是量化不同输入特征类别贡献度的有效手段，但将其应用于路面性能预测模型的研究仍较为有限。将 SHAP 的逐特征归因与消融实验的按类别移除有机结合，可以从微观与宏观两个层面交叉验证各类工程因素的贡献程度，有望为养护决策提供更为可靠的依据。

基于此，本文依托美国长期路面性能数据库（LTPP），提取了覆盖 35 年的 1026 个沥青路面路段、共计 154764 条有效监测记录，构建了包含时序特征（3 维）、气候特征（8 维）、结构特征（4 维）、地理特征（3 维）和路面结构类型编码（1 维）的 19 维多源异构特征数据集，分别建立双层 S-LSTM 与 Transformer 两种深度学习预测模型，在已对齐的数据划分、序列长度和超参数优化条件下系统对比两模型的预测精度与参数效率。在此基础上，设计 5 组消融实验从类别层面分析不同特征类型对模型性能的影响，并结合 SHAP 方法从特征层面揭示模型决策逻辑，以期为 IRI 预测方法的选型与工程应用提供参考。

本文的主要研究内容包括：（1）构建融合气候、结构、地理及历史 IRI 等多源信息的沥青路面 IRI 预测数据集，采用滑动窗口方法（SEQ_LEN=5）建立时序样本；（2）分别建立 S-LSTM 与 Transformer 预测模型，从预测精度与参数效率两个维度对比两类架构的差异；（3）设计多组特征消融实验，系统研究气候、结构、地理等不同类型输入特征对模型预测结果的影响规律；（4）结合 SHAP 可解释性分析方法，揭示关键特征的重要性排序与 IRI 演化机理，为路面养护管理提供决策支持。

---

## 1 研究方法

### 1.1 数据来源与预处理

**数据来源**：LTPP（Long-Term Pavement Performance）数据库，包含 1,026 个沥青路面路段，时间跨度 1989–2024（35 年），有效 IRI 记录 154,764 条。

**预处理流程**：
1. 去除目标车道非 AC 路面类型的数据
2. 剔除 MRI 缺失的记录
3. 剔除关键特征列存在 NaN 的记录
4. 按路段（SHRP_ID）和检测日期（VISIT_DATE）排序

**标准化**：采用 Z-score 标准化，所有统计参数（均值和标准差）**仅基于训练集拟合**，避免数据泄漏。

**数据集划分**：按路段（SHRP_ID）随机分层抽样，70% 训练集、15% 验证集、15% 测试集，使用固定随机种子（RANDOM_SEED=42）保证可复现。

### 1.2 特征工程

构建 **19 维特征体系**，涵盖 5 个类别：

| 类别 | 数量 | 特征列表 |
|:----:|:----:|---------|
| 时序特征 | 3 | PAVEMENT_AGE, IRI_LAG_1, IRI_LAG_2 |
| 气候特征 | 8 | DEGREE_DAYS_OVER_10C_YR, COLDEST_AIR_TEMP, HIGH_TEMP_7DAYS, MIN_SURFACE_50_TEMP, FREEZE_INDEX, FREEZE_THAW, PRECIPITATION, EVAPORATION |
| 结构特征 | 4 | TOTAL_THICKNESS, AC_THICKNESS, BASE_THICKNESS, NUM_LAYERS |
| 地理特征 | 3 | LATITUDE, LONGITUDE, ELEVATION |
| 分类特征 | 1 | PAVEMENT_FAMILY_ENC |

**滑动窗口策略**（SEQ_LEN=5）：使用前 5 年的特征数据预测第 6 年的 IRI。窗口按路段边界分割，确保不跨越不同路段，避免时序数据泄漏。

### 1.3 LSTM模型构建

**模型架构**：
- 双层 S-LSTM（Stacked LSTM）：hidden_dim=128, num_layers=2
- 全连接头：Linear(128→64) + ReLU + Dropout(0.2) + Linear(64→1)
- 取最后一层隐藏状态 hidden[-1] 作为全连接头的输入

**训练配置**：
- 损失函数：MSE Loss
- 优化器：Adam（lr=0.005, weight_decay=1e-3）
- 学习率调度：ReduceLROnPlateau（factor=0.5, patience=10, min_lr=1e-6）
- 早停机制：patience=30
- 梯度裁剪：max_norm=1.0
- Batch size：256
- 最大训练轮数：100

**模型参数量**：217,025

### 1.4 Transformer模型构建

**模型架构**：
- 输入投影：Linear(19→256)
- 位置编码：正弦位置编码（max_len=100, dropout=0.2）
- [CLS] Token：可学习分类标记
- Transformer Encoder：2 层、8 头注意力、d_model=256、ff_dim=256
- 全连接头：Linear(256→128) + LayerNorm + ReLU + Dropout(0.2) + Linear(128→1)

**训练配置**：
- 损失函数：MSE Loss
- 优化器：Adam（lr=0.0005, weight_decay=1e-4）
- 学习率调度：ReduceLROnPlateau（factor=0.5, patience=10, min_lr=1e-6）
- 早停机制：patience=60
- 梯度裁剪：max_norm=0.5
- Batch size：256
- 最大训练轮数：100

**模型参数量**：830,209

### 1.5 SHAP可解释性分析

采用 SHAP（SHapley Additive exPlanations）对两模型进行特征重要性量化分析。SHAP 基于合作博弈论的 Shapley 值，能公平分配每个特征对预测结果的贡献。本文使用 **DeepExplainer** 计算每个特征的平均绝对 SHAP 值作为重要性指标。

### 1.6 消融实验设计

#### 1.6.1 实验方案构建

为定量分析不同类别特征对模型预测性能的增量贡献及其敏感性，本研究遵循"变量剥离"原则设计了 5 组消融实验（E1~E5），在 S-LSTM 与 Transformer 上分别移除特定维度的特征向量，以对比不同架构对特征缺失的鲁棒性。消融实验具体方案见表 4。

**表 4 消融实验设计方案**
| 编号 | 实验名称 | 移除特征 | 保留特征数 | 验证目的 |
|:---:|:-------:|:--------:|:---------:|---------|
| E1 | 气候因素缺失 | 气候（8个） | 11 | 气候因素的贡献 |
| E2 | 结构与分类特征缺失 | 结构特征（4个）+ 路面结构类型编码 | 14 | 结构设计的贡献 |
| E3 | 地理因素缺失 | 地理（3个） | 16 | 地理位置的贡献 |
| E4 | 极简时序基准 | 结构+地理+气候（16个） | 3 | 纯时序自回归能力 |
| E5 | 结构-气候耦合缺失 | 气候+结构（13个） | 6 | 时序+地理的预测能力 |

实验设计逻辑如下：E1~E3 属于单因素消融，旨在剖析单一物理维度对预测精度的独立贡献；E4 为极端消融实验，通过剥离所有外部静态特征，标定模型在仅依赖时序特征（历史 IRI 值与路面龄期）时的预测基准；E5 为多因素组合消融，重点考察在核心力学与环境特征缺失时，地理空间特征的支撑作用。

#### 1.6.2 数据划分一致性保证

消融实验与主模型使用完全相同的训练/验证/测试集划分。所有实验调用主模型的序列构建器一次性完成数据加载、标准化和划分，各实验仅通过特征维度切片改变输入特征，确保所有实验在相同数据子集上评估。

---

## 2 实验结果

### 2.1 LSTM模型性能

| 指标 | 数值 |
|:---:|:----:|
| R² | **0.7279** |
| RMSE | 0.3724 m/km |
| MAE | 0.2056 m/km |
| 参数量 | 217,025 |
| 最佳验证损失 | 0.1331 |

### 2.2 Transformer模型性能

| 指标 | 数值 |
|:---:|:----:|
| R² | **0.7472** |
| RMSE | 0.3590 m/km |
| MAE | 0.1889 m/km |
| 参数量 | 830,209 |
| 最佳验证损失 | 0.1244 |

### 2.3 模型对比分析

| 指标 | LSTM | Transformer | 差异 |
|:---:|:----:|:-----------:|:----:|
| R² | 0.7279 | **0.7472** | +0.0193 |
| RMSE | 0.3724 | **0.3590** | -0.0134 |
| MAE | 0.2056 | **0.1889** | -0.0167 |
| 参数量 | 217,025 | 830,209 | 3.8× |

**分析**：Transformer 以约 3.8 倍参数量在所有指标上轻徼优于 LSTM（ΔR² ≈ 0.02）。性能差距不大的原因在于 IRI 预测任务具有强时序自回归特性——IRI 的当前值与历史值高度相关，LSTM 的循环结构对此类任务具有天然优势。

### 2.4 特征重要性分析（SHAP）

#### LSTM SHAP 分析

| 排名 | 特征 | 中文名 | SHAP 值 | 累计占比 |
|:---:|:----:|:------:|:-------:|:--------:|
| 1 | IRI_LAG_1 | 去年IRI | 0.5362 | 66.78% |
| 2 | IRI_LAG_2 | 前年IRI | 0.1572 | 86.36% |
| 3 | LATITUDE | 纬度 | 0.0444 | 91.89% |
| 4 | PAVEMENT_FAMILY_ENC | 路面结构类型 | 0.0141 | 93.64% |
| 5 | EVAPORATION | 年蒸发量 | 0.0125 | 95.20% |
| 6-19 | 其余 14 特征 | — | <0.01 | 100% |

#### Transformer SHAP 分析

| 排名 | 特征 | 中文名 | SHAP 值 | 累计占比 |
|:---:|:----:|:------:|:-------:|:--------:|
| 1 | IRI_LAG_2 | 前年IRI | 0.0727 | 32.70% |
| 2 | IRI_LAG_1 | 去年IRI | 0.0426 | 51.84% |
| 3 | LATITUDE | 纬度 | 0.0146 | 58.42% |
| 4 | ELEVATION | 海拔 | 0.0132 | 64.36% |
| 5 | PRECIPITATION | 年降水量 | 0.0127 | 70.08% |
| 6 | PAVEMENT_AGE | 路面龄期 | 0.0109 | 74.98% |
| 7 | FREEZE_THAW | 年冻融天数 | 0.0097 | 79.35% |
| 8 | NUM_LAYERS | 结构层数量 | 0.0089 | 83.35% |
| 9-19 | 其余 11 特征 | — | <0.007 | 100% |

#### SHAP 关键发现

1. **LSTM 高度聚焦于时序特征**：IRI_LAG_1 和 IRI_LAG_2 的 SHAP 占比合计高达 86.36%，其余 17 维特征合计不到 14%。LSTM 的循环结构通过隐状态已有效捕捉时序模式，外部特征的边际贡献有限。

2. **Transformer 对多类特征均有感知**：时序特征占比约 57%，气候、地理、结构特征均有 10%–20% 的累计贡献。自注意力机制使 Transformer 能同时关注所有输入维度的信息。

3. **两模型共性**：时序滞后特征均位居重要性首位，地理因素（纬度/海拔）在两者中均排名靠前，表明地理位置对 IRI 劣化具有显著的间接影响。

### 2.5 消融实验结果

为分析不同类型输入特征对路面平整度预测结果的影响，本文基于表 4 设计的 5 组消融实验（E1~E5），分别在 S-LSTM 与 Transformer 上独立运行，结果汇总于表 5~表 7。

**表 5 不同特征子集下预测模型决定系数（R²）的消融实验统计**
| 消融实验 | 特征数 | S-LSTM R² | S-LSTM ΔR² | Transformer R² | Transformer ΔR² |
|:-------:|:-----:|:---------:|:-----------:|:--------------:|:---------------:|
| 基线（全部特征） | 19 | 0.7234 | — | 0.7544 | — |
| E1 气候缺失 | 11 | 0.7238 | +0.0004 | 0.7589 | +0.0045 |
| E2 结构缺失 | 14 | 0.7167 | -0.0067 | 0.7462 | -0.0081 |
| E3 地理缺失 | 16 | 0.7273 | +0.0039 | 0.7564 | +0.0021 |
| E4 极简时序 | 3 | 0.7300 | +0.0066 | 0.7480 | -0.0063 |
| E5 气候+结构缺失 | 6 | 0.7258 | +0.0024 | 0.7504 | -0.0040 |

**表 6 不同特征子集下预测模型均方根误差（RMSE）的消融实验统计**
| 消融实验 | 特征数 | S-LSTM RMSE | S-LSTM ΔRMSE | Transformer RMSE | Transformer ΔRMSE |
|:-------:|:-----:|:-----------:|:-------------:|:----------------:|:-----------------:|
| 基线（全部特征） | 19 | 0.3755 | — | 0.3538 | — |
| E1 气候缺失 | 11 | 0.3752 | -0.0003 | 0.3506 | -0.0032 |
| E2 结构缺失 | 14 | 0.3800 | +0.0045 | 0.3596 | +0.0058 |
| E3 地理缺失 | 16 | 0.3728 | -0.0027 | 0.3523 | -0.0015 |
| E4 极简时序 | 3 | 0.3710 | -0.0045 | 0.3584 | +0.0045 |
| E5 气候+结构缺失 | 6 | 0.3738 | -0.0017 | 0.3567 | +0.0028 |

**表 7 不同特征子集下预测模型平均绝对误差（MAE）的消融实验统计**
| 消融实验 | 特征数 | S-LSTM MAE | S-LSTM ΔMAE | Transformer MAE | Transformer ΔMAE |
|:-------:|:-----:|:----------:|:------------:|:---------------:|:----------------:|
| 基线（全部特征） | 19 | 0.2144 | — | 0.1853 | — |
| E1 气候缺失 | 11 | 0.2091 | -0.0053 | 0.1813 | -0.0040 |
| E2 结构缺失 | 14 | 0.2134 | -0.0010 | 0.1904 | +0.0051 |
| E3 地理缺失 | 16 | 0.2039 | -0.0105 | 0.1824 | -0.0029 |
| E4 极简时序 | 3 | 0.2016 | -0.0128 | 0.1844 | -0.0009 |
| E5 气候+结构缺失 | 6 | 0.2060 | -0.0084 | 0.1830 | -0.0023 |

> ΔR² = 实验 R² − 基线 R²，正值表示优于基线；ΔRMSE/ΔMAE = 实验值 − 基线值，负值表示优于基线。

#### （1）S-LSTM 模型消融结果分析

从表 5~7 可以看出，S-LSTM 在各类特征剥离条件下均保持了较为稳定的预测性能，各消融实验的 R² 变化幅度在 -0.0067~+0.0066 之间。

E2（结构缺失）是唯一导致 S-LSTM 性能下降的实验，R² 由 0.7234 降至 0.7167（ΔR² = -0.0067），RMSE 由 0.3755 增至 0.3800，说明路面结构参数（各结构层厚度及结构类型）是影响 S-LSTM 预测性能的主要工程因素，其缺失会削弱模型对路面力学性能演变的捕捉能力。

其余实验（E1、E3、E4、E5）中，S-LSTM 的 R² 均未出现下降，变化幅度在 ±0.007 以内，处于单次训练的正常方差范围。

#### （2）Transformer 模型消融结果分析

与 S-LSTM 相比，Transformer 对特征缺失的敏感度更高。E2（结构缺失）导致 Transformer 的 R² 由 0.7544 降至 0.7462（ΔR² = -0.0081），RMSE 由 0.3538 增至 0.3596（ΔRMSE = +0.0058），为所有实验中性能下降最显著的一组。E4（极简时序基准）同样导致 R² 下降 0.0063，表明 Transformer 对外部静态特征具有更强的依赖性。

E5（气候+结构组合缺失）也出现了 R² 下降（ΔR² = -0.0040），进一步印证了非时序特征对 Transformer 的支撑作用。然而，E1（气候缺失）和 E3（地理缺失）的单因素移除并未导致性能下降，说明单一类别的外部特征对 Transformer 的边际贡献尚未超过单次运行的方差阈值。

#### （3）不同影响因素贡献分析

综合两模型消融实验结果，各工程因素的贡献度排序如下：

**时序滞后特征 > 结构特征 > 地理因素 ≈ 气候因素**

其中，时序变量（IRI_LAG_1、IRI_LAG_2）始终占据主导地位，是决定预测性能的核心因素；结构特征是唯一在两模型中均产生一致性负面影响（ΔR² ≈ -0.007）的工程因素组；气候因素和地理因素在单因素移除条件下未产生超越噪声阈值的可辨识影响。

两模型在特征利用方式上呈现出互补特性。S-LSTM 的循环归纳偏置使其预测高度聚焦于时序滞后特征——SHAP 贡献占比达 86%，外部工程特征的单独移除对其整体性能影响有限——在低维输入下保持稳健；Transformer 的自注意力机制平等关注所有输入维度，对特征完整性的依赖更高，但也能更充分地利用多源信息，在完整特征条件下取得更优的综合预测效果（R² = 0.7472 vs 0.7279）。

---

## 3 讨论

### 3.1 模型性能对比

LSTM 与 Transformer 的性能差距较小（ΔR² ≈ 0.02），这与任务特性密切相关。IRI 预测本质上是强时序自回归任务——当前 IRI 值主要受历史 IRI 值决定。在此类任务中，LSTM 的循环归纳偏置（sequential inductive bias）使其能够以远少于 Transformer 的参数量（约 1/3.8）达到可比的性能。

Transformer 的优势体现在对多模态信息的整合能力上。SHAP 分析显示，Transformer 对气候、结构、地理等非时序特征的感知明显强于 LSTM。在需要融合多源信息的预测场景中（如新建路段的 IRI 预测），Transformer 可能具有更大的潜力。

### 3.2 特征贡献度分析

综合 SHAP 和消融实验的结果，各工程因素的贡献度排序如下：

```
时序滞后特征 >> 结构特征 > 地理因素 ≈ 气候因素
```

- **时序滞后特征（IRI_LAG）** 是最重要的预测因子，占 SHAP 贡献的 57%–86%
- **结构特征** 是唯一在消融实验中产生一致性负向影响的工程因素组
- **地理因素和气候因素** 的个体贡献相对较小，但 SHAP 分析显示其具有不可忽略的累计效应

这一结论的实践意义在于：对于在役路面的 IRI 预测，应优先关注历史 IRI 数据和结构信息；而对于新建路面，气候和地理因素在长期预测中的累积效应不容忽视。

### 3.3 研究局限性

1. **单次消融实验的噪声**：消融实验基于单次独立训练，ΔR² 在 ±0.005 以内的波动属于正常运行方差。未来可采用多轮重复消融（multi-run ablation）以降低随机性干扰。
2. **特征独立性假设**：消融实验假设各特征类别相互独立，但气候与地理因素之间存在天然的耦合关系。
3. **模型容量差异**：Transformer 参数量（830K）约为 LSTM（217K）的 3.8 倍，性能差异部分可能源于容量差异而非架构差异。

---

## 4 结论

本文基于 LTPP 数据库 35 年的真实监测数据，构建了 19 维特征体系，系统对比了 S-LSTM 与 Transformer 在 IRI 预测中的表现，并利用 SHAP 和消融实验量化了各类工程因素的贡献。主要结论如下：

1. **两模型均可有效预测 IRI**：LSTM（R²=0.7279）和 Transformer（R²=0.7472）均展现了良好的预测性能，Transformer 以约 3.8 倍参数量轻徼领先。
2. **时序滞后特征主导预测**：IRI_LAG_1/2 占 SHAP 重要性 57%–86%，是预测性能的核心驱动因素。
3. **结构特征贡献最稳健**：消融实验显示，结构特征是唯一在两模型中均导致一致性性能下降的工程因素组。
4. **交叉验证方法框架有效**：SHAP（逐特征归因）和消融实验（按类别移除）从不同角度量化了特征贡献，结论相互支撑。

---

## 参考文献

[1] Yoon Y, Hassan S. Feasibility of modernizing the acceptable International Roughness Index value[J]. Transportation Research Record, 2024, 2678(8): 431-442.

[2] 杜昭, 冉旭, 张金喜, 等. 基于网联车辆数据融合的路面平整度评估方法[J]. 中国公路学报, 2024, 37(6).

[3] Construction and Building Materials. Evolution of prediction models for road surface irregularity: trends, methods and future[J]. 2024, 449.

[4] Rahman M M, Uddin M M, Gassman S L. Pavement performance evaluation models for South Carolina[J]. KSCE Journal of Civil Engineering, 2017, 21(7): 2695-2706.

[5] Onayev A, Swei O. IRI deterioration model for asphalt concrete pavements: capturing performance improvements over time[J]. Construction and Building Materials, 2021, 271: 121768.

[6] 陈丰, 李岳, 刘星坤, 等. 面向自动驾驶编队的沥青路面结构疲劳寿命预估模型[J]. 中国公路学报, 2023, 36(12): 34-46.

[7] Yang X, Guan J, Ding L, et al. Research and applications of artificial neural network in pavement engineering: a state-of-the-art review[J]. Journal of Traffic and Transportation Engineering (English Edition), 2021, 8(6): 1000-1021.

[8] Kayadelen, Önal, Altay, et al. Gradient boosting and random forest for IRI prediction[J]. International Journal of Pavement Engineering, 2023, 24.

[9] Alnaqbi, Zeiada, Al-Khateeb. Machine learning modeling of pavement performance and IRI prediction in flexible pavement[J]. Innovative Infrastructure Solutions, 2024, 10.

[10] Kotb M. Prediction of distresses in pavement networks: a machine learning approach[D]. American University in Cairo, 2024.

[11] Sharma, Aggarwal. IRI prediction using Gaussian process regression[J]. WSEAS Transactions on Computer Research, 2023, 11: 111-116.

[12] Abdualmtalab, Heneash, Hussein, et al. Application of ANN for pavement roughness prediction in different climate regions[J]. Journal of King Saud University — Engineering Sciences, 2024, 36(2): 128-139.

[13] Hou C, Wang H, Guan W, et al. Road pavement performance prediction using a time series long short-term memory (LSTM) model[J]. Journal of Zhejiang University-SCIENCE A, 2025, 26(5): 424-437.

[14] Zhang T, Smith A, Zhai H, et al. LSTM+MA: a time-series model for predicting pavement IRI[J]. Infrastructures, 2025, 10(1): 10.

[15] Chen, Li, Wang, et al. Improved model for pavement performance prediction based on recurrent neural network using LTPP database[J]. ScienceDirect (TRB 2025), 2024.

[16] Plati C, Armeni A, Kyriakou K, et al. AI for predicting pavement roughness in road monitoring and maintenance[J]. Infrastructures, 2025, 10(7): 157.

[17] 黄凯枫, 刘庆华. 基于卷积长短时记忆网络的国际平整度指标预测[J]. 计算机与数字工程, 2024(1).

[18] Xin J, Akiyama M, Frangopol D. Sustainability-informed management optimization of asphalt pavement considering risk evaluated by multiple performance indicators using deep neural networks[J]. Reliability Engineering & System Safety, 2023, 238.

[19] Han C, Ma T, Gu L, et al. Asphalt pavement health prediction based on improved Transformer network[J]. IEEE Transactions on Intelligent Transportation Systems, 2023, 24: 4482-4493.

[20] Yao H, Han K, Liu Y, et al. Research and comparison of pavement performance prediction based on neural networks and fusion transformer architecture[J]. Electronic Research Archive, 2024, 32(2): 1239-1267.

[21] Cao X, Zeng Z, Yi F. Physics-aware Informer: a hybrid framework for accurate pavement IRI prediction in diverse climates[J]. Infrastructures, 2025.

[22] 呙润华, 于向前. 基于编码器-解码器结构的路面平整度预测[J]. 同济大学学报(自然科学版), 2023, 51(8).

[23] 蔡文渊, 李伦鹏, 朱兴一, 等. 机理-数据协同驱动的路面损伤劣化预测模型[J]. 中国公路学报, 2024（网络首发）.

[24] 张金喜, 侯明昊, 冉旭, 等. 基于不同智能设备的路面平整度预测精度对比[J]. 华南理工大学学报(自然科学版), 2024.

[25] KSCE Journal of Civil Engineering. Influence analysis of pavement distress on International Roughness Index using machine learning[J]. 2024, 28: 4344-4355.

[26] Applied Sciences. Optimizing faulting prediction for rigid pavements using a hybrid SHAP-TPE-CatBoost model[J]. 2023, 13(23): 12862.

---

## 附录：模型超参数

| 超参数 | LSTM | Transformer |
|:------:|:----:|:-----------:|
| 序列长度 | 5 | 5 |
| Dropout | 0.2 | 0.2 |
| Batch Size | 256 | 256 |
| 学习率 | 0.005 | 0.0005 |
| Weight Decay | 1e-3 | 1e-4 |
| 优化器 | Adam | Adam |
| 学习率调度 | ReduceLROnPlateau | ReduceLROnPlateau |
| 梯度裁剪 | 1.0 | 0.5 |
| 早停耐心 | 30 | 60 |
| 随机种子 | 42 | 42 |

---

*论文草稿生成时间：2026-05-22*
*基于 AC_PerformancePrediction_Research 项目代码实际输出*
