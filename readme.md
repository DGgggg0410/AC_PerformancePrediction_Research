
# 沥青混合料疲劳寿命预测研究：LSTM vs. Transformer

本研究项目基于 Python 与 PyTorch 框架，对比了 **LSTM (长短期记忆网络)** 与 **Transformer (自注意力模型)** 在沥青路面疲劳寿命预测中的应用。项目不仅关注预测精度，还通过 **SHAP (Shapley Additive Explanations)** 算法量化了各材料参数（如温度、油石比等）对寿命的影响。

📂 目录结构说明

AC_PerformancePrediction_Research/ (项目根目录)
├── .gitignore                      # 忽略规则文件
├── README.md                       # 本说明文档
├── _post_1_model_comparison.py     # 跨模型指标汇总
├── _post_2_residual_comparison.py   # 残差分布统计分析
├── _post_3_advanced_model_analysis.py # 综合科研对比绘图
├── Asphalt_LSTM_SHAP_Sequential/   # 子目录1：LSTM 模块
│   ├── _1_config.py ~ _8_shap_analyzer.py
│   └── interim/                    # LSTM 运行产生的中间数据
└── Asphalt_Transformer_SHAP_Coupling/ # 子目录2：Transformer 模块
    ├── _1_config.py ~ _8_shap_analyzer.py
    └── interim/                    # Transformer 运行产生的中间数据



## 📂 项目模块与运行环境

为确保路径引用的准确性，执行不同阶段的代码时必须切换到对应的**根目录**：

实验阶段  |	根目录名称	                                        |   说明
第一组：  |  LSTM	Asphalt_LSTM_SHAP_Sequential	           |   运行脚本 _1 至 _8
第二组：  |  Transformer	Asphalt_Transformer_SHAP_Coupling  |  运行脚本 _1 至 _8
对比分析  |	AC_PerformancePrediction_Research	               |   运行 _post 脚本
---

## 🚀 详细运行步骤

### 第一阶段：LSTM 模型全流程

**根目录切换至：`Asphalt_LSTM_SHAP_Sequential**`

1. **`_1_config.py`**：初始化实验环境，创建 `data/`、`interim/`、`output/` 文件夹并配置全局参数。
2. **`_2_data_processor.py`**：生成或加载沥青样本数据，执行 Min-Max 归一化，并保存为 `.pt` 张量。
3. **`_3_lstm_model.py`**：定义 `AsphaltLSTM` 模型类，自动校验输入输出张量维度。
4. **`_4_lstm_hyperparam_tuning.py`**：执行自动化网格搜索，寻找最优的学习率、层数及隐藏层维度，并保存最优权重。
5. **`_5_lstm_predictor.py`**：加载最优模型权重，对测试集进行预测，并保存预测结果至中间文件夹。
6. **`_6_lstm_trainer.py`**：执行标准的模型训练流程，记录每轮（Epoch）的损失值。
7. **`_7_lstm_evaluator.py`**：评估模型性能，绘制 Loss 曲线图及预测散点图（Regression Plot）。
8. **`_8_lstm_shap_analyzer.py`**：利用 SHAP 库解释模型，产出特征贡献度蜂群图及依赖分析图。

---

### 第二阶段：Transformer 模型全流程

**根目录切换至：`Asphalt_Transformer_SHAP_Coupling**`

1. **`_1_config.py`**：初始化 Transformer 专属的目录结构与配置。
2. **`_2_data_processor.py`**：处理 Transformer 训练所需的时序数据（保持与 LSTM 组数据逻辑一致）。
3. **`_3_transformer_model.py`**：定义包含位置编码（Positional Encoding）的 `AsphaltTransformer` 模型。
4. **`_4_transformer_hyperparam_tuning.py`**：调优 Transformer 核心参数（如多头注意力头数 `NHEAD`、编码器层数等）。
5. **`_5_predict_with_best_model.py`**：利用最优 Transformer 模型生成预测序列并反归一化。
6. **`_6_transformer_trainer.py`**：执行训练逻辑，捕捉 Transformer 在大数据量下的收敛特性。
7. **`_7_transformer_evaluator.py`**：计算 Transformer 的 、RMSE 等指标，绘制性能可视化图表。
8. **`_8_transformer_shap_analyzer.py`**：分析自注意力机制对沥青材料特征的关注分布。

---

### 第三阶段：综合对比分析（Post-Analysis）

**根目录切换至：`AC_PerformancePrediction_Research**`

1. **`_post_1_model_comparison_metrics.py`**：跨目录读取两个模型的 CSV 结果，生成综合性能评估对比报表。
2. **`_post_2_residual_distribution_comparison.py`**：对比两模型的预测残差（Residuals），通过核密度估计（KDE）分析误差稳定性。
3. **`_post_3_advanced_model_analysis.py`**：**核心汇总脚本**。生成全实验最终对比大图，包含：
* 两模型预测精度（）直观对比图。
* 两模型特征关注度（SHAP Importance）科学对比图。



---

## 📝 产出说明

* **图像资源**：各阶段生成的 `.png` 图表均存放在对应根目录的 `output/` 文件夹下。
* **中间数据**：`.pth`（权重）、`.joblib`（归一化器）及 `.pt`（张量数据）存放在 `interim/` 文件夹。
* **最终报告**：最终对比图表保存在 `output_final_comparison/` 文件夹中，可直接用于学术论文撰写。

---

**提示：** 运行前请确保安装了必备库：`torch`, `shap`, `seaborn`, `joblib`, `pandas`, `matplotlib`, `scikit-learn`。

---
 联系方式：17635150410@163.com