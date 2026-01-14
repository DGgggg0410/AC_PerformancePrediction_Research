

# 沥青混合料疲劳寿命预测研究 (AC_PerformancePrediction_Research)

本项目旨在利用深度学习模型（**LSTM** 与 **Transformer**）对沥青混合料的疲劳寿命进行预测，并引入 **SHAP (SHapley Additive exPlanations)** 解释性框架，从全局和局部维度分析输入特征对模型决策的影响。

## 📂 项目目录结构

```text
AC_PerformancePrediction_Research/
├── Asphalt_LSTM_SHAP_Sequential/         # LSTM 模型全流程文件夹
│   ├── data/                             # 原始数据集
│   ├── interim/                          # 中间计算结果 (.pt, .joblib, .npy)
│   ├── output/                           # 模型训练产物 (权重, 训练日志)
│   └── _1_config.py ~ _8_shap_analyzer.py # 核心脚本 (Step 1-8)
├── Asphalt_Transformer_SHAP_Coupling/    # Transformer 模型全流程文件夹
│   ├── (结构同上)
├── Model_Comparison_Analysis/            # 跨模型对比分析模块
│   ├── _post_1_model_comparison_metrics.py # 性能指标 PK
│   ├── _post_2_residual_distribution_comparison.py # 残差分布对比
│   └── _post_3_advanced_model_analysis.py  # 综合性能与 SHAP 关注度对比
├── venv/                                 # Python 虚拟环境
├── .gitignore                            # Git 忽略文件配置
└── readme.md                             # 项目说明文档

```

## 🚀 运行指南

本项目分为 **模型独立运行** 和 **跨模型对比** 两个阶段。两个模型文件夹内的脚本命名规则一致（Step 1-8），请务必按编号顺序执行。

### 第一阶段：模型开发 (Step 1 - 4)

在 `Asphalt_LSTM_...` 或 `Asphalt_Transformer_...` 文件夹下依次运行：

1. **`_1_config.py`**: 初始化路径与全局参数。
2. **`_2_data_preprocessor.py`**: 数据清洗、归一化及序列化处理。
3. **`_3_model_definition.py`**: 定义模型架构。
4. **`_4_hyperparameter_tuner.py`**: 自动寻找最优超参数组合。

### ⚠️ 第二阶段：手动调参衔接 (Step 5 - 8)

**重要提示**：在完成 Step 4 自动调优后，你需要根据 `output` 文件夹中生成的 `best_params.txt` 或日志，**手动修改后续脚本中的参数**。

* **`_5_final_trainer.py`**:
* **操作**：手动修改代码开头的 `HIDDEN_SIZE`、`LR`、`EPOCHS` 等参数为 Step 4 得到的最佳值。
* **目的**：使用最优参数训练最终模型并保存权重。


* **`_6_evaluator.py`**:
* **操作**：确保加载的权重路径指向 Step 5 生成的 `.pth` 文件。
* **目的**：计算测试集 R²、RMSE、MAE 等指标。


* **`_7_visualizer.py`**:
* **操作**：根据需要调整绘图的坐标轴范围或保存名称。
* **目的**：生成拟合效果散点图。


* **`_8_shap_analyzer.py`**:
* **操作**：根据模型实际的输入维度（Samples, Seq_Len, Features），调整 SHAP Explainer 的初始化参数。
* **目的**：计算并保存全局 SHAP 值 (`shap_values.npy`)。



### 第三阶段：跨模型对比分析 (Post 1 - 3)

当两个子项目的 Step 8 全部运行完成后，进入 `Model_Comparison_Analysis` 文件夹运行对比脚本：

1. **`_post_1_...`**: 生成 `final_model_comparison_report.csv` 指标对比表。
2. **`_post_2_...`**: 绘制残差概率密度曲线，对比模型稳定性。
3. **`_post_3_...`**: 绘制特征关注度对比图，分析 LSTM 与 Transformer 决策机制的差异。

## 🛠️ 环境要求

1. **Python 版本**: 3.8+
2. **核心库**: `torch`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `shap`, `scikit-learn`, `joblib`。
3. **安装命令**:
```bash
pip install -r requirements.txt

```



## 📌 注意事项

* **路径引用**：项目采用了动态路径定位，请始终在 VS Code 中打开 `AC_PerformancePrediction_Research` **根目录**作为工作区，以保证 `sys.path.append` 逻辑正常工作。
* **Git 提交**：在上传至 GitHub 前，请务必检查 `.gitignore` 是否已包含 `venv/`、`data/` 及各类 `.pt` 权重文件，避免仓库体积过大。

---

**作者**: [你的名字/ID]

**日期**: 2026-01-14
