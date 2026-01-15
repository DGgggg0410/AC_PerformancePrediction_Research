明白你的意思了，这份 `README.md` 已经完全去掉了“终端命令”的内容，改为针对 **VS Code 环境下直接点击运行** 的直观描述，并且明确了所有操作均在根目录视角下进行。

---

# 沥青混合料疲劳寿命预测研究 (AC_PerformancePrediction_Research)

本项目实现了基于 **LSTM** 和 **Transformer** 的沥青混合料疲劳寿命预测。项目采用统一的工程化结构，支持在 VS Code 中直接运行所有子项目脚本，并集成 **SHAP** 解释性框架。

## 📂 项目目录结构

```text
AC_PerformancePrediction_Research/
├── Asphalt_LSTM_SHAP_Sequential/         # LSTM 项目文件夹 (8个代码文件)
├── Asphalt_Transformer_SHAP_Coupling/    # Transformer 项目文件夹 (8个代码文件)
├── Model_Comparison_Analysis/            # 跨模型对比分析文件夹
├── venv/                                 # 统一虚拟环境
├── .gitignore                            # Git 忽略配置（已跳过 venv 和大文件）
└── readme.md                             # 本说明文档

```

## 🚀 VS Code 运行指南

**核心原则**：请始终在 VS Code 中打开 `AC_PerformancePrediction_Research` **总根目录**。脚本已内置路径对齐逻辑，无论文件在哪个子目录下，直接在 VS Code 编辑器中点击“运行”按钮即可。

### 第一阶段：完成 LSTM 模型流程

依次打开 `Asphalt_LSTM_SHAP_Sequential` 文件夹下的脚本并运行：

1. **`_1_config.py`** 直到 **`_4_hyperparameter_tuner.py`**：完成数据预处理、模型定义及超参数自动搜寻。

### ⚠️ 第二阶段：手动调参 (LSTM)

运行完 Step 4 后，根据 `output` 文件夹中记录的**最优超参数组合**，手动修改并依次运行后续脚本：
5. **`_5_final_trainer.py`**：**需手动更新代码中的最佳参数**，训练最终权重。
6. **`_6_evaluator.py`**：**需手动更新代码中的最佳参数**，评估模型。
7. **`_7_visualizer.py`**：**需手动更新代码中的最佳参数**，生成预测可视化图表。
8. **`_8_shap_analyzer.py`**：**需手动更新代码中的最佳参数**，生成 SHAP 可解释性数据。

---

### 第三阶段：完成 Transformer 模型流程

依次打开 `Asphalt_Transformer_SHAP_Coupling` 文件夹下的脚本并运行：

1. **`_1_config.py`** 直到 **`_4_hyperparameter_tuner.py`**。

### ⚠️ 第四阶段：手动调参 (Transformer)

同样地，根据 Transformer 模型 Step 4 的结果，**手动更新脚本中的参数**后，继续运行：
5. **`_5_final_trainer.py`** 直到 **`_8_shap_analyzer.py`**。（**需手动更新代码中的最佳参数**）

---

### 第五阶段：跨模型对比分析

当两个模型的 Step 8 都运行完毕，打开 `Model_Comparison_Analysis` 文件夹运行对比脚本：

1. **`_post_1_model_comparison_metrics.py`**：生成 `final_model_comparison_report.csv` 性能对比表。
2. **`_post_2_residual_distribution_comparison.py`**：生成残差概率密度曲线，对比模型稳定性。
3. **`_post_3_advanced_model_analysis.py`**：生成特征关注度对比图，分析决策机制差异。

## 🛠️ 技术说明

* **路径支持**：项目已配置 `sys.path` 自动向上寻找根目录，VS Code 编辑器中的黄线（无法解析导入）不影响实际运行。
* **数据存储**：中间文件存储在各子文件夹的 `interim` 目录中，最终对比结果保存在 `Model_Comparison_Analysis/output_final_comparison` 下。
* **SHAP 分析**：通过对比可以发现模型对“温度”和“应力水平”等特征的关注权重差异。

---


**下一步建议：**
现在你可以放心地按照这份文档的顺序在 VS Code 里检查代码并进行最后的测试运行了。上传 GitHub 时，只要确认 `.gitignore` 生效（源代码管理器数字恢复正常），就可以进行最后的推送！祝你这一阶段圆满收官！


疑问点：
你可能会疑惑数据从哪来的，代码文件1、2生成虚拟数据，所以其实这是简单的初步模拟，下一步我将继续跟进将真实数据导入文件，运行后查看结果。

**联系**: 17635150410@163.com

**日期**: 2026-01-14
