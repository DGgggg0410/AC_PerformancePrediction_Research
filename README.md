# 基于LSTM与Transformer的沥青路面IRI预测研究

> Pavement IRI Prediction Research — LSTM vs Transformer with SHAP Interpretability

基于美国 LTPP（Long-Term Pavement Performance）数据库 35 年真实监测数据，构建 LSTM 和 Transformer 两种深度学习模型进行路面国际平整度指数（IRI）预测，并结合 SHAP 可解释性分析和系统性消融实验量化各类工程因素的贡献度。

---

## 核心结果

| 模型 | R² | RMSE (m/km) | MAE (m/km) | 参数量 |
|------|:---:|:-----------:|:----------:|:------:|
| **LSTM** | **0.7279** | 0.3724 | 0.2056 | **217K** |
| **Transformer** | **0.7472** | 0.3590 | 0.1889 | 830K |

### 关键发现

- **Transformer 以约 3.8 倍参数量轻徼优于 LSTM**（ΔR² ≈ 0.02）——IRI 预测的强时序自回归特性使 LSTM 的循环归纳偏置能以较少参数达到可比性能
- **时序滞后特征主导预测**——IRI_LAG_1/2 占 SHAP 重要性 57%–86%，位居特征重要性首位
- **结构特征贡献最稳健**——消融实验中唯一在两模型均产生一致性性能下降的工程因素组（ΔR² ≈ -0.007）
- **两模型差异**：LSTM 高度聚焦时序特征（SHAP 86%），Transformer 对多类特征均有感知（气候+地理+结构合计约 43%）

---

## 项目结构

```
AC_PerformancePrediction_Research/
├── AC_LSTM/                    # LSTM 模型完整代码
│   ├── _0_ltpp_data_loader.py  # 数据加载与预处理
│   ├── _1_config.py            # 配置文件
│   ├── _2_sequence_builder.py  # 滑动窗口序列构建
│   ├── _3_lstm_model.py        # LSTM 模型定义
│   ├── _4_hyperparam_tuning.py # 超参数随机搜索
│   ├── _5_trainer.py           # 模型训练
│   ├── _6_predictor.py         # 预测与评估
│   ├── _7_shap_analyzer.py     # SHAP 可解释性分析
│   ├── output/                 # 训练输出
│   ├── output_5yr/             # 5 年预测输出
│   └── ablation/               # 消融实验代码
│
├── AC_Transformer/             # Transformer 模型完整代码
│   ├── ...                     # （结构同 AC_LSTM）
│   ├── output/                 # 训练输出
│   ├── output_5yr/
│   └── ablation/
│
├── ablation_analysis.py        # 消融实验综合对比分析
├── ablation_figures/           # 消融实验图表
└── output_quick_test/          # 交通数据快速验证
```

## 19维特征体系

| 类别 | 数量 | 说明 |
|------|:----:|------|
| 时序特征 | 3 | PAVEMENT_AGE, IRI_LAG_1, IRI_LAG_2 |
| 气候特征 | 8 | 度日数、极端温度、冻融指标、降水量、蒸发量 |
| 结构特征 | 5 | 总厚度、沥青层厚度、基层厚度、层数、路面类型 |
| 地理特征 | 3 | 纬度、经度、海拔 |

## 运行顺序

```
E:\Visual Studio Code2025\python_program\AC_PerformancePrediction_Research\AC_LSTM
_0_ltpp_data_loader
_1_config
_2_sequence_builder
_3_lstm_model
_4_lstm_hyperparam_tuning
_5_trainer
_6_predictor
_7_shap_analyzer

↓↓↓↓↓↓↓↓↓↓↓↓

E:\Visual Studio Code2025\python_program\AC_PerformancePrediction_Research\AC_Transformer
_0_ltpp_data_loader
_1_config
_2_sequence_builder
_3_lstm_model
_4_lstm_hyperparam_tuning
_5_trainer
_6_predictor
_7_shap_analyzer

↓↓↓↓↓↓↓↓↓↓↓↓

E:\Visual Studio Code2025\python_program\AC_PerformancePrediction_Research
fix_ablation_configs

↓↓↓↓↓↓↓↓↓↓↓↓

E:\Visual Studio Code2025\python_program\AC_PerformancePrediction_Research\AC_LSTM\ablation
_run_all_lstm_ablation

↓↓↓↓↓↓↓↓↓↓↓↓

E:\Visual Studio Code2025\python_program\AC_PerformancePrediction_Research\AC_Transformer\ablation
_run_all_transformer_ablation

↓↓↓↓↓↓↓↓↓↓↓↓

E:\Visual Studio Code2025\python_program\AC_PerformancePrediction_Research
ablation_analysis

↓↓↓↓↓↓↓↓↓↓↓↓
E:\Visual Studio Code2025\python_program\AC_PerformancePrediction_Research\AC_LSTM\ablation
_aadtt_analysis
```

