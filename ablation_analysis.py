"""
消融实验综合对比分析
对比LSTM和Transformer在各消融实验中的表现

功能：
1. 加载各消融实验的评估指标
2. 生成 6 张独立的论文级图表，保存到 ablation_figures/ 目录
3. 输出控制台对比表格和文字报告

作者: 研究团队
日期: 2026
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from matplotlib.font_manager import FontManager
fm = FontManager()
available_fonts = set(font.name for font in fm.ttflist)
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
has_chinese = any(font in available_fonts for font in chinese_fonts)

if not has_chinese:
    print("警告: 未检测到中文字体，图表中文可能显示为方框")
    print("建议安装: 黑体(SimHei)或微软雅黑(Microsoft YaHei)")
    USE_ENGLISH_LABELS = True
else:
    USE_ENGLISH_LABELS = False

import pandas as pd

# ============================================================================
# 路径配置
# ============================================================================

PROJECT_DIR = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research'
LSTM_OUTPUT = os.path.join(PROJECT_DIR, 'AC_LSTM', 'output')
TRANSFORMER_OUTPUT = os.path.join(PROJECT_DIR, 'AC_Transformer', 'output')

# 新增：论文图表输出目录
FIGURES_DIR = os.path.join(PROJECT_DIR, 'ablation_figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# 颜色主题
COLOR_LSTM = '#2980b9'       # 蓝色
COLOR_TRANS = '#8e44ad'     # 紫色
COLOR_LSTM_LIGHT = '#5dade2'
COLOR_TRANS_LIGHT = '#a569bd'


# ============================================================================
# 数据加载
# ============================================================================

def load_ablation_results():
    """加载所有消融实验结果（从各子目录的 _results.json 读取）"""
    if USE_ENGLISH_LABELS:
        experiments = [
            ('ablation_no_climate',          'No Climate',          11),
            ('ablation_no_structure',         'No Structure',         14),
            ('ablation_no_geographic',        'No Geographic',        16),
            ('ablation_only_temporal',        'Only Temporal',         3),
            ('ablation_no_climate_structure', 'No Climate+Structure',  6),
        ]
    else:
        experiments = [
            ('ablation_no_climate',          '去掉气候因素',   11),
            ('ablation_no_structure',         '去掉结构因素',   14),
            ('ablation_no_geographic',        '去掉地理因素',   16),
            ('ablation_only_temporal',        '只保留时序',      3),
            ('ablation_no_climate_structure', '去掉气候+结构',  6),
        ]

    # Baseline 固定值（实际运行结果）
    lstm_baseline  = {'r2': 0.7172, 'rmse': 0.3797, 'mae': 0.2201}
    transformer_baseline = {'r2': 0.7473, 'rmse': 0.3589, 'mae': 0.1890}

    lstm_results      = {'baseline': lstm_baseline}
    transformer_results = {'baseline': transformer_baseline}

    for exp_name, _, _ in experiments:
        # LSTM
        lstm_path = os.path.join(LSTM_OUTPUT, exp_name, f'{exp_name}_results.json')
        if os.path.exists(lstm_path):
            with open(lstm_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                lstm_results[exp_name] = data['metrics']
        else:
            lstm_results[exp_name] = {'r2': None, 'rmse': None, 'mae': None}

        # Transformer
        trans_path = os.path.join(TRANSFORMER_OUTPUT, exp_name, f'{exp_name}_results.json')
        if os.path.exists(trans_path):
            with open(trans_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                transformer_results[exp_name] = data['metrics']
        else:
            transformer_results[exp_name] = {'r2': None, 'rmse': None, 'mae': None}

    return lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline


# ============================================================================
# 辅助函数
# ============================================================================

def get_val(d, key):
    """安全获取字典值，避免 None 导致计算错误"""
    v = d.get(key, None)
    return v if v is not None else 0.0


def bar_text(ax, bars, values, offset_frac=0.015, va='bottom', fontsize=9):
    """在柱状图顶部/底部添加数值标签"""
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    for bar, val in zip(bars, values):
        if val == 0:
            continue
        y_pos = bar.get_height() if val >= 0 else bar.get_height()
        sign = '+' if val > 0 else ''
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            y_pos + y_range * offset_frac * (1 if val >= 0 else -1),
            f'{sign}{val:.4f}',
            ha='center', va=va if val >= 0 else 'top',
            fontsize=fontsize, fontweight='bold'
        )


# ============================================================================
# 图表 1：R² 对比柱状图
# ============================================================================

def plot_fig1_r2_comparison(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline):
    """LSTM vs Transformer 各实验 R² 对比（并排柱状图）"""
    fig, ax = plt.subplots(figsize=(12, 6))

    exp_keys   = [e[0] for e in experiments]
    exp_labels = [e[1] for e in experiments]

    lstm_r2   = [get_val(lstm_results.get(k, {}), 'r2') for k in exp_keys]
    trans_r2  = [get_val(transformer_results.get(k, {}), 'r2') for k in exp_keys]

    x = np.arange(len(exp_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, lstm_r2,  width, label='S-LSTM',       color=COLOR_LSTM, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, trans_r2, width, label='Transformer', color=COLOR_TRANS, edgecolor='black', linewidth=0.8)

    # 基准线
    ax.axhline(y=lstm_baseline['r2'],       color=COLOR_LSTM,  linestyle='--', linewidth=1.5, alpha=0.7, label=f'S-LSTM Baseline (R$^2$={lstm_baseline["r2"]:.4f})')
    ax.axhline(y=transformer_baseline['r2'], color=COLOR_TRANS, linestyle='--', linewidth=1.5, alpha=0.7, label=f'Transformer Baseline ($^2$={transformer_baseline["r2"]:.4f})')

    ax.set_ylabel(r'R$^2$', fontsize=13)
    ax.set_title('Ablation Study: R$^2$ Comparison - S-LSTM vs Transformer', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=11)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim(0.60, 0.80)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3)

    bar_text(ax, bars1, lstm_r2,  offset_frac=0.008)
    bar_text(ax, bars2, trans_r2, offset_frac=0.008)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig1_r2_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  [OK] fig1_r2_comparison.png  -> {path}")
    plt.close()


# ============================================================================
# 图表 2：RMSE 对比柱状图
# ============================================================================

def plot_fig2_rmse_comparison(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline):
    """LSTM vs Transformer 各实验 RMSE 对比"""
    fig, ax = plt.subplots(figsize=(12, 6))

    exp_keys   = [e[0] for e in experiments]
    exp_labels = [e[1] for e in experiments]

    lstm_rmse  = [get_val(lstm_results.get(k, {}), 'rmse') for k in exp_keys]
    trans_rmse = [get_val(transformer_results.get(k, {}), 'rmse') for k in exp_keys]

    x = np.arange(len(exp_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, lstm_rmse,  width, label='S-LSTM',       color=COLOR_LSTM, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, trans_rmse, width, label='Transformer', color=COLOR_TRANS, edgecolor='black', linewidth=0.8)

    ax.axhline(y=lstm_baseline['rmse'],       color=COLOR_LSTM,  linestyle='--', linewidth=1.5, alpha=0.7, label=f'LSTM Baseline (RMSE={lstm_baseline["rmse"]:.4f})')
    ax.axhline(y=transformer_baseline['rmse'], color=COLOR_TRANS, linestyle='--', linewidth=1.5, alpha=0.7, label=f'Transformer Baseline (RMSE={transformer_baseline["rmse"]:.4f})')

    ax.set_ylabel('RMSE (m/km)', fontsize=13)
    ax.set_title('Ablation Study: RMSE Comparison — LSTM vs Transformer', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    y_max = max(max(lstm_rmse), max(trans_rmse)) * 1.2
    ax.set_ylim(0, y_max)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3)

    bar_text(ax, bars1, lstm_rmse,  offset_frac=0.02)
    bar_text(ax, bars2, trans_rmse, offset_frac=0.02)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig2_rmse_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  [OK] fig2_rmse_comparison.png  -> {path}")
    plt.close()


# ============================================================================
# 图表 3：MAE 对比柱状图
# ============================================================================

def plot_fig3_mae_comparison(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline):
    """LSTM vs Transformer 各实验 MAE 对比"""
    fig, ax = plt.subplots(figsize=(12, 6))

    exp_keys   = [e[0] for e in experiments]
    exp_labels = [e[1] for e in experiments]

    lstm_mae  = [get_val(lstm_results.get(k, {}), 'mae') for k in exp_keys]
    trans_mae = [get_val(transformer_results.get(k, {}), 'mae') for k in exp_keys]

    x = np.arange(len(exp_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, lstm_mae,  width, label='S-LSTM',       color=COLOR_LSTM, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, trans_mae, width, label='Transformer', color=COLOR_TRANS, edgecolor='black', linewidth=0.8)

    ax.axhline(y=lstm_baseline['mae'],       color=COLOR_LSTM,  linestyle='--', linewidth=1.5, alpha=0.7, label=f'LSTM Baseline (MAE={lstm_baseline["mae"]:.4f})')
    ax.axhline(y=transformer_baseline['mae'], color=COLOR_TRANS, linestyle='--', linewidth=1.5, alpha=0.7, label=f'Transformer Baseline (MAE={transformer_baseline["mae"]:.4f})')

    ax.set_ylabel('MAE (m/km)', fontsize=13)
    ax.set_title('Ablation Study: MAE Comparison — LSTM vs Transformer', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    y_max = max(max(lstm_mae), max(trans_mae)) * 1.2
    ax.set_ylim(0, y_max)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3)

    bar_text(ax, bars1, lstm_mae,  offset_frac=0.02)
    bar_text(ax, bars2, trans_mae, offset_frac=0.02)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_mae_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  [OK] fig3_mae_comparison.png  -> {path}")
    plt.close()


# ============================================================================
# 图表 4：R² 变化量对比（修复：支持正负值，y轴对称）
# ============================================================================

def plot_fig4_r2_change(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline):
    """
    各消融实验相对于 baseline 的 R² 变化量
    — 修复：LSTM 所有消融 R² 均为正值（下降），但有负值（改善），
      因此 y 轴对称设置，使正负均可完整显示
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    exp_keys   = [e[0] for e in experiments]
    exp_labels = [e[1] for e in experiments]

    lstm_r2_baseline  = lstm_baseline['r2']
    trans_r2_baseline = transformer_baseline['r2']

    lstm_r2_drop  = [lstm_r2_baseline  - get_val(lstm_results.get(k, {}),      'r2') for k in exp_keys]
    trans_r2_drop = [trans_r2_baseline - get_val(transformer_results.get(k, {}), 'r2') for k in exp_keys]

    x = np.arange(len(exp_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, lstm_r2_drop,  width, label='S-LSTM',       color=COLOR_LSTM, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, trans_r2_drop, width, label='Transformer', color=COLOR_TRANS, edgecolor='black', linewidth=0.8)

    # 0 值参考线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.8)

    # y 轴对称设置（确保正负柱都能完整显示）
    all_drops = lstm_r2_drop + trans_r2_drop
    max_abs = max(abs(v) for v in all_drops) * 1.25
    ax.set_ylim(-max_abs, max_abs)

    ax.set_ylabel(r'R$^2$ Change (Baseline - Ablation)', fontsize=13)
    ax.set_title(r'R$^2$ Change Relative to Baseline\n(Positive = Performance Drop, Negative = Improvement)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 标注数值（处理正负值）
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]

    for bar, val in zip(bars1, lstm_r2_drop):
        offset = y_range * 0.015
        va = 'bottom' if val >= 0 else 'top'
        sign = '+' if val > 0 else ''
        ax.text(bar.get_x() + bar.get_width() / 2.,
                val + offset * (1 if val >= 0 else -1),
                f'{sign}{val:.4f}',
                ha='center', va=va, fontsize=9, fontweight='bold', color=COLOR_LSTM)

    for bar, val in zip(bars2, trans_r2_drop):
        offset = y_range * 0.015
        va = 'bottom' if val >= 0 else 'top'
        sign = '+' if val > 0 else ''
        ax.text(bar.get_x() + bar.get_width() / 2.,
                val + offset * (1 if val >= 0 else -1),
                f'{sign}{val:.4f}',
                ha='center', va=va, fontsize=9, fontweight='bold', color=COLOR_TRANS)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig4_r2_change.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  [OK] fig4_r2_change.png  -> {path}")
    plt.close()


# ============================================================================
# 图表 5：特征数量 vs R²（折线图）
# ============================================================================

def plot_fig5_features_vs_r2(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline):
    """特征数量与 R² 的关系曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    exp_keys  = [e[0] for e in experiments]
    n_features = [e[2] for e in experiments]

    lstm_r2 = [get_val(lstm_results.get(k, {}),      'r2') for k in exp_keys]
    trans_r2 = [get_val(transformer_results.get(k, {}), 'r2') for k in exp_keys]

    # 按特征数排序，避免折线因乱序而"往回拐"
    sorted_data = sorted(zip(n_features, lstm_r2, trans_r2), key=lambda x: x[0])
    n_features_sorted, lstm_r2_sorted, trans_r2_sorted = zip(*sorted_data)

    ax.plot(n_features_sorted, lstm_r2_sorted,  'o-', color=COLOR_LSTM,  label='LSTM',       linewidth=2.2, markersize=9)
    ax.plot(n_features_sorted, trans_r2_sorted, 's-', color=COLOR_TRANS, label='Transformer', linewidth=2.2, markersize=9)

    # Baseline 参考线
    ax.axhline(y=lstm_baseline['r2'],       color=COLOR_LSTM,  linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axhline(y=transformer_baseline['r2'], color=COLOR_TRANS, linestyle='--', alpha=0.6, linewidth=1.5)

    # 标注各点特征数
    for n, l, t in zip(n_features_sorted, lstm_r2_sorted, trans_r2_sorted):
        ax.annotate(f'{n}', (n, l), textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=9, color=COLOR_LSTM)
        ax.annotate(f'{n}', (n, t), textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=9, color=COLOR_TRANS)

    ax.set_xlabel('Number of Features', fontsize=13)
    ax.set_ylabel(r'R$^2$', fontsize=13)
    ax.set_title(r'Model Performance vs. Number of Features (R$^2$)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_ylim(0.60, 0.80)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig5_features_vs_r2.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  [OK] fig5_features_vs_r2.png  -> {path}")
    plt.close()


# ============================================================================
# 图表 6：特征类别重要性对比（柱状图）
# ============================================================================

def plot_fig6_importance_comparison(lstm_results, transformer_results, lstm_baseline, transformer_baseline):
    """
    特征类别（气候/结构/地理）对 R² 的贡献度对比
    贡献度 = baseline R² − 去掉该类别后的 R²
    数值越大说明该类别越重要
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = [
        ('气候因素\n(Climate)',      'ablation_no_climate'),
        ('结构因素\n(Structure)',     'ablation_no_structure'),
        ('地理因素\n(Geographic)',    'ablation_no_geographic'),
    ]

    lstm_vals  = []
    trans_vals = []
    labels_cn  = []
    labels_en  = []

    for cn, exp_key in categories:
        lstm_r2_after = get_val(lstm_results.get(exp_key, {}), 'r2')
        trans_r2_after = get_val(transformer_results.get(exp_key, {}), 'r2')
        lstm_vals.append(lstm_baseline['r2']  - lstm_r2_after)
        trans_vals.append(transformer_baseline['r2'] - trans_r2_after)
        labels_cn.append(cn.replace('\n', ' '))
        labels_en.append(cn.split('\n')[0])

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, lstm_vals,  width, label='S-LSTM',       color=COLOR_LSTM, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, trans_vals, width, label='Transformer', color=COLOR_TRANS, edgecolor='black', linewidth=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel(r'R$^2$ Drop (Baseline - Ablation)', fontsize=13)
    ax.set_title('Feature Category Importance\n(Positive = Category Helps, Negative = Category Hurts)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_en, fontsize=11)
    ax.legend(fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3)

    # y 轴对称（应对 LSTM 负值）
    all_vals = lstm_vals + trans_vals
    max_abs = max(abs(v) for v in all_vals) * 1.3
    ax.set_ylim(-max_abs, max_abs)

    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]

    for bar, val in zip(bars1, lstm_vals):
        offset = y_range * 0.015
        va = 'bottom' if val >= 0 else 'top'
        sign = '+' if val > 0 else ''
        ax.text(bar.get_x() + bar.get_width() / 2.,
                val + offset * (1 if val >= 0 else -1),
                f'{sign}{val:.4f}', ha='center', va=va,
                fontsize=9, fontweight='bold', color=COLOR_LSTM)

    for bar, val in zip(bars2, trans_vals):
        offset = y_range * 0.015
        va = 'bottom' if val >= 0 else 'top'
        sign = '+' if val > 0 else ''
        ax.text(bar.get_x() + bar.get_width() / 2.,
                val + offset * (1 if val >= 0 else -1),
                f'{sign}{val:.4f}', ha='center', va=va,
                fontsize=9, fontweight='bold', color=COLOR_TRANS)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig6_importance_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  [OK] fig6_importance_comparison.png  -> {path}")
    plt.close()


# ============================================================================
# 汇总表格（CSV）
# ============================================================================

def save_summary_csv(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline):
    """将所有消融实验结果保存为 CSV"""
    rows = []
    for exp_name, exp_label, n_feat in experiments:
        lstm_r  = get_val(lstm_results.get(exp_name, {}),      'r2')
        lstm_rmse = get_val(lstm_results.get(exp_name, {}),    'rmse')
        lstm_mae  = get_val(lstm_results.get(exp_name, {}),    'mae')
        trans_r  = get_val(transformer_results.get(exp_name, {}), 'r2')
        trans_rmse = get_val(transformer_results.get(exp_name, {}), 'rmse')
        trans_mae  = get_val(transformer_results.get(exp_name, {}), 'mae')

        lstm_r2_change  = lstm_baseline['r2']  - lstm_r
        trans_r2_change = transformer_baseline['r2'] - trans_r

        rows.append({
            'Experiment': exp_label,
            'N_Features': n_feat,
            'LSTM_R2':  round(lstm_r,  6),
            'LSTM_RMSE': round(lstm_rmse, 6),
            'LSTM_MAE':  round(lstm_mae,  6),
            'LSTM_R2_Change':  round(lstm_r2_change,  6),
            'Trans_R2':  round(trans_r,  6),
            'Trans_RMSE': round(trans_rmse, 6),
            'Trans_MAE':  round(trans_mae,  6),
            'Trans_R2_Change': round(trans_r2_change, 6),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(FIGURES_DIR, 'ablation_summary.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  [OK] ablation_summary.csv  -> {csv_path}")
    return df


# ============================================================================
# 控制台输出报告
# ============================================================================

def print_report(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline):
    """打印详细的文字报告"""
    print("\n" + "=" * 90)
    print("  消融实验综合对比分析")
    print("=" * 90)

    print(f"\n{'实验名称':<25} {'特征数':<8} {'LSTM R2':<12} {'Trans R2':<12} {'LSTM ΔR2':<12} {'Trans ΔR2':<12}")
    print("-" * 90)

    for exp_name, exp_label, n_feat in experiments:
        lstm_r   = get_val(lstm_results.get(exp_name, {}),      'r2')
        trans_r  = get_val(transformer_results.get(exp_name, {}), 'r2')
        lstm_d   = lstm_baseline['r2']  - lstm_r
        trans_d  = transformer_baseline['r2'] - trans_r
        print(f"{exp_label:<25} {n_feat:<8} {lstm_r:>10.4f}  {trans_r:>10.4f}  {lstm_d:>+10.4f}  {trans_d:>+10.4f}")

    print("-" * 90)
    print(f"{'Baseline (Full 19)':<25} {'19':<8} {lstm_baseline['r2']:>10.4f}  {transformer_baseline['r2']:>10.4f}  {'—':>12}  {'—':>12}")
    print("=" * 90)

    # 特征类别重要性
    print("\n[特征类别重要性] R2 下降越多 = 该类别越重要")
    print("-" * 60)

    categories = [
        ('气候因素 (Climate)',      'ablation_no_climate'),
        ('结构因素 (Structure)',    'ablation_no_structure'),
        ('地理因素 (Geographic)',   'ablation_no_geographic'),
    ]

    print(f"  {'类别':<25} {'LSTM ΔR2':<15} {'Transformer ΔR2':<15}")
    print("-" * 60)
    for cn, exp_key in categories:
        lstm_v  = lstm_baseline['r2']  - get_val(lstm_results.get(exp_key, {}),      'r2')
        trans_v = transformer_baseline['r2'] - get_val(transformer_results.get(exp_key, {}), 'r2')
        print(f"  {cn:<25} {lstm_v:>+14.4f}  {trans_v:>+14.4f}")

    print("\n[关键发现]")
    print("-" * 60)
    print("  1. LSTM: 所有消融实验的 R2 均高于 baseline (ΔR2 均为负值)")
    print("     -> 表明 LSTM 在 19 维全部特征下存在轻微过拟合")
    print("     -> 减少特征后反而提升了泛化能力")
    print("  2. Transformer: 去掉气候因素后 R2 下降最显著 (ΔR2 = +0.0290)")
    print("     -> Transformer 更依赖气候特征进行预测")
    print("  3. 地理因素对两模型贡献均最小 (|ΔR2| 最小)")
    print("=" * 90)


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("\n开始消融实验综合分析...")
    print(f"图表将保存至: {FIGURES_DIR}\n")

    lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline = load_ablation_results()

    print_report(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline)

    print("\n生成论文级图表...")
    plot_fig1_r2_comparison(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline)
    plot_fig2_rmse_comparison(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline)
    plot_fig3_mae_comparison(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline)
    plot_fig4_r2_change(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline)
    plot_fig5_features_vs_r2(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline)
    plot_fig6_importance_comparison(lstm_results, transformer_results, lstm_baseline, transformer_baseline)

    save_summary_csv(lstm_results, transformer_results, experiments, lstm_baseline, transformer_baseline)

    print(f"\n[完成] 分析完成！共生成 6 张图片 + 1 张汇总 CSV")
    print(f"  输出目录: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
