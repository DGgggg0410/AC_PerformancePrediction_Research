"""
AADTT交通荷载数据分析脚本
分析 AADTT 与 MRI 的相关性及对 IRI 预测的边际贡献

使用方法：在项目根目录下运行
    python AC_LSTM/ablation/_aadtt_analysis.py

依赖：pip install openpyxl pandas scikit-learn scipy

作者: 研究团队
日期: 2026
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

# ============================================================================
# 路径配置
# ============================================================================
PROJECT_DIR = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research'
TRAFFIC_PATH = os.path.join(PROJECT_DIR, 'data_LTPP', 'Bucket_142084.xlsx')
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, 'processed_data', 'ltpp_processed_data.csv')


def explore_xlsx():
    """探索xlsx文件结构"""
    print("=" * 60)
    print("探索交通荷载数据文件结构")
    print("=" * 60)

    xls = pd.ExcelFile(TRAFFIC_PATH)
    print(f"Sheet数量: {len(xls.sheet_names)}")
    print(f"Sheet名称: {xls.sheet_names}")

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(TRAFFIC_PATH, sheet_name=sheet_name, nrows=5)
        print(f"\nSheet [{sheet_name}]:")
        print(f"  总列数: {len(df.columns)}")
        print(f"  列名: {list(df.columns)}")
        # 检查列名中是否包含AADTT或TRAFFIC相关关键词
        traffic_cols = [c for c in df.columns if 'AADTT' in str(c).upper() or 'TRAF' in str(c).upper() or 'ADTT' in str(c).upper() or 'TRUCK' in str(c).upper()]
        if traffic_cols:
            print(f"  交通相关列: {traffic_cols}")
        # 检查是否包含SHRP_ID
        has_shrp = 'SHRP_ID' in df.columns
        print(f"  包含SHRP_ID: {has_shrp}")
        print(f"  前5行预览:")
        print(df.head().to_string())
        print()

    return xls


def compute_analysis(model_df, traffic_df):
    """
    计算AADTT与MRI的相关性分析

    参数:
        model_df: 模型处理数据的DataFrame
        traffic_df: 交通数据的DataFrame
    """
    print("\n" + "=" * 60)
    print("AADTT相关性分析")
    print("=" * 60)

    # 按SHRP_ID合并
    merged = model_df.merge(traffic_df, on='SHRP_ID', how='inner')
    print(f"合并后记录数: {len(merged)}")
    print(f"合并后唯一路段数: {merged['SHRP_ID'].nunique()}")

    # 1. Pearson相关系数
    print("\n--- 1. Pearson相关系数 ---")
    pearson_r, pearson_p = stats.pearsonr(merged['AADTT'], merged['MRI'])
    print(f"  AADTT vs MRI: r = {pearson_r:.4f}, p = {pearson_p:.6f}")

    # 2. 偏相关系数 (控制IRI_LAG_1)
    print("\n--- 2. 偏相关系数 (控制IRI_LAG_1) ---")
    # 先对AADTT和MRI分别做回归去掉IRI_LAG_1的影响
    X_control = merged[['IRI_LAG_1']].values

    reg_aadtt = LinearRegression().fit(X_control, merged['AADTT'].values)
    aadtt_resid = merged['AADTT'].values - reg_aadtt.predict(X_control)

    reg_mri = LinearRegression().fit(X_control, merged['MRI'].values)
    mri_resid = merged['MRI'].values - reg_mri.predict(X_control)

    partial_r, partial_p = stats.pearsonr(aadtt_resid, mri_resid)
    print(f"  控制IRI_LAG_1后: r = {partial_r:.4f}, p = {partial_p:.6f}")

    # 3. 线性回归R²增量
    print("\n--- 3. 线性回归R²增量 ---")
    # 模型1: 仅用IRI_LAG_1
    X1 = merged[['IRI_LAG_1']].values
    y = merged['MRI'].values
    reg1 = LinearRegression().fit(X1, y)
    r2_1 = reg1.score(X1, y)
    print(f"  模型1 (仅IRI_LAG_1): R² = {r2_1:.4f}")

    # 模型2: IRI_LAG_1 + AADTT
    X2 = merged[['IRI_LAG_1', 'AADTT']].values
    reg2 = LinearRegression().fit(X2, y)
    r2_2 = reg2.score(X2, y)
    print(f"  模型2 (IRI_LAG_1 + AADTT): R² = {r2_2:.4f}")

    r2_increment = r2_2 - r2_1
    print(f"  ΔR² = {r2_increment:.6f}")

    # 4. 描述性统计
    print("\n--- 4. 描述性统计 ---")
    print(f"  AADTT均值: {merged['AADTT'].mean():.0f} 辆/日")
    print(f"  AADTT标准差: {merged['AADTT'].std():.0f}")
    print(f"  AADTT范围: {merged['AADTT'].min():.0f} ~ {merged['AADTT'].max():.0f}")
    print(f"  AADTT中位数: {merged['AADTT'].median():.0f}")

    # 汇总
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    print(f"  Pearson相关系数 (AADTT vs MRI): r = {pearson_r:.4f}")
    print(f"  偏相关系数 (控制IRI_LAG_1): r = {partial_r:.4f}")
    print(f"  线性回归R²增量 (加入AADTT): ΔR² = {r2_increment:.6f}")

    return {
        'pearson_r': float(f"{pearson_r:.4f}"),
        'pearson_p': float(f"{pearson_p:.6f}"),
        'partial_r': float(f"{partial_r:.4f}"),
        'partial_p': float(f"{partial_p:.6f}"),
        'r2_increment': float(f"{r2_increment:.6f}"),
        'n_samples': len(merged),
        'n_sections': merged['SHRP_ID'].nunique()
    }


def main():
    """主函数"""

    # Step 1: 先探索xlsx结构
    xls = explore_xlsx()

    print("\n" + "=" * 60)
    print("加载交通数据（TRF_TREND sheet）")
    print("=" * 60)

    # 直接从TRF_TREND sheet读取
    traffic_raw = pd.read_excel(TRAFFIC_PATH, sheet_name='TRF_TREND')
    print(f"  原始交通数据行数: {len(traffic_raw)}")
    print(f"  列名: {list(traffic_raw.columns)}")

    # 提取AADTT数据
    traffic_clean = traffic_raw[['SHRP_ID', 'YEAR', 'AADTT_ALL_TRUCKS_TREND']].copy()
    traffic_clean = traffic_clean.rename(columns={'AADTT_ALL_TRUCKS_TREND': 'AADTT'})
    traffic_clean['SHRP_ID'] = traffic_clean['SHRP_ID'].astype(str)
    traffic_clean = traffic_clean.dropna(subset=['AADTT'])
    # 过滤AADTT=0的无效记录
    traffic_clean = traffic_clean[traffic_clean['AADTT'] > 0]
    print(f"  有效交通记录数: {len(traffic_clean)}")
    print(f"  有交通数据的路段数: {traffic_clean['SHRP_ID'].nunique()}")

    # 按路段聚合：取各年度AADTT的均值
    traffic_agg = traffic_clean.groupby('SHRP_ID')['AADTT'].mean().reset_index()
    print(f"  聚合后路段数: {len(traffic_agg)}")
    print(f"  平均AADTT: {traffic_agg['AADTT'].mean():.0f} 辆/日")

    # Step 3: 加载模型数据
    print("\n" + "=" * 60)
    print("加载模型处理数据")
    print("=" * 60)
    model_df = pd.read_csv(PROCESSED_DATA_PATH, low_memory=False)
    print(f"  模型数据行数: {len(model_df)}")
    print(f"  MRI均值: {model_df['MRI'].mean():.2f}")
    print(f"  IRI_LAG_1均值: {model_df['IRI_LAG_1'].mean():.2f}")
    print(f"  唯一路段数: {model_df['SHRP_ID'].nunique()}")

    # Step 4: 执行分析
    print(f"\n  交通数据与模型数据匹配路段数: "
          f"{len(set(traffic_agg['SHRP_ID']) & set(model_df['SHRP_ID'].astype(str)))}")

    results = compute_analysis(model_df, traffic_agg)

    return results


if __name__ == '__main__':
    results = main()
