"""
LTPP数据加载器
从Access数据库提取IRI、气候、坐标、结构层数据并进行关联

功能说明：
1. 从主数据库(Bucket_141347.mdb)加载IRI监测数据、路段信息、地理坐标和结构层厚度
2. 从气候数据库(Bucket_141348_1.accdb)加载气候数据
3. 对各数据源进行关联和整合
4. 构建IRI滞后特征(历史IRI值)
5. 对分类特征进行编码
6. 处理缺失值并输出清洗后的数据

作者: 研究团队
日期: 2024
"""

import os
import pyodbc  # 用于连接Microsoft Access数据库
import pandas as pd  # 数据处理库
import numpy as np  # 数值计算库
from datetime import datetime  # 日期时间处理

# ============================================================================
# 数据库路径配置
# ============================================================================

# 数据目录路径
DATA_DIR = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/data_LTPP'

# 主数据库路径 - 包含IRI、路段信息、坐标、结构层等核心数据
MDB_PATH = os.path.join(DATA_DIR, 'Bucket_141347.mdb')

# 气候数据数据库路径 - 包含MERRA气象再分析数据
ACCDB_CLIMATE_PATH = os.path.join(DATA_DIR, 'Bucket_141348_1.accdb')

# 输出目录 - 存放处理后的数据
OUTPUT_DIR = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/processed_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 如果目录不存在则创建


# ============================================================================
# 数据库连接函数
# ============================================================================

def get_db_connection(db_path):
    """
    获取Access数据库连接

    参数:
        db_path: 数据库文件路径

    返回:
        pyodbc连接对象
    """
    # 构建ODBC连接字符串，使用Access ODBC驱动
    conn_str = r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' + db_path + ';'
    return pyodbc.connect(conn_str)


# ============================================================================
# 数据加载函数 - 从数据库读取原始数据
# ============================================================================

def load_iri_data(conn):
    """
    加载IRI(国际平整度指数)监测数据

    IRI是评估路面行驶质量的关键指标，单位为m/km
    数值越小表示路面越平整

    返回的DataFrame包含:
        - SHRP_ID: 路段唯一标识符
        - STATE_CODE: 州代码
        - VISIT_DATE: 监测日期
        - CONSTRUCTION_NO: 施工编号(用于区分同一路段的不同施工段)
        - MRI: 路面平整度指数(均方根IRI)
        - IRI_LEFT_WHEEL_PATH/IRI_RIGHT_WHEEL_PATH: 左右轮迹带IRI
        - PAVEMENT_FAMILY/PAVEMENT_FAMILY_EXP: 路面结构家族及其描述
    """
    print("加载IRI数据...")
    sql = """
    SELECT
        SHRP_ID,
        STATE_CODE,
        VISIT_DATE,
        CONSTRUCTION_NO,
        MRI,
        IRI_LEFT_WHEEL_PATH,
        IRI_RIGHT_WHEEL_PATH,
        PAVEMENT_FAMILY,
        PAVEMENT_FAMILY_EXP
    FROM ANALYSIS_IRI
    WHERE MRI IS NOT NULL
    ORDER BY SHRP_ID, VISIT_DATE
    """
    df = pd.read_sql(sql, conn)  # 执行SQL查询并转换为DataFrame
    df['VISIT_DATE'] = pd.to_datetime(df['VISIT_DATE'])  # 将日期字符串转换为datetime类型
    print(f"  IRI记录数: {len(df)}")  # 打印加载的记录数量
    return df


def load_climate_data(conn):
    """
    加载气候数据(来自MERRA气象再分析数据)

    气候因素对路面性能有重要影响：
        - 度日数影响沥青老化
        - 极端温度影响路面开裂
        - 地表温度影响路面结构稳定性
        - 冻融循环影响路面疲劳损坏

    返回的DataFrame包含:
        - SHRP_ID: 路段标识符
        - DEGREE_DAYS_OVER_10C_YR: 年度度日数(大于10°C的累积温度)
        - COLDEST_AIR_TEMP: 最冷空气温度
        - HIGH_TEMP_7DAYS: 连续7天最高温度
        - MAX_AVG_TEMP_HIGH_7DAYS: 7天最高平均温度
        - MAX_TEMP_HIGH_7DAYS: 最高地表温度
        - MIN_SURFACE_50_TEMP: 最低地表温度(50%分位)
        - MIN_SURFACE_98_TEMP: 最低地表温度(98%分位)
        - STDDEV_HIGH_TEMP_7DAYS: 7天高温标准差
        - STDDEV_COLDEST_AIR_TEMP: 冬季低温标准差
    """
    print("加载气候数据...")
    sql = """
    SELECT
        SHRP_ID,
        STATE_CODE,
        DEGREE_DAYS_OVER_10C_YR,
        COLDEST_AIR_TEMP,
        HIGH_TEMP_7DAYS,
        MAX_AVG_TEMP_HIGH_7DAYS,
        MAX_TEMP_HIGH_7DAYS,
        MIN_SURFACE_50_TEMP,
        MIN_SURFACE_98_TEMP,
        STDDEV_HIGH_TEMP_7DAYS,
        STDDEV_COLDEST_AIR_TEMP
    FROM VW_MERRA_BIND_CLIMATE_DATA
    """
    df = pd.read_sql(sql, conn)
    print(f"  气候数据记录数: {len(df)}")
    return df


def load_year_climate_data(conn):
    """
    加载年尺度气候数据（文献补充特征）

    新增特征：
        - FREEZE_INDEX: 年冷冻指数（度日）
        - FREEZE_THAW: 年冻融天数
        - PRECIPITATION: 年降水量(mm)
        - PRECIP_DAYS: 年降水天数
        - EVAPORATION: 年蒸发量(mm)

    这些特征在文献中提到，对路面性能有重要影响：
        - 冻融循环导致路面内部水分结冰膨胀，引起裂缝
        - 降水量影响路基含水量和强度
    """
    print("加载年尺度气候数据...")
    # 先获取SHRP_ID到MERRA_ID的映射
    print("  获取MERRA_ID映射...")
    mapping_sql = """
    SELECT DISTINCT SHRP_ID, MERRA_ID
    FROM VW_MERRA_BIND_CLIMATE_DATA
    WHERE MERRA_ID IS NOT NULL
    """
    mapping_df = pd.read_sql(mapping_sql, conn)
    print(f"    MERRA_ID映射数: {len(mapping_df)}")

    # 加载年温度数据
    print("  加载年温度数据...")
    temp_sql = """
    SELECT
        MERRA_ID,
        YEAR,
        FREEZE_INDEX,
        FREEZE_THAW
    FROM VW_MERRA_TEMP_YEAR
    """
    temp_df = pd.read_sql(temp_sql, conn)
    print(f"    年温度记录数: {len(temp_df)}")

    # 加载年降水量数据
    print("  加载年降水量数据...")
    precip_sql = """
    SELECT
        MERRA_ID,
        YEAR,
        PRECIPITATION,
        PRECIP_DAYS,
        EVAPORATION
    FROM VW_MERRA_PRECIP_YEAR
    """
    precip_df = pd.read_sql(precip_sql, conn)
    print(f"    年降水量记录数: {len(precip_df)}")

    # 合并年温度和降水量
    year_df = temp_df.merge(precip_df, on=['MERRA_ID', 'YEAR'], how='outer')

    # 关联SHRP_ID
    year_df = year_df.merge(mapping_df, on='MERRA_ID', how='left')

    # 按SHRP_ID聚合（取均值）
    agg_df = year_df.groupby('SHRP_ID').agg({
        'FREEZE_INDEX': 'mean',    # 年均冷冻指数
        'FREEZE_THAW': 'mean',     # 年均冻融天数
        'PRECIPITATION': 'mean',   # 年均降水量
        'PRECIP_DAYS': 'mean',     # 年均降水天数
        'EVAPORATION': 'mean'       # 年均蒸发量
    }).reset_index()

    print(f"  年气候数据汇总后记录数: {len(agg_df)}")
    return agg_df


def load_section_info(conn):
    """
    加载路段基本信息

    返回的DataFrame包含:
        - SHRP_ID: 路段唯一标识符
        - START_DATE: 路面建设/重建日期
        - FUNC_CLASS/FUNC_CLASS_EXP: 道路功能分类(高速公路、主干道等)
        - SRO_CLASS/SRO_CLASS_EXP: 路面状况评级分类
    """
    print("加载路段信息...")
    sql = """
    SELECT
        SHRP_ID,
        START_DATE,
        FUNC_CLASS,
        FUNC_CLASS_EXP,
        SRO_CLASS,
        SRO_CLASS_EXP
    FROM SHRP_INFO
    """
    df = pd.read_sql(sql, conn)
    df['START_DATE'] = pd.to_datetime(df['START_DATE'])
    print(f"  路段信息记录数: {len(df)}")
    return df


def load_coordinates(conn):
    """
    加载路段地理坐标

    地理因素影响路面性能：
        - 纬度影响气候条件(冻融循环次数)
        - 海拔影响温度和降水
        - 经纬度用于空间分析和可视化

    返回的DataFrame包含:
        - SHRP_ID: 路段标识符
        - LATITUDE: 纬度
        - LONGITUDE: 经度
        - ELEVATION: 海拔高度(m)
    """
    print("加载地理坐标...")
    sql = """
    SELECT
        SHRP_ID,
        LATITUDE,
        LONGITUDE,
        ELEVATION
    FROM SECTION_COORDINATES
    """
    df = pd.read_sql(sql, conn)
    print(f"  坐标数据记录数: {len(df)}")
    return df


def load_layer_thickness(conn):
    """
    加载路面结构层厚度信息

    路面结构层组成(从上到下):
        - 沥青混凝土层(AC): 直接承受车辆荷载
        - 基层(Base): 分散荷载
        - 底基层(Subbase): 进一步分散荷载
        - 路基(Subgrade): 土壤基础

    厚度是影响路面耐久性的关键因素

    返回的DataFrame包含:
        - SHRP_ID: 路段标识符
        - CONSTRUCTION_NO: 施工编号
        - LAYER_NO: 层编号
        - LAYER_TYPE/LAYER_TYPE_EXP: 层类型及其描述
        - REPR_THICKNESS: 代表性厚度(mm)
    """
    print("加载结构层厚度...")
    # 不过滤RECORD_STATUS，获取所有记录以确保完整性
    sql = """
    SELECT
        SHRP_ID,
        CONSTRUCTION_NO,
        LAYER_NO,
        LAYER_TYPE,
        LAYER_TYPE_EXP,
        REPR_THICKNESS
    FROM TST_L05B
    """
    df = pd.read_sql(sql, conn)
    print(f"  结构层记录数: {len(df)}")
    return df


# ============================================================================
# 数据聚合和特征构建函数
# ============================================================================

def aggregate_layer_thickness(layer_df):
    """
    汇总路面结构层厚度

    处理逻辑:
        1. 按路段(SHRP_ID)和施工编号(CONSTRUCTION_NO)分组
        2. 计算总厚度和层数
        3. 根据层类型关键词分别计算沥青层和基层厚度

    参数:
        layer_df: 原始结构层数据

    返回:
        包含汇总后厚度信息的DataFrame:
            - TOTAL_THICKNESS: 总厚度
            - NUM_LAYERS: 结构层数量
            - AC_THICKNESS: 沥青层厚度
            - BASE_THICKNESS: 基层厚度
    """
    print("汇总结构层厚度...")

    # 处理空数据情况
    if len(layer_df) == 0:
        return pd.DataFrame(columns=['SHRP_ID', 'CONSTRUCTION_NO', 'TOTAL_THICKNESS', 'NUM_LAYERS', 'AC_THICKNESS', 'BASE_THICKNESS'])

    # 按路段和施工编号汇总
    agg_df = layer_df.groupby(['SHRP_ID', 'CONSTRUCTION_NO']).agg({
        'REPR_THICKNESS': 'sum',  # 各层厚度求和得到总厚度
        'LAYER_NO': 'count'       # 统计层数
    }).reset_index()
    agg_df.columns = ['SHRP_ID', 'CONSTRUCTION_NO', 'TOTAL_THICKNESS', 'NUM_LAYERS']

    # 根据LAYER_TYPE_EXP中的关键词识别不同层类型
    # 沥青层关键词：asphalt, ac, bituminous
    ac_keywords = ['asphalt', 'ac', 'bituminous']
    # 基层关键词：base, subbase, sub grade
    base_keywords = ['base', 'subbase', 'sub grade']

    def get_layer_thickness(layer_df, keywords):
        """
        根据关键词筛选层并计算厚度

        参数:
            layer_df: 结构层数据
            keywords: 匹配关键词列表

        返回:
            按路段和施工编号汇总的厚度数据
        """
        # 将层类型描述转为小写进行匹配
        mask = layer_df['LAYER_TYPE_EXP'].str.lower().str.contains('|'.join(keywords), na=False)
        filtered = layer_df[mask]  # 筛选匹配的层
        # 按路段和施工编号汇总厚度
        result = filtered.groupby(['SHRP_ID', 'CONSTRUCTION_NO'])['REPR_THICKNESS'].sum().reset_index()
        result.columns = ['SHRP_ID', 'CONSTRUCTION_NO', 'THICKNESS']
        return result

    # 分别计算沥青层和基层厚度
    ac_thickness = get_layer_thickness(layer_df, ac_keywords)
    ac_thickness.columns = ['SHRP_ID', 'CONSTRUCTION_NO', 'AC_THICKNESS']

    base_thickness = get_layer_thickness(layer_df, base_keywords)
    base_thickness.columns = ['SHRP_ID', 'CONSTRUCTION_NO', 'BASE_THICKNESS']

    # 合并到主表
    agg_df = agg_df.merge(ac_thickness, on=['SHRP_ID', 'CONSTRUCTION_NO'], how='left')
    agg_df = agg_df.merge(base_thickness, on=['SHRP_ID', 'CONSTRUCTION_NO'], how='left')

    # 填充缺失值为0(某类层可能不存在)
    agg_df['AC_THICKNESS'] = agg_df['AC_THICKNESS'].fillna(0)
    agg_df['BASE_THICKNESS'] = agg_df['BASE_THICKNESS'].fillna(0)

    print(f"  汇总后路段数: {len(agg_df)}")
    return agg_df


def build_pavement_age(iri_df, section_df):
    """
    计算路面龄期

    路面龄期是影响IRI的重要特征：
        - 新路面通常平整度好
        - 随着时间推移，路面会老化、产生车辙和裂缝
        - 龄期与交通量、环境因素共同影响路面性能

    计算方法: 监测日期 - 路面建设日期

    参数:
        iri_df: IRI监测数据
        section_df: 路段信息(包含START_DATE)

    返回:
        添加了PAVEMENT_AGE列的IRI DataFrame
    """
    print("计算路面龄期...")
    # 合并获取START_DATE
    merged = iri_df.merge(
        section_df[['SHRP_ID', 'START_DATE']],  # 只保留SHRP_ID和START_DATE
        on='SHRP_ID',
        how='left'
    )
    # 计算龄期(天)转换为年
    merged['PAVEMENT_AGE'] = (merged['VISIT_DATE'] - merged['START_DATE']).dt.days / 365.25
    # 确保龄期非负(处理数据中的异常情况)
    merged['PAVEMENT_AGE'] = merged['PAVEMENT_AGE'].clip(lower=0)
    return merged


def build_iri_lags(iri_df, n_lags=2):
    """
    构建IRI历史特征(滞后特征)

    滞后特征是时序预测的关键：
        - IRI_LAG_1: 去年IRI值
        - IRI_LAG_2: 前年IRI值

    时序依赖性：
        - 路面性能具有时序相关性
        - 过去的状态可以预测未来的状态
        - 例如：去年IRI高的路段今年通常也较高

    参数:
        iri_df: IRI数据
        n_lags: 滞后阶数(默认2年)

    返回:
        添加了IRI_LAG_1, IRI_LAG_2列的DataFrame
    """
    print(f"构建IRI滞后特征 (lag={n_lags})...")
    # 确保数据按路段和时间排序
    df = iri_df.sort_values(['SHRP_ID', 'VISIT_DATE']).copy()

    # 构建滞后特征
    for lag in range(1, n_lags + 1):
        # shift(lag)将MRI向上移动lag行，实现获取历史值的效果
        df[f'IRI_LAG_{lag}'] = df.groupby('SHRP_ID')['MRI'].shift(lag)

    # 用组内均值填充第一个观测的滞后值(路段的第一个记录没有历史数据)
    iri_mean = df['MRI'].mean()  # 全局均值作为备选
    for lag in range(1, n_lags + 1):
        # 先用0填充，然后每个路段用该路段均值填充，最后用全局均值填充剩余
        df[f'IRI_LAG_{lag}'] = df.groupby('SHRP_ID')[f'IRI_LAG_{lag}'].transform(
            lambda x: x.fillna(iri_mean)
        )

    return df


def encode_categorical(df):
    """
    对分类特征进行数值编码

    机器学习模型需要数值输入，因此将分类变量转为数值：
        - PAVEMENT_FAMILY: 路面结构家族
        - FUNC_CLASS: 道路功能分类

    使用标签编码：将类别映射到0, 1, 2, ...

    参数:
        df: 原始数据

    返回:
        添加了编码后列的DataFrame
    """
    print("编码分类特征...")

    # PAVEMENT_FAMILY编码 - 路面结构家族
    if 'PAVEMENT_FAMILY' in df.columns:
        # 创建从类别到整数的映射
        pave_mapping = {v: i for i, v in enumerate(df['PAVEMENT_FAMILY'].dropna().unique())}
        df['PAVEMENT_FAMILY_ENC'] = df['PAVEMENT_FAMILY'].map(pave_mapping)
        print(f"  路面结构类型数: {len(pave_mapping)}")

    # FUNC_CLASS编码 - 道路功能分类
    if 'FUNC_CLASS' in df.columns:
        func_mapping = {v: i for i, v in enumerate(df['FUNC_CLASS'].dropna().unique())}
        df['FUNC_CLASS_ENC'] = df['FUNC_CLASS'].map(func_mapping)
        print(f"  道路功能分类数: {len(func_mapping)}")

    return df


# ============================================================================
# 主函数 - 数据整合流程
# ============================================================================

def main():
    """
    主函数：加载并整合所有数据

    完整流程:
        1. 连接主数据库和气候数据库
        2. 加载各模块数据
        3. 数据整合:
            - 汇总结构层厚度
            - 计算路面龄期
            - 构建IRI滞后特征
            - 合并坐标数据
            - 合并气候数据(按路段聚合)
            - 合并结构层厚度
            - 编码分类特征
        4. 数据清洗:
            - 删除高缺失行
            - 用中位数填充剩余缺失值
        5. 保存处理后的数据

    返回:
        处理完成的DataFrame
    """
    print("=" * 60)
    print("LTPP数据加载器")
    print("=" * 60)

    # 连接主数据库
    print("\n连接主数据库...")
    conn_mdb = get_db_connection(MDB_PATH)

    # 加载基础数据
    iri_df = load_iri_data(conn_mdb)  # IRI监测数据
    section_df = load_section_info(conn_mdb)  # 路段信息
    coords_df = load_coordinates(conn_mdb)  # 地理坐标
    layer_df = load_layer_thickness(conn_mdb)  # 结构层厚度

    conn_mdb.close()  # 关闭连接

    # 连接气候数据库
    print("\n连接气候数据库...")
    conn_climate = get_db_connection(ACCDB_CLIMATE_PATH)
    climate_df = load_climate_data(conn_climate)

    # 加载年尺度气候数据（文献补充特征）
    year_climate_df = load_year_climate_data(conn_climate)
    conn_climate.close()

    # 数据整合
    print("\n" + "=" * 60)
    print("数据整合")
    print("=" * 60)

    # 1. 汇总结构层厚度
    layer_agg = aggregate_layer_thickness(layer_df)

    # 2. 计算路面龄期
    df = build_pavement_age(iri_df, section_df)

    # 3. 构建IRI滞后特征
    df = build_iri_lags(df, n_lags=2)

    # 4. 合并坐标
    print("合并坐标数据...")
    # 同一路段可能有多条坐标记录，只保留第一条
    coords_unique = coords_df.drop_duplicates(subset='SHRP_ID', keep='first')
    df = df.merge(coords_unique, on='SHRP_ID', how='left')

    # 5. 合并气候数据（通过SHRP_ID，按路段聚合气候特征取均值）
    # 注意：同一路段可能有多年的气候数据，取均值代表该路段的气候特征
    print("合并气候数据...")
    climate_agg = climate_df.groupby('SHRP_ID').agg({
        'DEGREE_DAYS_OVER_10C_YR': 'mean',  # 年度度日数均值
        'COLDEST_AIR_TEMP': 'mean',  # 最冷气温均值
        'HIGH_TEMP_7DAYS': 'mean',  # 最高7日气温均值
        'MAX_AVG_TEMP_HIGH_7DAYS': 'mean',  # 7天最高平均气温均值
        'MAX_TEMP_HIGH_7DAYS': 'mean',  # 最高地表温度均值
        'MIN_SURFACE_50_TEMP': 'mean',  # 最低地表温度(50分位)均值
        'MIN_SURFACE_98_TEMP': 'mean',  # 最低地表温度(98分位)均值
        'STDDEV_HIGH_TEMP_7DAYS': 'mean',  # 高温标准差均值
        'STDDEV_COLDEST_AIR_TEMP': 'mean'  # 低温标准差均值
    }).reset_index()
    df = df.merge(climate_agg, on='SHRP_ID', how='left')

    # 6. 合并年尺度气候数据（文献补充特征：冻融、降水量）
    print("合并年尺度气候数据...")
    df = df.merge(year_climate_df, on='SHRP_ID', how='left')

    # 6. 合并结构层厚度（通过SHRP_ID和CONSTRUCTION_NO）
    print("合并结构层厚度...")
    df = df.merge(layer_agg, on=['SHRP_ID', 'CONSTRUCTION_NO'], how='left')

    # 7. 编码分类特征
    df = encode_categorical(df)

    # 选择最终特征列
    feature_cols = [
        'SHRP_ID', 'VISIT_DATE', 'CONSTRUCTION_NO',  # 标识列
        'MRI',  # 目标变量
        'PAVEMENT_AGE',  # 路面龄期
        'IRI_LAG_1', 'IRI_LAG_2',  # IRI滞后特征
        'PAVEMENT_FAMILY_ENC',  # 路面结构类型编码
        # 注意: FUNC_CLASS_ENC 未被模型使用，已移除
        'LATITUDE', 'LONGITUDE', 'ELEVATION',  # 地理特征
        # 基本气候特征
        'DEGREE_DAYS_OVER_10C_YR',  # 年度度日数
        'COLDEST_AIR_TEMP',  # 最冷气温
        'HIGH_TEMP_7DAYS',  # 最高7日气温
        'MIN_SURFACE_50_TEMP',  # 最低地表温度
        # 文献补充气候特征（冻融、降水）
        'FREEZE_INDEX',  # 年冷冻指数
        'FREEZE_THAW',  # 年冻融天数
        'PRECIPITATION',  # 年降水量
        'PRECIP_DAYS',  # 年降水天数
        'EVAPORATION',  # 年蒸发量
        # 结构特征
        'TOTAL_THICKNESS',  # 总路面厚度
        'AC_THICKNESS',  # 沥青层厚度
        'BASE_THICKNESS',  # 基层厚度
        'NUM_LAYERS'  # 结构层数量
    ]

    # 只保留存在的列(容错处理)
    available_cols = [c for c in feature_cols if c in df.columns]
    df_final = df[available_cols].copy()

    # 数据清洗：删除高缺失行
    print(f"\n原始数据行数: {len(df_final)}")
    initial_rows = len(df_final)

    # 计算每行的缺失值比例
    missing_ratio = df_final.isnull().sum(axis=1) / len(df_final.columns)
    # 删除缺失超过30%的行
    df_final = df_final[missing_ratio < 0.3]

    print(f"删除高缺失行后: {len(df_final)} (删除了 {initial_rows - len(df_final)} 行)")

    # 用中位数填充剩余缺失值
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_final[col].isnull().any():
            median_val = df_final[col].median()  # 计算中位数
            if pd.isna(median_val):  # 处理全为NaN的情况
                median_val = 0
            df_final[col] = df_final[col].fillna(median_val)

    # 保存处理后的数据
    output_path = os.path.join(OUTPUT_DIR, 'ltpp_processed_data.csv')
    df_final.to_csv(output_path, index=False)
    print(f"\n数据已保存到: {output_path}")

    # 数据统计信息
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    print(f"总样本数: {len(df_final)}")
    print(f"唯一路段数: {df_final['SHRP_ID'].nunique()}")
    print(f"特征数: {len(df_final.columns) - 4}")  # 减去ID、日期、施工编号、目标变量
    print(f"\n特征列:")
    for col in df_final.columns:
        print(f"  - {col}")

    # MRI统计
    print(f"\nMRI统计:")
    print(f"  范围: {df_final['MRI'].min():.2f} ~ {df_final['MRI'].max():.2f}")
    print(f"  均值: {df_final['MRI'].mean():.2f}")

    print("\n" + "=" * 60)
    print("数据加载完成!")
    print("=" * 60)

    return df_final


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == '__main__':
    # 执行主函数
    df = main()
