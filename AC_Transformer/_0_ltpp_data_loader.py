"""
LTPP数据加载器
直接从LSTM项目复用数据
"""
import os
import sys

# 链接到LSTM的数据加载器
ltpp_loader_path = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/AC_LSTM/_0_ltpp_data_loader.py'
exec(open(ltpp_loader_path, encoding='utf-8').read())