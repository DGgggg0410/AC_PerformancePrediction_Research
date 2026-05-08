"""自动生成的配置更新脚本
将最佳超参数应用到 _1_config.py
"""
import re

CONFIG_PATH = r'e:/Visual Studio Code2025/python_program/AC_PerformancePrediction_Research/AC_LSTM/_1_config.py'

# 最佳超参数
BEST_PARAMS = {'HIDDEN_DIM': 256, 'NUM_LAYERS': 2, 'DROPOUT': 0.1, 'LEARNING_RATE': 0.001, 'BATCH_SIZE': 256}

def update_config():
    """更新配置文件"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换各超参数
    replacements = [
        ('HIDDEN_DIM = \\d+', 'HIDDEN_DIM = 256'),
        ('NUM_LAYERS = \\d+', 'NUM_LAYERS = 2'),
        ('DROPOUT = 0\\.\\d+', 'DROPOUT = 0.1'),
        ('LEARNING_RATE = 0\\.\\d+', 'LEARNING_RATE = 0.001'),
        ('BATCH_SIZE = \\d+', 'BATCH_SIZE = 256'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(content)

    print("配置已更新!")
    print("最佳超参数:")
    for k, v in BEST_PARAMS.items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    update_config()
