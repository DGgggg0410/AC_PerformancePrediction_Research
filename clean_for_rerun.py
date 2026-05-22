"""
======================================================================
清理脚本：删除所有已训练的输出文件，保留源代码和源数据
======================================================================
作用：删除模型权重、评估报告、预测结果、图片、缩放器等所有训练产物，
      保留 LSTM/Transformer/消融实验的 .py 源代码、源数据（.mdb/.accdb/.xlsx）、
      Markdown/.docx 文档等，便于从头重新运行整个流程。
======================================================================
"""
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ====================================================================
# 待删除的路径清单
# ====================================================================
TARGETS = [
    # ── LSTM 全部输出 ──
    "AC_LSTM/output",

    # ── Transformer 全部输出 ──
    "AC_Transformer/output",

    # ── 预处理后的数据（数据加载器已修改，需重新生成） ──
    "processed_data",

    # ── 快速测试临时输出 ──
    "output_quick_test",

    # ── 消融实验汇总图表 ──
    "ablation_figures",
    "ablation_figures_chinese",
    "pdf_content.txt",
]

# ====================================================================
# 辅助工具
# ====================================================================
def format_size(path: str) -> str:
    """计算路径总大小（目录递归）"""
    if os.path.isfile(path):
        size = os.path.getsize(path)
    elif os.path.isdir(path):
        size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    size += os.path.getsize(fp)
                except OSError:
                    pass
    else:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def scan_targets(abs_paths: list) -> list:
    """只返回真实存在的路径"""
    existing = []
    for p in abs_paths:
        if os.path.exists(p):
            existing.append(p)
    return existing


def delete_path(path: str):
    """删除文件或目录"""
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        return True
    except Exception as e:
        print(f"  [失败] {path} → {e}")
        return False


# ====================================================================
# 主流程
# ====================================================================
def main():
    abs_targets = [os.path.join(PROJECT_ROOT, t) for t in TARGETS]
    existing = scan_targets(abs_targets)

    if not existing:
        print("=" * 60)
        print("  没有找到任何待删除的输出文件，项目已处于干净状态。")
        print("=" * 60)
        return

    print("=" * 60)
    print("  以下文件/目录将被删除（不可恢复）：")
    print("=" * 60)
    total_size = 0
    for p in existing:
        rel = os.path.relpath(p, PROJECT_ROOT)
        sz = format_size(p)
        print(f"  • {rel}  ({sz})")
    print("-" * 60)

    # 确认
    confirm = input("  确认删除以上所有内容？(yes/no): ").strip().lower()
    if confirm not in ("yes", "y"):
        print("  操作已取消。")
        return

    # 执行删除
    success_count = 0
    fail_count = 0
    for p in existing:
        rel = os.path.relpath(p, PROJECT_ROOT)
        if delete_path(p):
            print(f"  [已删除] {rel}")
            success_count += 1
        else:
            fail_count += 1

    # 结果汇总
    print("=" * 60)
    print(f"  清理完成：成功删除 {success_count} 项", end="")
    if fail_count:
        print(f"，失败 {fail_count} 项", end="")
    print("")
    print("=" * 60)
    print("  现在可以直接从头运行流程：")
    print(f"  1. python AC_LSTM/_0_ltpp_data_loader.py")
    print(f"  2. python AC_LSTM/_2_sequence_builder.py")
    print(f"  3. python AC_LSTM/_5_trainer.py")
    print(f"  4. python AC_LSTM/_6_predictor.py")
    print(f"  5. python AC_LSTM/_7_shap_analyzer.py")
    print(f"  6. python AC_Transformer/_2_sequence_builder.py")
    print(f"  7. python AC_Transformer/_5_trainer.py")
    print(f"  8. python AC_Transformer/_6_predictor.py")
    print(f"  9. python AC_Transformer/_7_shap_analyzer.py")
    print(f" 10. python ablation_analysis_cn.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
