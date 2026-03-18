import sys
import os
from pathlib import Path
import polars as pl

# 确保能找到 src 和 config
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_loader.daily_loader import DailyDataLoader
from src.label_generator.example_returns import NextNDaysReturnLabel

def test_loader_and_label():
    print("1. 初始化 DataLoader...")
    # 我们只需要 2023 年的数据，并且只要 000021 和 000099
    # 如果原始文件中的 date 是字符串类型，需要将过滤条件也设为字符串
    filter_expr = (pl.col("date") >= "2023-01-01") & (pl.col("code").is_in(["000021", "000099"]))
    
    loader = DailyDataLoader(
        data_footprint_filter=filter_expr,
        columns=["date", "code", "close", "volume"],
        prefix_code=True
    )
    
    # 2. 获取 LazyFrame (瞬间完成)
    print("2. 扫描数据生成执行计划...")
    lf = loader.load_lazy()
    print("LazyFrame 结构:", lf.collect_schema())
    
    # 3. 接入 Label Generator (瞬间完成)
    print("3. 添加标签计算逻辑...")
    label_gen = NextNDaysReturnLabel(n_days=5)
    lf_with_label = label_gen(lf)
    print("LazyFrame (with label) 结构:", lf_with_label.collect_schema())
    
    # 4. 触发真实计算 (collect)
    print("4. 触发执行 (.collect())，读取硬盘并计算...")
    df_result = lf_with_label.collect()
    
    print("\n--- 最终结果预览 ---")
    print(df_result.head(10))
    print(f"数据总行数: {len(df_result)}")

if __name__ == "__main__":
    test_loader_and_label()
