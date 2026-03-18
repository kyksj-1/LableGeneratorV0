import time
import polars as pl
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader.daily_loader import DailyDataLoader
from src.utils.logger import logger

def main():
    print("开始测试数据加载性能...")
    
    # 1. 实例化 DailyDataLoader (默认加载 10 个字段)
    loader = DailyDataLoader()
    
    # 2. 获取懒加载计算图
    lf = loader.load_lazy()
    print("已生成计算图:", lf)
    
    # 3. 测量完全加载到内存的时间 (触发 action: collect)
    start_time = time.time()
    df = lf.collect()
    end_time = time.time()
    
    load_time = end_time - start_time
    print(f"成功将全部数据加载到 DataFrame 中!")
    print(f"加载耗时: {load_time:.4f} 秒")
    print(f"数据总行数: {df.height}, 总列数: {df.width}")
    print("数据预览:")
    print(df.head())
    
    # 4. 编写并测试 polars 筛选表达式
    # 假设我们要筛选:
    # - 2020年以后的数据
    # - code 包含 "600" 的股票
    # - pe_ratio 大于 0 (剔除亏损)
    # - turnover_rate 大于 0.05 (活跃度较高)
    print("\n开始测试基于 Polars 的表达式筛选 (Predicate Pushdown 测试)...")
    
    filter_expr = (
        (pl.col("date") >= "2020-01-01") & 
        (pl.col("code").cast(pl.Utf8).str.starts_with("600")) & 
        (pl.col("pe_ratio") > 0) & 
        (pl.col("turnover_rate") > 0.05)
    )
    
    # 我们也可以直接在 loader 中传入过滤表达式，利用懒加载的谓词下推
    filtered_loader = DailyDataLoader(data_footprint_filter=filter_expr)
    filtered_lf = filtered_loader.load_lazy()
    
    start_time_filter = time.time()
    filtered_df = filtered_lf.collect()
    end_time_filter = time.time()
    
    filter_time = end_time_filter - start_time_filter
    print(f"过滤后数据加载耗时: {filter_time:.4f} 秒")
    print(f"过滤后数据总行数: {filtered_df.height}")
    print("过滤后数据预览:")
    print(filtered_df.head())

if __name__ == "__main__":
    main()
