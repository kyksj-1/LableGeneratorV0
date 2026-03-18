import os
import polars as pl
from typing import Optional, List
from pathlib import Path

from src.data_loader.base import DataLoader
from src.utils.config_manager import config
from src.utils.logger import logger

class DailyDataLoader(DataLoader):
    """
    日频数据加载器，实现对 list of parquet 文件的统一懒加载与过滤
    """
    
    # 默认加载的字段 (根据需求规范)
    DEFAULT_COLUMNS = [
        'date', 'code', 'open', 'close', 'high', 'low', 
        'volume', 'amount', 'turnover_rate', 'pe_ratio'
    ]

    def __init__(
        self, 
        data_footprint_filter: Optional[pl.Expr] = None,
        columns: Optional[List[str]] = None,
        prefix_code: bool = False
    ):
        """
        :param data_footprint_filter: 数据指纹过滤器
        :param columns: 需要加载的列，如果为 None，则使用 DEFAULT_COLUMNS
        :param prefix_code: 是否为 code 列添加前缀 (sh/sz)，默认关闭
        """
        # 如果未指定列，则使用需求中规定的默认列
        actual_columns = columns if columns is not None else self.DEFAULT_COLUMNS
        
        super().__init__(data_footprint_filter, actual_columns)
        self.prefix_code = prefix_code
        self.raw_data_dir = config.raw_data_dir

    def load_lazy(self) -> pl.LazyFrame:
        """
        使用 Polars 的 scan_parquet 实现高性能懒加载
        """
        # Parquet 文件的统配路径
        # Polars 会利用 Rust 引擎并发且延迟地扫描这些文件
        glob_path = str(self.raw_data_dir / "*.parquet")
        
        try:
            # 1. 扫描文件构建查询计划
            lf = pl.scan_parquet(glob_path)
            
            # 2. 列裁剪 (Projection Pushdown)
            # 如果指定了 columns，我们必须确保 'code' 和 'date' 这种指纹列也在其中
            if self.columns is not None:
                # 确保基础主键一定被包含，避免后续 join 报错
                required_cols = {"date", "code"}
                final_cols = list(set(self.columns).union(required_cols))
                lf = lf.select(final_cols)
                
            # 3. 谓词下推 (Predicate Pushdown)
            if self.data_footprint_filter is not None:
                lf = lf.filter(self.data_footprint_filter)
                
            # 4. 可选功能: 股票代码加前缀
            # (如果代码以 6 开头加 sh，否则加 sz，这里用 Polars 的 when.then 表达式)
            if self.prefix_code:
                lf = lf.with_columns(
                    pl.when(pl.col("code").cast(pl.Utf8).str.starts_with("6"))
                    .then(pl.lit("sh") + pl.col("code").cast(pl.Utf8))
                    .otherwise(pl.lit("sz") + pl.col("code").cast(pl.Utf8))
                    .alias("code")
                )
                
            return lf
            
        except Exception as e:
            logger.error(f"加载 Parquet 文件失败，路径 {glob_path}: {e}")
            raise e
