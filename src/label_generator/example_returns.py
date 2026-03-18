import polars as pl
from .base import LabelGenerator

class NextNDaysReturnLabel(LabelGenerator):
    """
    计算未来 N 天的连续收益率标签 (回归)
    """
    def __init__(self, n_days: int = 5, label_name: str = "target_ret_5d"):
        super().__init__(label_name=label_name, is_discrete=False)
        self.n_days = n_days

    def generate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        利用 Polars 的 window function (over 'code') 实现组内时序偏移计算
        """
        # 注意：数据应该先按 date 排序，保证 shift 操作是按时间顺序
        lf = lf.sort(["code", "date"])
        
        return lf.with_columns(
            (
                pl.col("close").shift(-self.n_days).over("code") / pl.col("close") - 1.0
            ).alias(self.label_name)
        )
