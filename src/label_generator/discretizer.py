"""
离散化组件

职责：
  将连续标签列转换为离散分类标签。
  例如将收益率排序分转为 Top/Mid/Bottom 三分类。

支持的离散化方式：
  - quantile: 等频分箱（基于截面分位数）
  - threshold: 固定阈值分类（如 top 20% = 1, bottom 20% = -1, 中间 = 0）

设计原则：
  - 独立组件，与标签生成和标准化解耦
  - 分箱操作在截面（每天）维度上进行
"""

import polars as pl
from typing import Optional, List

from src.utils.logger import logger


class Discretizer:
    """
    标签离散化器

    将连续值标签转为离散分类标签。

    使用示例:
        # 三分类: top/mid/bottom
        disc = Discretizer(method="quantile", n_bins=3)
        lf = disc.transform(lf, source_col="ret_open1_close5_rank")

        # 自定义阈值: top 20% = 1, bottom 20% = -1
        disc = Discretizer(method="threshold", top_pct=0.2, bottom_pct=0.2)
        lf = disc.transform(lf, source_col="ret_open1_close5_rank")
    """

    VALID_METHODS = {"quantile", "threshold"}

    def __init__(
        self,
        method: str = "quantile",
        n_bins: int = 3,
        top_pct: float = 0.2,
        bottom_pct: float = 0.2,
        date_col: str = "date",
        suffix: Optional[str] = None,
    ):
        """
        :param method: 离散化方式 ("quantile" / "threshold")
        :param n_bins: 分箱数（仅 quantile 模式，默认 3）
        :param top_pct: 顶部百分比阈值（仅 threshold 模式，默认 0.2）
        :param bottom_pct: 底部百分比阈值（仅 threshold 模式，默认 0.2）
        :param date_col: 日期列名（截面分组依据）
        :param suffix: 输出列名后缀
        """
        assert method in self.VALID_METHODS, \
            f"method 须为 {self.VALID_METHODS} 之一, 收到: {method}"
        assert n_bins >= 2, f"n_bins 须 >= 2, 收到: {n_bins}"
        assert 0 < top_pct < 1, f"top_pct 须在 (0, 1) 之间, 收到: {top_pct}"
        assert 0 < bottom_pct < 1, f"bottom_pct 须在 (0, 1) 之间, 收到: {bottom_pct}"

        self.method = method
        self.n_bins = n_bins
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.date_col = date_col

        # 默认后缀
        if suffix is None:
            if method == "quantile":
                self.suffix = f"_{n_bins}cls"
            else:
                self.suffix = f"_cls"
        else:
            self.suffix = suffix

    def transform(
        self,
        lf: pl.LazyFrame,
        source_col: str,
        output_col: Optional[str] = None,
    ) -> pl.LazyFrame:
        """
        对指定列执行截面离散化

        :param lf: 输入 LazyFrame
        :param source_col: 待离散化的源列名
        :param output_col: 输出列名，默认为 "{source_col}{suffix}"
        :return: 附加了离散标签列的 LazyFrame
        """
        target_col = output_col or f"{source_col}{self.suffix}"

        logger.info(
            f"离散化: {source_col} → {target_col} "
            f"(方式: {self.method}, n_bins={self.n_bins})"
        )

        if self.method == "quantile":
            return self._quantile_bin(lf, source_col, target_col)
        elif self.method == "threshold":
            return self._threshold_classify(lf, source_col, target_col)
        else:
            raise ValueError(f"未知的离散化方式: {self.method}")

    def _quantile_bin(
        self,
        lf: pl.LazyFrame,
        source_col: str,
        target_col: str,
    ) -> pl.LazyFrame:
        """
        等频分箱（基于截面排序位置）

        原理:
          1. 计算截面内的排序百分位: rank / count → [0, 1]
          2. 乘以 n_bins 后向下取整，得到 bin 编号 [0, n_bins-1]
          3. 最大值 clip 到 n_bins-1（处理 rank/count 恰好为 1.0 的边界情况）

        分箱含义:
          0 = 最差组, n_bins-1 = 最好组
        """
        return lf.with_columns(
            (
                # 截面排序百分位
                pl.col(source_col).rank("ordinal").over(self.date_col)
                / pl.col(source_col).count().over(self.date_col)
                # 乘以 n_bins 向下取整
                * self.n_bins
            )
            .floor()
            # 边界 clip: 确保最大值不超过 n_bins-1
            .clip(0, self.n_bins - 1)
            .cast(pl.Int32)
            .alias(target_col)
        )

    def _threshold_classify(
        self,
        lf: pl.LazyFrame,
        source_col: str,
        target_col: str,
    ) -> pl.LazyFrame:
        """
        阈值分类

        基于截面排序百分位:
          - 排名在 top_pct 以上 → 1 (买入信号)
          - 排名在 bottom_pct 以下 → -1 (卖出信号)
          - 中间 → 0 (持有/观望)
        """
        # 先计算截面排序百分位
        pct_col = f"__{source_col}_pct__"

        return (
            lf.with_columns(
                (
                    pl.col(source_col).rank("ordinal").over(self.date_col)
                    / pl.col(source_col).count().over(self.date_col)
                ).alias(pct_col)
            )
            .with_columns(
                pl.when(pl.col(pct_col) >= (1.0 - self.top_pct))
                .then(pl.lit(1))        # 顶部: 买入信号
                .when(pl.col(pct_col) <= self.bottom_pct)
                .then(pl.lit(-1))       # 底部: 卖出信号
                .otherwise(pl.lit(0))   # 中间: 观望
                .cast(pl.Int32)
                .alias(target_col)
            )
            .drop(pct_col)  # 清理中间列
        )
