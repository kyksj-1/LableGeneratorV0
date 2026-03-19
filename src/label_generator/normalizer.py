"""
截面标准化组件

职责：
  对已计算好的标签列进行截面（Cross-Sectional）标准化。
  即每个交易日内，对所有股票的标签值进行统一变换。

支持的标准化方式：
  - rank: 截面排序分，归一化到 [0, 1]
  - zscore: 截面 Z-Score 标准化
  - industry_neutral: 行业中性化（减去行业均值后再 zscore）

设计原则：
  - 独立组件，不继承 LabelGenerator（它是变换器，不是生成器）
  - 输入 LazyFrame + 列名，输出 LazyFrame
  - 与 MetadataProvider 解耦：行业中性化时需要 industry_id 列已存在于数据中
"""

import polars as pl
from typing import Optional

from src.utils.logger import logger


class CrossSectionalNormalizer:
    """
    截面标准化器

    对指定列在每个交易日的截面上进行标准化变换。

    使用示例:
        normalizer = CrossSectionalNormalizer(method="rank")
        lf = normalizer.transform(lf, source_col="ret_open1_close5")
        # 生成新列 "ret_open1_close5_rank"
    """

    # 支持的标准化方式
    VALID_METHODS = {"rank", "zscore", "industry_neutral"}

    def __init__(
        self,
        method: str = "rank",
        date_col: str = "date",
        industry_col: str = "industry_id",
        suffix: Optional[str] = None,
    ):
        """
        :param method: 标准化方式 ("rank" / "zscore" / "industry_neutral")
        :param date_col: 日期列名（截面的分组依据）
        :param industry_col: 行业列名（仅 industry_neutral 模式需要）
        :param suffix: 输出列名后缀，默认使用 method 名（如 "_rank"）
        """
        assert method in self.VALID_METHODS, \
            f"method 须为 {self.VALID_METHODS} 之一, 收到: {method}"

        self.method = method
        self.date_col = date_col
        self.industry_col = industry_col
        self.suffix = suffix or f"_{method}"

    def transform(
        self,
        lf: pl.LazyFrame,
        source_col: str,
        output_col: Optional[str] = None,
    ) -> pl.LazyFrame:
        """
        对指定列执行截面标准化

        :param lf: 输入 LazyFrame（需包含 date_col 和 source_col）
        :param source_col: 待标准化的源列名（通常是收益率标签列）
        :param output_col: 输出列名，默认为 "{source_col}{suffix}"
        :return: 附加了标准化列的 LazyFrame
        """
        # 输出列名
        target_col = output_col or f"{source_col}{self.suffix}"

        logger.info(
            f"截面标准化: {source_col} → {target_col} (方式: {self.method})"
        )

        if self.method == "rank":
            return self._rank_normalize(lf, source_col, target_col)
        elif self.method == "zscore":
            return self._zscore_normalize(lf, source_col, target_col)
        elif self.method == "industry_neutral":
            return self._industry_neutral(lf, source_col, target_col)
        else:
            raise ValueError(f"未知的标准化方式: {self.method}")

    def _rank_normalize(
        self,
        lf: pl.LazyFrame,
        source_col: str,
        target_col: str,
    ) -> pl.LazyFrame:
        """
        截面排序分标准化

        每个交易日内，对所有股票的值排序，归一化到 [0, 1]。
        公式: rank(x) / count(non-null) , 其中 rank 从 1 开始

        优势: 消除极端值影响，天然适合截面模型
        """
        return lf.with_columns(
            (
                # rank("ordinal") 按出现顺序排名，从 1 开始
                # 除以当天的非空样本数，归一化到 (0, 1]
                pl.col(source_col).rank("ordinal").over(self.date_col)
                / pl.col(source_col).count().over(self.date_col)
            ).alias(target_col)
        )

    def _zscore_normalize(
        self,
        lf: pl.LazyFrame,
        source_col: str,
        target_col: str,
    ) -> pl.LazyFrame:
        """
        截面 Z-Score 标准化

        每个交易日内: (x - mean) / std
        处理 std=0 的情况（如当天只有一只股票）：输出 0.0
        """
        return lf.with_columns(
            (
                (pl.col(source_col) - pl.col(source_col).mean().over(self.date_col))
                / pl.col(source_col).std().over(self.date_col)
            )
            # 当 std 为 0 或 NaN 时（如截面内所有值相同），填充 0
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias(target_col)
        )

    def _industry_neutral(
        self,
        lf: pl.LazyFrame,
        source_col: str,
        target_col: str,
    ) -> pl.LazyFrame:
        """
        行业中性化标准化

        步骤:
          1. 每天每行业计算均值: industry_mean = mean(x).over(date, industry)
          2. 减去行业均值: x_demean = x - industry_mean
          3. 对残差做截面 Z-Score: (x_demean - mean) / std over date

        前提: LazyFrame 中必须已包含 industry_id 列
              （由 MetadataProvider.join_industry() 事先完成）
        """
        # 中间列名（计算完成后会被丢弃）
        demean_col = f"__{source_col}_demean__"

        lf_with_demean = lf.with_columns(
            # 步骤 1+2: 减去行业均值
            (
                pl.col(source_col)
                - pl.col(source_col).mean().over([self.date_col, self.industry_col])
            ).alias(demean_col)
        ).with_columns(
            # 步骤 3: 对残差做截面 Z-Score
            (
                (pl.col(demean_col) - pl.col(demean_col).mean().over(self.date_col))
                / pl.col(demean_col).std().over(self.date_col)
            )
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias(target_col)
        ).drop(demean_col)  # 清理中间列

        return lf_with_demean
