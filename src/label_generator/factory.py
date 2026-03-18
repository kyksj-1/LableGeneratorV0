"""
标签工厂

职责：
  将 ReturnLabelGenerator + CrossSectionalNormalizer + Discretizer 组合起来，
  基于"配方"（recipe）一步到位地生成完整的标签列。

设计理念：
  不是写 60 个子类，而是用 (收益方式, 时间窗口, 标准化, 离散化) 四元组
  驱动标签生成。一个配方 = 一组参数 → 自动组装组件 → 输出标签列。

  配方示例:
    {
      "buy_offset": 1, "sell_offset": 5,
      "buy_price": "open", "sell_price": "close",
      "return_type": "simple",
      "normalization": "rank",        # None / "rank" / "zscore" / "industry_neutral"
      "discretization": None,         # None / {"method": "quantile", "n_bins": 3}
    }
"""

import polars as pl
from typing import Dict, Any, Optional, List

from src.label_generator.returns import ReturnLabelGenerator
from src.label_generator.normalizer import CrossSectionalNormalizer
from src.label_generator.discretizer import Discretizer
from src.utils.logger import logger


class LabelFactory:
    """
    标签工厂

    基于配方参数自动组装标签生成流水线。

    使用示例:
        factory = LabelFactory()

        # 方式 1: 直接构建单个标签
        lf = factory.create_label(lf, recipe={
            "buy_offset": 1, "sell_offset": 5,
            "normalization": "rank",
        })

        # 方式 2: 批量生成多个标签
        lf = factory.create_labels(lf, recipes={
            "ret_5d_rank": {"sell_offset": 5, "normalization": "rank"},
            "ret_10d_zscore": {"sell_offset": 10, "normalization": "zscore"},
        })
    """

    # 默认配方参数
    DEFAULT_RECIPE = {
        "buy_offset": 1,
        "sell_offset": 5,
        "buy_price": "open",
        "sell_price": "close",
        "return_type": "simple",
        "normalization": None,          # None / "rank" / "zscore" / "industry_neutral"
        "discretization": None,         # None / {"method": "quantile", "n_bins": 3}
        "code_col": "code",
        "date_col": "date",
        "industry_col": "industry_id",
    }

    def create_label(
        self,
        lf: pl.LazyFrame,
        recipe: Dict[str, Any],
        label_name: Optional[str] = None,
    ) -> pl.LazyFrame:
        """
        根据单个配方生成标签

        流水线:
          1. ReturnLabelGenerator → 原始收益率
          2. CrossSectionalNormalizer → 截面标准化（可选）
          3. Discretizer → 离散化（可选）

        :param lf: 输入 LazyFrame（需包含 OHLCV 数据）
        :param recipe: 标签配方字典
        :param label_name: 最终标签列名（覆盖自动生成的名称）
        :return: 附加了标签列的 LazyFrame
        """
        # 合并默认参数
        params = {**self.DEFAULT_RECIPE, **recipe}

        code_col = params["code_col"]
        date_col = params["date_col"]
        industry_col = params["industry_col"]

        # ---- 步骤 1: 生成原始收益率 ----
        return_gen = ReturnLabelGenerator(
            buy_offset=params["buy_offset"],
            sell_offset=params["sell_offset"],
            buy_price=params["buy_price"],
            sell_price=params["sell_price"],
            return_type=params["return_type"],
            code_col=code_col,
            date_col=date_col,
        )
        lf = return_gen(lf)

        # 当前的标签列名（由 ReturnLabelGenerator 自动生成）
        current_col = return_gen.label_name

        # ---- 步骤 2: 截面标准化（可选）----
        normalization = params.get("normalization")
        if normalization is not None:
            normalizer = CrossSectionalNormalizer(
                method=normalization,
                date_col=date_col,
                industry_col=industry_col,
            )
            lf = normalizer.transform(lf, source_col=current_col)
            current_col = f"{current_col}_{normalization}"

        # ---- 步骤 3: 离散化（可选）----
        discretization = params.get("discretization")
        if discretization is not None:
            disc_params = discretization if isinstance(discretization, dict) else {}
            discretizer = Discretizer(
                method=disc_params.get("method", "quantile"),
                n_bins=disc_params.get("n_bins", 3),
                top_pct=disc_params.get("top_pct", 0.2),
                bottom_pct=disc_params.get("bottom_pct", 0.2),
                date_col=date_col,
            )
            lf = discretizer.transform(lf, source_col=current_col)
            # 离散化后的列名
            current_col = f"{current_col}{discretizer.suffix}"

        # ---- 可选: 重命名最终列 ----
        if label_name is not None and label_name != current_col:
            lf = lf.rename({current_col: label_name})
            logger.info(f"最终标签列重命名: {current_col} → {label_name}")

        return lf

    def create_labels(
        self,
        lf: pl.LazyFrame,
        recipes: Dict[str, Dict[str, Any]],
    ) -> pl.LazyFrame:
        """
        批量生成多个标签

        :param lf: 输入 LazyFrame
        :param recipes: 配方字典 {标签名: 配方参数}
        :return: 附加了所有标签列的 LazyFrame
        """
        logger.info(f"批量生成 {len(recipes)} 个标签: {list(recipes.keys())}")

        for label_name, recipe in recipes.items():
            lf = self.create_label(lf, recipe=recipe, label_name=label_name)

        return lf

    @staticmethod
    def get_preset_recipes() -> Dict[str, Dict[str, Any]]:
        """
        获取预设的标签配方集合

        这些是我们头脑风暴中确定的核心标签:
          1. 5日可交易收益(截面排序) — 最基础的截面选股标签
          2. 5日可交易收益(行业中性化) — Alpha 标签
          3. 10日可交易收益(截面排序) — 更长周期
          4. 5日收益三分类 — 分类任务用
        """
        return {
            # ---- 核心标签: 5日可交易收益，截面排序分 ----
            # 用途: 截面选股模型的主标签
            "ret_5d_rank": {
                "buy_offset": 1,
                "sell_offset": 5,
                "buy_price": "open",
                "sell_price": "close",
                "normalization": "rank",
            },

            # ---- Alpha 标签: 5日行业中性化收益 ----
            # 用途: 纯选股 alpha 信号（需先 join 行业信息）
            "alpha_5d": {
                "buy_offset": 1,
                "sell_offset": 5,
                "buy_price": "open",
                "sell_price": "close",
                "normalization": "industry_neutral",
            },

            # ---- 较长周期: 10日可交易收益 ----
            "ret_10d_rank": {
                "buy_offset": 1,
                "sell_offset": 10,
                "buy_price": "open",
                "sell_price": "close",
                "normalization": "rank",
            },

            # ---- 分类标签: 5日收益三分类 ----
            # 用途: 分类任务 (top 33% = 2, mid = 1, bottom = 0)
            "ret_5d_3cls": {
                "buy_offset": 1,
                "sell_offset": 5,
                "buy_price": "open",
                "sell_price": "close",
                "normalization": None,
                "discretization": {"method": "quantile", "n_bins": 3},
            },
        }
