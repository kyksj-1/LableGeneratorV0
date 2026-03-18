"""
可交易收益率标签生成器

核心设计：
  - 严格遵循 A股 T+1 约束：T日看到数据 → T+1 开盘最早买入 → T+2 最早卖出
  - 买入价/卖出价类型可配置（open / close / vwap）
  - 买入/卖出时间偏移可配置
  - 支持简单收益 (simple) 和对数收益 (log)

公式：
  tradable_return[T] = sell_price[T + sell_offset] / buy_price[T + buy_offset] - 1

  默认参数（5日可交易收益）：
    buy_offset=1, buy_price="open", sell_offset=5, sell_price="close"
    即: close[T+5] / open[T+1] - 1
"""

import polars as pl
from typing import Optional

from src.label_generator.base import LabelGenerator
from src.utils.logger import logger


class ReturnLabelGenerator(LabelGenerator):
    """
    可交易收益率标签生成器

    根据 T+1 交易制度生成实际可交易的收益率标签。
    支持多种买卖价格类型和时间窗口的组合。

    使用示例:
        # 5日可交易收益: buy@T+1 open, sell@T+5 close
        gen = ReturnLabelGenerator(buy_offset=1, sell_offset=5)
        lf_with_label = gen(lf)

        # 1日隔夜收益: buy@T+1 open, sell@T+1 close
        gen = ReturnLabelGenerator(buy_offset=1, sell_offset=1,
                                   sell_price="close", label_name="overnight_ret")
    """

    # 支持的价格类型
    VALID_PRICE_TYPES = {"open", "close", "vwap"}

    def __init__(
        self,
        buy_offset: int = 1,
        sell_offset: int = 5,
        buy_price: str = "open",
        sell_price: str = "close",
        return_type: str = "simple",
        label_name: Optional[str] = None,
        code_col: str = "code",
        date_col: str = "date",
    ):
        """
        :param buy_offset: 买入日相对 T 的偏移量（默认 1，即 T+1，符合 A股 T+1 制度）
        :param sell_offset: 卖出日相对 T 的偏移量（默认 5，即 T+5）
        :param buy_price: 买入价格类型 ("open" / "close" / "vwap")
        :param sell_price: 卖出价格类型 ("open" / "close" / "vwap")
        :param return_type: 收益率类型 ("simple" 简单收益 / "log" 对数收益)
        :param label_name: 标签列名，默认自动生成如 "ret_open1_close5"
        :param code_col: 股票代码列名
        :param date_col: 日期列名
        """
        # 参数校验
        assert buy_price in self.VALID_PRICE_TYPES, \
            f"buy_price 须为 {self.VALID_PRICE_TYPES} 之一, 收到: {buy_price}"
        assert sell_price in self.VALID_PRICE_TYPES, \
            f"sell_price 须为 {self.VALID_PRICE_TYPES} 之一, 收到: {sell_price}"
        assert buy_offset >= 1, \
            f"buy_offset 须 >= 1（A股 T+1 制度，最早 T+1 买入）, 收到: {buy_offset}"
        assert sell_offset >= buy_offset, \
            f"sell_offset({sell_offset}) 须 >= buy_offset({buy_offset})"
        assert return_type in ("simple", "log"), \
            f"return_type 须为 'simple' 或 'log', 收到: {return_type}"

        # 自动生成标签名: 如 "ret_open1_close5"
        auto_name = label_name or f"ret_{buy_price}{buy_offset}_{sell_price}{sell_offset}"

        super().__init__(label_name=auto_name, is_discrete=False)

        self.buy_offset = buy_offset
        self.sell_offset = sell_offset
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.return_type = return_type
        self.code_col = code_col
        self.date_col = date_col

    def generate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        基于 Polars LazyFrame 生成可交易收益率标签

        计算逻辑：
          1. 按 (code, date) 排序，保证 shift 操作在时间序列上正确
          2. 构建买入价表达式: buy_price_col.shift(-buy_offset).over(code)
          3. 构建卖出价表达式: sell_price_col.shift(-sell_offset).over(code)
          4. 计算收益率: sell / buy - 1 (simple) 或 ln(sell / buy) (log)

        :param lf: 包含 OHLCV 数据的 LazyFrame
        :return: 附加了收益率标签列的 LazyFrame
        """
        # 1. 确保数据按 code + date 排序
        lf = lf.sort([self.code_col, self.date_col])

        # 2. 构建买入价和卖出价的 Polars 表达式
        buy_expr = self._build_price_expr(self.buy_price, self.buy_offset)
        sell_expr = self._build_price_expr(self.sell_price, self.sell_offset)

        # 3. 计算收益率
        if self.return_type == "simple":
            # 简单收益: sell / buy - 1
            ret_expr = (sell_expr / buy_expr - 1.0).alias(self.label_name)
        else:
            # 对数收益: ln(sell / buy)
            ret_expr = (sell_expr / buy_expr).log().alias(self.label_name)

        logger.info(
            f"生成收益率标签 '{self.label_name}': "
            f"buy={self.buy_price}[T+{self.buy_offset}], "
            f"sell={self.sell_price}[T+{self.sell_offset}], "
            f"type={self.return_type}"
        )

        return lf.with_columns(ret_expr)

    def _build_price_expr(self, price_type: str, offset: int) -> pl.Expr:
        """
        构建偏移后的价格表达式

        :param price_type: 价格类型 ("open" / "close" / "vwap")
        :param offset: 时间偏移量（正数表示未来）
        :return: Polars 表达式
        """
        if price_type == "vwap":
            # VWAP = 成交额 / (成交量 * 100)
            # A股 volume 单位为"手"(1手=100股)，amount 单位为"元"
            base_expr = pl.col("amount") / (pl.col("volume") * 100)
        else:
            # open 或 close 直接取对应列
            base_expr = pl.col(price_type)

        # shift(-offset) 表示取未来第 offset 行的值
        return base_expr.shift(-offset).over(self.code_col)
