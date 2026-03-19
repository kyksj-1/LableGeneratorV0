"""
价格比较标签生成器

核心设计：
  覆盖"移位价格条件比较"这一类标签，弥补 ReturnLabelGenerator 只能生成收益率比值的局限。
  例如：
    - 跳空信号: open[T+2] > high[T+1] → 1, open[T+2] < low[T+1] → -1, 否则 → 0
    - 相对差量: (open[T+2] - high[T+1]) / close[T+1]  (可接入截面 rank)

两种输出模式：
  - continuous（连续模式）: 输出 (target - reference) / normalizer 的相对差量
  - discrete（离散模式）: 按多组条件顺序判断，首个命中则输出对应值

价格偏移约定：
  offset=0 表示当日(T)，offset=1 表示 T+1，以此类推。
  内部通过 shift(-offset).over(code_col) 实现，与 ReturnLabelGenerator 一致。
"""

import polars as pl
from typing import Optional, List, Tuple, Dict, Any

from src.label_generator.base import LabelGenerator
from src.utils.logger import logger


class PriceComparisonLabel(LabelGenerator):
    """
    价格比较标签生成器

    支持两种模式:
      1. continuous: 输出相对差量 (target - reference) / normalizer
      2. discrete: 按条件列表顺序匹配，输出首个命中的离散值

    使用示例:
        # 连续模式: (open[T+2] - high[T+1]) / close[T+1]
        gen = PriceComparisonLabel(
            target=("open", 2),
            reference=("high", 1),
            normalizer_price=("close", 1),
            mode="continuous",
        )

        # 离散模式: 跳空信号
        gen = PriceComparisonLabel(
            target=("open", 2),
            mode="discrete",
            conditions=[
                {"ref_col": "high", "ref_offset": 1, "op": ">", "value": 1},
                {"ref_col": "low",  "ref_offset": 1, "op": "<", "value": -1},
            ],
            default_value=0,
        )
    """

    # 支持的输出模式
    VALID_MODES = {"continuous", "discrete"}

    # 支持的比较运算符
    VALID_OPS = {">", "<", ">=", "<=", "==", "!="}

    # 支持的价格列名（含 vwap 虚拟列）
    VALID_PRICE_COLS = {"open", "close", "high", "low", "vwap"}

    def __init__(
        self,
        target: Tuple[str, int],
        mode: str = "continuous",
        # --- continuous 模式参数 ---
        reference: Optional[Tuple[str, int]] = None,
        normalizer_price: Optional[Tuple[str, int]] = None,
        # --- discrete 模式参数 ---
        conditions: Optional[List[Dict[str, Any]]] = None,
        default_value: int = 0,
        # --- 通用参数 ---
        label_name: Optional[str] = None,
        code_col: str = "code",
        date_col: str = "date",
    ):
        """
        :param target: 目标价格 (被比较方)，格式 (列名, 偏移)，如 ("open", 2) 表示 open[T+2]
        :param mode: 输出模式 ("continuous" / "discrete")
        :param reference: 参照价格 (比较对象)，格式同 target。continuous 模式必需
        :param normalizer_price: 差量分母，格式同 target。None 则不归一化（仅 continuous 模式）
        :param conditions: 离散模式的条件列表，每项为 dict:
            {
                "ref_col": str,      # 参照列名 (如 "high")
                "ref_offset": int,   # 参照偏移 (如 1 表示 T+1)
                "op": str,           # 比较运算符 (">", "<", ">=", "<=", "==", "!=")
                "value": int/float,  # 命中时输出的标签值
            }
            按顺序检查，首个命中的输出对应 value
        :param default_value: 离散模式下所有条件都不命中时的默认标签值
        :param label_name: 标签列名，默认自动生成
        :param code_col: 股票代码列名
        :param date_col: 日期列名
        """
        # 模式校验
        assert mode in self.VALID_MODES, \
            f"mode 须为 {self.VALID_MODES} 之一, 收到: {mode}"

        # target 校验
        self._validate_price_spec(target, "target")

        # continuous 模式校验
        if mode == "continuous":
            assert reference is not None, \
                "continuous 模式必须指定 reference 参数"
            self._validate_price_spec(reference, "reference")
            if normalizer_price is not None:
                self._validate_price_spec(normalizer_price, "normalizer_price")

        # discrete 模式校验
        if mode == "discrete":
            assert conditions is not None and len(conditions) > 0, \
                "discrete 模式必须指定至少一个 condition"
            for i, cond in enumerate(conditions):
                self._validate_condition(cond, i)

        # 自动生成标签名
        if label_name is None:
            label_name = self._auto_label_name(target, mode, reference, conditions)

        # 确定 is_discrete
        is_discrete = (mode == "discrete")
        super().__init__(label_name=label_name, is_discrete=is_discrete)

        self.target = target
        self.mode = mode
        self.reference = reference
        self.normalizer_price = normalizer_price
        self.conditions = conditions
        self.default_value = default_value
        self.code_col = code_col
        self.date_col = date_col

    def generate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        基于 Polars LazyFrame 生成价格比较标签

        :param lf: 包含 OHLCV 数据的 LazyFrame
        :return: 附加了标签列的 LazyFrame
        """
        # 确保数据按 code + date 排序（shift 操作的前提）
        lf = lf.sort([self.code_col, self.date_col])

        if self.mode == "continuous":
            return self._generate_continuous(lf)
        else:
            return self._generate_discrete(lf)

    def _generate_continuous(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        连续模式: 计算 (target - reference) / normalizer

        公式:
          无 normalizer:  target_price - reference_price
          有 normalizer:  (target_price - reference_price) / normalizer_price
        """
        target_col, target_offset = self.target
        ref_col, ref_offset = self.reference

        # 构建 target 和 reference 的价格表达式
        target_expr = self._build_price_expr(target_col, target_offset)
        ref_expr = self._build_price_expr(ref_col, ref_offset)

        # 差量
        diff_expr = target_expr - ref_expr

        # 是否归一化
        if self.normalizer_price is not None:
            norm_col, norm_offset = self.normalizer_price
            norm_expr = self._build_price_expr(norm_col, norm_offset)
            result_expr = (diff_expr / norm_expr).alias(self.label_name)
            logger.info(
                f"生成连续价格比较标签 '{self.label_name}': "
                f"({target_col}[T+{target_offset}] - {ref_col}[T+{ref_offset}]) "
                f"/ {norm_col}[T+{norm_offset}]"
            )
        else:
            result_expr = diff_expr.alias(self.label_name)
            logger.info(
                f"生成连续价格比较标签 '{self.label_name}': "
                f"{target_col}[T+{target_offset}] - {ref_col}[T+{ref_offset}]"
            )

        return lf.with_columns(result_expr)

    def _generate_discrete(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        离散模式: 按条件列表顺序匹配，首个命中输出对应 value

        实现方式:
          使用 pl.when(...).then(...).when(...).then(...)...otherwise(default)
          构建链式条件表达式
        """
        target_col, target_offset = self.target
        target_expr = self._build_price_expr(target_col, target_offset)

        # 构建链式 when/then 表达式
        # 从最后一个条件开始反向构建，保证按列表顺序优先匹配
        chain_expr = None
        for cond in reversed(self.conditions):
            ref_expr = self._build_price_expr(cond["ref_col"], cond["ref_offset"])
            comparison = self._build_comparison(target_expr, ref_expr, cond["op"])

            if chain_expr is None:
                # 最内层: when(...).then(value).otherwise(default)
                chain_expr = (
                    pl.when(comparison)
                    .then(pl.lit(cond["value"]))
                    .otherwise(pl.lit(self.default_value))
                )
            else:
                # 外层: when(...).then(value).otherwise(内层)
                chain_expr = (
                    pl.when(comparison)
                    .then(pl.lit(cond["value"]))
                    .otherwise(chain_expr)
                )

        # 条件描述日志
        cond_desc = ", ".join(
            f"{target_col}[T+{target_offset}] {c['op']} {c['ref_col']}[T+{c['ref_offset']}] → {c['value']}"
            for c in self.conditions
        )
        logger.info(
            f"生成离散价格比较标签 '{self.label_name}': "
            f"条件=[{cond_desc}], 默认={self.default_value}"
        )

        return lf.with_columns(chain_expr.alias(self.label_name))

    def _build_price_expr(self, price_col: str, offset: int) -> pl.Expr:
        """
        构建偏移后的价格表达式

        :param price_col: 价格列名 ("open" / "close" / "high" / "low" / "vwap")
        :param offset: 时间偏移量（0=当日T，1=T+1，...）
        :return: Polars 表达式
        """
        if price_col == "vwap":
            # VWAP = 成交额 / (成交量 * 100)
            # A股 volume 单位为"手"(1手=100股)，amount 单位为"元"
            base_expr = pl.col("amount") / (pl.col("volume") * 100)
        else:
            base_expr = pl.col(price_col)

        # offset=0 表示当日，无需 shift
        if offset == 0:
            return base_expr

        # shift(-offset) 表示取未来第 offset 行的值
        return base_expr.shift(-offset).over(self.code_col)

    @staticmethod
    def _build_comparison(
        left: pl.Expr,
        right: pl.Expr,
        op: str,
    ) -> pl.Expr:
        """
        根据运算符构建比较表达式

        :param left: 左侧表达式 (target)
        :param right: 右侧表达式 (reference)
        :param op: 比较运算符
        :return: 布尔型 Polars 表达式
        """
        if op == ">":
            return left > right
        elif op == "<":
            return left < right
        elif op == ">=":
            return left >= right
        elif op == "<=":
            return left <= right
        elif op == "==":
            return left == right
        elif op == "!=":
            return left != right
        else:
            raise ValueError(f"不支持的比较运算符: {op}")

    def _validate_price_spec(
        self,
        spec: Tuple[str, int],
        param_name: str,
    ) -> None:
        """
        校验价格规格参数

        :param spec: (列名, 偏移) 元组
        :param param_name: 参数名（用于报错信息）
        """
        assert isinstance(spec, (tuple, list)) and len(spec) == 2, \
            f"{param_name} 须为 (列名, 偏移) 的二元组, 收到: {spec}"
        col_name, offset = spec
        assert col_name in self.VALID_PRICE_COLS, \
            f"{param_name} 列名须为 {self.VALID_PRICE_COLS} 之一, 收到: {col_name}"
        assert isinstance(offset, int) and offset >= 0, \
            f"{param_name} 偏移须为非负整数, 收到: {offset}"

    def _validate_condition(self, cond: Dict[str, Any], index: int) -> None:
        """
        校验单个离散条件

        :param cond: 条件字典
        :param index: 条件在列表中的序号（用于报错信息）
        """
        required_keys = {"ref_col", "ref_offset", "op", "value"}
        missing = required_keys - set(cond.keys())
        assert len(missing) == 0, \
            f"conditions[{index}] 缺少必需键: {missing}"
        assert cond["ref_col"] in self.VALID_PRICE_COLS, \
            f"conditions[{index}].ref_col 须为 {self.VALID_PRICE_COLS} 之一, 收到: {cond['ref_col']}"
        assert isinstance(cond["ref_offset"], int) and cond["ref_offset"] >= 0, \
            f"conditions[{index}].ref_offset 须为非负整数, 收到: {cond['ref_offset']}"
        assert cond["op"] in self.VALID_OPS, \
            f"conditions[{index}].op 须为 {self.VALID_OPS} 之一, 收到: {cond['op']}"

    @staticmethod
    def _auto_label_name(
        target: Tuple[str, int],
        mode: str,
        reference: Optional[Tuple[str, int]],
        conditions: Optional[List[Dict[str, Any]]],
    ) -> str:
        """
        自动生成标签名

        连续模式: "cmp_{target_col}{offset}_vs_{ref_col}{offset}"
        离散模式: "cmp_{target_col}{offset}_disc"

        :return: 标签列名字符串
        """
        target_col, target_offset = target

        if mode == "continuous" and reference is not None:
            ref_col, ref_offset = reference
            return f"cmp_{target_col}{target_offset}_vs_{ref_col}{ref_offset}"
        else:
            return f"cmp_{target_col}{target_offset}_disc"

    def _get_cache_params(self) -> Dict[str, Any]:
        """
        返回价格比较生成器的完整参数，用于缓存 meta.json

        :return: 包含所有配置参数的字典
        """
        params: Dict[str, Any] = {
            "label_name": self.label_name,
            "is_discrete": self.is_discrete,
            "target": list(self.target),
            "mode": self.mode,
            "code_col": self.code_col,
            "date_col": self.date_col,
        }

        if self.reference is not None:
            params["reference"] = list(self.reference)
        if self.normalizer_price is not None:
            params["normalizer_price"] = list(self.normalizer_price)
        if self.conditions is not None:
            params["conditions"] = self.conditions
            params["default_value"] = self.default_value

        return params
