"""
价格比较标签生成器测试

测试流水线:
  1. 连续模式: 验证差量计算正确性（手动验证几行数据）
  2. 离散模式: 验证跳空信号标签的 1/-1/0 分布
  3. 连续模式 + Normalizer 串联: 验证可以接入截面 rank
  4. 用 2 只股票 + 小日期范围，快速验证
"""

import sys
from pathlib import Path

# 确保能找到 src 和 config
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import polars as pl

from src.data_loader.daily_loader import DailyDataLoader
from src.label_generator.price_comparison import PriceComparisonLabel
from src.label_generator.normalizer import CrossSectionalNormalizer


# 测试用公共过滤器: 2 只股票, 2023年6月
TEST_FILTER = (
    (pl.col("date") >= "2023-06-01")
    & (pl.col("date") <= "2023-06-30")
    & (pl.col("code").is_in(["000021", "600519"]))
)


def _load_test_data() -> pl.LazyFrame:
    """加载测试用小数据集"""
    loader = DailyDataLoader(data_footprint_filter=TEST_FILTER)
    return loader.load_lazy()


def test_continuous_basic():
    """测试1: 连续模式基本差量计算"""
    print("=" * 60)
    print("测试 1: 连续模式 - (open[T+2] - high[T+1]) / close[T+1]")
    print("=" * 60)

    gen = PriceComparisonLabel(
        target=("open", 2),
        reference=("high", 1),
        normalizer_price=("close", 1),
        mode="continuous",
    )

    lf = _load_test_data()
    lf_result = gen(lf)
    df = lf_result.collect()

    print(f"标签列名: {gen.label_name}")
    print(f"数据行数: {len(df)}")
    print(f"标签非空数: {df[gen.label_name].drop_nulls().len()}")

    # 手动验证: 取 000021 的前几行, 逐行核算
    sample = df.filter(pl.col("code") == "000021").sort("date")
    print(f"\n000021 前 5 行数据:")
    print(sample.select(["date", "code", "open", "high", "close", gen.label_name]).head(5))

    if len(sample) >= 3:
        # T=0 行的标签 = (open[T+2] - high[T+1]) / close[T+1]
        open_t2 = sample["open"][2]      # 第2行的 open (即 T+2)
        high_t1 = sample["high"][1]      # 第1行的 high (即 T+1)
        close_t1 = sample["close"][1]    # 第1行的 close (即 T+1)
        expected = (open_t2 - high_t1) / close_t1
        actual = sample[gen.label_name][0]

        print(f"\n手动验证 (000021 第1行):")
        print(f"  open[T+2] = {open_t2}")
        print(f"  high[T+1] = {high_t1}")
        print(f"  close[T+1] = {close_t1}")
        print(f"  期望值 = ({open_t2} - {high_t1}) / {close_t1} = {expected:.8f}")
        print(f"  实际值 = {actual:.8f}")
        match = abs(expected - actual) < 1e-8
        print(f"  一致: {match}")
        assert match, f"手动验证失败: 期望 {expected}, 实际 {actual}"

    print()
    return True


def test_continuous_no_normalizer():
    """测试2: 连续模式无归一化"""
    print("=" * 60)
    print("测试 2: 连续模式 - open[T+2] - high[T+1] (无归一化)")
    print("=" * 60)

    gen = PriceComparisonLabel(
        target=("open", 2),
        reference=("high", 1),
        normalizer_price=None,
        mode="continuous",
    )

    lf = _load_test_data()
    df = gen(lf).collect()

    sample = df.filter(pl.col("code") == "600519").sort("date")
    print(f"标签列名: {gen.label_name}")

    if len(sample) >= 3:
        open_t2 = sample["open"][2]
        high_t1 = sample["high"][1]
        expected = open_t2 - high_t1
        actual = sample[gen.label_name][0]

        print(f"\n手动验证 (600519 第1行):")
        print(f"  open[T+2] = {open_t2}, high[T+1] = {high_t1}")
        print(f"  期望值 = {expected:.4f}")
        print(f"  实际值 = {actual:.4f}")
        match = abs(expected - actual) < 1e-6
        print(f"  一致: {match}")
        assert match, f"手动验证失败: 期望 {expected}, 实际 {actual}"

    print()
    return True


def test_discrete_gap_signal():
    """测试3: 离散模式 - 跳空信号"""
    print("=" * 60)
    print("测试 3: 离散模式 - 跳空信号 (open[T+2] vs high/low[T+1])")
    print("=" * 60)

    gen = PriceComparisonLabel(
        target=("open", 2),
        mode="discrete",
        conditions=[
            # 条件1: open[T+2] > high[T+1] → 跳空高开 → 1
            {"ref_col": "high", "ref_offset": 1, "op": ">", "value": 1},
            # 条件2: open[T+2] < low[T+1] → 跳空低开 → -1
            {"ref_col": "low", "ref_offset": 1, "op": "<", "value": -1},
        ],
        default_value=0,
    )

    lf = _load_test_data()
    df = gen(lf).collect()

    print(f"标签列名: {gen.label_name}")
    valid = df[gen.label_name].drop_nulls()
    print(f"标签非空数: {valid.len()}")

    # 分布统计
    vc = valid.value_counts().sort(gen.label_name)
    print(f"\n标签分布:\n{vc}")

    # 跳空不应太频繁: 默认值0应该占多数
    count_0 = vc.filter(pl.col(gen.label_name) == 0)["count"]
    if len(count_0) > 0:
        ratio_0 = count_0[0] / valid.len()
        print(f"\n默认值(0)占比: {ratio_0:.2%} (应占多数)")
        assert ratio_0 > 0.5, f"默认值占比过低: {ratio_0:.2%}"

    # 手动验证几行
    sample = df.filter(pl.col("code") == "000021").sort("date")
    if len(sample) >= 3:
        open_t2 = sample["open"][2]
        high_t1 = sample["high"][1]
        low_t1 = sample["low"][1]
        actual = sample[gen.label_name][0]

        if open_t2 > high_t1:
            expected = 1
        elif open_t2 < low_t1:
            expected = -1
        else:
            expected = 0

        print(f"\n手动验证 (000021 第1行):")
        print(f"  open[T+2]={open_t2}, high[T+1]={high_t1}, low[T+1]={low_t1}")
        print(f"  期望标签={expected}, 实际标签={actual}")
        assert expected == actual, f"离散标签不一致: 期望 {expected}, 实际 {actual}"

    print()
    return True


def test_continuous_with_normalizer():
    """测试4: 连续模式 + 截面 rank 串联"""
    print("=" * 60)
    print("测试 4: 连续模式 + CrossSectionalNormalizer (截面 rank)")
    print("=" * 60)

    # 使用全市场6月数据，才有足够样本做截面
    full_filter = (
        (pl.col("date") >= "2023-06-01")
        & (pl.col("date") <= "2023-06-15")
    )
    loader = DailyDataLoader(data_footprint_filter=full_filter)
    lf = loader.load_lazy()

    # 生成连续标签
    gen = PriceComparisonLabel(
        target=("open", 2),
        reference=("high", 1),
        normalizer_price=("close", 1),
        mode="continuous",
    )
    lf = gen(lf)

    # 截面 rank 标准化
    rank_norm = CrossSectionalNormalizer(method="rank")
    lf = rank_norm.transform(lf, source_col=gen.label_name)
    df = lf.collect()

    rank_col = f"{gen.label_name}_rank"
    valid = df[rank_col].drop_nulls()
    print(f"rank 列非空数: {valid.len()}")

    if valid.len() > 0:
        mean_val = valid.mean()
        min_val = valid.min()
        max_val = valid.max()
        print(f"rank 范围: [{min_val:.4f}, {max_val:.4f}]")
        print(f"rank 均值: {mean_val:.4f} (应接近 0.5)")
        # rank 均值应在 0.45~0.55 之间
        assert 0.4 < mean_val < 0.6, \
            f"rank 均值异常: {mean_val:.4f}, 应接近 0.5"

    print()
    return True


def test_vwap_target():
    """测试5: vwap 作为目标价格"""
    print("=" * 60)
    print("测试 5: vwap 作为目标价格")
    print("=" * 60)

    gen = PriceComparisonLabel(
        target=("vwap", 1),
        reference=("close", 0),
        normalizer_price=("close", 0),
        mode="continuous",
        label_name="vwap_vs_close",
    )

    lf = _load_test_data()
    df = gen(lf).collect()

    sample = df.filter(pl.col("code") == "600519").sort("date")
    print(f"标签列名: {gen.label_name}")

    if len(sample) >= 2:
        # T=0 行: vwap[T+1] = amount[T+1] / (volume[T+1] * 100)
        amount_t1 = sample["amount"][1]
        volume_t1 = sample["volume"][1]
        vwap_t1 = amount_t1 / (volume_t1 * 100)
        close_t0 = sample["close"][0]
        expected = (vwap_t1 - close_t0) / close_t0
        actual = sample[gen.label_name][0]

        print(f"\n手动验证 (600519 第1行):")
        print(f"  vwap[T+1] = {amount_t1} / ({volume_t1} * 100) = {vwap_t1:.4f}")
        print(f"  close[T+0] = {close_t0}")
        print(f"  期望值 = {expected:.8f}")
        print(f"  实际值 = {actual:.8f}")
        match = abs(expected - actual) < 1e-6
        print(f"  一致: {match}")
        assert match, f"vwap 验证失败: 期望 {expected}, 实际 {actual}"

    print()
    return True


if __name__ == "__main__":
    results = {}

    for test_fn in [
        test_continuous_basic,
        test_continuous_no_normalizer,
        test_discrete_gap_signal,
        test_continuous_with_normalizer,
        test_vwap_target,
    ]:
        name = test_fn.__doc__.strip().split(":")[0] if test_fn.__doc__ else test_fn.__name__
        try:
            ok = test_fn()
            results[name] = "PASS" if ok else "FAIL"
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = f"ERROR: {e}"

    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, status in results.items():
        print(f"  [{status}] {name}")
