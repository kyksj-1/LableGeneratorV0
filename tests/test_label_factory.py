"""
标签工厂集成测试

测试流水线:
  1. DataLoader 加载原始数据
  2. MetadataProvider join 行业信息
  3. LabelFactory 生成多种标签
  4. 验证结果正确性
"""

import sys
from pathlib import Path

# 确保能找到 src 和 config
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import polars as pl

from src.data_loader.daily_loader import DailyDataLoader
from src.metadata.provider import MetadataProvider
from src.label_generator.returns import ReturnLabelGenerator
from src.label_generator.normalizer import CrossSectionalNormalizer
from src.label_generator.discretizer import Discretizer
from src.label_generator.factory import LabelFactory


def test_step1_return_label():
    """测试1: ReturnLabelGenerator 单独使用"""
    print("=" * 60)
    print("测试 1: ReturnLabelGenerator (可交易收益率)")
    print("=" * 60)

    # 只加载 2 只股票、2023年的数据，快速验证
    filter_expr = (
        (pl.col("date") >= "2023-01-01")
        & (pl.col("code").is_in(["000021", "600519"]))
    )
    loader = DailyDataLoader(data_footprint_filter=filter_expr)
    lf = loader.load_lazy()

    # 5日可交易收益: buy@T+1 open, sell@T+5 close
    gen = ReturnLabelGenerator(buy_offset=1, sell_offset=5)
    lf_with_ret = gen(lf)
    df = lf_with_ret.collect()

    print(f"标签列名: {gen.label_name}")
    print(f"数据行数: {len(df)}")
    print(f"标签非空数: {df[gen.label_name].drop_nulls().len()}")
    print(f"标签均值: {df[gen.label_name].mean():.6f}")
    print(f"标签标准差: {df[gen.label_name].std():.6f}")
    print("\n前 5 行:")
    print(df.select(["date", "code", "open", "close", gen.label_name]).head(5))

    # 手动验证第一行: ret = close[T+5] / open[T+1] - 1
    sample = df.filter(pl.col("code") == "000021").sort("date").head(10)
    if len(sample) >= 6:
        t0_open_t1 = sample["open"][1]  # T+1 open
        t0_close_t5 = sample["close"][5]  # T+5 close
        expected_ret = t0_close_t5 / t0_open_t1 - 1
        actual_ret = sample[gen.label_name][0]
        print(f"\n手动验证 (000021 第1行):")
        print(f"  open[T+1] = {t0_open_t1}, close[T+5] = {t0_close_t5}")
        print(f"  期望收益率 = {expected_ret:.6f}")
        print(f"  实际收益率 = {actual_ret:.6f}")
        print(f"  一致: {abs(expected_ret - actual_ret) < 1e-6}")

    print()
    return True


def test_step2_metadata_join():
    """测试2: MetadataProvider join 行业信息"""
    print("=" * 60)
    print("测试 2: MetadataProvider (行业信息 join)")
    print("=" * 60)

    meta = MetadataProvider(
        stock_ids_path=project_root / "stock_ids.json",
        dicts_path=project_root / "dicts.json",
    )

    print(f"行业数: {meta.get_industry_count()}")
    print(f"股票数: {len(meta.stock_meta_df)}")
    print(f"行业示例: {meta.get_industry_name(57)}")  # 57=酿酒行业

    # join 到行情数据
    filter_expr = (
        (pl.col("date") == "2023-06-01")
        & (pl.col("code").is_in(["000021", "600519"]))
    )
    loader = DailyDataLoader(data_footprint_filter=filter_expr)
    lf = loader.load_lazy()

    lf_joined = meta.join_industry(lf, add_name=True)
    df = lf_joined.collect()

    print("\njoin 后的数据:")
    print(df.select(["date", "code", "close", "industry_id", "industry_name"]))

    print()
    return True


def test_step3_normalizer():
    """测试3: CrossSectionalNormalizer"""
    print("=" * 60)
    print("测试 3: CrossSectionalNormalizer (截面标准化)")
    print("=" * 60)

    # 加载一段时间的全市场数据，确保收益率能计算出来
    filter_expr = (pl.col("date") >= "2023-06-01") & (pl.col("date") <= "2023-06-30")
    loader = DailyDataLoader(data_footprint_filter=filter_expr)
    lf = loader.load_lazy()

    # 先生成收益率
    gen = ReturnLabelGenerator(buy_offset=1, sell_offset=5)
    lf = gen(lf)

    # rank 标准化
    rank_norm = CrossSectionalNormalizer(method="rank")
    lf_ranked = rank_norm.transform(lf, source_col=gen.label_name)
    df = lf_ranked.collect()

    rank_col = f"{gen.label_name}_rank"
    valid = df[rank_col].drop_nulls()
    print(f"rank 列非空数: {valid.len()}")
    if valid.len() > 0:
        print(f"rank 范围: [{valid.min():.4f}, {valid.max():.4f}]")
        print(f"rank 均值: {valid.mean():.4f} (应接近 0.5)")

    # zscore 标准化
    zscore_norm = CrossSectionalNormalizer(method="zscore")
    lf_zscored = zscore_norm.transform(lf, source_col=gen.label_name)
    df_z = lf_zscored.collect()

    z_col = f"{gen.label_name}_zscore"
    valid_z = df_z[z_col].drop_nulls()
    if valid_z.len() > 0:
        print(f"zscore 均值: {valid_z.mean():.6f} (应接近 0)")
        print(f"zscore 标准差: {valid_z.std():.4f} (应接近 1)")

    print()
    return True


def test_step4_discretizer():
    """测试4: Discretizer"""
    print("=" * 60)
    print("测试 4: Discretizer (离散化)")
    print("=" * 60)

    filter_expr = (pl.col("date") >= "2023-06-01") & (pl.col("date") <= "2023-06-30")
    loader = DailyDataLoader(data_footprint_filter=filter_expr)
    lf = loader.load_lazy()

    gen = ReturnLabelGenerator(buy_offset=1, sell_offset=5)
    lf = gen(lf)

    # 三分类
    disc = Discretizer(method="quantile", n_bins=3)
    lf_disc = disc.transform(lf, source_col=gen.label_name)
    df = lf_disc.collect()

    cls_col = f"{gen.label_name}_3cls"
    valid = df[cls_col].drop_nulls()
    print(f"分类分布:\n{valid.value_counts().sort('count', descending=True)}")

    # 阈值分类
    disc_th = Discretizer(method="threshold", top_pct=0.2, bottom_pct=0.2)
    lf_th = disc_th.transform(lf, source_col=gen.label_name)
    df_th = lf_th.collect()

    th_col = f"{gen.label_name}_cls"
    valid_th = df_th[th_col].drop_nulls()
    print(f"\n阈值分类分布:\n{valid_th.value_counts().sort(th_col)}")

    print()
    return True


def test_step5_factory_e2e():
    """测试5: LabelFactory 端到端流水线"""
    print("=" * 60)
    print("测试 5: LabelFactory 端到端 (完整流水线)")
    print("=" * 60)

    # 加载数据
    filter_expr = (
        (pl.col("date") >= "2023-06-01")
        & (pl.col("date") <= "2023-06-30")
    )
    loader = DailyDataLoader(data_footprint_filter=filter_expr)
    lf = loader.load_lazy()

    # join 行业信息 (解耦步骤: 独立于 DataLoader 和 LabelFactory)
    meta = MetadataProvider(
        stock_ids_path=project_root / "stock_ids.json",
        dicts_path=project_root / "dicts.json",
    )
    lf = meta.join_industry(lf)

    # 使用工厂批量生成标签
    factory = LabelFactory()
    recipes = LabelFactory.get_preset_recipes()
    lf = factory.create_labels(lf, recipes)

    # 触发计算
    df = lf.collect()

    print(f"最终数据形状: {df.shape}")
    print(f"列名: {df.columns}")
    print(f"\n各标签列统计:")
    for col in ["ret_5d_rank", "alpha_5d", "ret_10d_rank", "ret_5d_3cls"]:
        if col in df.columns:
            valid = df[col].drop_nulls()
            print(f"  {col}: 非空={valid.len()}, 均值={valid.mean():.4f}, "
                  f"min={valid.min():.4f}, max={valid.max():.4f}")

    print(f"\n数据预览 (前5行, 关键列):")
    preview_cols = ["date", "code", "close", "industry_id"]
    preview_cols += [c for c in ["ret_5d_rank", "alpha_5d", "ret_5d_3cls"] if c in df.columns]
    print(df.select(preview_cols).head(5))

    print()
    return True


if __name__ == "__main__":
    results = {}

    for test_fn in [
        test_step1_return_label,
        test_step2_metadata_join,
        test_step3_normalizer,
        test_step4_discretizer,
        test_step5_factory_e2e,
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
