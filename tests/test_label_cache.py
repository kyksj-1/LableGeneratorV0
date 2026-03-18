"""
标签缓存落盘测试

测试内容:
  1. ReturnLabelGenerator 的 save_cache / load_cache / meta.json
  2. PriceComparisonLabel 的缓存
  3. LabelFactory 的 save_result / load_result
  4. yaml 开关 enabled=false 时跳过落盘
  5. minimal_columns 模式下只保存索引+标签列
"""

import sys
import json
import shutil
import tempfile
from pathlib import Path

# 确保能找到 src 和 config
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import polars as pl

from src.data_loader.daily_loader import DailyDataLoader
from src.label_generator.returns import ReturnLabelGenerator
from src.label_generator.price_comparison import PriceComparisonLabel
from src.label_generator.factory import LabelFactory

# 测试用临时目录（每次运行重建）
TEST_CACHE_DIR = Path(tempfile.mkdtemp(prefix="label_cache_test_"))

# 测试用小数据集: 2 只股票, 2023年6月
TEST_FILTER = (
    (pl.col("date") >= "2023-06-01")
    & (pl.col("date") <= "2023-06-15")
    & (pl.col("code").is_in(["000021", "600519"]))
)


def _load_test_data() -> pl.LazyFrame:
    """加载测试用小数据集"""
    loader = DailyDataLoader(data_footprint_filter=TEST_FILTER)
    return loader.load_lazy()


def test_return_label_cache():
    """测试1: ReturnLabelGenerator 的 save / load / meta"""
    print("=" * 60)
    print("测试 1: ReturnLabelGenerator 缓存")
    print("=" * 60)

    gen = ReturnLabelGenerator(buy_offset=1, sell_offset=5)
    lf = _load_test_data()
    lf_with_label = gen(lf)

    # 保存缓存 (强制 enabled, 保存全部列)
    path = gen.save_cache(
        lf_with_label,
        cache_dir=TEST_CACHE_DIR,
        enabled=True,
        columns=None,  # 保存全部列以便验证
    )
    print(f"缓存路径: {path}")
    assert path is not None and path.exists(), "parquet 文件应存在"

    # 检查 meta.json
    meta_path = path.parent / "meta.json"
    assert meta_path.exists(), "meta.json 应存在"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"meta.json 内容:")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    # 验证 meta 字段完整性
    assert meta["generator_class"] == "ReturnLabelGenerator"
    assert meta["label_name"] == "ret_open1_close5"
    assert meta["params"]["buy_offset"] == 1
    assert meta["params"]["sell_offset"] == 5
    assert "created_at" in meta
    assert "data_shape" in meta

    # 从缓存加载
    loaded = gen.load_cache(cache_dir=TEST_CACHE_DIR, as_lazy=False)
    assert loaded is not None, "应成功加载缓存"
    assert gen.label_name in loaded.columns, "加载的数据应包含标签列"
    print(f"加载行数: {loaded.height}, 列: {loaded.columns}")

    # cache_exists 检查
    assert gen.cache_exists(cache_dir=TEST_CACHE_DIR), "cache_exists 应返回 True"

    # get_cache_meta
    meta2 = gen.get_cache_meta(cache_dir=TEST_CACHE_DIR)
    assert meta2 is not None
    assert meta2["params"]["buy_price"] == "open"

    print()
    return True


def test_price_comparison_cache():
    """测试2: PriceComparisonLabel 的缓存"""
    print("=" * 60)
    print("测试 2: PriceComparisonLabel 缓存")
    print("=" * 60)

    gen = PriceComparisonLabel(
        target=("open", 2),
        reference=("high", 1),
        normalizer_price=("close", 1),
        mode="continuous",
    )

    lf = _load_test_data()
    lf_with_label = gen(lf)

    path = gen.save_cache(
        lf_with_label,
        cache_dir=TEST_CACHE_DIR,
        enabled=True,
    )
    assert path is not None and path.exists()

    # 验证 meta.json 参数
    meta = gen.get_cache_meta(cache_dir=TEST_CACHE_DIR)
    assert meta["generator_class"] == "PriceComparisonLabel"
    assert meta["params"]["target"] == ["open", 2]
    assert meta["params"]["reference"] == ["high", 1]
    assert meta["params"]["mode"] == "continuous"
    print(f"meta 参数: {json.dumps(meta['params'], indent=2)}")

    # 加载并验证
    loaded = gen.load_cache(cache_dir=TEST_CACHE_DIR, as_lazy=False)
    assert loaded is not None
    assert gen.label_name in loaded.columns
    print(f"加载行数: {loaded.height}")

    print()
    return True


def test_factory_cache():
    """测试3: LabelFactory 的 save_result / load_result"""
    print("=" * 60)
    print("测试 3: LabelFactory 缓存")
    print("=" * 60)

    lf = _load_test_data()

    factory = LabelFactory()
    recipes = {
        "ret_5d_rank": {
            "buy_offset": 1,
            "sell_offset": 5,
            "normalization": "rank",
        },
    }
    lf_result = factory.create_labels(lf, recipes)

    # 保存
    path = factory.save_result(
        lf_result,
        cache_name="test_batch",
        recipes=recipes,
        cache_dir=TEST_CACHE_DIR,
    )
    assert path is not None and path.exists()

    # 检查 meta.json
    meta_path = path.parent / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["factory_class"] == "LabelFactory"
    assert meta["cache_name"] == "test_batch"
    assert "ret_5d_rank" in meta["recipes"]
    print(f"工厂 meta 配方: {list(meta['recipes'].keys())}")

    # 加载
    loaded = factory.load_result(
        cache_name="test_batch",
        cache_dir=TEST_CACHE_DIR,
        as_lazy=False,
    )
    assert loaded is not None
    assert "ret_5d_rank" in loaded.columns
    print(f"加载行数: {loaded.height}, 列数: {loaded.width}")

    print()
    return True


def test_enabled_switch():
    """测试4: enabled=False 跳过落盘"""
    print("=" * 60)
    print("测试 4: enabled=False 跳过落盘")
    print("=" * 60)

    gen = ReturnLabelGenerator(
        buy_offset=1, sell_offset=3, label_name="ret_test_skip"
    )
    lf = _load_test_data()
    lf_with_label = gen(lf)

    # 显式传 enabled=False
    path = gen.save_cache(
        lf_with_label,
        cache_dir=TEST_CACHE_DIR,
        enabled=False,
    )
    assert path is None, "enabled=False 时应返回 None"
    assert not gen.cache_exists(cache_dir=TEST_CACHE_DIR), "不应产生缓存文件"

    print("跳过落盘: 通过")
    print()
    return True


def test_minimal_columns():
    """测试5: minimal_columns 模式"""
    print("=" * 60)
    print("测试 5: minimal_columns 只保存索引+标签列")
    print("=" * 60)

    gen = ReturnLabelGenerator(
        buy_offset=1, sell_offset=5, label_name="ret_minimal_test"
    )
    lf = _load_test_data()
    lf_with_label = gen(lf)
    df_full = lf_with_label.collect()
    print(f"完整数据列数: {df_full.width}, 列: {df_full.columns}")

    # 手动传 columns 模拟 minimal 模式
    path = gen.save_cache(
        df_full,
        cache_dir=TEST_CACHE_DIR,
        columns=["date", "code", gen.label_name],
        enabled=True,
    )
    assert path is not None

    loaded = gen.load_cache(cache_dir=TEST_CACHE_DIR, as_lazy=False)
    assert loaded is not None
    assert loaded.columns == ["date", "code", gen.label_name]
    print(f"最小列集: {loaded.columns} ({loaded.width} 列)")

    # meta.json 应记录保存的列
    meta = gen.get_cache_meta(cache_dir=TEST_CACHE_DIR)
    assert meta["saved_columns"] == ["date", "code", gen.label_name]
    print(f"meta.saved_columns: {meta['saved_columns']}")

    print()
    return True


def test_cache_not_found():
    """测试6: 加载不存在的缓存返回 None"""
    print("=" * 60)
    print("测试 6: 缓存不存在时返回 None")
    print("=" * 60)

    gen = ReturnLabelGenerator(
        buy_offset=1, sell_offset=99, label_name="nonexistent_label"
    )
    result = gen.load_cache(cache_dir=TEST_CACHE_DIR)
    assert result is None, "不存在的缓存应返回 None"
    assert not gen.cache_exists(cache_dir=TEST_CACHE_DIR)
    print("通过: 返回 None")

    print()
    return True


if __name__ == "__main__":
    results = {}

    try:
        for test_fn in [
            test_return_label_cache,
            test_price_comparison_cache,
            test_factory_cache,
            test_enabled_switch,
            test_minimal_columns,
            test_cache_not_found,
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
    finally:
        # 清理临时目录
        shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)
        print(f"\n临时目录已清理: {TEST_CACHE_DIR}")

    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, status in results.items():
        print(f"  [{status}] {name}")
