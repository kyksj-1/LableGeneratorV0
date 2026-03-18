# 数据加载性能测试与 Polars 筛选表达式说明

## 1. 性能测试结果概览

我编写了性能测试脚本 `scripts/test_load_performance.py`，模拟了从 2891 个 Parquet 碎片文件中加载截面全量数据并进行内存映射的场景：

- **无过滤条件加载全部数据**
  - **总行数**: 5,957,178 行
  - **总列数**: 10 列
  - **实际耗时**: ~0.75 秒
  - **原理**: Polars 在背后使用了 Rust 的 Arrow 内存模型，并发拉取并映射了大量 Parquet 文件。

- **带 Polars 表达式过滤加载数据**
  - **过滤条件**: 日期 >= "2020-01-01"，代码以 "600" 开头，PE > 0，换手率 > 0.05
  - **过滤后行数**: 865,607 行
  - **实际耗时**: ~1.95 秒
  - **原理**: 这利用了 Polars 的 **谓词下推 (Predicate Pushdown)** 机制，在引擎扫描文件层面直接跳过不需要的数据页（Row Groups）或者在反序列化前直接进行筛选，极大地节省了内存。耗时虽然因为大量字符串正则匹配与对比有所增加，但对近600万条全量金融截面数据的处理依然保持在了 2 秒以内的极致性能。

## 2. 核心测试代码与表达式

完整的测试脚本位于：`scripts/test_load_performance.py`。

这里的核心是我们所写的 Polars 过滤表达式 (`pl.Expr`)：

```python
import polars as pl

# 编写用于筛选的 Polars 表达式
filter_expr = (
    (pl.col("date") >= "2020-01-01") & 
    (pl.col("code").cast(pl.Utf8).str.starts_with("600")) & 
    (pl.col("pe_ratio") > 0) & 
    (pl.col("turnover_rate") > 0.05)
)

# 将表达式传给 DataLoader 实例
from src.data_loader.daily_loader import DailyDataLoader
filtered_loader = DailyDataLoader(data_footprint_filter=filter_expr)

# 执行 LazyFrame 计算图并触发 Collect
filtered_df = filtered_loader.load_lazy().collect()
```

### 表达式详解：
1. `pl.col("date") >= "2020-01-01"`: 直接对比字符串格式的日期（因为格式标准，字符串比对可以直接过滤时间）。
2. `pl.col("code").cast(pl.Utf8).str.starts_with("600")`: 将 code 显式转换为 utf8 字符串，利用 `str.starts_with` 找出所有沪市主板的股票。
3. `pl.col("pe_ratio") > 0`: 剔除市盈率为负（即亏损）的标的。
4. `pl.col("turnover_rate") > 0.05`: 筛选换手率大于 5% 的活跃标的。
5. 所有条件通过位运算符 `&` 连接，并在最外层用圆括号包裹以保证运算优先级。
