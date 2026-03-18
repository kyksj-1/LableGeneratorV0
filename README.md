# LabelGeneratorV0

高性能量化金融数据处理模块，主要负责股票历史数据的极速加载与标签（Label）生成。

## 核心特性

- **极致性能**: 弃用 Pandas，全面拥抱 **Polars** 库。利用底层 Rust 引擎实现多线程读取和列式计算。
- **懒加载架构 (Lazy Evaluation)**: DataLoader 默认返回 `LazyFrame`，支持“谓词下推”与“列裁剪”，只在最后 `.collect()` 时读取真正需要的数据，大幅降低内存占用。
- **高度解耦**: 配置文件、工具类、核心逻辑严格分离。

## 目录结构

```text
LabelGeneratorV0/
├── config/                 # 配置目录
│   ├── config_base.py      # 静态系统级配置 (如数据路径)
│   └── config.yaml         # 动态运行时配置
├── src/                    # 核心源代码
│   ├── data_loader/        # 数据加载器模块
│   ├── label_generator/    # 标签生成器模块
│   └── utils/              # 通用工具 (日志、配置解析)
├── tests/                  # 测试代码
└── scripts/                # 运行脚本 (规划中)
```

## 关键类与使用方法

### 1. DataLoader & DailyDataLoader

位于 `src/data_loader/`。提供全市场股票数据的统一下载接口。

```python
import polars as pl
from src.data_loader.daily_loader import DailyDataLoader

# 1. 定义数据指纹过滤器 (只加载特定时间或特定股票的数据)
filter_expr = (pl.col("date") >= "2023-01-01") & (pl.col("code").is_in(["000021", "000099"]))

# 2. 初始化加载器
# 默认会加载需求中指定的 10 个核心字段: ['date', 'code', 'open', 'close', 'high', 'low', 'volume', 'amount', 'turnover_rate', 'pe_ratio']
# 也可以通过 columns 参数覆盖
loader = DailyDataLoader(
    data_footprint_filter=filter_expr,
    prefix_code=True  # 可选: 自动为股票代码添加 sh/sz 前缀
)

# 3. 懒加载 (瞬间完成，不占内存)
lazy_df = loader.load_lazy()

# 4. 贪婪加载 (触发真实硬盘 I/O)
df = loader.load()
```

### 2. LabelGenerator

位于 `src/label_generator/`。提供标签计算的抽象接口。支持分类与回归标签。

```python
from src.label_generator.example_returns import NextNDaysReturnLabel

# 1. 实例化标签生成器
label_gen = NextNDaysReturnLabel(n_days=5, label_name="target_ret_5d")

# 2. 将 LazyFrame 传入生成器，附加标签计算逻辑 (同样是懒执行)
lazy_df_with_label = label_gen(lazy_df)

# 3. 最终收集结果
final_df = lazy_df_with_label.collect()
```

## 环境与依赖

本项目依赖 `conda` 环境运行，主要包如下：
- `polars >= 0.20.0`
- `pyyaml >= 6.0.0`

建议在专门的量化环境 (如 `quant_fin_env`) 中执行。
