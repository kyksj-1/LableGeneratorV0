# LabelGeneratorV0

高性能量化金融数据处理与标签工程模块。负责股票历史数据的极速加载、元数据管理、以及可复用的标签工厂系统。

## 核心特性

- **极致性能**: 全面使用 **Polars** 库，底层 Rust 引擎实现多线程读取和列式计算
- **懒加载架构**: DataLoader 返回 `LazyFrame`，支持谓词下推与列裁剪，最终 `.collect()` 才触发 I/O
- **标签工厂**: 配方驱动的标签生成系统，`(收益方式, 时间窗口, 标准化, 离散化)` 四元组组合出任意标签
- **A股 T+1 合规**: 收益率标签严格遵循 T+1 交易制度（T+1 开盘买入，T+N 收盘卖出）
- **Alpha + Beta 双轨**: 同时支持总收益、行业超额收益（Alpha）、行业平均收益（Beta）标签
- **高度解耦**: 数据加载 → 元数据匹配 → 标签生成 → 标准化 → 离散化，每一步独立可替换

## 目录结构

```text
LabelGeneratorV0/
├── config/                     # 配置目录
│   ├── config_base.py          # 静态系统级配置 (数据路径等)
│   └── config.yaml             # 动态运行时配置 + 标签配方定义
├── src/                        # 核心源代码
│   ├── data_loader/            # 数据加载器模块
│   │   ├── base.py             # DataLoader 抽象基类
│   │   └── daily_loader.py     # DailyDataLoader 日频数据加载
│   ├── metadata/               # 元数据管理模块 (行业/概念)
│   │   └── provider.py         # MetadataProvider
│   ├── label_generator/        # 标签生成器模块
│   │   ├── base.py             # LabelGenerator 抽象基类
│   │   ├── returns.py          # ReturnLabelGenerator 可交易收益标签
│   │   ├── normalizer.py       # CrossSectionalNormalizer 截面标准化
│   │   ├── discretizer.py      # Discretizer 离散化
│   │   ├── factory.py          # LabelFactory 标签工厂
│   │   └── example_returns.py  # 示例 (早期 demo)
│   └── utils/                  # 通用工具 (日志、配置解析)
├── tests/                      # 测试代码
│   ├── test_pipeline.py        # DataLoader 基础测试
│   └── test_label_factory.py   # 标签工厂集成测试
├── stock_ids.json              # 股票 → 行业/概念 映射表
├── dicts.json                  # 行业/概念 ID ↔ 名称 字典
└── scripts/                    # 运行脚本 (规划中)
```

## 标签工厂使用方法

### 完整流水线（推荐）

```python
import polars as pl
from src.data_loader.daily_loader import DailyDataLoader
from src.metadata.provider import MetadataProvider
from src.label_generator.factory import LabelFactory

# 1. 加载原始行情数据
filter_expr = (pl.col("date") >= "2023-01-01") & (pl.col("date") <= "2023-12-31")
loader = DailyDataLoader(data_footprint_filter=filter_expr)
lf = loader.load_lazy()

# 2. 独立步骤: join 行业元数据 (解耦于 DataLoader)
meta = MetadataProvider("stock_ids.json", "dicts.json")
lf = meta.join_industry(lf)

# 3. 标签工厂: 批量生成标签
factory = LabelFactory()
lf = factory.create_labels(lf, recipes={
    # 5日可交易收益 截面排序分
    "ret_5d_rank": {
        "buy_offset": 1, "sell_offset": 5,
        "buy_price": "open", "sell_price": "close",
        "normalization": "rank",
    },
    # 5日行业中性化收益 (Alpha 标签)
    "alpha_5d": {
        "buy_offset": 1, "sell_offset": 5,
        "normalization": "industry_neutral",
    },
    # 5日收益三分类
    "ret_5d_3cls": {
        "buy_offset": 1, "sell_offset": 5,
        "discretization": {"method": "quantile", "n_bins": 3},
    },
})

# 4. 触发计算
df = lf.collect()
```

### 各组件独立使用

```python
# 单独使用收益率生成器
from src.label_generator.returns import ReturnLabelGenerator
gen = ReturnLabelGenerator(buy_offset=1, sell_offset=5, buy_price="open", sell_price="close")
lf = gen(lf)  # 生成 "ret_open1_close5" 列

# 单独使用截面标准化
from src.label_generator.normalizer import CrossSectionalNormalizer
norm = CrossSectionalNormalizer(method="rank")
lf = norm.transform(lf, source_col="ret_open1_close5")  # 生成 "ret_open1_close5_rank" 列

# 单独使用离散化
from src.label_generator.discretizer import Discretizer
disc = Discretizer(method="quantile", n_bins=3)
lf = disc.transform(lf, source_col="ret_open1_close5")  # 生成 "ret_open1_close5_3cls" 列
```

## 预设标签配方

| 配方名 | 含义 | 用途 |
|--------|------|------|
| `ret_5d_rank` | 5日可交易收益截面排序分 [0,1] | 截面选股模型主标签 |
| `alpha_5d` | 5日行业中性化收益 (Z-Score) | 纯 Alpha 选股信号 |
| `ret_10d_rank` | 10日可交易收益截面排序分 | 较长周期选股 |
| `ret_5d_3cls` | 5日收益三分类 (0/1/2) | 分类任务 |

## T+1 约束说明

```
T日收盘 → T+1日开盘(最早买入) → T+N日收盘(卖出)

可交易收益 = close[T+N] / open[T+1] - 1

注意: close[T+N]/close[T]-1 是不可交易的, 本项目不使用这种标签
```

## 环境与依赖

- conda 环境: `quant_fin_env`
- 核心依赖: `polars >= 0.20.0`, `pyyaml >= 6.0.0`
- 数据路径: 环境变量 `KALMAN_DATA_ROOT_AKS` 指向数据目录

## 运行测试

```bash
conda activate quant_fin_env
python tests/test_label_factory.py
```

## License

MIT License. Author: kyksj-1
