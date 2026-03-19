# 本次会话工作摘要：价格比较标签 + 落盘缓存系统

> 会话日期: 2026-03-19
> 分支: `feat/label-factory`

---

## 一、新增功能概览

本次会话完成了两项功能：

| 功能 | 文件 | 说明 |
|------|------|------|
| PriceComparisonLabel | `src/label_generator/price_comparison.py` | 价格比较标签生成器 |
| 标签落盘缓存系统 | `base.py`, `factory.py`, `config_base.py`, `config.yaml`, `config_manager.py` | 所有标签生成器的 save/load 能力 |

---

## 二、PriceComparisonLabel 价格比较标签生成器

### 2.1 解决的问题

现有 `ReturnLabelGenerator` 只能生成 **收益率比值** (sell/buy - 1)，无法表达：
- 跳空信号：`open[T+2] > high[T+1]` → 离散标签 1/-1/0
- 价格差量：`(open[T+2] - high[T+1]) / close[T+1]` → 连续标签

`PriceComparisonLabel` 覆盖了 **"移位价格条件比较"** 这一类标签。

### 2.2 两种输出模式

#### continuous（连续模式）

输出相对差量，可直接接入 `CrossSectionalNormalizer` 做截面 rank。

```
公式: (target_price - reference_price) / normalizer_price
```

```python
gen = PriceComparisonLabel(
    target=("open", 2),            # open[T+2]
    reference=("high", 1),         # high[T+1]
    normalizer_price=("close", 1), # close[T+1] 作为分母
    mode="continuous",
)
# 输出列名: "cmp_open2_vs_high1"

# 可串联截面 rank:
lf = gen(lf)
lf = CrossSectionalNormalizer(method="rank").transform(lf, "cmp_open2_vs_high1")
```

#### discrete（离散模式）

按条件列表顺序匹配，首个命中输出对应离散值。

```python
gen = PriceComparisonLabel(
    target=("open", 2),
    mode="discrete",
    conditions=[
        {"ref_col": "high", "ref_offset": 1, "op": ">",  "value": 1},   # 跳空高开
        {"ref_col": "low",  "ref_offset": 1, "op": "<",  "value": -1},  # 跳空低开
    ],
    default_value=0,  # 未命中
)
# 输出列名: "cmp_open2_disc"
```

### 2.3 支持的价格列和参数

| 参数 | 可选值 | 说明 |
|------|--------|------|
| 价格列名 | `open`, `close`, `high`, `low`, `vwap` | vwap = amount/(volume*100) |
| 偏移量 | 任意非负整数 | 0=当日T, 1=T+1, ... |
| 比较运算符 | `>`, `<`, `>=`, `<=`, `==`, `!=` | 仅 discrete 模式 |

### 2.4 更多实例化示例

```python
# 收盘突破前日高/低点
gen = PriceComparisonLabel(
    target=("close", 1),
    mode="discrete",
    conditions=[
        {"ref_col": "high", "ref_offset": 0, "op": ">",  "value": 1},
        {"ref_col": "low",  "ref_offset": 0, "op": "<",  "value": -1},
    ],
    default_value=0,
    label_name="breakout_signal",
)

# 隔夜 vwap 偏离度
gen = PriceComparisonLabel(
    target=("vwap", 1),
    reference=("close", 0),
    normalizer_price=("close", 0),
    mode="continuous",
    label_name="vwap_deviation",
)

# 无归一化的绝对价差
gen = PriceComparisonLabel(
    target=("high", 3),
    reference=("low", 1),
    normalizer_price=None,
    mode="continuous",
)
```

---

## 三、标签落盘缓存系统

### 3.1 解决的问题

此前所有标签生成器（包括基类 `LabelGenerator`）都没有落盘能力。
每次使用都需要重新计算，对于全市场数据量大时非常浪费。

### 3.2 架构设计

```
{LABEL_CACHE_DIR}/                          ← 由环境变量或 config_base.py 决定
  ├── ReturnLabelGenerator/                 ← 按生成器类名分目录
  │   └── ret_open1_close5/                 ← 按标签名分子目录
  │       ├── data.parquet                  ← 标签数据 (Polars 高效列式存储)
  │       └── meta.json                     ← 落盘参数记录
  ├── PriceComparisonLabel/
  │   └── cmp_open2_vs_high1/
  │       ├── data.parquet
  │       └── meta.json
  └── LabelFactory/                         ← 工厂级批量缓存
      └── my_batch/
          ├── data.parquet
          └── meta.json
```

### 3.3 配置层级

#### config_base.py（静态配置，不常变）

```python
# 优先读环境变量，否则与 raw/ 并列
LABEL_CACHE_DIR = Path(
    os.environ.get("LABEL_CACHE_DIR", str(DATA_DIR / "label_cache"))
)
```

#### config.yaml（动态配置，随时调整）

```yaml
label_cache:
  enabled: true          # 全局开关
  minimal_columns: true  # true: 只保存 [date, code, label_name]
                         # false: 保存全部列
```

#### 调用时参数（最高优先级）

| 调用时 `enabled=` | yaml `enabled` | 实际行为 |
|---|---|---|
| `None`（默认） | `true` | 保存 |
| `None`（默认） | `false` | 跳过 |
| `True` | 任意 | **强制保存** |
| `False` | 任意 | **强制跳过** |

### 3.4 API 说明

#### LabelGenerator 基类方法（所有子类继承）

```python
gen = ReturnLabelGenerator(buy_offset=1, sell_offset=5)
lf = gen(lf)  # 生成标签

# 保存缓存 (受 yaml label_cache.enabled 控制)
path = gen.save_cache(lf)

# 强制保存（忽略 yaml 开关）
path = gen.save_cache(lf, enabled=True)

# 指定保存列
path = gen.save_cache(lf, columns=["date", "code", gen.label_name])

# 加载缓存 (返回 LazyFrame, 延迟执行)
cached = gen.load_cache()          # 返回 LazyFrame 或 None
cached = gen.load_cache(as_lazy=False)  # 返回 DataFrame

# 快速检测缓存是否存在
if gen.cache_exists():
    lf = gen.load_cache()

# 读取 meta.json
meta = gen.get_cache_meta()
print(meta["params"])       # 生成器参数
print(meta["created_at"])   # 创建时间
print(meta["data_shape"])   # {"rows": ..., "cols": ...}
```

#### LabelFactory 工厂方法

```python
factory = LabelFactory()
recipes = LabelFactory.get_preset_recipes()
lf = factory.create_labels(lf, recipes)

# 保存工厂输出
factory.save_result(lf, cache_name="full_2023", recipes=recipes)

# 加载工厂输出
cached = factory.load_result(cache_name="full_2023")
```

### 3.5 meta.json 示例

```json
{
  "generator_class": "ReturnLabelGenerator",
  "label_name": "ret_open1_close5",
  "is_discrete": false,
  "params": {
    "label_name": "ret_open1_close5",
    "is_discrete": false,
    "buy_offset": 1,
    "sell_offset": 5,
    "buy_price": "open",
    "sell_price": "close",
    "return_type": "simple",
    "code_col": "code",
    "date_col": "date"
  },
  "created_at": "2026-03-19T05:11:25.703536",
  "data_shape": { "rows": 22, "cols": 3 },
  "saved_columns": ["date", "code", "ret_open1_close5"]
}
```

---

## 四、Git 提交记录

| 序号 | commit | 说明 |
|------|--------|------|
| 1 | `c277559` | `feat(label): 添加价格比较标签生成器 PriceComparisonLabel` |
| 2 | `eef85e5` | `test(label): 添加价格比较标签测试` (5/5 PASS) |
| 3 | `7a03c25` | `feat(cache): 添加标签落盘缓存系统` |
| 4 | `0eaa4ee` | `test(cache): 添加标签缓存落盘测试` (6/6 PASS) |

---

## 五、测试结果

### PriceComparisonLabel 测试 (5/5 PASS)

| 测试项 | 结果 |
|--------|------|
| 连续模式 + 归一化 (手动验证) | PASS |
| 连续模式 无归一化 | PASS |
| 离散模式 跳空信号 (0 占比 77.5%) | PASS |
| 连续模式 + rank 串联 (均值 0.5002) | PASS |
| vwap 虚拟列 (手动验证) | PASS |

### 缓存系统测试 (6/6 PASS)

| 测试项 | 结果 |
|--------|------|
| ReturnLabelGenerator save/load/meta | PASS |
| PriceComparisonLabel 缓存参数记录 | PASS |
| LabelFactory save_result/load_result | PASS |
| enabled=False 跳过落盘 | PASS |
| minimal_columns 只保存索引+标签 | PASS |
| 缓存不存在返回 None | PASS |

---

## 六、文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `config/config_base.py` | 修改 | 新增 `LABEL_CACHE_DIR` |
| `config/config.yaml` | 修改 | 新增 `label_cache` 配置段 |
| `src/utils/config_manager.py` | 修改 | 新增 `label_cache_enabled` / `label_cache_minimal` 属性 |
| `src/label_generator/base.py` | 重写 | 新增 `save_cache` / `load_cache` / `cache_exists` / `get_cache_meta` |
| `src/label_generator/returns.py` | 修改 | 覆写 `_get_cache_params` |
| `src/label_generator/price_comparison.py` | 新增+修改 | 新生成器 + 覆写 `_get_cache_params` |
| `src/label_generator/factory.py` | 修改 | 新增 `save_result` / `load_result` |
| `src/label_generator/__init__.py` | 修改 | 导出 `PriceComparisonLabel` |
| `tests/test_price_comparison.py` | 新增 | 5 项测试 |
| `tests/test_label_cache.py` | 新增 | 6 项测试 |
