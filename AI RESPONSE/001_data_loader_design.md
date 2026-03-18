# 数据加载器(DataLoader)设计哲学与工程实现

**扮演专家**: Wes McKinney (Pandas 与 Apache Arrow 创始人)
**选择理由**: 你当前面临的核心问题是“结构化数据的内存管理”、“索引效率”以及“存储格式(Parquet)的IO优化”。作为 Pandas 的创作者，我深知在 Python 中处理金融时序数据的每一个坑（Pitfall）和性能瓶颈（Bottleneck）。

---

## 1. 为什么要加交易所前缀 (Exchange Prefix)?

你可能会觉得 `600000` 已经足够代表浦发银行了，但在构建**通用**金融基础设施时，**“隐式假设”是万恶之源**。

*   **命名空间冲突 (Namespace Collision)**:
    *   `000001` 在深交所是“平安银行”。
    *   `000001` 在上交所是“上证指数”。
    *   如果不加前缀，你的系统必须在每一处代码都隐含“这是股票”还是“这是指数”的上下文，这增加了耦合。
*   **未来扩展性 (Future Proofing)**:
    *   如果未来你要做跨市场套利（引入港股 `00700` 或美股 `AAPL`），单纯的数字代码瞬间失效。
    *   **标准化建议**: 使用 `Exchange.Ticker` 格式 (e.g., `SH.600000`, `SZ.000001`)。这是金融数据的“全限定名 (Fully Qualified Name)”。

---

## 2. 日级 vs 秒级：DataLoader 的通用性边界

你问：“这个 DataLoader 对秒级数据也可行吗？”

**答案是：接口 (Interface) 可以通用，但实现 (Implementation) 必须分层。**

*   **量级差异**:
    *   **日线**: 全市场 5000 只股票 x 20 年 $\approx$ 2500 万行。内存占用 < 5GB。**完全可以一次性 Load 进 RAM**，这是最高效的。
    *   **秒级 (Tick/3s Snapshot)**: 每天产生的数据量就是 GB 级别。全量加载会直接撑爆内存 (OOM)。
*   **抽象策略**:
    *   设计一个基类 `BaseDataLoader`，定义标准接口 `get_batch(start, end)`。
    *   **DailyLoader**: 内部维护一个巨大的 `pd.DataFrame` 或 `pa.Table`，直接切片返回。
    *   **HighFreqLoader**: 内部维护的是**文件索引 (File Registry)**。当调用 `get_batch` 时，它采用 **Lazy Loading (惰性加载)** —— 只读取该时间段对应的 Parquet 文件块，用完即扔。

**结论**: 不要试图用一种内存策略吃遍所有频率。写一个通用的**API**，但针对日线数据（你目前的重点）做**全内存优化**。

---

## 3. "键" (Key) 的实现与高效性保证

在 Pandas/Numpy 生态中，"键"不仅是标识，更是**物理存储的组织方式**。

### A. 逻辑层：MultiIndex 是唯一真理
不要把 `date` 和 `code` 当作普通列 (Column)，要当作索引 (Index)。
```python
# 推荐的数据结构 (Pandas MultiIndex)
                       Open   Close  Volume
Date       Code                             
2023-01-01 SH.600000   10.0   10.2   1000
           SZ.000001   15.0   14.8   2000
```
**优势**: Pandas 在底层对 Index 有哈希优化，`.loc[date]` 的查找速度远快于布尔掩码 `df[df['date'] == date]`。

### B. 物理层：排序 (Sorting) 决定性能
这是做“时间截面”深度学习最关键的一点。

*   **场景**: 你的模型训练时，是一个 Batch 一个 Batch 地取数据。
    *   **时间截面模型 (Cross-Sectional)**: 每次取**某一天**的所有股票。
    *   **时序模型 (Time-Series)**: 每次取**某只股票**的一段历史。
*   **优化方案**:
    *   如果你主要做**截面** (如 AlphaNet, 选股): 数据必须 **按 Date 主序，Code 次序** 排序 (`Sort by Date, then Code`)。
    *   这样在磁盘上，同一天的数据是**连续存储**的。读取 `2023-01-01` 的数据是一次连续 IO，而不是 5000 次随机 Seek。

### C. 极速优化技巧 (针对 Parquet)
1.  **整数化代码**: 字符串 `SH.600000` 在内存中占用大且比较慢。
    *   **Trick**: 维护一个 `stock_ids.json` 映射表 `{'SH.600000': 1, 'SZ.000001': 2}`。
    *   在 DataFrame 中只存 `int32` 类型的 ID。性能提升显著。
2.  **日期整数化**: 使用 `20230101` (int) 代替 `datetime64` 对象，有时候能避免很多时区 (TimeZone) 带来的麻烦，且序列化更快。

---

## 总结建议

1.  **ID**: 坚持使用 `SH.xxxxxx` 格式，但在 DataFrame 内部映射为 `Int` 以提升性能。
2.  **Loader**: 优先实现 `InMemoryDailyLoader`，通过 `MultiIndex(Date, Code)` 管理数据。
3.  **IO**: 利用 Parquet 的特性，预先将数据按 `Date` 排序重写一遍 (Re-partition)，这样训练时的 IO 速度可以提升 10-50 倍。
