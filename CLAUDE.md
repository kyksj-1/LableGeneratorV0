## 前提须知

我需要做量化金融项目,这个子项目负责数据处理,主要包括加载和股票标签.
在开始会话首次回答前, 请你先选择一个这方面 (量化金融\*数据处理) 的真实专业人士进行扮演, 并告诉我为何选ta

## 用户喜好

1. 源代码 执行代码 配置文件解耦 可分为src/ scripts/ config/ 等目录.
2. 最基本的配置用 config\_base.py 实现, 例如后续scripts 中的数据目录等内容. 一般这个config\_base.py的配置是不动的(只要给定一个用户,就不用改变. 只有当程序给到新用户时, 才有可能根据每个用户不同而改变的变量放这里). 而常见的动态配置, 如日期范围 进程数 加载的股票数 是否落盘缓存等, 则用 config.yaml 文件管理配置.
3. 环境变量 KALMAN\_DATA\_ROOT\_AKS 指向的目录为本版本项目统一放数据的位置
4. 如果git提交, 提交者为:kyksj-1
5. 目前的原始parquet文件在环境变量 KALMAN\_DATA\_ROOT\_AKS 指向的目录下的raw文件夹.
6. 输出的回答讲解, 如无特殊说明,请用中文, 并放在 'AI RESPONSE' 文件夹中.
7. 为实现高性能, 考虑用 Polars 库代替 Pandas 库
8. 开发环境为conda的quant\_fin\_env


注意, 打了勾\[x]的是已经完成的任务.

### 初步加载

- [x] 写一个抽象类 DataLoader, 定义了加载数据的基本方法, 注意必须是非常通用的. 未来可以用于加载日数据 分钟数据 行业数据 指数数据等
- [x] 写一个继承 DataLoader 的类 DailyDataLoader, 实现对list of parquet文件的加载

  默认加载的字段为 \['date', 'code', 'open', 'close', 'high', 'low', 'volume', 'amount', 'turnover\_rate', 'pe\_ratio'] 其中, 'date''code' 可以视为每一条截面数据的"指纹" (注意可选功能: code加前缀"sh"或"sz"以保证唯一性. 默认关闭) 其余可视为原始特征

  数据示例: {"date":"2014-01-02","code":"600519","open":127.99,"close":125.98,"high":127.99,"low":125.6,"volume":21976,"amount":277430224,"turnover\_rate":0.211684,"pe\_ratio":9.370103} (原文件为parquet, 为做示意我改成json格式)

  注意内存的高效性

### 标签生成器

- [x] 写一个抽象类 LabelGenerator, 定义了生成标签的基本属性: 输入一组(内有多条)数据, 经过一个生成方法, 得到一个标签. 可以选择是离散的标签还是连续的标签 (用于区分分类-回归). 子类需要提供的就是具体的标签生成逻辑
  需考虑效率, 需可并行处理.

### 标签工厂系统

- [x] MetadataProvider: 管理行业/概念元数据, 提供 join 接口将行业信息附加到行情数据 (与 DataLoader 完全解耦)
- [x] ReturnLabelGenerator: 可交易收益率标签, 严格遵循 A股 T+1 约束 (buy\_offset/sell\_offset/价格类型可配置)
- [x] CrossSectionalNormalizer: 截面标准化组件 (rank/zscore/industry\_neutral 三种方式)
- [x] Discretizer: 离散化组件 (quantile 等频分箱 / threshold 阈值分类)
- [x] LabelFactory: 标签工厂, 配方驱动批量生成标签 (四元组: 收益方式×时间窗口×标准化×离散化)

## 要求

- 代码要注释详细, 包括类 方法 函数 变量 等. 推荐使用hint type, 并在函数定义中使用
- **git工作流**, 推荐使用 branch 或 worktree 的git工作流. 需详细撰写提交信息, 包括提交的目的, 所修改的文件, 以及可能的影响等
- 代码需按照工业标准编写, 包括但不限于命名规范, 代码格式, 注释规范, 业务解耦, 可复用性, 可扩展性, **鲁棒性**(防报错+可溯源), 性能优化...
- TDD原则: 包含运行的代码需通过测试. 测试在 tests/ 文件夹下进行
- 需在完成后撰写/更新 README.md 文档, 介绍目前位置项目的功能, 关键代码, 使用方法
- 完成工作后我反馈，以及后续的建议

## 可参考

- 如遇障碍, 可使用MCP工具查阅资料 (GitHub, Brave搜索等). 且你有MCP工具直接去读数据库 (D:\KalmanStockData\AKS\raw) 中的parquet文件、获取数据的项目(D:\AShareQuant\DataFetcherV0)等
- 可直接向我提问

## MEMORY

你可以在这里记录下**有价值**的记忆, 方便后续开发者接着你的工作进行

### 架构决策
- 标签系统采用"配方驱动的组合式设计", 不使用继承式 (避免类爆炸)
- 组件解耦顺序: DataLoader → MetadataProvider(join) → ReturnLabelGenerator → Normalizer → Discretizer
- 收益率标签必须遵循 T+1 约束: 默认 buy=open[T+1], sell=close[T+N]
- A股 volume 单位为"手"(1手=100股), VWAP = amount / (volume * 100)

### 数据说明
- stock_ids.json: 3984只股票, 行业ID为float需转int, 概念IDs为list
- dicts.json: INDUSTRY 86个行业, CONCEPT 440个概念, 格式为 {名称: ID}
- 原始parquet: 2891个文件(每只股票一个), date为字符串格式
