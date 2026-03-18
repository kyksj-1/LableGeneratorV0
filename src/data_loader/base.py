from abc import ABC, abstractmethod
import polars as pl
from typing import Optional, List, Union

class DataLoader(ABC):
    """
    数据加载器抽象基类
    """
    def __init__(
        self, 
        data_footprint_filter: Optional[pl.Expr] = None,
        columns: Optional[List[str]] = None
    ):
        """
        初始化 DataLoader
        
        :param data_footprint_filter: 指纹过滤器 (Polars 表达式)，如不指定则全量加载
            例如：pl.col("date") >= pl.date(2020, 1, 1) & pl.col("code").is_in(["000001", "600000"])
        :param columns: 仅加载特定列，如果不指定则加载所有列
        """
        self.data_footprint_filter = data_footprint_filter
        self.columns = columns

    @abstractmethod
    def load_lazy(self) -> pl.LazyFrame:
        """
        懒加载模式 (返回 LazyFrame)
        在此模式下，数据尚未真正读入内存，仅生成查询计划 (Query Plan)。
        适合需要在此基础上继续添加过滤或计算逻辑的下游任务。
        """
        pass

    def load(self) -> pl.DataFrame:
        """
        贪婪加载模式 (返回 DataFrame)
        触发真实的 I/O 操作和内存分配，将最终过滤、选择后的数据完整加载到内存中。
        适合直接将数据传入不支持懒加载的算法模型的场景。
        """
        # 调用具体的 lazy 加载逻辑，并立刻执行 (.collect())
        return self.load_lazy().collect()
