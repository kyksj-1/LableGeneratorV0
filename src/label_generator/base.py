from abc import ABC, abstractmethod
import polars as pl
from typing import Union

class LabelGenerator(ABC):
    """
    标签生成器抽象基类
    """
    def __init__(self, label_name: str, is_discrete: bool):
        """
        :param label_name: 生成的标签列名 (如 'target_ret_5d', 'is_up')
        :param is_discrete: 标签类型 (True: 分类/离散, False: 回归/连续)
        """
        self.label_name = label_name
        self.is_discrete = is_discrete

    @abstractmethod
    def generate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        核心生成方法
        
        基于 Polars LazyFrame 进行计算，通过 .with_columns() 或聚合操作附加标签列。
        子类需要实现具体的业务逻辑表达式。
        
        :param lf: 包含原始特征的 LazyFrame (由 DataLoader 提供)
        :return: 包含标签列的新 LazyFrame
        """
        pass

    def __call__(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        使得生成器对象可以像函数一样被调用: generator(lf)
        """
        return self.generate(lf)
