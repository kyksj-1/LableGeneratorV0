from abc import ABC, abstractmethod
import polars as pl
from typing import Optional, List, Union, Any

# 引入我们刚才定义的通用类型
from src.utils.expression_converter import FilterConditionType, ExpressionConverter

class DataLoader(ABC):
    """
    数据加载器抽象基类
    """
    def __init__(
        self, 
        data_footprint_filter: Optional[FilterConditionType] = None,
        columns: Optional[List[str]] = None
    ):
        """
        初始化 DataLoader
        
        :param data_footprint_filter: 指纹过滤器
            支持传入原生 Polars 表达式 (pl.Expr)，也支持字典、列表等通用 Python 对象
            (具体解析逻辑由 converter 决定，当前主要支持 pl.Expr)
        :param columns: 仅加载特定列，如果不指定则加载所有列
        """
        self.raw_filter = data_footprint_filter
        self.columns = columns
        
        # 尝试转换并缓存最终的 Polars 表达式
        self.filter_expr = self._convert_filter(data_footprint_filter)

    def _convert_filter(self, condition: Any) -> Optional[pl.Expr]:
        """
        内部辅助方法：将用户输入的任意类型 condition 转为标准的 pl.Expr
        """
        # 如果本来就是 pl.Expr，直接用
        if isinstance(condition, pl.Expr):
            return condition
            
        # 如果不是 pl.Expr (例如 dict/list)，这里预留转换逻辑
        # 目前暂时不支持自动转换，直接返回 None 或抛出警告
        # 后续可以在这里引入 ExpressionConverter 工具函数
        if condition is not None:
            # print("Warning: 传入了非 pl.Expr 类型的过滤器，目前暂未实现自动转换，该过滤条件将被忽略。")
            pass
            
        return None

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
