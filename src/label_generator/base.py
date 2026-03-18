"""
标签生成器抽象基类

职责:
  1. 定义 generate() 抽象接口，子类实现具体标签计算逻辑
  2. 提供落盘缓存能力 (save_cache / load_cache)，所有子类自动继承

缓存目录结构:
  {LABEL_CACHE_DIR}/
    {GeneratorClassName}/
      {label_name}/
        data.parquet        ← 标签数据（parquet 格式，高效列式存储）
        meta.json           ← 本次落盘参数记录（生成器类名、参数、时间、数据形状等）
"""

import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

# 确保能导入 config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config_base import LABEL_CACHE_DIR
from src.utils.logger import logger
from src.utils.config_manager import config as yaml_config


class LabelGenerator(ABC):
    """
    标签生成器抽象基类

    所有标签生成器的父类，提供:
      - generate(): 子类实现的标签计算逻辑
      - __call__(): 函数式调用语法
      - save_cache(): 将生成结果落盘为 parquet + meta.json
      - load_cache(): 从缓存加载已有标签
      - cache_exists(): 快速检测缓存是否存在
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

    # ================================================================
    # 落盘缓存相关方法
    # ================================================================

    def save_cache(
        self,
        lf_or_df: Union[pl.LazyFrame, pl.DataFrame],
        cache_dir: Optional[Path] = None,
        columns: Optional[List[str]] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[Path]:
        """
        将标签数据落盘为 parquet，并写入 meta.json 记录参数

        落盘行为受 config.yaml 中 label_cache.enabled 控制:
          - yaml 中 enabled=false 且调用时未显式传 enabled=True → 跳过保存
          - 调用时 enabled=True → 强制保存（忽略 yaml）
          - 调用时 enabled=False → 强制跳过

        列选择受 config.yaml 中 label_cache.minimal_columns 控制:
          - minimal_columns=true 且未显式传 columns → 只保存索引+标签列
          - 显式传 columns → 以传入为准

        目录结构:
          {cache_dir}/{ClassName}/{label_name}/data.parquet
          {cache_dir}/{ClassName}/{label_name}/meta.json

        :param lf_or_df: 生成标签后的 LazyFrame 或 DataFrame
        :param cache_dir: 缓存根目录，默认使用 config_base.LABEL_CACHE_DIR
        :param columns: 指定要保存的列名列表。
                        默认 None 表示遵循 yaml 的 minimal_columns 配置。
        :param enabled: 是否执行落盘。None=由 yaml 配置决定，True/False=强制覆盖
        :return: parquet 文件的完整路径，若跳过则返回 None
        """
        # 判断是否执行落盘
        should_save = enabled if enabled is not None else yaml_config.label_cache_enabled
        if not should_save:
            logger.info(
                f"标签缓存已跳过 (label_cache.enabled={yaml_config.label_cache_enabled}, "
                f"显式参数={enabled}): {self.label_name}"
            )
            return None

        # 解析缓存子目录
        sub_dir = self._resolve_cache_dir(cache_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)

        # LazyFrame → DataFrame (落盘必须物化)
        if isinstance(lf_or_df, pl.LazyFrame):
            logger.info(f"缓存落盘: 正在 collect LazyFrame...")
            df = lf_or_df.collect()
        else:
            df = lf_or_df

        # 列选择: 显式传入 > yaml minimal_columns > 全部列
        if columns is not None:
            # 校验列名存在性
            missing = set(columns) - set(df.columns)
            if missing:
                raise ValueError(f"指定的列不存在于数据中: {missing}")
            df = df.select(columns)
        elif yaml_config.label_cache_minimal:
            # 自动选择最小列集: 索引列 + 标签列
            minimal = self._get_minimal_columns(df)
            df = df.select(minimal)

        # 写入 parquet
        parquet_path = sub_dir / "data.parquet"
        df.write_parquet(parquet_path)

        # 构建并写入 meta.json
        meta = {
            "generator_class": self.__class__.__name__,
            "label_name": self.label_name,
            "is_discrete": self.is_discrete,
            "params": self._get_cache_params(),
            "created_at": datetime.now().isoformat(),
            "data_shape": {"rows": df.height, "cols": df.width},
            "saved_columns": df.columns,
        }
        meta_path = sub_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(
            f"标签缓存已保存: {parquet_path} "
            f"({df.height} 行, {df.width} 列, "
            f"标签={self.label_name})"
        )
        return parquet_path

    def load_cache(
        self,
        cache_dir: Optional[Path] = None,
        as_lazy: bool = True,
    ) -> Optional[Union[pl.LazyFrame, pl.DataFrame]]:
        """
        从缓存加载已有标签数据

        :param cache_dir: 缓存根目录，默认使用 config_base.LABEL_CACHE_DIR
        :param as_lazy: True 返回 LazyFrame (推荐, 延迟执行); False 返回 DataFrame
        :return: 缓存数据，若缓存不存在则返回 None
        """
        sub_dir = self._resolve_cache_dir(cache_dir)
        parquet_path = sub_dir / "data.parquet"
        meta_path = sub_dir / "meta.json"

        if not parquet_path.exists():
            logger.info(f"未找到标签缓存: {parquet_path}")
            return None

        # 读取 meta.json 并记录日志
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            logger.info(
                f"标签缓存命中: {parquet_path} "
                f"(创建于 {meta.get('created_at', '未知')}, "
                f"{meta['data_shape']['rows']} 行)"
            )
        else:
            logger.info(f"标签缓存命中: {parquet_path} (无 meta.json)")

        if as_lazy:
            return pl.scan_parquet(parquet_path)
        else:
            return pl.read_parquet(parquet_path)

    def cache_exists(self, cache_dir: Optional[Path] = None) -> bool:
        """
        快速检测缓存是否存在

        :param cache_dir: 缓存根目录
        :return: True 表示缓存文件存在
        """
        sub_dir = self._resolve_cache_dir(cache_dir)
        return (sub_dir / "data.parquet").exists()

    def get_cache_meta(
        self,
        cache_dir: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        读取缓存的 meta.json 信息

        :param cache_dir: 缓存根目录
        :return: meta 字典，若不存在返回 None
        """
        sub_dir = self._resolve_cache_dir(cache_dir)
        meta_path = sub_dir / "meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_cache_dir(self, cache_dir: Optional[Path] = None) -> Path:
        """
        解析缓存子目录路径

        规则: {cache_root}/{ClassName}/{label_name}/

        :param cache_dir: 用户指定的缓存根目录，None 则使用全局默认
        :return: 完整的缓存子目录路径
        """
        root = Path(cache_dir) if cache_dir is not None else LABEL_CACHE_DIR
        return root / self.__class__.__name__ / self.label_name

    def _get_cache_params(self) -> Dict[str, Any]:
        """
        返回本生成器的参数字典，用于写入 meta.json

        子类应覆写此方法，补充自身的特定参数。
        便于后续加载缓存时验证参数是否匹配。

        :return: 参数字典
        """
        return {
            "label_name": self.label_name,
            "is_discrete": self.is_discrete,
        }

    def _get_minimal_columns(self, df: pl.DataFrame) -> List[str]:
        """
        确定最小保存列集: 索引列 (date, code) + 标签列

        自动检测 df 中是否存在常见索引列名，保证 join 回原始数据时有主键。
        子类可覆写此方法以自定义最小列集。

        :param df: 待保存的 DataFrame
        :return: 列名列表
        """
        # 候选索引列名（按优先级）
        index_candidates = ["date", "code"]
        cols = []
        for c in index_candidates:
            if c in df.columns:
                cols.append(c)

        # 确保标签列存在
        if self.label_name in df.columns:
            cols.append(self.label_name)

        # 兜底: 如果一个索引列都没找到，退回保存全部列
        if len(cols) <= 1:
            logger.warning(
                f"未找到足够的索引列 (date/code)，将保存全部 {df.width} 列"
            )
            return df.columns

        return cols
