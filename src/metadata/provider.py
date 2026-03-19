"""
股票元数据管理模块

职责：
  - 加载 stock_ids.json（股票代码 → 行业ID/概念IDs 映射）
  - 加载 dicts.json（行业名称 ↔ ID、概念名称 ↔ ID 的双向字典）
  - 提供将行业信息 join 到任意 LazyFrame 的能力

设计原则：
  - 与 DataLoader 完全解耦，不依赖数据加载流程
  - 仅关注元数据的管理和查询
  - 使用 Polars 保持与主流水线一致的数据格式
"""

import json
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.utils.logger import logger


class MetadataProvider:
    """
    股票元数据提供器

    管理行业/概念等静态元数据，提供 join 接口将元数据附加到行情数据上。

    使用示例:
        meta = MetadataProvider("stock_ids.json", "dicts.json")
        lf_with_industry = meta.join_industry(lf, code_col="code")
    """

    def __init__(
        self,
        stock_ids_path: str | Path,
        dicts_path: str | Path
    ):
        """
        :param stock_ids_path: stock_ids.json 的路径（股票 → 行业/概念映射）
        :param dicts_path: dicts.json 的路径（名称 ↔ ID 字典）
        """
        self.stock_ids_path = Path(stock_ids_path)
        self.dicts_path = Path(dicts_path)

        # 懒加载缓存
        self._stock_meta_df: Optional[pl.DataFrame] = None
        self._industry_dict: Optional[Dict[int, str]] = None  # ID → 名称
        self._concept_dict: Optional[Dict[int, str]] = None   # ID → 名称

    # ========================================================================
    #  公开属性：行业/概念字典
    # ========================================================================

    @property
    def industry_dict(self) -> Dict[int, str]:
        """行业 ID → 名称 的映射字典"""
        if self._industry_dict is None:
            self._load_dicts()
        return self._industry_dict

    @property
    def concept_dict(self) -> Dict[int, str]:
        """概念 ID → 名称 的映射字典"""
        if self._concept_dict is None:
            self._load_dicts()
        return self._concept_dict

    @property
    def stock_meta_df(self) -> pl.DataFrame:
        """
        股票元数据 DataFrame

        结构: code(Utf8) | industry_id(Int32) | concept_ids(List[Int32])
        """
        if self._stock_meta_df is None:
            self._load_stock_ids()
        return self._stock_meta_df

    # ========================================================================
    #  核心方法：将行业信息 join 到 LazyFrame
    # ========================================================================

    def join_industry(
        self,
        lf: pl.LazyFrame,
        code_col: str = "code",
        add_name: bool = False
    ) -> pl.LazyFrame:
        """
        将行业ID列 join 到目标 LazyFrame 上

        :param lf: 目标 LazyFrame（需包含股票代码列）
        :param code_col: LazyFrame 中股票代码列的列名
        :param add_name: 是否同时附加行业名称列（默认仅附加 industry_id）
        :return: 附加了 industry_id 列的 LazyFrame
        """
        # 构建 join 用的行业映射表（只取 code + industry_id）
        industry_map = self.stock_meta_df.select(["code", "industry_id"]).lazy()

        # 左连接：保留原始数据的所有行，匹配不上的行业ID为 null
        lf_joined = lf.join(
            industry_map,
            left_on=code_col,
            right_on="code",
            how="left"
        )

        # 可选：将 industry_id 映射为行业名称
        if add_name:
            # 构建 ID→名称 的映射表
            id_name_pairs = [
                (id_, name) for id_, name in self.industry_dict.items()
            ]
            name_map_df = pl.DataFrame({
                "industry_id": [p[0] for p in id_name_pairs],
                "industry_name": [p[1] for p in id_name_pairs]
            }).lazy()

            lf_joined = lf_joined.join(
                name_map_df,
                on="industry_id",
                how="left"
            )

        return lf_joined

    def join_concepts(
        self,
        lf: pl.LazyFrame,
        code_col: str = "code"
    ) -> pl.LazyFrame:
        """
        将概念IDs列 join 到目标 LazyFrame 上

        :param lf: 目标 LazyFrame
        :param code_col: 股票代码列名
        :return: 附加了 concept_ids (List[Int32]) 列的 LazyFrame
        """
        concept_map = self.stock_meta_df.select(["code", "concept_ids"]).lazy()

        return lf.join(
            concept_map,
            left_on=code_col,
            right_on="code",
            how="left"
        )

    # ========================================================================
    #  查询辅助方法
    # ========================================================================

    def get_industry_name(self, industry_id: int) -> Optional[str]:
        """根据行业ID查询行业名称"""
        return self.industry_dict.get(industry_id)

    def get_stocks_by_industry(self, industry_id: int) -> List[str]:
        """获取某个行业下的所有股票代码"""
        return (
            self.stock_meta_df
            .filter(pl.col("industry_id") == industry_id)
            .get_column("code")
            .to_list()
        )

    def get_industry_count(self) -> int:
        """获取行业总数"""
        return len(self.industry_dict)

    # ========================================================================
    #  内部加载逻辑
    # ========================================================================

    def _load_dicts(self) -> None:
        """加载 dicts.json，构建 ID → 名称 的反向映射"""
        logger.info(f"加载元数据字典: {self.dicts_path}")

        with open(self.dicts_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 原始格式: {"INDUSTRY": {"行业名": ID, ...}, "CONCEPT": {"概念名": ID, ...}}
        # 转为反向映射: {ID: "行业名", ...}
        industry_raw = raw.get("INDUSTRY", {})
        concept_raw = raw.get("CONCEPT", {})

        self._industry_dict = {int(v): k for k, v in industry_raw.items()}
        self._concept_dict = {int(v): k for k, v in concept_raw.items()}

        logger.info(
            f"字典加载完成: {len(self._industry_dict)} 个行业, "
            f"{len(self._concept_dict)} 个概念"
        )

    def _load_stock_ids(self) -> None:
        """
        加载 stock_ids.json，构建股票元数据 DataFrame

        原始格式 (每条记录):
        {
            "板块": "科创板",
            "代码": "688001",
            "股票代码格式": "sh.688001",
            "行业ID": 24.0,
            "概念IDs": [1, 27, 32, ...]
        }
        """
        logger.info(f"加载股票元数据: {self.stock_ids_path}")

        with open(self.stock_ids_path, "r", encoding="utf-8") as f:
            raw_list: List[Dict[str, Any]] = json.load(f)

        # 提取为三列: code, industry_id, concept_ids
        codes = []
        industry_ids = []
        concept_ids_list = []

        for item in raw_list:
            code = str(item.get("代码", ""))
            industry_id = item.get("行业ID")
            concepts = item.get("概念IDs", [])

            # 跳过数据不完整的记录
            if not code or industry_id is None:
                continue

            codes.append(code)
            # 行业ID在json中是float（如24.0），转为int
            industry_ids.append(int(industry_id))
            concept_ids_list.append([int(c) for c in concepts])

        self._stock_meta_df = pl.DataFrame({
            "code": pl.Series(codes, dtype=pl.Utf8),
            "industry_id": pl.Series(industry_ids, dtype=pl.Int32),
            "concept_ids": pl.Series(concept_ids_list, dtype=pl.List(pl.Int32)),
        })

        logger.info(
            f"股票元数据加载完成: {len(self._stock_meta_df)} 只股票, "
            f"{self._stock_meta_df['industry_id'].n_unique()} 个行业"
        )
