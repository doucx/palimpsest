"""
Palimpsest - 搜索引擎抽象基类与标准结果契约
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    """标准检索命中实体"""
    file_id: str
    score: float
    title: str = ""
    matched_turn_index: int | None = None
    highlighted_snippet: str = ""
    sub_scores: dict[str, float] = field(default_factory=dict)
    extra_meta: dict[str, Any] = field(default_factory=dict)


class BaseSearchEngine(ABC):
    """所有候选搜索引擎必须实现的统一抽象接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """返回引擎唯一代号，如 'baseline_sqlite_fts', 'tantivy_bm25'"""
        pass

    @abstractmethod
    def build_index(self, corpus_db_path: str, force_rebuild: bool = False):
        """从提纯数据库构建引擎专属的索引数据"""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """执行检索，返回按综合相关性降序排列的 SearchHit 列表"""
        pass

    @abstractmethod
    def get_index_size_mb(self) -> float:
        """获取当前引擎物理索引文件占用的磁盘大小 (MB)"""
        pass