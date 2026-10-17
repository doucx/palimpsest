"""
Palimpsest - 搜索算法实验引擎注册中心
"""

from typing import Type
from .base import BaseSearchEngine, SearchHit
from .engine_0_baseline_fts import BaselineFtsEngine
from .engine_1_opt_sqlite import OptSqliteEngine
from .engine_2a_tantivy_raw import TantivyRawEngine
from .engine_2b_tantivy_rescore import TantivyRescoreEngine

# 引擎集中注册表 (代号 -> 构造类)
ENGINE_REGISTRY: dict[str, Type[BaseSearchEngine]] = {
    "baseline": BaselineFtsEngine,
    "opt_sqlite": OptSqliteEngine,
    "tantivy_raw": TantivyRawEngine,
    "tantivy_rescore": TantivyRescoreEngine,
}


def list_registered_engines() -> list[str]:
    """返回所有已注册引擎的代号列表"""
    return list(ENGINE_REGISTRY.keys())


def create_engine(engine_name: str) -> BaseSearchEngine:
    """根据代号动态创建引擎实例"""
    if engine_name not in ENGINE_REGISTRY:
        raise ValueError(
            f"未注册的引擎代号: '{engine_name}'. 可选范围: {list_registered_engines()}"
        )
    return ENGINE_REGISTRY[engine_name]()


__all__ = [
    "BaseSearchEngine",
    "SearchHit",
    "ENGINE_REGISTRY",
    "list_registered_engines",
    "create_engine",
    "BaselineFtsEngine",
    "OptSqliteEngine",
    "TantivyRawEngine",
    "TantivyRescoreEngine",
]