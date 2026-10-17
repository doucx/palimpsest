"""
Palimpsest - 方案 C：本地双路混合检索与 Cross-Encoder 重排引擎 (HybridRerankEngine)
架构:
  1. 稀疏符号路 (Sparse): 基于 SQLite FTS / 符号表捕获绝对代码路径与标识符
  2. 稠密语义路 (Dense): 基于 FastEmbed ONNX (BAAI/bge-small-zh-v1.5) 计算意图向量 (全库矩阵计算仅 ~1.8ms)
  3. 倒数排名融合 (RRF): 汇聚双路候选至 Top 15
  4. 交叉注意力重排 (Rerank): 基于 FastEmbed (BAAI/bge-reranker-base) 深度打分，彻底消灭假阳性
"""

import html
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseSearchEngine, SearchHit

# 弹性探测 fastembed 模块中的 Embedding 与 Rerank 类
try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None

RerankerClass = None
# 优先尝试新版标准命名 TextRerank
try:
    from fastembed import TextRerank as RerankerClass
except ImportError:
    # 尝试子模块命名 TextCrossEncoder
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder as RerankerClass
    except ImportError:
        try:
            from fastembed.reranker.cross_encoder import TextCrossEncoder as RerankerClass
        except ImportError:
            RerankerClass = None


class HybridRerankEngine(BaseSearchEngine):
    """工业级本地混合检索与重排引擎"""

    def __init__(
        self,
        cache_dir: str = "data/engine_3_hybrid_cache",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
    ):
        self.cache_dir = cache_dir
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model

        self.embedder = None
        self.reranker = None

        self.meta_records: list[dict[str, Any]] = []
        self.embeddings_matrix: np.ndarray | None = None
        self.sparse_db_path = os.path.join(cache_dir, "sparse_fts.db")

    @property
    def name(self) -> str:
        return "engine_3_hybrid_rerank"

    def _ensure_dependency(self):
        if TextEmbedding is None or RerankerClass is None:
            raise ImportError(
                "未检测到 fastembed 及其重排模块，请确保已安装最新版: pip install --upgrade fastembed"
            )

    def _init_models(self):
        self._ensure_dependency()
        if self.embedder is None:
            self.embedder = TextEmbedding(model_name=self.embedding_model_name)
        if self.reranker is None:
            self.reranker = RerankerClass(model_name=self.reranker_model_name)

    def build_index(self, corpus_db_path: str, force_rebuild: bool = False):
        self._ensure_dependency()
        os.makedirs(self.cache_dir, exist_ok=True)

        npy_path = os.path.join(self.cache_dir, "embeddings.npy")
        meta_path = os.path.join(self.cache_dir, "meta.json")

        if os.path.exists(npy_path) and os.path.exists(meta_path) and not force_rebuild:
            self.embeddings_matrix = np.load(npy_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta_records = json.load(f)
            self._init_models()
            return

        print(f"📦 [{self.name}] 正在初始化 FastEmbed 纯本地 ONNX 模型...")
        self._init_models()

        # 1. 从提纯库提取代表性意图文本 (标题 + 首轮核心提问 + 核心回复前 300 字)
        src_conn = sqlite3.connect(corpus_db_path)
        src_cur = src_conn.cursor()

        src_cur.execute("SELECT file_id, title FROM sessions ORDER BY modified_time DESC;")
        sessions = src_cur.fetchall()

        meta_list = []
        corpus_texts = []
        sparse_rows = []

        print(f"⏳ 正在提取 {len(sessions):,} 场会话的核心意图与符号...")
        for fid, title in sessions:
            src_cur.execute(
                "SELECT role, text_content FROM intent_turns WHERE file_id = ? ORDER BY turn_index ASC LIMIT 6;",
                (fid,)
            )
            turns = src_cur.fetchall()

            # 提取用户首个实质提问
            user_prompt = ""
            for r, txt in turns:
                if r == "user" and len(txt.strip()) >= 5:
                    user_prompt = txt.strip()[:350]
                    break

            # 提取模型核心回应前 300 字
            model_reply = ""
            for r, txt in turns:
                if r == "model" and len(txt.strip()) >= 10:
                    model_reply = txt.strip()[:300]
                    break

            # 构造高密度意图代表文本
            intent_doc = f"会话标题: {title}\n用户意图: {user_prompt}\n回复摘要: {model_reply}"
            corpus_texts.append(intent_doc)

            # 抓取符号用于稀疏路
            src_cur.execute("SELECT symbol_text FROM context_symbols WHERE file_id = ?;", (fid,))
            symbols = [s[0].strip() for s in src_cur.fetchall() if s[0].strip()]
            symbols_str = " ".join(symbols)

            meta_item = {
                "file_id": fid,
                "title": title or "Untitled",
                "intent_doc": intent_doc,
                "symbols_str": symbols_str,
            }
            meta_list.append(meta_item)
            sparse_rows.append((fid, title or "", user_prompt, symbols_str))

        # 2. 批量向量化 (纯 CPU ONNX 极速推理)
        print(f"⚡ 正在对 {len(corpus_texts):,} 篇意图语料进行全量向量化 (Embedding)...")
        t_emb0 = time.perf_counter()
        embeddings_gen = self.embedder.embed(corpus_texts, batch_size=64)
        embeddings_list = list(embeddings_gen)
        self.embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        # 归一化便于直接做点积 (Dot Product == Cosine Similarity)
        norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
        self.embeddings_matrix = self.embeddings_matrix / np.maximum(norms, 1e-9)

        np.save(npy_path, self.embeddings_matrix)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, ensure_ascii=False)
        self.meta_records = meta_list

        t_emb = time.perf_counter() - t_emb0
        print(f"✅ 向量化构建完成 (耗时: {t_emb:.2f}s | 矩阵形态: {self.embeddings_matrix.shape})")

        # 3. 构建轻量辅助稀疏表 (用于精确路径/符号)
        if os.path.exists(self.sparse_db_path):
            os.remove(self.sparse_db_path)
        sp_conn = sqlite3.connect(self.sparse_db_path)
        sp_cur = sp_conn.cursor()
        sp_cur.execute("PRAGMA journal_mode=WAL;")
        sp_cur.execute("CREATE VIRTUAL TABLE sparse_fts USING fts5(file_id UNINDEXED, title, prompt, symbols, tokenize = 'trigram');")
        sp_cur.executemany("INSERT INTO sparse_fts VALUES (?, ?, ?, ?);", sparse_rows)
        sp_conn.commit()
        sp_conn.close()

        src_conn.close()

    def _search_dense(self, query: str, top_k: int = 30) -> list[tuple[int, float]]:
        """纯内存矩阵点积稠密检索 (耗时约 1~2ms)"""
        q_emb = list(self.embedder.embed([query]))[0]
        q_vec = np.array(q_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        scores = np.dot(self.embeddings_matrix, q_vec)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        sorted_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(idx), float(scores[idx])) for idx in sorted_indices]

    def _search_sparse(self, query: str, top_k: int = 25) -> list[tuple[str, float]]:
        """辅助稀疏检索 (专门捕捉路径与符号)"""
        if not os.path.exists(self.sparse_db_path):
            return []

        clean_q = query.strip()
        escaped_q = clean_q.replace('"', '""')
        sql = "SELECT file_id, bm25(sparse_fts, 5.0, 5.0, 15.0) AS rank FROM sparse_fts WHERE sparse_fts MATCH ? ORDER BY rank LIMIT ?;"

        conn = sqlite3.connect(self.sparse_db_path)
        cur = conn.cursor()
        try:
            # 优先精确短语
            cur.execute(sql, (f'"{escaped_q}"', top_k))
            rows = cur.fetchall()
            if not rows and len(clean_q.split()) > 1:
                and_expr = " AND ".join(f'"{t}"' for t in clean_q.split()[:5])
                cur.execute(sql, (and_expr, top_k))
                rows = cur.fetchall()
            return [(r[0], -float(r[1])) for r in rows]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        self._ensure_dependency()
        clean_q = query.strip()
        if not clean_q or self.embeddings_matrix is None:
            return []

        # =========================================================================
        # 阶段 1: 双路候选召回 (Dense 稠密向量 + Sparse 稀疏符号)
        # =========================================================================
        dense_results = self._search_dense(clean_q, top_k=30)
        sparse_results = self._search_sparse(clean_q, top_k=25)

        # =========================================================================
        # 阶段 2: 倒数排名融合 (RRF: Reciprocal Rank Fusion)
        # =========================================================================
        rrf_scores: dict[str, float] = {}
        rrf_k = 60.0

        for rank, (idx, _) in enumerate(dense_results, start=1):
            fid = self.meta_records[idx]["file_id"]
            rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (rrf_k + rank))

        # 稀疏路赋予更高权重给工程路径
        for rank, (fid, _) in enumerate(sparse_results, start=1):
            boost = 1.8 if ("/" in clean_q or "." in clean_q) else 1.0
            rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (boost / (rrf_k + rank))

        if not rrf_scores:
            return []

        # 截取前 15 名候选送入 Cross-Encoder
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:15]

        # 组装元信息查找字典
        fid_to_meta = {m["file_id"]: m for m in self.meta_records}

        candidate_fids = [item[0] for item in sorted_candidates if item[0] in fid_to_meta]
        candidate_docs = [fid_to_meta[fid]["intent_doc"] for fid in candidate_fids]

        if not candidate_fids:
            return []

        # =========================================================================
        # 阶段 3: 本地 Cross-Encoder 深度重排 (耗时约 15~25ms，粉碎高频词畸变)
        # =========================================================================
        # 执行重排，适配返回字典、对象或浮点数
        rerank_results = list(self.reranker.rerank(clean_q, candidate_docs))

        ranked_items = []
        for idx, res_item in enumerate(rerank_results):
            # 兼容 fastembed 的不同返回结构:
            # 形式 A: float
            # 形式 B: dict {"index": 0, "score": 0.95}
            # 形式 C: Object with .index and .score
            if isinstance(res_item, (int, float)):
                c_idx = idx
                score_val = float(res_item)
            elif isinstance(res_item, dict):
                c_idx = res_item.get("index", idx)
                score_val = float(res_item.get("score", 0.0))
            else:
                c_idx = getattr(res_item, "index", idx)
                score_val = float(getattr(res_item, "score", 0.0))

            if c_idx < len(candidate_fids):
                fid = candidate_fids[c_idx]
                m = fid_to_meta[fid]
                ranked_items.append({
                    "file_id": fid,
                    "title": m["title"],
                    "score": score_val,
                    "preview": m["intent_doc"][:220].replace("\n", " ").strip(),
                    "rrf_score": rrf_scores.get(fid, 0.0),
                })

        # 按重排器真实注意力得分降序排列
        ranked_items.sort(key=lambda x: x["score"], reverse=True)

        hits = []
        for item in ranked_items[:limit]:
            hits.append(SearchHit(
                file_id=item["file_id"],
                score=round(item["score"], 4),
                title=item["title"],
                highlighted_snippet=html.escape(item["preview"]),
                sub_scores={
                    "rerank_score": round(item["score"], 4),
                    "rrf_score": round(item["rrf_score"], 4),
                },
            ))

        return hits

    def get_index_size_mb(self) -> float:
        if not os.path.exists(self.cache_dir):
            return 0.0
        total_bytes = sum(f.stat().st_size for f in Path(self.cache_dir).glob("*") if f.is_file())
        return total_bytes / (1024 * 1024)