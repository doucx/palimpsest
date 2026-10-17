"""
Palimpsest - Baseline 对照组：现网 SQLite FTS5 Trigram 搜索引擎
"""

import html
import math
import os
import sqlite3
import time
from typing import Any

from .base import BaseSearchEngine, SearchHit


class BaselineFtsEngine(BaseSearchEngine):
    """
    原样复刻当前生产环境的 SQLite FTS5 实现：
    1. Trigram 字符分词；
    2. 候选表达式早停机制 (Exact -> AND -> OR)；
    3. 仅剥离 ASCII 标点，缺少中文全角标点处理；
    4. BM25 配合时间衰减因子的混合排序。
    """

    def __init__(self, index_db_path: str = "data/engine_0_baseline.db"):
        self.index_db_path = index_db_path
        self._conn = None

    @property
    def name(self) -> str:
        return "engine_0_baseline_fts"

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.index_db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def build_index(self, corpus_db_path: str, force_rebuild: bool = False):
        if os.path.exists(self.index_db_path) and not force_rebuild:
            return

        if os.path.exists(self.index_db_path):
            os.remove(self.index_db_path)

        os.makedirs(os.path.dirname(os.path.abspath(self.index_db_path)), exist_ok=True)
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")

        # 复刻现网 session_fts 虚表结构
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5(
                file_id,
                title,
                system_instruction,
                content,
                tokenize = 'trigram'
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS session_meta (
                file_id TEXT PRIMARY KEY,
                title TEXT,
                modified_time TEXT
            );
        """)

        # 从提纯语料库抽取并灌入 FTS 虚表
        src_conn = sqlite3.connect(corpus_db_path)
        src_cur = src_conn.cursor()

        src_cur.execute("SELECT file_id, title, modified_time FROM sessions;")
        sessions = src_cur.fetchall()

        fts_rows = []
        meta_rows = []

        for fid, title, mtime in sessions:
            meta_rows.append((fid, title, mtime))
            # 聚合意图对话文本
            src_cur.execute(
                "SELECT role, turn_index, text_content FROM intent_turns WHERE file_id = ? ORDER BY turn_index ASC;",
                (fid,)
            )
            turns = src_cur.fetchall()
            turn_lines = [f"[{r} #{idx}]: {txt}" for r, idx, txt in turns]
            full_content = "\n".join(turn_lines)
            fts_rows.append((fid, title, "", full_content))

        cur.executemany("INSERT INTO session_meta VALUES (?, ?, ?);", meta_rows)
        cur.executemany("INSERT INTO session_fts (file_id, title, system_instruction, content) VALUES (?, ?, ?, ?);", fts_rows)

        conn.commit()
        cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        src_conn.close()

    @staticmethod
    def _clean_code_token(tok: str) -> str:
        """现网代码原样复刻：仅剥离 ASCII 标点，缺少对中文全角标点的清洗"""
        stripped = tok.strip(" \t\r\n;,:'\"`()[]{}")
        return stripped.replace('"', '""')

    def _build_fts_query_candidates(self, query: str) -> list[str]:
        """现网查询编译器逻辑原样复刻"""
        if not query or not query.strip():
            return []

        raw_lines = [line.strip() for line in query.strip().splitlines() if line.strip()]
        if not raw_lines:
            return []

        # 分支 1：多行文本
        if len(raw_lines) > 1:
            valid_subphrases = []
            for line in raw_lines:
                cleaned = line.strip()
                if len(set(cleaned)) <= 2 and len(cleaned) > 5:
                    continue
                if len(cleaned) >= 2:
                    escaped = cleaned.replace('"', '""')
                    valid_subphrases.append(f'"{escaped}"')

            if not valid_subphrases:
                valid_subphrases = [f'"{raw_lines[0].replace(chr(34), chr(34)*2)}"']

            and_expr = " AND ".join(valid_subphrases[:6])
            or_expr = " OR ".join(valid_subphrases[:8])
            candidates = [and_expr]
            if or_expr != and_expr:
                candidates.append(or_expr)
            return candidates

        # 分支 2：单行查询
        single = raw_lines[0]
        escaped_single = single.replace('"', '""')

        raw_words = single.split()
        clean_tokens = []
        for w in raw_words:
            c = self._clean_code_token(w)
            if len(c) >= 2:
                clean_tokens.append(c)

        candidates = []
        candidates.append(f'"{escaped_single}"')

        if len(clean_tokens) > 1:
            and_expr = " AND ".join(f'"{t}"' for t in clean_tokens[:8])
            if and_expr not in candidates:
                candidates.append(and_expr)
            or_expr = " OR ".join(f'"{t}"' for t in clean_tokens[:10])
            if or_expr not in candidates:
                candidates.append(or_expr)
        elif len(clean_tokens) == 1 and clean_tokens[0] != escaped_single:
            candidates.append(f'"{clean_tokens[0]}"')

        return candidates

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        candidates = self._build_fts_query_candidates(query)
        if not candidates:
            return []

        conn = self._get_connection()
        cur = conn.cursor()

        fetch_limit = max(50, limit + 20)

        # 现网早停机制：依序尝试候选表达式，一旦某一级命中结果立即返回，不再看下一级
        for expr in candidates:
            sql = """
                SELECT 
                    f.file_id,
                    bm25(session_fts, 0.0, 6.0, 1.2, 1.0) AS raw_rank,
                    snippet(session_fts, 3, '\u0001', '\u0002', '...', 28) AS raw_snippet,
                    m.title,
                    m.modified_time
                FROM session_fts f
                JOIN session_meta m ON f.file_id = m.file_id
                WHERE session_fts MATCH ?
                ORDER BY raw_rank
                LIMIT ?;
            """
            try:
                cur.execute(sql, (expr, fetch_limit))
                rows = cur.fetchall()
                if rows:
                    hits = []
                    for r in rows:
                        fid = r["file_id"]
                        raw_rank = r["raw_rank"]
                        title = r["title"]
                        mtime = r["modified_time"]
                        raw_snip = r["raw_snippet"] or ""

                        # 现网时间衰减
                        days_ago = 60.0
                        if mtime:
                            try:
                                dt = time.mktime(time.strptime(mtime[:19], "%Y-%m-%dT%H:%M:%S"))
                                days_ago = max(0.0, (time.time() - dt) / 86400.0)
                            except (ValueError, TypeError):
                                pass
                        time_penalty = math.log1p(days_ago) * 0.35
                        composite_score = raw_rank + time_penalty

                        snip_html = html.escape(raw_snip).replace("\u0001", "<mark>").replace("\u0002", "</mark>")

                        hits.append(SearchHit(
                            file_id=fid,
                            score=-composite_score,  # 转为正数：分值越大越优
                            title=title,
                            highlighted_snippet=snip_html,
                            sub_scores={"bm25_raw": raw_rank, "time_penalty": time_penalty, "candidate_expr": expr},
                        ))

                    hits.sort(key=lambda h: h.score, reverse=True)
                    return hits[:limit]

            except sqlite3.OperationalError:
                continue

        return []

    def get_index_size_mb(self) -> float:
        if os.path.exists(self.index_db_path):
            return os.path.getsize(self.index_db_path) / (1024 * 1024)
        return 0.0