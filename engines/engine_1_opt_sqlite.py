"""
Palimpsest - 方案 A：SQLite FTS5 原地极致调优搜索引擎 (OptSqliteEngine)
"""

import html
import math
import os
import re
import sqlite3
import time
from typing import Any

from .base import BaseSearchEngine, SearchHit

# 中英文全角半角标点统一清洗字符集
CJK_PUNCTUATION_PATTERN = re.compile(
    r"[。？！，、；：“”‘’（）《》【】…—～\s\t\r\n;,:'\"`()\[\]{}*#<>=_+\-|\\]+"
)


class OptSqliteEngine(BaseSearchEngine):
    """
    方案 A: 纯 SQLite 极致优化版
    1. 字段解耦物化 (title: 8.0, intent: 10.0, symbols: 2.0)
    2. CJK 标点全量清洗与中英混合智能分词
    3. 废除 First-Match 早停，精确短语与 AND 多路融合
    4. 词项覆盖率 (Term Coverage) 惩罚机制，粉碎单高频词污染
    5. 平滑弱化时间衰减因子
    """

    def __init__(self, index_db_path: str = "data/engine_1_opt_sqlite.db"):
        self.index_db_path = index_db_path
        self._conn = None

    @property
    def name(self) -> str:
        return "engine_1_opt_sqlite"

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

        # 多字段解耦 FTS5 虚表: title, intent_content, symbol_content
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5(
                file_id,
                title,
                intent_content,
                symbol_content,
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

        src_conn = sqlite3.connect(corpus_db_path)
        src_cur = src_conn.cursor()

        src_cur.execute("SELECT file_id, title, modified_time FROM sessions;")
        sessions = src_cur.fetchall()

        fts_rows = []
        meta_rows = []

        for fid, title, mtime in sessions:
            meta_rows.append((fid, title, mtime))

            # 1. 抓取意图层对话
            src_cur.execute(
                "SELECT role, turn_index, text_content FROM intent_turns WHERE file_id = ? ORDER BY turn_index ASC;",
                (fid,)
            )
            turns = src_cur.fetchall()
            intent_lines = [f"[{r} #{idx}]: {txt}" for r, idx, txt in turns]
            intent_text = "\n".join(intent_lines)

            # 2. 抓取背景代码库与符号表
            src_cur.execute(
                "SELECT symbol_type, symbol_text FROM context_symbols WHERE file_id = ?;",
                (fid,)
            )
            symbols = src_cur.fetchall()
            symbol_lines = [f"{stype}:{stext}" for stype, stext in symbols]
            symbol_text = " ".join(symbol_lines)

            fts_rows.append((fid, title, intent_text, symbol_text))

        cur.executemany("INSERT INTO session_meta VALUES (?, ?, ?);", meta_rows)
        cur.executemany(
            "INSERT INTO session_fts (file_id, title, intent_content, symbol_content) VALUES (?, ?, ?, ?);",
            fts_rows
        )

        conn.commit()
        cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        src_conn.close()

    @classmethod
    def clean_query_tokens(cls, query: str) -> list[str]:
        """对输入长句与代码进行中英文全角标点清洗，提取有价值的纯净词元"""
        if not query:
            return []

        # 用空格替换所有 CJK 和 ASCII 标点
        cleaned = CJK_PUNCTUATION_PATTERN.sub(" ", query).strip()
        raw_tokens = cleaned.split()

        tokens = []
        for t in raw_tokens:
            t = t.strip()
            if len(t) >= 2:
                tokens.append(t.replace('"', '""'))
        return tokens

    def _compile_queries(self, query: str) -> tuple[str | None, str | None, list[str]]:
        """
        编译多级查询表达式：
        1. exact_phrase: 清洗首尾后的紧凑短语
        2. and_expr: 核心标识符与 CJK 词元 AND 共现
        3. tokens: 参与计算覆盖率的基准词元列表
        """
        tokens = self.clean_query_tokens(query)
        if not tokens:
            return None, None, []

        # 精确短语：取前 50 个字符（去除极度特殊的换行）
        clean_oneline = re.sub(r'[\r\n]+', ' ', query).strip()
        escaped_exact = clean_oneline.replace('"', '""')
        exact_phrase = f'"{escaped_exact}"' if len(escaped_exact) >= 3 else None

        # AND 表达式：最多取 6 个最有区分度的词元
        and_expr = " AND ".join(f'"{t}"' for t in tokens[:6]) if len(tokens) > 1 else None

        return exact_phrase, and_expr, tokens

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        exact_phrase, and_expr, query_tokens = self._compile_queries(query)
        if not query_tokens:
            return []

        conn = self._get_connection()
        cur = conn.cursor()

        candidates_map: dict[str, dict[str, Any]] = {}
        fetch_limit = max(40, limit * 3)

        # 构造执行队列: (expr, match_type, multiplier)
        query_plans = []
        if exact_phrase:
            query_plans.append((exact_phrase, "exact", 2.2))
        if and_expr:
            query_plans.append((and_expr, "and", 1.5))

        # 若词数较多，构造宽容 OR 召回候选（但严格附加覆盖率过滤）
        if len(query_tokens) >= 2:
            or_expr = " OR ".join(f'"{t}"' for t in query_tokens[:8])
            query_plans.append((or_expr, "or", 0.8))

        # 多路执行并融合候选（彻底废黜 First-Match 早停）
        for expr, m_type, multiplier in query_plans:
            # 权重矩阵: file_id(0.0), title(8.0), intent_content(10.0), symbol_content(2.0)
            sql = """
                SELECT 
                    f.file_id,
                    bm25(session_fts, 0.0, 8.0, 10.0, 2.0) AS raw_rank,
                    snippet(session_fts, 2, '\u0001', '\u0002', '...', 28) AS raw_snippet,
                    m.title,
                    m.modified_time,
                    f.intent_content
                FROM session_fts f
                JOIN session_meta m ON f.file_id = m.file_id
                WHERE session_fts MATCH ?
                ORDER BY raw_rank
                LIMIT ?;
            """
            try:
                cur.execute(sql, (expr, fetch_limit))
                rows = cur.fetchall()
                for r in rows:
                    fid = r["file_id"]
                    raw_rank = r["raw_rank"]  # 负数，如 -15.0

                    if fid not in candidates_map:
                        candidates_map[fid] = {
                            "file_id": fid,
                            "title": r["title"],
                            "modified_time": r["modified_time"],
                            "raw_snippet": r["raw_snippet"] or "",
                            "intent_sample": (r["intent_content"] or "")[:2000].lower(),
                            "best_raw_rank": raw_rank,
                            "match_types": {m_type},
                            "multiplier": multiplier,
                        }
                    else:
                        item = candidates_map[fid]
                        item["match_types"].add(m_type)
                        if multiplier > item["multiplier"]:
                            item["multiplier"] = multiplier
                        if raw_rank < item["best_raw_rank"]:
                            item["best_raw_rank"] = raw_rank
                            if r["raw_snippet"]:
                                item["raw_snippet"] = r["raw_snippet"]
            except sqlite3.OperationalError:
                continue

        if not candidates_map:
            return []

        # =========================================================================
        # 核心算分：计算词项覆盖率 (Term Coverage) + 融合打分
        # =========================================================================
        hits = []
        total_q_tokens = max(1, len(query_tokens))

        for fid, item in candidates_map.items():
            intent_text = item["intent_sample"]
            title_text = (item["title"] or "").lower()

            # 计算查询词项覆盖度 (Coverage Count)
            hit_token_count = 0
            for tok in query_tokens:
                tok_lower = tok.lower()
                if tok_lower in intent_text or tok_lower in title_text:
                    hit_token_count += 1

            coverage_ratio = hit_token_count / total_q_tokens

            # 核心防线：如果是多词长查询，但候选只命中了 1 个词（且为 OR 召回），施加毁灭性惩罚
            if total_q_tokens >= 3 and hit_token_count <= 1 and "exact" not in item["match_types"]:
                coverage_penalty = 0.15  # 斩掉 85% 得分，粉碎单高频词污染
            elif coverage_ratio >= 0.8:
                coverage_penalty = 1.35  # 高覆盖度奖励
            else:
                coverage_penalty = max(0.4, coverage_ratio)

            # 基础 BM25 分转正数并施加匹配乘子
            base_score = (-item["best_raw_rank"]) * item["multiplier"] * coverage_penalty

            # 平滑时效性微调 (弱化为平局决胜因子，绝不颠覆相关性)
            days_ago = 60.0
            mtime = item["modified_time"]
            if mtime:
                try:
                    dt = time.mktime(time.strptime(mtime[:19], "%Y-%m-%dT%H:%M:%S"))
                    days_ago = max(0.0, (time.time() - dt) / 86400.0)
                except (ValueError, TypeError):
                    pass
            # 极微弱时间惩罚: 最多扣减 0.05 * ln(1 + days)
            tie_breaker = math.log1p(days_ago) * 0.05
            final_score = base_score - tie_breaker

            snip_html = html.escape(item["raw_snippet"]).replace("\u0001", "<mark>").replace("\u0002", "</mark>")

            hits.append(SearchHit(
                file_id=fid,
                score=final_score,
                title=item["title"],
                highlighted_snippet=snip_html,
                sub_scores={
                    "bm25_raw": item["best_raw_rank"],
                    "coverage": round(coverage_ratio, 2),
                    "multiplier": item["multiplier"],
                    "match_types": ",".join(item["match_types"]),
                },
            ))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def get_index_size_mb(self) -> float:
        if os.path.exists(self.index_db_path):
            return os.path.getsize(self.index_db_path) / (1024 * 1024)
        return 0.0