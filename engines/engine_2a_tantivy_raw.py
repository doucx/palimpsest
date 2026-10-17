"""
Palimpsest - 方案 2A：Tantivy 原生纯检索基准引擎 (TantivyRawEngine)
特征:
  - 纯粹的 CJK 2-Gram 分词 + Tantivy 原生 BM25
  - 独立持久化索引目录: data/index_2a_tantivy_raw
  - 极简 3ms 极速响应，作为纯倒排检索的标准性能基线
"""

import html
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from .base import BaseSearchEngine, SearchHit

try:
    import tantivy
except ImportError:
    tantivy = None

CJK_PUNCTUATION_PATTERN = re.compile(
    r"[。？！，、；：“”‘’（）《》【】…—～\s\t\r\n;,:'\"`()\[\]{}*#<>=_+\-|\\]+"
)
HAN_REGEX = re.compile(r"([\u4e00-\u9fa5]+)")
DIFF_NOISE_PATTERN = re.compile(r"^(\+{3}|\-{3}|@{2}|#{1,6}|`{3,})")


def cjk_bigram_tokenize(text: str) -> str:
    if not text:
        return ""

    def replace_han(match):
        han_str = match.group(1)
        if len(han_str) <= 1:
            return f" {han_str} "
        bigrams = [han_str[i:i+2] for i in range(len(han_str) - 1)]
        return " " + " ".join(bigrams) + " "

    expanded = HAN_REGEX.sub(replace_han, text)
    return " ".join(expanded.split())


class TantivyRawEngine(BaseSearchEngine):
    """方案 2A: 纯粹的 Tantivy 原生检索引擎"""

    def __init__(self, index_dir: str = "data/index_2a_tantivy_raw"):
        self.index_dir = index_dir
        self._index = None
        self._searcher = None

    @property
    def name(self) -> str:
        return "tantivy_raw"

    def _ensure_dependency(self):
        if tantivy is None:
            raise ImportError("未检测到 tantivy 依赖库，请先执行: pip install tantivy")

    def _build_schema(self) -> Any:
        self._ensure_dependency()
        sb = tantivy.SchemaBuilder()
        sb.add_text_field("file_id", stored=True)
        sb.add_text_field("title", stored=True, tokenizer_name="default")
        sb.add_text_field("intent_content", stored=True, tokenizer_name="default")
        sb.add_text_field("symbol_exact", stored=True, tokenizer_name="raw")
        sb.add_text_field("symbol_text", stored=False, tokenizer_name="default")
        return sb.build()

    def build_index(self, corpus_db_path: str, force_rebuild: bool = False):
        self._ensure_dependency()
        index_meta_file = os.path.join(self.index_dir, "meta.json")
        if os.path.exists(index_meta_file) and not force_rebuild:
            schema = self._build_schema()
            self._index = tantivy.Index(schema, path=self.index_dir)
            self._searcher = self._index.searcher()
            return

        os.makedirs(self.index_dir, exist_ok=True)
        for f in Path(self.index_dir).glob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass

        schema = self._build_schema()
        self._index = tantivy.Index(schema, path=self.index_dir)
        writer = self._index.writer(heap_size=128 * 1024 * 1024)

        src_conn = sqlite3.connect(corpus_db_path)
        src_cur = src_conn.cursor()

        src_cur.execute("SELECT file_id, title FROM sessions;")
        sessions = src_cur.fetchall()

        for fid, title in sessions:
            src_cur.execute(
                "SELECT role, turn_index, text_content FROM intent_turns WHERE file_id = ? ORDER BY turn_index ASC;",
                (fid,)
            )
            turns = src_cur.fetchall()
            intent_lines = [f"[{r} #{idx}]: {txt}" for r, idx, txt in turns]
            raw_intent = "\n".join(intent_lines)

            cjk_intent = cjk_bigram_tokenize(raw_intent)
            cjk_title = cjk_bigram_tokenize(title or "")

            src_cur.execute(
                "SELECT symbol_type, symbol_text FROM context_symbols WHERE file_id = ?;",
                (fid,)
            )
            symbols = src_cur.fetchall()
            unique_symbols = {s[1].strip() for s in symbols if s[1].strip()}

            doc = tantivy.Document()
            doc.add_text("file_id", fid)
            doc.add_text("title", cjk_title)
            doc.add_text("intent_content", cjk_intent)

            for sym in unique_symbols:
                doc.add_text("symbol_exact", sym)
                if "/" in sym:
                    basename = sym.split("/")[-1]
                    if len(basename) >= 3:
                        doc.add_text("symbol_exact", basename)
                doc.add_text("symbol_text", sym.replace("/", " ").replace(".", " "))

            writer.add_document(doc)

        writer.commit()
        self._index.reload()
        self._searcher = self._index.searcher()
        src_conn.close()

    @classmethod
    def extract_safe_terms(cls, raw_query: str) -> list[str]:
        if not raw_query:
            return []

        lines = raw_query.splitlines()
        clean_words = []
        for line in lines:
            line_str = line.strip()
            if DIFF_NOISE_PATTERN.match(line_str) and len(line_str) <= 8:
                continue
            cjk_line = cjk_bigram_tokenize(line_str)
            words = CJK_PUNCTUATION_PATTERN.split(cjk_line)
            for w in words:
                w_str = w.strip()
                if len(w_str) >= 2:
                    sanitized = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fa5]', '', w_str)
                    if len(sanitized) >= 2:
                        clean_words.append(sanitized)
        return clean_words

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        self._ensure_dependency()
        if not self._searcher:
            self._searcher = self._index.searcher()

        clean_q = query.strip()
        if not clean_q:
            return []

        has_cjk_chars = bool(re.search(r"[\u4e00-\u9fa5]", clean_q))
        has_cjk_punct = bool(re.search(r"[。？！，、；：“”‘’（）《》【】…—～]", clean_q))

        is_path_query = (
            not has_cjk_chars
            and not has_cjk_punct
            and not ("\n" in clean_q)
            and ("/" in clean_q or clean_q.startswith("."))
            and len(clean_q.split()) <= 2
            and (
                bool(re.search(r"\.[a-zA-Z0-9_\-]+$", clean_q))
                or clean_q.endswith("/")
                or clean_q.startswith(("./", "../", "/"))
            )
        )

        if is_path_query:
            escaped_path = clean_q.replace('"', '\\"')
            basename = clean_q.split("/")[-1].replace('"', '\\"')
            path_query_str = (
                f'symbol_exact:"{escaped_path}"^25.0 '
                f'OR symbol_exact:"{basename}"^12.0 '
                f'OR symbol_text:({clean_q.replace("/", " ").replace(".", " ")})^3.0'
            )
            try:
                parsed_path_q = self._index.parse_query(
                    path_query_str,
                    default_field_names=["symbol_exact", "symbol_text"]
                )
                res = self._searcher.search(parsed_path_q, limit)
                if res.hits:
                    hits = []
                    for score, doc_address in res.hits:
                        doc = self._searcher.doc(doc_address)
                        hits.append(SearchHit(
                            file_id=doc.get_first("file_id"),
                            score=round(score, 3),
                            title=doc.get_first("title") or "Untitled",
                            highlighted_snippet=html.escape(doc.get_first("intent_content")[:200]),
                            sub_scores={"path_score": round(score, 3)},
                        ))
                    return hits
            except Exception:
                pass

        terms = self.extract_safe_terms(clean_q)
        if not terms:
            return []

        query_terms_str = " ".join(terms[:15])
        safe_query_expr = f"intent_content:({query_terms_str})^10.0 OR title:({query_terms_str})^2.5 OR symbol_text:({query_terms_str})^2.0"

        try:
            parsed_query = self._index.parse_query(
                safe_query_expr,
                default_field_names=["intent_content", "title", "symbol_text"]
            )
        except Exception:
            try:
                parsed_query = self._index.parse_query(
                    query_terms_str,
                    default_field_names=["intent_content", "title"]
                )
            except Exception:
                return []

        search_result = self._searcher.search(parsed_query, limit)
        hits = []

        for score, doc_address in search_result.hits:
            doc = self._searcher.doc(doc_address)
            preview = (doc.get_first("intent_content") or "")[:220].replace("\n", " ").strip()
            hits.append(SearchHit(
                file_id=doc.get_first("file_id"),
                score=round(score, 3),
                title=doc.get_first("title") or "Untitled",
                highlighted_snippet=html.escape(preview),
                sub_scores={"tantivy_score": round(score, 3)},
            ))

        return hits

    def get_index_size_mb(self) -> float:
        if not os.path.exists(self.index_dir):
            return 0.0
        total_bytes = sum(f.stat().st_size for f in Path(self.index_dir).glob("*") if f.is_file())
        return total_bytes / (1024 * 1024)