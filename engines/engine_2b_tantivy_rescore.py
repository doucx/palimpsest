"""
Palimpsest - 方案 2B：Tantivy 两阶段短语与内存微重排引擎 (TantivyRescoreEngine v3.2)
修复重点:
  1. 扩充 raw_intent 上限至 85,000 字符，彻底解决 12 万字超长文档深轮次原句截断问题
  2. 子串匹配剔除通用模板标题蹭分，仅允许实质业务正文触发满额置顶 (+70.0)
  3. 补全路径分支诊断字段，彻底消灭 KeyError
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

# 递增 Schema 版本触发自动无缝重建
SCHEMA_VERSION = "3.2"

CJK_PUNCTUATION_PATTERN = re.compile(
    r"[。？！，、；：“”‘’（）《》【】…—～\s\t\r\n;,:'\"`()\[\]{}*#<>=_+\-|\\]+"
)
HAN_REGEX = re.compile(r"([\u4e00-\u9fa5]+)")
DIFF_NOISE_PATTERN = re.compile(r"^(\+{3}|\-{3}|@{2}|#{1,6}|`{3,})")
CAMEL_CASE_PATTERN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)")


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


class TantivyRescoreEngine(BaseSearchEngine):
    """方案 2B: Tantivy + 内存微重排双阶段引擎 (v3.2 稳固版)"""

    def __init__(self, index_dir: str = "data/index_2b_tantivy_rescore"):
        self.index_dir = index_dir
        self._index = None
        self._searcher = None

    @property
    def name(self) -> str:
        return "tantivy_rescore"

    def _ensure_dependency(self):
        if tantivy is None:
            raise ImportError("未检测到 tantivy 依赖库，请先执行: pip install tantivy")

    def _build_schema(self) -> Any:
        self._ensure_dependency()
        sb = tantivy.SchemaBuilder()
        sb.add_text_field("file_id", stored=True)
        sb.add_text_field("title", stored=True, tokenizer_name="default")
        sb.add_text_field("intent_content", stored=False, tokenizer_name="default")
        sb.add_text_field("raw_intent", stored=True, tokenizer_name="raw")
        sb.add_text_field("symbol_exact", stored=True, tokenizer_name="raw")
        sb.add_text_field("symbol_text", stored=False, tokenizer_name="default")
        return sb.build()

    def _check_version_match(self) -> bool:
        ver_file = os.path.join(self.index_dir, "schema_version.txt")
        if not os.path.exists(ver_file):
            return False
        try:
            with open(ver_file, "r", encoding="utf-8") as f:
                return f.read().strip() == SCHEMA_VERSION
        except OSError:
            return False

    def build_index(self, corpus_db_path: str, force_rebuild: bool = False):
        self._ensure_dependency()
        index_meta_file = os.path.join(self.index_dir, "meta.json")

        if os.path.exists(index_meta_file) and self._check_version_match() and not force_rebuild:
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
            # 扩容至 85,000 字符，轻松覆盖 12 万字超长文档后置轮次
            doc.add_text("raw_intent", raw_intent[:85000])

            for sym in unique_symbols:
                doc.add_text("symbol_exact", sym)
                if "/" in sym:
                    basename = sym.split("/")[-1]
                    if len(basename) >= 3:
                        doc.add_text("symbol_exact", basename)
                doc.add_text("symbol_text", sym.replace("/", " ").replace(".", " "))

            writer.add_document(doc)

        writer.commit()

        ver_file = os.path.join(self.index_dir, "schema_version.txt")
        with open(ver_file, "w", encoding="utf-8") as f:
            f.write(SCHEMA_VERSION)

        self._index.reload()
        self._searcher = self._index.searcher()
        src_conn.close()

    @classmethod
    def extract_safe_terms(cls, raw_query: str) -> list[str]:
        if not raw_query:
            return []

        lines = [l.strip() for l in raw_query.splitlines() if l.strip()]
        if not lines:
            return []

        clean_words = []
        for line_str in lines:
            if DIFF_NOISE_PATTERN.match(line_str) and len(line_str) <= 8:
                continue
            cjk_line = cjk_bigram_tokenize(line_str)
            words = CJK_PUNCTUATION_PATTERN.split(cjk_line)
            for w in words:
                w_str = w.strip()
                if len(w_str) >= 2:
                    sanitized = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fa5]', '', w_str)
                    if len(sanitized) >= 2 and sanitized not in clean_words:
                        clean_words.append(sanitized)
                if len(clean_words) >= 45:
                    break
            if len(clean_words) >= 45:
                break

        return clean_words

    @classmethod
    def build_phrase_expression(cls, raw_query: str) -> str | None:
        lines = [l.strip() for l in raw_query.splitlines() if l.strip()]
        if not lines:
            return None

        candidate_lines = []
        for l in lines:
            cleaned_l = re.sub(r"^(?:#{1,6}|[-*+>]|\d+\.)\s*", "", l).strip()
            cleaned_l = re.sub(r"(\*\*|__|\*|_|`)(.*?)\1", r"\2", cleaned_l).strip()
            if len(cleaned_l) >= 12 and not cleaned_l.startswith(("---", "+++", "@@")):
                candidate_lines.append(cleaned_l)

        if not candidate_lines:
            candidate_lines = [lines[0]]

        target_line = candidate_lines[0]
        for cl in candidate_lines:
            if re.search(r"[\u4e00-\u9fa5]", cl) and len(cl) >= 16:
                target_line = cl
                break

        clean_line = CJK_PUNCTUATION_PATTERN.sub(" ", target_line).strip()
        if not clean_line:
            return None

        cjk_expanded = cjk_bigram_tokenize(clean_line)
        tokens = cjk_expanded.split()
        if len(tokens) >= 2:
            safe = [re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fa5]", "", t) for t in tokens[:16]]
            safe = [t for t in safe if t]
            if len(safe) >= 2:
                return " ".join(safe)
        return None

    def explain_query(self, query: str) -> dict[str, Any]:
        terms = self.extract_safe_terms(query)
        phrase = self.build_phrase_expression(query)
        query_terms_str = " ".join(terms[:36])

        query_clauses = []
        if phrase:
            query_clauses.append(f'intent_content:"{phrase}"^25.0')
        query_clauses.append(f'intent_content:({query_terms_str})^6.0')
        query_clauses.append(f'title:({query_terms_str})^4.0')
        query_clauses.append(f'symbol_text:({query_terms_str})^2.0')

        return {
            "query": query,
            "terms": terms,
            "phrase_expression": phrase,
            "full_query_dsl": " OR ".join(query_clauses),
        }

    def search_with_trace(
        self,
        query: str,
        limit: int = 10,
    ) -> tuple[list[SearchHit], dict[str, Any]]:
        self._ensure_dependency()
        if not self._searcher:
            self._searcher = self._index.searcher()

        clean_q = query.strip()
        trace: dict[str, Any] = {
            "query": query,
            "is_path_query": False,
            "compiler": {},
            "raw_candidates_count": 0,
            "all_scored_candidates": [],
        }

        if not clean_q:
            return [], trace

        # =========================================================================
        # 分支 1: 路径与工程符号探测通道 (严格排他性防误伤)
        # =========================================================================
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
        trace["is_path_query"] = is_path_query

        if is_path_query:
            escaped_path = clean_q.replace('"', '\\"')
            basename = clean_q.split("/")[-1]
            stem_name = basename.split(".")[0]
            escaped_base = basename.replace('"', '\\"')

            camel_tokens = CAMEL_CASE_PATTERN.findall(stem_name)
            if not camel_tokens:
                camel_tokens = [stem_name]
            camel_expr = " ".join(camel_tokens)

            path_query_str = (
                f'symbol_exact:"{escaped_path}"^35.0 '
                f'OR symbol_exact:"{escaped_base}"^20.0 '
                f'OR title:({camel_expr})^15.0 '
                f'OR symbol_text:({clean_q.replace("/", " ").replace(".", " ")})^3.0'
            )
            trace["compiler"]["path_query_dsl"] = path_query_str
            try:
                parsed_path_q = self._index.parse_query(
                    path_query_str,
                    default_field_names=["symbol_exact", "title", "symbol_text"]
                )
                res = self._searcher.search(parsed_path_q, max(15, limit * 2))
                trace["raw_candidates_count"] = len(res.hits)

                scored_candidates = []
                for raw_rank, (score, doc_address) in enumerate(res.hits, start=1):
                    doc = self._searcher.doc(doc_address)
                    fid = doc.get_first("file_id")
                    title = doc.get_first("title") or "Untitled"
                    raw_preview = doc.get_first("raw_intent") or ""

                    title_boost = 0.0
                    title_lower = title.lower()

                    camel_hits = sum(1 for tok in camel_tokens if len(tok) >= 3 and tok.lower() in title_lower)
                    title_boost += camel_hits * 25.0

                    is_core_design = any(k in title_lower for k in ["设计", "方案", "架构", "实现", "规范", "spec", "design"])
                    if is_core_design and (camel_hits >= 1 or stem_name.lower() in raw_preview[:600].lower()):
                        title_boost += 60.0

                    final_score = float(score) + title_boost

                    # 补全全部字段，彻底杜绝 KeyError
                    scored_candidates.append({
                        "file_id": fid,
                        "title": title,
                        "final_score": round(final_score, 2),
                        "tantivy_raw_bm25": round(score, 2),
                        "raw_rank_in_coarse": raw_rank,
                        "hit_terms_count": f"{camel_hits}/{len(camel_tokens)}",
                        "exact_matched": False,
                        "exact_reason": "Path & Component focus",
                        "exact_boost": 0.0,
                        "code_boost": 0.0,
                        "title_boost": title_boost,
                        "coverage_ratio": 1.0,
                        "coverage_multiplier": 1.0,
                        "title_alignment": title_boost,
                        "preview": raw_preview[:200],
                    })

                scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
                trace["all_scored_candidates"] = scored_candidates

                hits = []
                for item in scored_candidates[:limit]:
                    hits.append(SearchHit(
                        file_id=item["file_id"],
                        score=item["final_score"],
                        title=item["title"],
                        highlighted_snippet=html.escape(item["preview"]),
                        sub_scores={"path_score": item["tantivy_raw_bm25"], "title_boost": item["title_boost"]},
                        debug_info=item,
                    ))
                return hits, trace
            except Exception as e:
                trace["compiler"]["error"] = str(e)

        # =========================================================================
        # 分支 2: 阶段 1 - 多级 Tantivy 倒排粗召回 (Top 35)
        # =========================================================================
        compiler_info = self.explain_query(clean_q)
        trace["compiler"] = compiler_info
        terms = compiler_info["terms"]
        if not terms:
            return [], trace

        full_query_str = compiler_info["full_query_dsl"]
        try:
            parsed_query = self._index.parse_query(
                full_query_str,
                default_field_names=["intent_content", "title", "symbol_text"]
            )
        except Exception:
            query_terms_str = " ".join(terms[:36])
            parsed_query = self._index.parse_query(
                query_terms_str,
                default_field_names=["intent_content", "title"]
            )

        fetch_limit = max(35, limit * 3)
        search_result = self._searcher.search(parsed_query, fetch_limit)
        trace["raw_candidates_count"] = len(search_result.hits)
        if not search_result.hits:
            return [], trace

        # =========================================================================
        # 阶段 2: 内存极速微重排 (杜绝标题蹭分 + 扩大文本视野)
        # =========================================================================
        normalized_q = re.sub(r"\s+", " ", clean_q).strip().lower()
        q_lines = [l.strip().lower() for l in clean_q.splitlines() if len(l.strip()) >= 4]

        distinct_terms = [t.lower() for t in set(terms) if len(t) >= 2]
        total_terms = max(1, len(distinct_terms))

        scored_candidates = []

        for raw_rank, (score, doc_address) in enumerate(search_result.hits, start=1):
            doc = self._searcher.doc(doc_address)
            fid = doc.get_first("file_id")
            title = doc.get_first("title") or "Untitled"
            raw_text = doc.get_first("raw_intent") or ""
            raw_lower = raw_text.lower()
            title_lower = title.lower()

            base_score = float(score)

            # A. 原生子串完全包含 (剔除通用标题蹭分)
            exact_matched = False
            exact_reason = "No substring match"
            if len(normalized_q) >= 12 and (normalized_q in raw_lower or normalized_q in title_lower):
                exact_matched = True
                exact_reason = "Full query substring matched"
            elif q_lines:
                # 寻找第一条实质性的长正文行 (剥离列表标记与通用章节大纲)
                substantive_line = None
                for l in q_lines:
                    cleaned_line = re.sub(r"^(?:#{1,6}|[-*+>]|\d+\.)\s*", "", l).strip()
                    cleaned_line = re.sub(r"(\*\*|__|\*|_|`)(.*?)\1", r"\2", cleaned_line).strip()
                    # 排除诸如 "第 1 章：动机与场景故事" 这种无业务特征的通用大纲行
                    is_generic_header = bool(re.search(r"^(第\s*\d+\s*章|motivation|background|overview)", cleaned_line))
                    if not is_generic_header and len(cleaned_line) >= 16:
                        substantive_line = cleaned_line
                        break

                if substantive_line and substantive_line in raw_lower:
                    exact_matched = True
                    exact_reason = f"Substantive line matched: '{substantive_line[:30]}...'"

            exact_boost = 70.0 if exact_matched else 0.0

            # B. 词项覆盖率门槛
            hit_terms_list = [t for t in distinct_terms if (t in raw_lower or t in title_lower)]
            hit_terms = len(hit_terms_list)
            coverage_ratio = hit_terms / total_terms

            if total_terms >= 4 and coverage_ratio < 0.35 and not exact_matched:
                coverage_multiplier = 0.2
            elif coverage_ratio >= 0.75:
                coverage_multiplier = 1.35
            else:
                coverage_multiplier = max(0.5, coverage_ratio)

            # C. 代码块结构保真度累进加权
            code_line_hits = 0
            total_matched_chars = 0
            if len(q_lines) >= 2:
                for line in q_lines:
                    core_code = line.strip(" ;{}()")
                    if len(core_code) >= 4 and core_code in raw_lower:
                        code_line_hits += 1
                        total_matched_chars += len(core_code)
                code_boost = (code_line_hits * 12.0) + (total_matched_chars * 0.5)
            else:
                code_boost = 0.0

            # D. 主题词聚焦微调
            title_alignment = 0.0
            for t in distinct_terms:
                if len(t) >= 3 and t in title_lower:
                    title_alignment += 5.0

            final_score = (base_score * coverage_multiplier) + exact_boost + code_boost + title_alignment
            preview = raw_text[:240].replace("\n", " ").strip()

            candidate_record = {
                "file_id": fid,
                "title": title,
                "final_score": round(final_score, 2),
                "tantivy_raw_bm25": round(base_score, 2),
                "raw_rank_in_coarse": raw_rank,
                "coverage_ratio": round(coverage_ratio, 2),
                "coverage_multiplier": coverage_multiplier,
                "hit_terms_count": f"{hit_terms}/{total_terms}",
                "exact_matched": exact_matched,
                "exact_reason": exact_reason,
                "exact_boost": exact_boost,
                "code_boost": round(code_boost, 2),
                "title_alignment": title_alignment,
                "preview": preview,
            }
            scored_candidates.append(candidate_record)

        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        trace["all_scored_candidates"] = scored_candidates

        hits = []
        for item in scored_candidates[:limit]:
            hits.append(SearchHit(
                file_id=item["file_id"],
                score=item["final_score"],
                title=item["title"],
                highlighted_snippet=html.escape(item["preview"]),
                sub_scores={
                    "tantivy_bm25": item["tantivy_raw_bm25"],
                    "coverage": item["coverage_ratio"],
                    "exact_boost": item["exact_boost"],
                    "code_boost": item["code_boost"],
                    "title_align": item["title_alignment"],
                },
                debug_info=item,
            ))

        return hits, trace

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        hits, _ = self.search_with_trace(query=query, limit=limit)
        return hits

    def get_index_size_mb(self) -> float:
        if not os.path.exists(self.index_dir):
            return 0.0
        total_bytes = sum(f.stat().st_size for f in Path(self.index_dir).glob("*") if f.is_file())
        return total_bytes / (1024 * 1024)