#!/usr/bin/env python3
"""
Palimpsest - 搜索算法量化评测与白盒诊断执行器 (Search IR Evaluator & Diagnostic Inspector)
用法:
  # 常规评测
  python benchmarks/evaluator.py --engine tantivy_rescore
  python benchmarks/evaluator.py --engine all
  # 白盒深度诊断特定 Case
  python benchmarks/evaluator.py --engine tantivy_rescore --explain TC-001-REGRESSION
  python benchmarks/evaluator.py --engine tantivy_rescore --explain TC-002-REGRESSION
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

# 注入项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engines import (
    BaseSearchEngine,
    ENGINE_REGISTRY,
    create_engine,
    list_registered_engines,
)


def inspect_corpus_target(corpus_db: str, target_fid: str, query: str) -> dict:
    """直接探测提纯语料库，查证目标会话的物理原始文本与子串匹配情况"""
    if not os.path.exists(corpus_db):
        return {"error": f"Corpus DB not found: {corpus_db}"}

    conn = sqlite3.connect(corpus_db)
    cur = conn.cursor()

    cur.execute("SELECT title, modified_time FROM sessions WHERE file_id = ?;", (target_fid,))
    s_row = cur.fetchone()
    if not s_row:
        conn.close()
        return {"exists_in_corpus": False}

    title, mtime = s_row
    cur.execute(
        "SELECT role, turn_index, text_content FROM intent_turns WHERE file_id = ? ORDER BY turn_index ASC;",
        (target_fid,)
    )
    turns = cur.fetchall()
    conn.close()

    full_texts = [f"[{r} #{idx}]: {txt}" for r, idx, txt in turns]
    joined_text = "\n".join(full_texts)

    clean_q = query.strip()
    clean_q_oneline = " ".join(clean_q.split())

    # 探测子串
    exact_in_corpus = clean_q in joined_text or clean_q_oneline in joined_text
    exact_in_lower = clean_q.lower() in joined_text.lower() or clean_q_oneline.lower() in joined_text.lower()

    # 探测具体发生在哪个 turn
    matched_turns = []
    for r, idx, txt in turns:
        if clean_q[:20] in txt or clean_q_oneline[:20] in txt:
            matched_turns.append((idx, r, txt[:160].replace("\n", " ")))

    return {
        "exists_in_corpus": True,
        "title": title,
        "modified_time": mtime,
        "turn_count": len(turns),
        "total_char_len": len(joined_text),
        "exact_in_corpus": exact_in_corpus,
        "exact_in_lower": exact_in_lower,
        "matched_turns_sample": matched_turns,
        "preview_head": joined_text[:300].replace("\n", " "),
    }


def explain_case(
    engine: BaseSearchEngine,
    case_id: str,
    queries_file: str = "benchmarks/queries.jsonl",
    corpus_db: str = "data/purified_corpus.db",
):
    """深度白盒诊断特定用例"""
    # 1. 查找用例
    target_case = None
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item["case_id"] == case_id:
                    target_case = item
                    break

    if not target_case:
        print(f"❌ 未找到用例 ID: '{case_id}'")
        return

    query = target_case["query"]
    target_fid = target_case["target_file_id"]
    target_title = target_case.get("target_title", "目标会话")
    category = target_case["category"]

    print("\n" + "=" * 90)
    print(f"🔬 [Palimpsest 白盒诊断透视] Case: {case_id}")
    print("=" * 90)
    print(f"📋 评测元信息:")
    print(f"   • 场景分类: {category}")
    print(f"   • 评测描述: {target_case.get('description', '无')}")
    print(f"   • 查询内容: {query!r}")
    print(f"   • 预期目标: {target_title} (ID: {target_fid})")
    print(f"   • 评测引擎: [{engine.name}]")

    # 确保索引构建
    engine.build_index(corpus_db_path=corpus_db)

    # 2. 检查引擎是否支持 trace
    if hasattr(engine, "search_with_trace"):
        hits, trace = engine.search_with_trace(query=query, limit=10)
    else:
        hits = engine.search(query=query, limit=10)
        trace = {"compiler": engine.explain_query(query), "all_scored_candidates": []}

    compiler = trace.get("compiler", {})

    print("\n🛠️  【Query 编译器剖析 (Query Compiler)】:")
    if "terms" in compiler:
        print(f"   • 提取有效词元 ({len(compiler['terms'])} 个): {compiler['terms']}")
    if "phrase_expression" in compiler:
        print(f"   • 精确短语表达式:    {compiler['phrase_expression']!r}")
    if "full_query_dsl" in compiler:
        print(f"   • 送入 Tantivy DSL:  {compiler['full_query_dsl']}")
    elif "path_query_dsl" in compiler:
        print(f"   • 路径专用 DSL:      {compiler['path_query_dsl']}")

    # 3. 候选池大盘
    all_scored = trace.get("all_scored_candidates", [])
    print(f"\n📊 【粗召回候选池重排大盘 (展示前 8 名，共召回 {len(all_scored)} 项)】:")
    print(f"{'Rank':<4} | {'总分':<8} | {'BM25':<7} | {'粗排':<5} | {'覆盖%':<6} | {'子串加权':<8} | {'代码加权':<8} | {'标题 (Title)':<24} | {'File ID'}")
    print("-" * 95)

    target_in_candidates = None
    target_rank = None

    for rank, cand in enumerate(all_scored, start=1):
        is_target = (cand["file_id"] == target_fid)
        if is_target:
            target_in_candidates = cand
            target_rank = rank

        if rank <= 8 or is_target:
            mark = "🎯 " if is_target else "   "
            title_display = cand["title"][:22]
            raw_coarse = cand.get("raw_rank_in_coarse", rank)
            hit_cnt = cand.get("hit_terms_count", "N/A")
            ex_boost = cand.get("exact_boost", 0.0)
            cd_boost = cand.get("code_boost", 0.0)
            print(
                f"{mark}#{rank:<2} | {cand['final_score']:<8.2f} | {cand['tantivy_raw_bm25']:<7.2f} | "
                f"#{raw_coarse:<4} | {hit_cnt:<6} | "
                f"{ex_boost:<8.1f} | {cd_boost:<8.1f} | "
                f"{title_display:<22} | {cand['file_id'][:8]}"
            )

    print("-" * 95)
    if not target_in_candidates:
        print(f"⚠️  警告: 目标文档 ({target_fid}) 完全没有进入 Tantivy 的粗排前 30 名候选池！")

    # 4. Top-1 vs Target 深度逐项对决
    top1_cand = all_scored[0] if all_scored else None

    if top1_cand and target_in_candidates:
        print("\n⚔️  【Top-1 误召回 vs 预期目标文档 深度对决】:")
        print(f"{'维度指标':<22} | {'Top-1 候选 (误召回)':<32} | {'Target 预期目标 (落后项)':<32}")
        print("-" * 90)
        print(f"{'会话标题':<22} | {top1_cand['title'][:30]:<32} | {target_in_candidates['title'][:30]:<32}")
        print(f"{'物理 File ID':<20} | {top1_cand['file_id']:<32} | {target_in_candidates['file_id']:<32}")
        print(f"{'最终排名 (Rank)':<20} | Rank #1{'':<25} | Rank #{target_rank}{'':<25}")
        diff_score = top1_cand['final_score'] - target_in_candidates['final_score']
        print(f"{'综合总分 (Score)':<20} | {top1_cand['final_score']:<32.2f} | {target_in_candidates['final_score']:<32.2f} (落后 {diff_score:.2f})")
        print(f"{'Tantivy BM25 原生分':<19} | {top1_cand['tantivy_raw_bm25']:<32.2f} | {target_in_candidates['tantivy_raw_bm25']:<32.2f}")
        print(f"{'粗排原始位置':<22} | 第 #{top1_cand['raw_rank_in_coarse']} 名{'':<25} | 第 #{target_in_candidates['raw_rank_in_coarse']} 名{'':<25}")
        print(f"{'词项覆盖命中':<22} | {top1_cand['hit_terms_count']:<32} | {target_in_candidates['hit_terms_count']:<32}")
        print(f"{'覆盖率惩罚/乘数':<20} | x{top1_cand['coverage_multiplier']:<31.2f} | x{target_in_candidates['coverage_multiplier']:<31.2f}")
        print(f"{'子串判定结果':<22} | {'命中 (+60)' if top1_cand['exact_matched'] else '未命中 (+0)' :<32} | {'命中 (+60)' if target_in_candidates['exact_matched'] else '未命中 (+0)':<32}")
        print(f"{'子串匹配原因':<22} | {top1_cand['exact_reason'][:30]:<32} | {target_in_candidates['exact_reason'][:30]:<32}")
        print("-" * 90)

    # 5. 深入语料库真实追溯 (Ground-Truth Corpus Inspection)
    print("\n🔍 【底层提纯语料库 (purified_corpus.db) 物理追溯】:")
    corpus_info = inspect_corpus_target(corpus_db, target_fid, query)
    if corpus_info.get("exists_in_corpus"):
        print(f"   • 数据库状态:     已确认存在 (总交互轮次: {corpus_info['turn_count']} 轮 | 全文总长: {corpus_info['total_char_len']:,} 字符)")
        print(f"   • 物理子串严格匹配: {'✅ 完全存在 (包含原句)' if corpus_info['exact_in_corpus'] else '❌ 严格不存在'}")
        print(f"   • 忽略大小写子串:   {'✅ 存在 (Case-insensitive)' if corpus_info['exact_in_lower'] else '❌ 不存在'}")
        if corpus_info["matched_turns_sample"]:
            print(f"   • 原文中命中的具体轮次:")
            for tidx, role, sample in corpus_info["matched_turns_sample"]:
                print(f"     -> [Turn #{tidx} ({role})]: {sample}")
        else:
            print(f"   • ⚠️  警告: 在该文档的全部历史轮次中，未搜寻到与查询词前 20 字符一致的片段！")
            print(f"     会话头部预览: {corpus_info['preview_head']}")
    else:
        print(f"   • ❌ 错误: 目标 File ID {target_fid} 竟然不存在于语料数据库中！")

    print("=" * 90 + "\n")


def run_benchmark(
    engine: BaseSearchEngine,
    queries_file: str = "benchmarks/queries.jsonl",
    corpus_db: str = "data/purified_corpus.db",
    verbose: bool = False,
) -> dict:
    if not os.path.exists(queries_file):
        raise FileNotFoundError(f"未找到黄金测试集: {queries_file}，请先执行 seed_ground_truth.py")

    print("=" * 85)
    print(f"📊 正在启动 Palimpsest 算法评测: [{engine.name}]")
    print("=" * 85)

    t_idx0 = time.perf_counter()
    print("⚙️  检查索引状态...")
    engine.build_index(corpus_db_path=corpus_db)
    t_idx = time.perf_counter() - t_idx0
    idx_size_mb = engine.get_index_size_mb()
    print(f"✅ 索引就绪 (耗时: {t_idx:.2f}s | 磁盘占用: {idx_size_mb:.1f} MB)\n")

    cases = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    results_by_cat = defaultdict(list)
    bad_cases = []
    all_latencies = []

    for idx, tc in enumerate(cases, start=1):
        case_id = tc["case_id"]
        cat = tc["category"]
        query = tc["query"]
        target_fid = tc["target_file_id"]

        t0 = time.perf_counter()
        hits = engine.search(query=query, limit=10)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        all_latencies.append(dur_ms)

        hit_ids = [h.file_id for h in hits]
        target_rank = (hit_ids.index(target_fid) + 1) if target_fid in hit_ids else None

        hit_top1 = (target_rank == 1)
        hit_top5 = (target_rank is not None and target_rank <= 5)
        reciprocal_rank = (1.0 / target_rank) if (target_rank and target_rank <= 5) else 0.0

        item_result = {
            "case_id": case_id,
            "category": cat,
            "query": query,
            "target_file_id": target_fid,
            "target_rank": target_rank,
            "hit_top1": hit_top1,
            "hit_top5": hit_top5,
            "rr": reciprocal_rank,
            "duration_ms": dur_ms,
            "top1_title": hits[0].title if hits else "None",
            "hits_count": len(hits),
        }
        results_by_cat[cat].append(item_result)

        if not hit_top1:
            bad_cases.append((tc, target_rank, hits[0] if hits else None))

        if verbose:
            tag = "🎯 Top-1" if hit_top1 else ("✨ Top-5" if hit_top5 else "❌ Miss")
            rank_str = f"Rank #{target_rank}" if target_rank else "Unranked"
            print(f"[{idx:02d}/{len(cases):02d}] {tag} ({dur_ms:4.1f}ms) [{rank_str}] {case_id}: {query[:36]!r}")

    print("\n" + "=" * 90)
    print(f"📈 评测报告: 引擎 [{engine.name}]")
    print("=" * 90)
    print(f"{'测试场景分类':<34} | {'用例数':<5} | {'MRR@5':<8} | {'Top-1 命中率':<12} | {'Top-5 召回率':<12} | {'平均耗时':<8}")
    print("-" * 90)

    total_cases = len(cases)
    total_top1 = 0
    total_top5 = 0
    total_rr = 0.0

    for cat in sorted(results_by_cat.keys()):
        items = results_by_cat[cat]
        c_len = len(items)
        c_top1 = sum(1 for x in items if x["hit_top1"])
        c_top5 = sum(1 for x in items if x["hit_top5"])
        c_mrr = sum(x["rr"] for x in items) / c_len if c_len else 0
        c_avg_t = sum(x["duration_ms"] for x in items) / c_len if c_len else 0

        total_top1 += c_top1
        total_top5 += c_top5
        total_rr += sum(x["rr"] for x in items)

        p_top1 = (c_top1 / c_len * 100) if c_len else 0
        p_top5 = (c_top5 / c_len * 100) if c_len else 0

        print(f"{cat:<32} | {c_len:<5} | {c_mrr:<8.3f} | {p_top1:>5.1f}% ({c_top1}/{c_len:<2}) | {p_top5:>5.1f}% ({c_top5}/{c_len:<2}) | {c_avg_t:>6.1f}ms")

    print("-" * 90)
    overall_mrr = total_rr / total_cases if total_cases else 0
    overall_top1 = (total_top1 / total_cases * 100) if total_cases else 0
    overall_top5 = (total_top5 / total_cases * 100) if total_cases else 0
    avg_latency = sum(all_latencies) / total_cases if total_cases else 0
    all_latencies.sort()
    p95_latency = all_latencies[int(len(all_latencies) * 0.95)] if all_latencies else 0

    print(
        f"{'⚡ 全维度综合总计 (Overall)':<32} | {total_cases:<5} | {overall_mrr:<8.3f} | "
        f"{overall_top1:>5.1f}% ({total_top1}/{total_cases}) | {overall_top5:>5.1f}% ({total_top5}/{total_cases}) | {avg_latency:>6.1f}ms"
    )
    print(f"⏱️  延时统计: 平均: {avg_latency:.1f}ms | P95: {p95_latency:.1f}ms | 索引体积: {idx_size_mb:.1f} MB")
    print("=" * 90)

    if bad_cases:
        print("\n🔍 典型失分 / 翻车 Case 诊断 (展示前 3 例):")
        for tc, rank, top1_hit in bad_cases[:3]:
            rank_display = f"Rank #{rank}" if rank else "未能进入 Top 10"
            print(f"• [{tc['category']}] {tc['case_id']}")
            print(f"  查询词:   {tc['query']!r}")
            print(f"  预期目标: {tc.get('target_title', '')} ({tc['target_file_id'][:8]})")
            print(f"  实际结果: {rank_display}")
            if top1_hit:
                print(f"  Top-1 误召回: {top1_hit.title} (得分: {top1_hit.score:.2f})")
            print()

    return {
        "engine": engine.name,
        "mrr@5": round(overall_mrr, 3),
        "top1_pct": round(overall_top1, 1),
        "top5_pct": round(overall_top5, 1),
        "p95_ms": round(p95_latency, 1),
        "avg_ms": round(avg_latency, 1),
        "index_mb": round(idx_size_mb, 1),
    }


def print_comparison_table(summary_list: list[dict]):
    print("\n" + "=" * 92)
    print("🏆 Palimpsest 搜索引擎横向打擂总决表 (Leaderboard)")
    print("=" * 92)
    print(f"{'引擎代号 (Engine)':<22} | {'MRR@5':<8} | {'Top-1 命中率':<13} | {'Top-5 召回率':<13} | {'平均耗时':<9} | {'索引体积':<10}")
    print("-" * 92)
    for s in summary_list:
        print(
            f"{s['engine']:<22} | {s['mrr@5']:<8.3f} | {s['top1_pct']:>5.1f}%       | "
            f"{s['top5_pct']:>5.1f}%       | {s['avg_ms']:>6.1f}ms   | {s['index_mb']:>6.1f} MB"
        )
    print("=" * 92 + "\n")


def main():
    registered = list_registered_engines()
    parser = argparse.ArgumentParser(description="Palimpsest 检索评测执行器")
    parser.add_argument(
        "--engine",
        default="tantivy_rescore",
        help=f"选择待评测的搜索引擎: {registered}，或 'all'，或逗号分隔多选",
    )
    parser.add_argument("--queries", default="benchmarks/queries.jsonl", help="测试集文件路径")
    parser.add_argument("--corpus-db", default="data/purified_corpus.db", help="提纯语料库路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细打印每条用例的执行状态")
    parser.add_argument("--explain", default=None, help="白盒诊断指定 Case ID (如: TC-001-REGRESSION)")
    args = parser.parse_args()

    # 分支 1: 单 Case 深度白盒透视模式
    if args.explain:
        engine_name = args.engine.split(",")[0]
        if engine_name == "all":
            engine_name = "tantivy_rescore"
        engine = create_engine(engine_name)
        explain_case(
            engine=engine,
            case_id=args.explain,
            queries_file=args.queries,
            corpus_db=args.corpus_db,
        )
        return

    # 分支 2: 常规评测 / 全量打擂模式
    if args.engine == "all":
        target_names = registered
    else:
        target_names = [e.strip() for e in args.engine.split(",") if e.strip()]

    for name in target_names:
        if name not in registered:
            print(f"❌ 错误: 未知引擎 '{name}'，可选引擎: {registered}")
            sys.exit(1)

    summaries = []
    for name in target_names:
        engine = create_engine(name)
        res = run_benchmark(
            engine=engine,
            queries_file=args.queries,
            corpus_db=args.corpus_db,
            verbose=args.verbose,
        )
        summaries.append(res)

    if len(summaries) > 1:
        print_comparison_table(summaries)


if __name__ == "__main__":
    main()