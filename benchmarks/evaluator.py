#!/usr/bin/env python3
"""
Palimpsest - 搜索算法量化评测流水线 (Search IR Evaluator)
用法:
  python benchmarks/evaluator.py --engine baseline
  python benchmarks/evaluator.py --engine baseline --verbose
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# 注入项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engines.base import BaseSearchEngine
from engines.engine_0_baseline_fts import BaselineFtsEngine
from engines.engine_1_opt_sqlite import OptSqliteEngine
from engines.engine_2_tantivy import TantivyEngine
from engines.engine_3_hybrid_rerank import HybridRerankEngine


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

    # 1. 构建或确认索引就绪
    t_idx0 = time.perf_counter()
    print("⚙️  检查索引状态...")
    engine.build_index(corpus_db_path=corpus_db)
    t_idx = time.perf_counter() - t_idx0
    idx_size_mb = engine.get_index_size_mb()
    print(f"✅ 索引就绪 (耗时: {t_idx:.2f}s | 磁盘占用: {idx_size_mb:.1f} MB)\n")

    # 2. 读取测试用例
    cases = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"🧪 加载测试用例: 共 {len(cases)} 个 (涵盖 {len(set(c['category'] for c in cases))} 类典型场景)\n")

    results_by_cat = defaultdict(list)
    bad_cases = []
    all_latencies = []

    for idx, tc in enumerate(cases, start=1):
        case_id = tc["case_id"]
        cat = tc["category"]
        query = tc["query"]
        target_fid = tc["target_file_id"]
        target_title = tc.get("target_title", "目标会话")

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

    # 3. 统计指标大盘
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

    # 4. 诊断最严重的 Bad Case (重点暴露翻车 Case)
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
                if "candidate_expr" in top1_hit.sub_scores:
                    print(f"  降级表达式:   {top1_hit.sub_scores['candidate_expr']!r}")
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


def main():
    parser = argparse.ArgumentParser(description="Palimpsest 检索评测执行器")
    parser.add_argument("--engine", choices=["baseline", "opt_sqlite", "tantivy", "hybrid_rerank"], default="baseline", help="选择待评测的搜索引擎")
    parser.add_argument("--queries", default="benchmarks/queries.jsonl", help="测试集文件路径")
    parser.add_argument("--corpus-db", default="data/purified_corpus.db", help="提纯语料库路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细打印每条用例的执行状态")
    args = parser.parse_args()

    if args.engine == "baseline":
        engine = BaselineFtsEngine()
    elif args.engine == "opt_sqlite":
        engine = OptSqliteEngine()
    elif args.engine == "tantivy":
        engine = TantivyEngine()
    elif args.engine == "hybrid_rerank":
        engine = HybridRerankEngine()
    else:
        raise ValueError(f"未知的引擎类型: {args.engine}")

    run_benchmark(
        engine=engine,
        queries_file=args.queries,
        corpus_db=args.corpus_db,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()